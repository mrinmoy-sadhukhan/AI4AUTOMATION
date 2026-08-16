#!/usr/bin/env python3
"""Two-adapter NL2PLC inference with bounded deterministic PLAN self-repair."""

from __future__ import annotations

import argparse
import gc
import json
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from nl2plc_common import (
    PLAN_SCHEMA_VERSION,
    canonical_plan_for_record,
    expected_construct_families,
    extract_instruction,
    extract_instruction_numbers,
    extract_plc_addresses,
    extract_reference_st,
    parse_json_object,
    read_jsonl,
    strip_outer_markdown_fence,
    summarize_issues,
    validate_address_contract,
    validate_plan,
    validate_st_structure,
    write_jsonl,
)
from two_adapter_prompts import (
    make_code_messages,
    make_plan_messages,
    make_repair_plan_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("two_adapter", "planner_only", "oracle_plan"),
        default="two_adapter",
    )
    parser.add_argument("--planner-adapter", default=None)
    parser.add_argument("--coder-adapter", default=None)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--submission-output",
        type=Path,
        default=None,
        help="Optional compact JSONL with id, language, solution_plan, and code.",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--plan-max-new-tokens", type=int, default=1400)
    parser.add_argument("--code-max-new-tokens", type=int, default=2200)
    parser.add_argument(
        "--plan-repair-attempts",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
    )
    parser.add_argument(
        "--code-retry-attempts",
        type=int,
        default=2,
        choices=(0, 1, 2, 3, 4),
        help="Regenerate invalid ST with deterministic errors; CODE SFT remains CODE-only.",
    )
    parser.add_argument(
        "--repair-temperature",
        type=float,
        default=0.20,
        help="Sampling temperature for repair calls only; initial calls remain deterministic.",
    )
    parser.add_argument("--repair-top-p", type=float, default=0.90)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.mode in {"two_adapter", "planner_only"} and not args.planner_adapter:
        raise ValueError(f"--planner-adapter is required for mode={args.mode}")
    if args.mode in {"two_adapter", "oracle_plan"} and not args.coder_adapter:
        raise ValueError(f"--coder-adapter is required for mode={args.mode}")
    for name in (
        "max_seq_length",
        "plan_max_new_tokens",
        "code_max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.repair_temperature < 0:
        raise ValueError("--repair-temperature cannot be negative")
    if not 0 < args.repair_top_p <= 1:
        raise ValueError("--repair-top-p must be in (0, 1]")


def local_adapter_base(adapter: str | None) -> str | None:
    if not adapter:
        return None
    config_path = Path(adapter) / "adapter_config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    value = payload.get("base_model_name_or_path")
    return str(value) if value else None


def assert_same_base(planner_adapter: str | None, coder_adapter: str | None) -> None:
    planner_base = local_adapter_base(planner_adapter)
    coder_base = local_adapter_base(coder_adapter)
    if planner_base and coder_base and planner_base != coder_base:
        raise ValueError(
            "Planner and coder adapters use different bases: "
            f"{planner_base!r} != {coder_base!r}"
        )


def normalized(text: str) -> str:
    return " ".join(text.split())


def compact_plan_steps(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict):
        return []
    steps = plan.get("ordered_logic")
    if not isinstance(steps, list):
        return []
    output: list[str] = []
    for item in steps:
        if isinstance(item, dict) and str(item.get("behavior", "")).strip():
            output.append(str(item["behavior"]).strip())
    return output


def load_generator(
    model_path: str,
    max_seq_length: int,
    seed: int,
) -> tuple[Any, Any, Callable[..., str]]:
    from unsloth import FastLanguageModel

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(
        messages: list[dict[str, str]],
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 0.90,
    ) -> str:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        if input_ids.shape[-1] + max_new_tokens > max_seq_length:
            raise ValueError(
                f"prompt ({input_ids.shape[-1]}) + output budget "
                f"({max_new_tokens}) exceeds max sequence length "
                f"({max_seq_length})"
            )
        attention_mask = torch.ones_like(input_ids)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            output = model.generate(**generation_kwargs)
        return tokenizer.decode(
            output[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()

    return model, tokenizer, generate


def unload_model(model: Any, tokenizer: Any) -> None:
    import torch

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def parse_and_check_plan(
    raw: str,
    instruction: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        plan = parse_json_object(raw)
    except ValueError as exc:
        return None, {
            "passed": False,
            "issues": [f"invalid JSON: {exc}"],
            "missing_addresses": [],
            "extra_addresses": [],
            "missing_literals": [],
            "extra_literals": [],
        }
    return plan, validate_plan(plan, instruction)


def _fallback_timer_type(instruction: str) -> str:
    lower = instruction.lower()
    if any(token in lower for token in ("off-delay", "off delay", "delay off")):
        return "TOF"
    if any(token in lower for token in ("pulse timer", "timed pulse", "one-shot pulse")):
        return "TP"
    return "TON"


def _fallback_counter_type(instruction: str) -> str:
    lower = instruction.lower()
    has_up = any(token in lower for token in ("increment", "count up", "up counter"))
    has_down = any(token in lower for token in ("decrement", "count down", "down counter"))
    if has_up and has_down:
        return "CTUD"
    if has_down:
        return "CTD"
    return "CTU"


def build_deterministic_fallback_plan(
    instruction: str,
    addresses: list[str] | None = None,
) -> dict[str, Any]:
    """Build a conservative, validator-valid plan without using test answers."""
    addresses = addresses if addresses is not None else extract_plc_addresses(instruction)
    hints = expected_construct_families(instruction)
    required_blocks: list[dict[str, str]] = []
    if "timer" in hints:
        required_blocks.append(
            {
                "type": _fallback_timer_type(instruction),
                "purpose": "Implement the duration and timing behavior stated in the instruction.",
            }
        )
    if "counter" in hints:
        required_blocks.append(
            {
                "type": _fallback_counter_type(instruction),
                "purpose": "Implement the counting behavior and threshold stated in the instruction.",
            }
        )
    if "rising_edge" in hints:
        required_blocks.append(
            {"type": "R_TRIG", "purpose": "Detect the requested rising edge."}
        )
    if "falling_edge" in hints:
        required_blocks.append(
            {"type": "F_TRIG", "purpose": "Detect the requested falling edge."}
        )

    io_contract = []
    for address in addresses:
        upper = address.upper()
        io_contract.append(
            {
                "address": address,
                "data_type": "BOOL" if upper.startswith(("%IX", "%QX")) else "INT",
                "role": "input" if upper.startswith("%I") else "output",
                "purpose": "Use this signal exactly as specified in the human instruction.",
            }
        )

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "objective": "Implement the complete human PLC instruction without changing its values or addresses.",
        "io_contract": io_contract,
        "literal_contract": extract_instruction_numbers(instruction),
        "required_blocks": required_blocks,
        "ordered_logic": [
            {"step": 1, "behavior": "Read all declared input signals once per scan."},
            {
                "step": 2,
                "behavior": "Implement every condition, value, timing rule, count, priority, and reset behavior in the human instruction.",
            },
            {"step": 3, "behavior": "Write every declared output from the resulting control state."},
        ],
        "priority_and_safety": [
            "Apply every explicit stop, reset, interlock, alarm, and priority condition before normal output behavior."
        ],
        "state_and_memory": "Retain only state required by timers, counters, latches, and edge detection in the instruction.",
        "reset_behavior": "Apply all reset and disable conditions exactly as stated in the instruction.",
        "scan_cycle_behavior": "Evaluate the behavior once per PLC scan while preserving required state between scans.",
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    assert_same_base(args.planner_adapter, args.coder_adapter)

    records = read_jsonl(args.input_jsonl)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    work: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        source_id = str(record.get("id", f"row_{index:04d}"))
        instruction = extract_instruction(record)
        work.append(
            {
                "id": source_id,
                "source_row": record.get("source_row"),
                "instruction": instruction,
                "addresses": extract_plc_addresses(instruction),
                "reference_st": extract_reference_st(record),
                "_record": record,
                "plan": None,
                "plan_check": {"passed": False, "issues": ["not generated"]},
                "initial_plan_passed": False,
                "plan_repair_attempts_used": 0,
                "plan_fallback_used": False,
                "plan_history": [],
            }
        )

    if args.mode == "oracle_plan":
        for item in work:
            plan = canonical_plan_for_record(item["_record"])
            check = validate_plan(plan, item["instruction"])
            item["plan"] = plan
            item["plan_check"] = check
            item["initial_plan_passed"] = check["passed"]
            item["plan_history"].append(
                {
                    "task": "ORACLE_PLAN",
                    "attempt": 0,
                    "parsed_plan": plan,
                    "check": check,
                }
            )
    else:
        model, tokenizer, generate_plan = load_generator(
            str(args.planner_adapter),
            args.max_seq_length,
            args.seed,
        )
        for index, item in enumerate(work, start=1):
            raw = generate_plan(
                make_plan_messages(item["instruction"], item["addresses"]),
                args.plan_max_new_tokens,
            )
            plan, check = parse_and_check_plan(raw, item["instruction"])
            item["plan_history"].append(
                {
                    "task": "PLAN",
                    "attempt": 0,
                    "raw_output": raw,
                    "parsed_plan": plan,
                    "check": check,
                }
            )
            item["initial_plan_passed"] = check["passed"]
            candidate_text: str | dict[str, Any] = plan if plan is not None else raw

            for repair_index in range(1, args.plan_repair_attempts + 1):
                if check["passed"]:
                    break
                errors = summarize_issues(check)
                repair_raw = generate_plan(
                    make_repair_plan_messages(
                        item["instruction"],
                        item["addresses"],
                        candidate_text,
                        errors,
                    ),
                    args.plan_max_new_tokens,
                    temperature=args.repair_temperature,
                    top_p=args.repair_top_p,
                )
                plan, check = parse_and_check_plan(
                    repair_raw,
                    item["instruction"],
                )
                item["plan_history"].append(
                    {
                        "task": "REPAIR_PLAN",
                        "attempt": repair_index,
                        "validator_errors_given": errors,
                        "raw_output": repair_raw,
                        "parsed_plan": plan,
                        "check": check,
                    }
                )
                item["plan_repair_attempts_used"] = repair_index
                candidate_text = plan if plan is not None else repair_raw

            if not check["passed"]:
                fallback = build_deterministic_fallback_plan(
                    item["instruction"], item["addresses"]
                )
                fallback_check = validate_plan(fallback, item["instruction"])
                item["plan_history"].append(
                    {
                        "task": "DETERMINISTIC_PLAN_FALLBACK",
                        "attempt": len(item["plan_history"]),
                        "parsed_plan": fallback,
                        "check": fallback_check,
                    }
                )
                if not fallback_check["passed"]:
                    raise RuntimeError(
                        f"{item['id']}: internal fallback plan failed validation: "
                        + "; ".join(fallback_check.get("issues", []))
                    )
                plan, check = fallback, fallback_check
                item["plan_fallback_used"] = True

            item["plan"] = plan
            item["plan_check"] = check
            print(
                f"[PLAN {index}/{len(work)}] {item['id']} "
                f"initial={item['initial_plan_passed']} "
                f"final={check['passed']} "
                f"repairs={item['plan_repair_attempts_used']} "
                f"fallback={item['plan_fallback_used']}"
            )
        unload_model(model, tokenizer)

    plan_phase_path = args.output_jsonl.with_suffix(".plans.jsonl")
    write_jsonl(
        plan_phase_path,
        [
            {key: value for key, value in item.items() if key != "_record"}
            for item in work
        ],
    )

    if args.mode == "planner_only":
        results = []
        for item in work:
            results.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"_record", "reference_st"}
                }
            )
        write_jsonl(args.output_jsonl, results)
        final_passed = [bool(row["plan_check"]["passed"]) for row in work]
        metrics = {
            "mode": args.mode,
            "examples": len(work),
            "initial_plan_pass_rate": mean(
                bool(row["initial_plan_passed"]) for row in work
            )
            if work
            else None,
            "final_plan_pass_rate": mean(final_passed) if final_passed else None,
            "repair_attempted_count": sum(
                row["plan_repair_attempts_used"] > 0 for row in work
            ),
            "repair_success_count": sum(
                not row["initial_plan_passed"]
                and row["plan_check"]["passed"]
                and not row.get("plan_fallback_used")
                for row in work
            ),
            "plan_fallback_count": sum(
                bool(row.get("plan_fallback_used")) for row in work
            ),
        }
        metrics_path = args.output_jsonl.with_suffix(".metrics.json")
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    model, tokenizer, generate_code = load_generator(
        str(args.coder_adapter),
        args.max_seq_length,
        args.seed,
    )
    results: list[dict[str, Any]] = []
    for index, item in enumerate(work, start=1):
        predicted_st = ""
        code_history: list[dict[str, Any]] = []
        address_check: dict[str, Any] = {
            "passed": False,
            "issues": ["code not generated"],
        }
        structure_check: dict[str, Any] = {
            "passed": False,
            "issues": ["code not generated"],
        }
        plan = item["plan"]
        if isinstance(plan, dict) and item["plan_check"]["passed"]:
            raw = generate_code(
                make_code_messages(item["instruction"], item["addresses"], plan),
                args.code_max_new_tokens,
            )
            predicted_st = strip_outer_markdown_fence(raw)
            address_check = validate_address_contract(
                item["instruction"],
                predicted_st,
            )
            structure_check = validate_st_structure(predicted_st)
            code_history.append(
                {
                    "attempt": 0,
                    "task": "CODE",
                    "raw_output": raw,
                    "address_check": address_check,
                    "structure_check": structure_check,
                }
            )
            for retry_index in range(1, args.code_retry_attempts + 1):
                if address_check["passed"] and structure_check["passed"]:
                    break
                errors = summarize_issues(address_check, structure_check)
                raw = generate_code(
                    make_code_messages(
                        item["instruction"],
                        item["addresses"],
                        plan,
                        previous_code=predicted_st,
                        validator_errors=errors,
                    ),
                    args.code_max_new_tokens,
                    temperature=args.repair_temperature,
                    top_p=args.repair_top_p,
                )
                predicted_st = strip_outer_markdown_fence(raw)
                address_check = validate_address_contract(
                    item["instruction"],
                    predicted_st,
                )
                structure_check = validate_st_structure(predicted_st)
                code_history.append(
                    {
                        "attempt": retry_index,
                        "task": "CODE_RETRY",
                        "validator_errors_given": errors,
                        "raw_output": raw,
                        "address_check": address_check,
                        "structure_check": structure_check,
                    }
                )

        result: dict[str, Any] = {
            "id": item["id"],
            "source_row": item["source_row"],
            "mode": args.mode,
            "instruction": item["instruction"],
            "addresses": item["addresses"],
            "generated_plan": plan,
            "plan_check": item["plan_check"],
            "initial_plan_passed": item["initial_plan_passed"],
            "plan_repair_attempts_used": item["plan_repair_attempts_used"],
            "plan_repaired_successfully": (
                not item["initial_plan_passed"]
                and item["plan_check"]["passed"]
                and not item.get("plan_fallback_used")
            ),
            "plan_fallback_used": bool(item.get("plan_fallback_used")),
            "plan_history": item["plan_history"],
            "code": predicted_st,
            "predicted_st": predicted_st,
            "code_history": code_history,
            "address_check": address_check,
            "structure_check": structure_check,
            "valid_generation": bool(
                item["plan_check"]["passed"]
                and address_check["passed"]
                and structure_check["passed"]
            ),
        }
        reference = item["reference_st"]
        if reference is not None:
            result["reference_st"] = reference
            result["exact_match"] = normalized(predicted_st) == normalized(reference)
            result["sequence_similarity"] = SequenceMatcher(
                None,
                normalized(reference),
                normalized(predicted_st),
                autojunk=False,
            ).ratio()
        results.append(result)
        # Keep a recoverable checkpoint if a later row or model call fails.
        write_jsonl(args.output_jsonl, results)
        print(
            f"[CODE {index}/{len(work)}] {item['id']} "
            f"plan={item['plan_check']['passed']} "
            f"address={address_check['passed']} "
            f"structure={structure_check['passed']} "
            f"attempts={len(code_history)}"
        )
    unload_model(model, tokenizer)

    write_jsonl(args.output_jsonl, results)
    if args.submission_output is not None:
        write_jsonl(
            args.submission_output,
            [
                {
                    "id": row["id"],
                    "language": "IEC 61131-3 Structured Text",
                    "solution_plan": compact_plan_steps(row["generated_plan"]),
                    "code": row["code"],
                }
                for row in results
            ],
        )

    comparable = [row for row in results if "reference_st" in row]
    repair_attempted = [
        row for row in results if row["plan_repair_attempts_used"] > 0
    ]
    metrics = {
        "mode": args.mode,
        "examples": len(results),
        "initial_plan_pass_rate": mean(
            bool(row["initial_plan_passed"]) for row in results
        )
        if results
        else None,
        "final_plan_pass_rate": mean(
            bool(row["plan_check"]["passed"]) for row in results
        )
        if results
        else None,
        "repair_attempted_count": len(repair_attempted),
        "repair_success_count": sum(
            row["plan_repaired_successfully"] for row in results
        ),
        "plan_fallback_count": sum(
            bool(row.get("plan_fallback_used")) for row in results
        ),
        "repair_success_rate_when_attempted": mean(
            bool(row["plan_repaired_successfully"]) for row in repair_attempted
        )
        if repair_attempted
        else None,
        "address_contract_pass_rate": mean(
            bool(row["address_check"]["passed"]) for row in results
        )
        if results
        else None,
        "static_structure_pass_rate": mean(
            bool(row["structure_check"]["passed"]) for row in results
        )
        if results
        else None,
        "valid_generation_rate": mean(
            bool(row["valid_generation"]) for row in results
        )
        if results
        else None,
        "exact_match_rate": mean(bool(row["exact_match"]) for row in comparable)
        if comparable
        else None,
        "mean_sequence_similarity": mean(
            float(row["sequence_similarity"]) for row in comparable
        )
        if comparable
        else None,
    }
    metrics_path = args.output_jsonl.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
