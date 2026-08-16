#!/usr/bin/env python3
"""Generate Rust from repaired PLAN + ST, then validate and repair once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from rust_common import (
    extract_instruction,
    extract_plan,
    extract_plc_addresses,
    extract_st,
    normalized,
    read_jsonl,
    sequence_similarity,
    strip_outer_markdown_fence,
    validate_rust_candidate,
    write_jsonl,
)
from rust_prompts import make_repair_rust_messages, make_rust_code_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-adapter", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument("--reference-jsonl", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=3500)
    parser.add_argument("--repair-attempts", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--cargo-timeout", type=int, default=120)
    parser.add_argument("--allow-missing-cargo", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.reference_jsonl is not None and not args.reference_jsonl.is_file():
        raise FileNotFoundError(args.reference_jsonl)
    if args.runtime_root is None:
        args.runtime_root = Path(__file__).resolve().parent / "rust_runtime"
    if not args.runtime_root.is_dir():
        raise FileNotFoundError(args.runtime_root)
    if args.max_seq_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sequence length and output budget must be positive")
    if args.max_new_tokens >= args.max_seq_length:
        raise ValueError("--max-new-tokens must be less than --max-seq-length")


def load_generator(
    model_path: str,
    max_seq_length: int,
    seed: int,
) -> tuple[Any, Any, Callable[[list[dict[str, str]], int], str]]:
    import torch
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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

    def generate(messages: list[dict[str, str]], max_new_tokens: int) -> str:
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
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()

    return model, tokenizer, generate


def compact_plan_steps(plan: dict[str, Any]) -> list[str]:
    steps = plan.get("ordered_logic", [])
    if not isinstance(steps, list):
        return []
    output: list[str] = []
    for item in steps:
        if isinstance(item, dict) and str(item.get("behavior", "")).strip():
            output.append(str(item["behavior"]).strip())
        elif str(item).strip():
            output.append(str(item).strip())
    return output


def reference_rust_by_id(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    output: dict[str, str] = {}
    for index, row in enumerate(read_jsonl(path), start=1):
        source_id = str(row.get("id", f"row_{index:04d}"))
        for key in ("final_rust", "reference_rust", "rust_code"):
            value = str(row.get(key, "")).strip()
            if value:
                output[source_id] = value
                break
    return output


def upstream_status(record: dict[str, Any]) -> dict[str, Any]:
    plan_check = record.get("plan_check")
    address_check = record.get("address_check")
    structure_check = record.get("structure_check")
    return {
        "plan_passed": plan_check.get("passed")
        if isinstance(plan_check, dict)
        else None,
        "st_address_passed": address_check.get("passed")
        if isinstance(address_check, dict)
        else None,
        "st_structure_passed": structure_check.get("passed")
        if isinstance(structure_check, dict)
        else None,
        "st_valid_generation": record.get("valid_generation"),
    }


def bounded_errors(check: dict[str, Any], max_chars: int = 8000) -> list[str]:
    output: list[str] = []
    used = 0
    for item in check.get("issues", []) or []:
        text = " ".join(str(item).split())
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        if text:
            output.append(text)
            used += len(text)
    return output or ["candidate failed Rust validation"]


def main() -> None:
    args = parse_args()
    validate_args(args)
    records = read_jsonl(args.input_jsonl)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    references = reference_rust_by_id(args.reference_jsonl)
    model, tokenizer, generate = load_generator(
        args.rust_adapter,
        args.max_seq_length,
        args.seed,
    )

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        source_id = str(record.get("id", f"row_{index:04d}"))
        instruction = extract_instruction(record)
        plan = extract_plan(record)
        st_code = extract_st(record)
        addresses = extract_plc_addresses(instruction)
        messages = make_rust_code_messages(instruction, addresses, plan, st_code)
        history: list[dict[str, Any]] = []
        raw = generate(messages, args.max_new_tokens)
        code = strip_outer_markdown_fence(raw)
        check = validate_rust_candidate(
            instruction,
            code,
            args.runtime_root,
            cargo_timeout=args.cargo_timeout,
            allow_missing_cargo=args.allow_missing_cargo,
        )
        history.append(
            {
                "attempt": 0,
                "task_type": "RUST_CODE",
                "raw_output": raw,
                "rust_code": code,
                "validation": check,
            }
        )
        initial_passed = bool(check["passed"])
        for repair_index in range(1, args.repair_attempts + 1):
            if check["passed"]:
                break
            errors = bounded_errors(check)
            messages = make_repair_rust_messages(
                instruction,
                addresses,
                plan,
                st_code,
                code,
                errors,
            )
            raw = generate(messages, args.max_new_tokens)
            code = strip_outer_markdown_fence(raw)
            check = validate_rust_candidate(
                instruction,
                code,
                args.runtime_root,
                cargo_timeout=args.cargo_timeout,
                allow_missing_cargo=args.allow_missing_cargo,
            )
            history.append(
                {
                    "attempt": repair_index,
                    "task_type": "REPAIR_RUST",
                    "validator_errors_given": errors,
                    "raw_output": raw,
                    "rust_code": code,
                    "validation": check,
                }
            )

        result: dict[str, Any] = {
            "id": source_id,
            "source_row": record.get("source_row"),
            "mode": "rust_adapter",
            "instruction": instruction,
            "addresses": addresses,
            "generated_plan": plan,
            "source_st": st_code,
            "upstream_status": upstream_status(record),
            "rust_code": code,
            "code": code,
            "rust_check": check,
            "initial_rust_passed": initial_passed,
            "rust_repair_attempts_used": len(history) - 1,
            "rust_repaired_successfully": not initial_passed and bool(check["passed"]),
            "valid_generation": bool(check["passed"]),
            "history": history,
        }
        reference = references.get(source_id)
        if reference is not None:
            result["reference_rust"] = reference
            result["exact_match"] = normalized(code) == normalized(reference)
            result["sequence_similarity"] = sequence_similarity(code, reference)
        results.append(result)
        write_jsonl(args.output_jsonl, results)
        print(
            f"[RUST {index}/{len(records)}] {source_id} "
            f"initial={initial_passed} final={check['passed']} "
            f"repairs={len(history) - 1}"
        )

    del model
    del tokenizer
    write_jsonl(args.output_jsonl, results)
    if args.submission_output is not None:
        write_jsonl(
            args.submission_output,
            [
                {
                    "id": row["id"],
                    "language": "Rust",
                    "solution_plan": compact_plan_steps(row["generated_plan"]),
                    "code": row["rust_code"],
                }
                for row in results
            ],
        )

    attempted = [row for row in results if row["rust_repair_attempts_used"] > 0]
    comparable = [row for row in results if "reference_rust" in row]
    metrics = {
        "mode": "rust_adapter",
        "examples": len(results),
        "initial_rust_pass_rate": mean(
            bool(row["initial_rust_passed"]) for row in results
        )
        if results
        else None,
        "final_rust_pass_rate": mean(
            bool(row["valid_generation"]) for row in results
        )
        if results
        else None,
        "repair_attempted_count": len(attempted),
        "repair_success_count": sum(
            bool(row["rust_repaired_successfully"]) for row in results
        ),
        "repair_success_rate_when_attempted": mean(
            bool(row["rust_repaired_successfully"]) for row in attempted
        )
        if attempted
        else None,
        "address_contract_pass_rate": mean(
            bool(row["rust_check"]["address_check"]["passed"]) for row in results
        )
        if results
        else None,
        "io_api_contract_pass_rate": mean(
            bool(row["rust_check"]["io_contract_check"]["passed"])
            for row in results
        )
        if results
        else None,
        "static_structure_pass_rate": mean(
            bool(row["rust_check"]["structure_check"]["passed"])
            for row in results
        )
        if results
        else None,
        "cargo_available_count": sum(
            bool(row["rust_check"]["compiler_check"]["available"])
            for row in results
        ),
        "cargo_pass_rate": mean(
            row["rust_check"]["compiler_check"]["passed"] is True
            for row in results
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

