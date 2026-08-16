#!/usr/bin/env python3
"""Run DIRECT, student PLAN->CODE, or oracle-PLAN diagnostic inference."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from nl2plc_validation import (
    extract_instruction,
    extract_plc_addresses,
    extract_reference_st,
    parse_json_object,
    read_jsonl,
    strip_outer_markdown_fence,
    validate_address_contract,
    validate_plan,
    validate_st_structure,
    write_jsonl,
)
from two_stage_prompts import make_code_prompt, make_direct_prompt, make_plan_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Unsloth final_adapter, merged model, or Hugging Face model id.",
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("two_stage", "direct", "oracle_plan"),
        default="two_stage",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--plan-max-new-tokens", type=int, default=1400)
    parser.add_argument("--code-max-new-tokens", type=int, default=2000)
    parser.add_argument("--plan-attempts", type=int, default=2)
    parser.add_argument("--code-attempts", type=int, default=1)
    parser.add_argument(
        "--fallback-direct",
        action="store_true",
        help="Use DIRECT only after the two-stage path fails validation.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized(text: str) -> str:
    return " ".join(text.split())


def issue_text(*checks: dict[str, Any]) -> str:
    messages: list[str] = []
    for check in checks:
        issues = check.get("issues")
        if isinstance(issues, list):
            messages.extend(str(item) for item in issues)
        for key in ("missing_addresses", "extra_addresses"):
            values = check.get(key)
            if values:
                messages.append(f"{key}: {', '.join(str(item) for item in values)}")
    return "; ".join(messages) or "output did not satisfy the required contract"


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    for name in (
        "max_seq_length",
        "plan_max_new_tokens",
        "code_max_new_tokens",
        "plan_attempts",
        "code_attempts",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    from unsloth import FastLanguageModel

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(args.seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
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
        if input_ids.shape[-1] + max_new_tokens > args.max_seq_length:
            raise ValueError(
                f"prompt ({input_ids.shape[-1]}) + requested output "
                f"({max_new_tokens}) exceeds --max-seq-length "
                f"({args.max_seq_length})"
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

    def direct(instruction: str, addresses: list[str]) -> tuple[str, str]:
        raw = generate(
            make_direct_prompt(instruction, addresses),
            args.code_max_new_tokens,
        )
        return strip_outer_markdown_fence(raw), raw

    records = read_jsonl(args.input_jsonl)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    predictions: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        source_id = str(record.get("id", f"row_{index:04d}"))
        instruction = extract_instruction(record)
        addresses = extract_plc_addresses(instruction)
        reference = extract_reference_st(record)
        plan: dict[str, Any] | None = None
        plan_raw_outputs: list[str] = []
        plan_check: dict[str, Any] = {
            "passed": args.mode == "direct",
            "issues": [],
        }
        code_raw_outputs: list[str] = []
        predicted_st = ""
        mode_used = args.mode

        if args.mode == "direct":
            predicted_st, raw = direct(instruction, addresses)
            code_raw_outputs.append(raw)
        else:
            if args.mode == "oracle_plan":
                candidate = record.get("algorithm")
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"{source_id}: oracle_plan mode requires an algorithm object"
                    )
                plan = candidate
                plan_check = validate_plan(plan, addresses)
            else:
                correction: str | None = None
                for _ in range(args.plan_attempts):
                    raw = generate(
                        make_plan_prompt(
                            instruction,
                            addresses,
                            correction=correction,
                        ),
                        args.plan_max_new_tokens,
                    )
                    plan_raw_outputs.append(raw)
                    try:
                        candidate = parse_json_object(raw)
                    except ValueError as exc:
                        plan_check = {
                            "passed": False,
                            "issues": [f"invalid JSON: {exc}"],
                        }
                    else:
                        plan_check = validate_plan(candidate, addresses)
                        if plan_check["passed"]:
                            plan = candidate
                            break
                    correction = issue_text(plan_check)

            if plan is not None and plan_check["passed"]:
                correction = None
                for _ in range(args.code_attempts):
                    raw = generate(
                        make_code_prompt(
                            instruction,
                            addresses,
                            plan,
                            correction=correction,
                        ),
                        args.code_max_new_tokens,
                    )
                    code_raw_outputs.append(raw)
                    predicted_st = strip_outer_markdown_fence(raw)
                    address_check = validate_address_contract(
                        instruction,
                        predicted_st,
                    )
                    structure_check = validate_st_structure(predicted_st)
                    if address_check["passed"] and structure_check["passed"]:
                        break
                    correction = issue_text(address_check, structure_check)

        address_check = validate_address_contract(instruction, predicted_st)
        structure_check = validate_st_structure(predicted_st)
        if (
            args.fallback_direct
            and args.mode != "direct"
            and not (
                plan is not None
                and plan_check["passed"]
                and address_check["passed"]
                and structure_check["passed"]
            )
        ):
            predicted_st, raw = direct(instruction, addresses)
            code_raw_outputs.append(raw)
            address_check = validate_address_contract(instruction, predicted_st)
            structure_check = validate_st_structure(predicted_st)
            mode_used = "direct_fallback"

        result: dict[str, Any] = {
            "id": source_id,
            "source_row": record.get("source_row"),
            "requested_mode": args.mode,
            "mode_used": mode_used,
            "instruction": instruction,
            "addresses": addresses,
            "generated_plan": plan,
            "plan_check": plan_check,
            "plan_raw_outputs": plan_raw_outputs,
            "predicted_st": predicted_st,
            "code_raw_outputs": code_raw_outputs,
            "address_check": address_check,
            "structure_check": structure_check,
            "valid_generation": (
                address_check["passed"] and structure_check["passed"]
            ),
        }
        if reference is not None:
            result["reference_st"] = reference
            result["exact_match"] = normalized(predicted_st) == normalized(reference)
            result["sequence_similarity"] = SequenceMatcher(
                None,
                normalized(reference),
                normalized(predicted_st),
                autojunk=False,
            ).ratio()
        predictions.append(result)
        print(
            f"[{index}/{len(records)}] {source_id} "
            f"mode={mode_used} plan={plan_check['passed']} "
            f"address={address_check['passed']} "
            f"structure={structure_check['passed']}"
        )

    write_jsonl(args.output_jsonl, predictions)
    comparable = [row for row in predictions if "reference_st" in row]
    plan_rows = [row for row in predictions if args.mode != "direct"]
    metrics = {
        "mode": args.mode,
        "examples": len(predictions),
        "plan_pass_rate": (
            mean(row["plan_check"]["passed"] for row in plan_rows)
            if plan_rows
            else None
        ),
        "address_contract_pass_rate": (
            mean(row["address_check"]["passed"] for row in predictions)
            if predictions
            else None
        ),
        "static_structure_pass_rate": (
            mean(row["structure_check"]["passed"] for row in predictions)
            if predictions
            else None
        ),
        "valid_generation_rate": (
            mean(row["valid_generation"] for row in predictions)
            if predictions
            else None
        ),
        "exact_match_rate": (
            mean(row["exact_match"] for row in comparable) if comparable else None
        ),
        "mean_sequence_similarity": (
            mean(row["sequence_similarity"] for row in comparable)
            if comparable
            else None
        ),
        "direct_fallback_count": sum(
            row["mode_used"] == "direct_fallback" for row in predictions
        ),
    }
    metrics_path = args.output_jsonl.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

