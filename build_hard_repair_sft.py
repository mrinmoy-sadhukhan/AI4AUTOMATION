#!/usr/bin/env python3
"""Convert real validator-detected planner failures into REPAIR_PLAN SFT rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nl2plc_common_ import (
    canonical_json,
    canonical_plan_for_record,
    extract_instruction,
    extract_plc_addresses,
    read_jsonl,
    summarize_issues,
    validate_plan,
    write_jsonl,
)
from two_adapter_prompts import make_repair_plan_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-predictions", type=Path, required=True)
    parser.add_argument("--canonical-master", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--all-failed-attempts",
        action="store_true",
        help="Use every failed attempt; default uses only the initial failed PLAN.",
    )
    return parser.parse_args()


def failed_attempts(
    prediction: dict[str, Any],
    all_failed_attempts: bool,
) -> list[dict[str, Any]]:
    history = prediction.get("plan_history")
    if not isinstance(history, list):
        return []
    output = []
    for item in history:
        if not isinstance(item, dict):
            continue
        check = item.get("check")
        if not isinstance(check, dict) or check.get("passed") is not False:
            continue
        if "raw_output" not in item and "parsed_plan" not in item:
            continue
        output.append(item)
        if not all_failed_attempts:
            break
    return output


def main() -> None:
    args = parse_args()
    for path in (args.planner_predictions, args.canonical_master):
        if not path.is_file():
            raise FileNotFoundError(path)

    master_rows = read_jsonl(args.canonical_master)
    master = {str(row.get("id")): row for row in master_rows}
    if len(master) != len(master_rows):
        raise ValueError("canonical master contains duplicate ids")

    output: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for prediction in read_jsonl(args.planner_predictions):
        source_id = str(prediction.get("id", ""))
        source = master.get(source_id)
        if source is None:
            rejected.append({"id": source_id, "reason": "missing canonical source"})
            continue
        instruction = extract_instruction(source)
        target = canonical_plan_for_record(source)
        target_check = validate_plan(target, instruction)
        if not target_check["passed"]:
            rejected.append(
                {
                    "id": source_id,
                    "reason": "canonical target failed validation",
                    "issues": target_check["issues"],
                }
            )
            continue
        attempts = failed_attempts(prediction, args.all_failed_attempts)
        for attempt_index, attempt in enumerate(attempts, start=1):
            invalid_plan: str | dict[str, Any]
            if isinstance(attempt.get("parsed_plan"), dict):
                invalid_plan = attempt["parsed_plan"]
            else:
                invalid_plan = str(attempt.get("raw_output", ""))
            check = attempt.get("check", {})
            errors = summarize_issues(check) if isinstance(check, dict) else []
            messages = make_repair_plan_messages(
                instruction,
                extract_plc_addresses(instruction),
                invalid_plan,
                errors or ["student plan failed deterministic validation"],
            )
            messages.append(
                {"role": "assistant", "content": canonical_json(target)}
            )
            output.append(
                {
                    "id": f"{source_id}__hard_repair__{attempt_index}",
                    "source_id": source_id,
                    "source_row": source.get("source_row"),
                    "split": source.get("split"),
                    "logic_family": source.get("logic_family"),
                    "task_type": "REPAIR_PLAN",
                    "repair_source": "student_prediction",
                    "messages": messages,
                }
            )

    write_jsonl(args.output_jsonl, output)
    rejected_path = args.output_jsonl.with_suffix(".rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    report = {
        "planner_predictions": str(args.planner_predictions),
        "canonical_master": str(args.canonical_master),
        "hard_repair_records": len(output),
        "rejected_records": len(rejected),
        "output": str(args.output_jsonl),
        "rejected_output": str(rejected_path),
    }
    report_path = args.output_jsonl.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
