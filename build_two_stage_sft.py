#!/usr/bin/env python3
"""Build prompt-consistent PLAN/CODE/DIRECT SFT files from enriched gold rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from nl2plc_validation import (
    extract_plc_addresses,
    read_jsonl,
    validate_address_contract,
    validate_plan,
    validate_st_structure,
    write_jsonl,
)
from two_stage_prompts import (
    canonical_algorithm,
    make_code_prompt,
    make_direct_prompt,
    make_plan_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-enriched", type=Path, required=True)
    parser.add_argument("--validation-enriched", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=None,
        help="Optional JSON with an exclude object mapping source ids to reasons.",
    )
    parser.add_argument(
        "--nemotron-reviews",
        type=Path,
        nargs="+",
        default=None,
        help="One or more outputs from audit_plans_nemotron.py.",
    )
    parser.add_argument(
        "--require-nemotron-pass",
        action="store_true",
        help="Keep only records whose Nemotron verdict is pass.",
    )
    parser.add_argument(
        "--direct-ratio",
        type=float,
        default=0.25,
        help="DIRECT examples per source example in the main mixed file (0..1).",
    )
    return parser.parse_args()


def load_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    excluded = payload.get("exclude", {})
    if not isinstance(excluded, dict):
        raise ValueError("review manifest 'exclude' must be an object")
    return {str(key): str(value) for key, value in excluded.items()}


def load_reviews(paths: list[Path] | None) -> dict[str, dict[str, Any]]:
    if paths is None:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            source_id = str(row["id"])
            if source_id in output:
                raise ValueError(f"duplicate Nemotron review id: {source_id}")
            output[source_id] = row
    return output


def source_score(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()


def validate_source(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    source_id = str(row.get("id", ""))
    instruction = row.get("instruction")
    algorithm = row.get("algorithm")
    gold_st = row.get("gold_st")
    addresses = row.get("addresses")
    if not source_id:
        issues.append("missing id")
    if not isinstance(instruction, str) or not instruction.strip():
        issues.append("missing instruction")
    if not isinstance(algorithm, dict):
        issues.append("missing algorithm object")
    if not isinstance(gold_st, str) or not gold_st.strip():
        issues.append("missing gold_st")
    if not isinstance(addresses, list):
        issues.append("addresses must be a list")
    if issues:
        return issues

    instruction_addresses = extract_plc_addresses(instruction)
    if set(addresses) != set(instruction_addresses):
        issues.append("stored addresses differ from instruction addresses")
    plan_check = validate_plan(algorithm, instruction_addresses)
    issues.extend(f"plan: {item}" for item in plan_check["issues"])
    address_check = validate_address_contract(instruction, gold_st)
    if not address_check["passed"]:
        issues.append("gold ST address contract failed")
    structure_check = validate_st_structure(gold_st)
    if not structure_check["passed"]:
        issues.append(
            "gold ST structure failed: " + ", ".join(structure_check["issues"])
        )
    verification = row.get("verification", {})
    if isinstance(verification, dict) and verification.get("passed") is False:
        issues.append("source verification.passed is false")
    return issues


def make_records(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_id = str(row["id"])
    instruction = str(row["instruction"]).strip()
    addresses = [str(item) for item in row["addresses"]]
    algorithm = row["algorithm"]
    gold_st = str(row["gold_st"]).strip()
    common = {
        "source_id": source_id,
        "source_row": row.get("source_row"),
        "split": row.get("split"),
        "logic_family": row.get("logic_family"),
        "addresses": addresses,
    }
    plan_messages = make_plan_prompt(instruction, addresses)
    plan_messages.append(
        {"role": "assistant", "content": canonical_algorithm(algorithm)}
    )
    code_messages = make_code_prompt(instruction, addresses, algorithm)
    code_messages.append({"role": "assistant", "content": gold_st})
    direct_messages = make_direct_prompt(instruction, addresses)
    direct_messages.append({"role": "assistant", "content": gold_st})
    return {
        "PLAN": {
            **common,
            "id": f"{source_id}__plan",
            "task_type": "PLAN",
            "messages": plan_messages,
        },
        "CODE": {
            **common,
            "id": f"{source_id}__code",
            "task_type": "CODE",
            "messages": code_messages,
        },
        "DIRECT": {
            **common,
            "id": f"{source_id}__direct",
            "task_type": "DIRECT",
            "messages": direct_messages,
        },
    }


def process_split(
    rows: list[dict[str, Any]],
    excluded: dict[str, str],
    reviews: dict[str, dict[str, Any]],
    require_nemotron_pass: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        source_id = str(row.get("id", ""))
        reasons: list[str] = []
        if source_id in seen:
            reasons.append("duplicate source id")
        seen.add(source_id)
        if source_id in excluded:
            reasons.append(f"manual exclusion: {excluded[source_id]}")
        reasons.extend(validate_source(row))
        if require_nemotron_pass:
            review = reviews.get(source_id)
            if review is None:
                reasons.append("missing Nemotron review")
            elif review.get("verdict") != "pass":
                reasons.append(
                    f"Nemotron verdict={review.get('verdict', 'unknown')}"
                )
        if reasons:
            rejected.append({"id": source_id, "reasons": reasons})
        else:
            accepted.append(row)
    return accepted, rejected


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row["task_type"]) for row in rows))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.direct_ratio <= 1.0:
        raise ValueError("--direct-ratio must be between 0 and 1")
    excluded = load_manifest(args.review_manifest)
    reviews = load_reviews(args.nemotron_reviews)
    train_rows = read_jsonl(args.train_enriched)
    validation_rows = read_jsonl(args.validation_enriched)
    train_ids = {str(row.get("id")) for row in train_rows}
    validation_ids = {str(row.get("id")) for row in validation_rows}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(f"train/validation source leakage: {overlap[:10]}")

    train_sources, train_rejected = process_split(
        train_rows,
        excluded,
        reviews,
        args.require_nemotron_pass,
    )
    validation_sources, validation_rejected = process_split(
        validation_rows,
        excluded,
        reviews,
        args.require_nemotron_pass,
    )
    if not train_sources:
        raise ValueError("no eligible training sources")
    if not validation_sources:
        raise ValueError("no eligible validation sources")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, list[dict[str, Any]]] = {}
    for split_name, sources in (
        ("train", train_sources),
        ("validation", validation_sources),
    ):
        records = [make_records(row) for row in sources]
        plan = [row["PLAN"] for row in records]
        code = [row["CODE"] for row in records]
        direct = [row["DIRECT"] for row in records]
        direct_count = round(len(direct) * args.direct_ratio)
        selected_direct_ids = {
            row["source_id"]
            for row in sorted(direct, key=lambda item: source_score(item["source_id"]))[
                :direct_count
            ]
        }
        selected_direct = [
            row for row in direct if row["source_id"] in selected_direct_ids
        ]
        mixed: list[dict[str, Any]] = []
        for group in zip(plan, code, strict=True):
            mixed.extend(group)
        mixed.extend(selected_direct)
        output[f"{split_name}_plan"] = plan
        output[f"{split_name}_code"] = code
        output[f"{split_name}_direct"] = direct
        output[f"{split_name}_two_stage"] = mixed

    for name, rows in output.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    write_jsonl(output_dir / "rejected_train.jsonl", train_rejected)
    write_jsonl(output_dir / "rejected_validation.jsonl", validation_rejected)

    report = {
        "eligible_train_sources": len(train_sources),
        "eligible_validation_sources": len(validation_sources),
        "rejected_train_sources": len(train_rejected),
        "rejected_validation_sources": len(validation_rejected),
        "direct_ratio_in_two_stage_file": args.direct_ratio,
        "files": {
            name: {
                "path": str(output_dir / f"{name}.jsonl"),
                "records": len(rows),
                "task_counts": task_counts(rows),
            }
            for name, rows in output.items()
        },
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
