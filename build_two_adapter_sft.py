#!/usr/bin/env python3
"""Build independent planner and coder SFT datasets from approved enriched rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from nl2plc_common_ import (
    canonical_json,
    canonical_plan_for_record,
    extract_instruction,
    extract_plc_addresses,
    extract_reference_st,
    read_jsonl,
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
    parser.add_argument("--train-enriched", type=Path, required=True)
    parser.add_argument("--validation-enriched", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, default=None)
    parser.add_argument("--nemotron-reviews", type=Path, nargs="*", default=None)
    parser.add_argument("--require-nemotron-pass", action="store_true")
    parser.add_argument(
        "--repairs-per-source",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help="Synthetic REPAIR_PLAN rows per approved source.",
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
    output: dict[str, dict[str, Any]] = {}
    for path in paths or []:
        for row in read_jsonl(path):
            source_id = str(row.get("id", ""))
            if not source_id:
                raise ValueError(f"{path}: review is missing id")
            if source_id in output:
                raise ValueError(f"duplicate Nemotron review id: {source_id}")
            output[source_id] = row
    return output


def stable_index(source_id: str, modulo: int) -> int:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def candidate_corruptions(
    source_id: str,
    plan: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Produce deterministic, validator-detectable bad plans."""
    candidates: list[tuple[str, dict[str, Any]]] = []

    io_contract = plan.get("io_contract", [])
    if isinstance(io_contract, list) and io_contract:
        bad = copy.deepcopy(plan)
        bad["io_contract"] = bad["io_contract"][1:]
        candidates.append(("missing_io_address", bad))

        bad = copy.deepcopy(plan)
        bad["io_contract"].append(
            {
                "address": "%MX999.999",
                "data_type": "BOOL",
                "role": "memory",
                "purpose": "Hallucinated address used only as a negative example.",
            }
        )
        candidates.append(("invented_io_address", bad))

        bad = copy.deepcopy(plan)
        first = bad["io_contract"][0]
        first["role"] = "output" if first.get("role") != "output" else "input"
        candidates.append(("incorrect_io_role", bad))

    literals = plan.get("literal_contract", [])
    if isinstance(literals, list) and literals:
        bad = copy.deepcopy(plan)
        old = str(bad["literal_contract"][0])
        try:
            replacement = str(float(old) + 1).rstrip("0").rstrip(".")
        except ValueError:
            replacement = old + "_wrong"
        bad["literal_contract"][0] = replacement
        candidates.append(("incorrect_protected_literal", bad))

    bad = copy.deepcopy(plan)
    bad.pop("reset_behavior", None)
    candidates.append(("missing_required_key", bad))

    bad = copy.deepcopy(plan)
    bad["ordered_logic"] = []
    candidates.append(("empty_ordered_logic", bad))

    bad = copy.deepcopy(plan)
    bad["schema_version"] = "nl2plc-plan-v1"
    candidates.append(("incorrect_schema_version", bad))

    bad = copy.deepcopy(plan)
    bad["program_name"] = "HallucinatedProgram"
    candidates.append(("implementation_name_leakage", bad))

    blocks = plan.get("required_blocks", [])
    if isinstance(blocks, list) and blocks:
        bad = copy.deepcopy(plan)
        bad["required_blocks"] = []
        candidates.append(("missing_required_block", bad))

    if not candidates:
        raise ValueError(f"{source_id}: unable to construct a repair example")
    offset = stable_index(source_id, len(candidates))
    return candidates[offset:] + candidates[:offset]


def make_repair_records(
    row: dict[str, Any],
    plan: dict[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    source_id = str(row["id"])
    instruction = extract_instruction(row)
    addresses = extract_plc_addresses(instruction)
    output: list[dict[str, Any]] = []
    for corruption_name, bad_plan in candidate_corruptions(source_id, plan):
        check = validate_plan(bad_plan, instruction)
        if check["passed"]:
            continue
        messages = make_repair_plan_messages(
            instruction,
            addresses,
            bad_plan,
            summarize_issues(check),
        )
        messages.append({"role": "assistant", "content": canonical_json(plan)})
        output.append(
            {
                "id": f"{source_id}__repair__{corruption_name}",
                "source_id": source_id,
                "source_row": row.get("source_row"),
                "split": row.get("split"),
                "logic_family": row.get("logic_family"),
                "task_type": "REPAIR_PLAN",
                "corruption_type": corruption_name,
                "addresses": addresses,
                "messages": messages,
            }
        )
        if len(output) >= count:
            break
    if len(output) != count:
        raise ValueError(
            f"{source_id}: requested {count} repair rows, built {len(output)}"
        )
    return output


def validate_source(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    source_id = str(row.get("id", ""))
    if not source_id:
        issues.append("missing id")
    try:
        instruction = extract_instruction(row)
    except ValueError as exc:
        issues.append(str(exc))
        instruction = ""
    gold_st = extract_reference_st(row)
    if gold_st is None:
        issues.append("missing gold ST")
    try:
        plan = canonical_plan_for_record(row)
    except ValueError as exc:
        issues.append(str(exc))
        plan = None

    if instruction and plan is not None:
        check = validate_plan(plan, instruction)
        issues.extend(f"canonical plan: {item}" for item in check["issues"])
    if instruction and gold_st:
        address_check = validate_address_contract(instruction, gold_st)
        if not address_check["passed"]:
            issues.append(
                "gold ST address contract: "
                + "; ".join(summarize_issues(address_check))
            )
        structure_check = validate_st_structure(gold_st)
        if not structure_check["passed"]:
            issues.append(
                "gold ST structure: " + "; ".join(summarize_issues(structure_check))
            )
    verification = row.get("verification")
    if isinstance(verification, dict) and verification.get("passed") is False:
        issues.append("source verification.passed is false")
    return plan, issues


def process_split(
    rows: list[dict[str, Any]],
    excluded: dict[str, str],
    reviews: dict[str, dict[str, Any]],
    require_nemotron_pass: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
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
        plan, source_issues = validate_source(row)
        reasons.extend(source_issues)
        if require_nemotron_pass:
            review = reviews.get(source_id)
            if review is None:
                reasons.append("missing Nemotron review")
            elif str(review.get("verdict", "")).lower() != "pass":
                reasons.append(
                    f"Nemotron verdict={review.get('verdict', 'unknown')}"
                )
        if reasons or plan is None:
            rejected.append({"id": source_id, "reasons": reasons})
        else:
            accepted.append((row, plan))
    return accepted, rejected


def make_plan_record(
    row: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    source_id = str(row["id"])
    instruction = extract_instruction(row)
    addresses = extract_plc_addresses(instruction)
    messages = make_plan_messages(instruction, addresses)
    messages.append({"role": "assistant", "content": canonical_json(plan)})
    return {
        "id": f"{source_id}__plan",
        "source_id": source_id,
        "source_row": row.get("source_row"),
        "split": row.get("split"),
        "logic_family": row.get("logic_family"),
        "task_type": "PLAN",
        "addresses": addresses,
        "messages": messages,
    }


def make_code_record(
    row: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    source_id = str(row["id"])
    instruction = extract_instruction(row)
    gold_st = extract_reference_st(row)
    if gold_st is None:
        raise ValueError(f"{source_id}: missing gold ST")
    addresses = extract_plc_addresses(instruction)
    messages = make_code_messages(instruction, addresses, plan)
    messages.append({"role": "assistant", "content": gold_st})
    return {
        "id": f"{source_id}__code",
        "source_id": source_id,
        "source_row": row.get("source_row"),
        "split": row.get("split"),
        "logic_family": row.get("logic_family"),
        "task_type": "CODE",
        "addresses": addresses,
        "messages": messages,
    }


def counts_by_task(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("task_type")) for row in rows))


def main() -> None:
    args = parse_args()
    for path in (args.train_enriched, args.validation_enriched):
        if not path.is_file():
            raise FileNotFoundError(path)
    excluded = load_manifest(args.review_manifest)
    reviews = load_reviews(args.nemotron_reviews)

    train_rows = read_jsonl(args.train_enriched)
    validation_rows = read_jsonl(args.validation_enriched)
    overlap = sorted(
        {str(row.get("id")) for row in train_rows}
        & {str(row.get("id")) for row in validation_rows}
    )
    if overlap:
        raise ValueError(f"train/validation source leakage: {overlap[:10]}")

    train, rejected_train = process_split(
        train_rows,
        excluded,
        reviews,
        args.require_nemotron_pass,
    )
    validation, rejected_validation = process_split(
        validation_rows,
        excluded,
        reviews,
        args.require_nemotron_pass,
    )
    if not train:
        raise ValueError("no eligible training sources")
    if not validation:
        raise ValueError("no eligible validation sources")

    train_plan = [make_plan_record(row, plan) for row, plan in train]
    train_repairs = [
        repair
        for row, plan in train
        for repair in make_repair_records(row, plan, args.repairs_per_source)
    ]
    validation_plan = [make_plan_record(row, plan) for row, plan in validation]
    validation_repairs = [
        repair
        for row, plan in validation
        for repair in make_repair_records(row, plan, args.repairs_per_source)
    ]
    train_coder = [make_code_record(row, plan) for row, plan in train]
    validation_coder = [make_code_record(row, plan) for row, plan in validation]

    canonical_train = [
        {
            **row,
            "canonical_plan": plan,
        }
        for row, plan in train
    ]
    canonical_validation = [
        {
            **row,
            "canonical_plan": plan,
        }
        for row, plan in validation
    ]
    outputs = {
        "train_planner": train_plan + train_repairs,
        "train_planner_plan_only": train_plan,
        "train_planner_repair_only": train_repairs,
        "validation_planner": validation_plan,
        "validation_planner_repair": validation_repairs,
        "train_coder": train_coder,
        "validation_coder": validation_coder,
        "canonical_train_master": canonical_train,
        "canonical_validation_master": canonical_validation,
        "rejected_train": rejected_train,
        "rejected_validation": rejected_validation,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    report = {
        "eligible_train_sources": len(train),
        "eligible_validation_sources": len(validation),
        "rejected_train_sources": len(rejected_train),
        "rejected_validation_sources": len(rejected_validation),
        "repairs_per_source": args.repairs_per_source,
        "files": {
            name: {
                "path": str(args.output_dir / f"{name}.jsonl"),
                "records": len(rows),
                "task_counts": counts_by_task(rows),
            }
            for name, rows in outputs.items()
        },
    }
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
