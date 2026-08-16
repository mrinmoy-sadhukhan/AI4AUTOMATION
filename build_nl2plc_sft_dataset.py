#!/usr/bin/env python3
"""Build strict, leakage-safe ST SFT JSONL files from the FIRE 2026 workbook."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from nl2plc_common import (
    SYSTEM_PROMPT,
    as_text,
    classify_logic_family,
    exact_input_group_key,
    make_user_prompt,
    record_fingerprint,
    validate_address_contract,
    validate_st_structure,
    write_jsonl,
)


REQUIRED_COLUMNS = ("TASK", "ST-CODE", "RUST-CODE")
SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TASK/ST-CODE rows to chat-format SFT JSONL while excluding "
            "examples that violate the literal PLC-address or ST structure contract."
        )
    )
    parser.add_argument("--excel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/nl2plc_st_sft"))
    parser.add_argument(
        "--sheet-name",
        default="0",
        help="Worksheet name or zero-based index. Default: first worksheet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.78)
    parser.add_argument("--validation-ratio", type=float, default=0.11)
    parser.add_argument("--test-ratio", type=float, default=0.11)
    parser.add_argument(
        "--allow-address-mismatch",
        action="store_true",
        help=(
            "Keep rows whose ST address set differs from TASK. This is unsafe for "
            "the competition and is disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-static-failures",
        action="store_true",
        help="Keep structurally incomplete/inconsistent ST rows. Disabled by default.",
    )
    return parser.parse_args()


def resolve_sheet_name(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def determine_targets(args: argparse.Namespace, total: int) -> dict[str, int]:
    explicit = (
        args.train_count,
        args.validation_count,
        args.test_count,
    )
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError(
                "Provide all three explicit counts: --train-count, "
                "--validation-count, and --test-count."
            )
        targets = {
            "train": int(args.train_count),
            "validation": int(args.validation_count),
            "test": int(args.test_count),
        }
        if any(value < 0 for value in targets.values()):
            raise ValueError("Split counts must be non-negative.")
        if sum(targets.values()) != total:
            raise ValueError(
                f"Explicit split counts sum to {sum(targets.values())}, but "
                f"{total} strict SFT records are available."
            )
        return targets

    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, found {ratio_sum}.")

    raw = {name: total * ratio for name, ratio in ratios.items()}
    targets = {name: int(raw[name]) for name in SPLIT_NAMES}
    remainder = total - sum(targets.values())
    for name in sorted(
        SPLIT_NAMES,
        key=lambda item: (raw[item] - targets[item], item == "train"),
        reverse=True,
    )[:remainder]:
        targets[name] += 1
    return targets


def choose_split(
    group: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    family_targets: dict[str, dict[str, float]],
) -> str:
    family_counts = Counter(row["logic_family"] for row in group)
    candidates: list[tuple[float, int, str]] = []
    for split_name in SPLIT_NAMES:
        current_size = len(split_rows[split_name])
        projected_size = current_size + len(group)
        overflow = max(0, projected_size - targets[split_name])
        size_gap = abs(targets[split_name] - projected_size)
        family_gap = 0.0
        existing = Counter(
            row["logic_family"] for row in split_rows[split_name]
        )
        for family, added in family_counts.items():
            desired = family_targets[split_name].get(family, 0.0)
            family_gap += abs(desired - (existing[family] + added))
        score = overflow * 10_000.0 + size_gap * 10.0 + family_gap
        candidates.append((score, current_size, split_name))
    return min(candidates)[2]


def split_grouped_records(
    records: list[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["input_group_id"]].append(record)

    rng = random.Random(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)

    overall_families = Counter(record["logic_family"] for record in records)
    family_targets = {
        split_name: {
            family: count * targets[split_name] / len(records)
            for family, count in overall_families.items()
        }
        for split_name in SPLIT_NAMES
    }
    result = {name: [] for name in SPLIT_NAMES}
    for group in grouped:
        split_name = choose_split(group, result, targets, family_targets)
        result[split_name].extend(group)

    actual = {name: len(rows) for name, rows in result.items()}
    if actual != targets:
        raise ValueError(
            "Leakage-safe grouping could not satisfy the requested exact counts. "
            f"Requested={targets}, actual={actual}. Use ratios or adjust counts."
        )

    for split_name, rows in result.items():
        rng.shuffle(rows)
        for row in rows:
            row["split"] = split_name
    return result


def build_records(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for dataframe_index, row in dataframe.iterrows():
        excel_row = int(dataframe_index) + 2
        task = as_text(row["TASK"])
        st_code = as_text(row["ST-CODE"])
        rust_code = as_text(row["RUST-CODE"])
        address_check = validate_address_contract(task, st_code)
        structure_check = validate_st_structure(st_code)
        reasons: list[str] = []
        if not task.strip():
            reasons.append("missing_task")
        if not st_code.strip():
            reasons.append("missing_st_code")
        if not rust_code.strip():
            reasons.append("missing_rust_code")
        if not address_check["passed"] and not args.allow_address_mismatch:
            reasons.append("address_contract_failed")
        if not structure_check["passed"] and not args.allow_static_failures:
            reasons.extend(structure_check["issues"])

        audit = {
            "source_row": excel_row,
            "task": task,
            "address_check": address_check,
            "structure_check": structure_check,
            "exclusion_reasons": sorted(set(reasons)),
        }
        if reasons:
            excluded.append(audit)
            continue

        logic_family = classify_logic_family(task, st_code)
        record_id = f"row_{excel_row:04d}"
        records.append(
            {
                "id": record_id,
                "source_row": excel_row,
                "logic_family": logic_family,
                "addresses": address_check["task_addresses"],
                "input_group_id": exact_input_group_key(task),
                "fingerprint": record_fingerprint(task, st_code),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": make_user_prompt(
                            task,
                            address_check["task_addresses"],
                        ),
                    },
                    {"role": "assistant", "content": st_code},
                ],
            }
        )
    return records, excluded


def verify_no_split_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    owners: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            group_id = row["input_group_id"]
            previous = owners.setdefault(group_id, split_name)
            if previous != split_name:
                raise AssertionError(
                    f"Input group {group_id} appears in {previous} and {split_name}."
                )


def main() -> None:
    args = parse_args()
    if not args.excel.is_file():
        raise FileNotFoundError(args.excel)

    dataframe = pd.read_excel(
        args.excel,
        sheet_name=resolve_sheet_name(args.sheet_name),
        dtype=object,
        keep_default_na=False,
    )
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Workbook is missing required columns {missing_columns}. "
            f"Found columns: {list(dataframe.columns)}"
        )

    records, excluded = build_records(dataframe, args)
    if not records:
        raise ValueError("No SFT-ready records remained after strict validation.")

    targets = determine_targets(args, len(records))
    splits = split_grouped_records(records, targets, args.seed)
    verify_no_split_leakage(splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        write_jsonl(args.output_dir / f"{split_name}.jsonl", rows)
    write_jsonl(args.output_dir / "all_strict.jsonl", records)
    write_jsonl(args.output_dir / "excluded_rows.jsonl", excluded)

    report = {
        "source_workbook": str(args.excel.resolve()),
        "sheet_name": args.sheet_name,
        "source_rows": int(len(dataframe)),
        "strict_sft_rows": len(records),
        "excluded_rows": len(excluded),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "logic_family_counts": dict(
            sorted(Counter(row["logic_family"] for row in records).items())
        ),
        "split_logic_family_counts": {
            name: dict(
                sorted(Counter(row["logic_family"] for row in rows).items())
            )
            for name, rows in splits.items()
        },
        "exclusion_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in excluded
                    for reason in row["exclusion_reasons"]
                ).items()
            )
        ),
        "address_policy": (
            "literal equality between TASK and ST address sets"
            if not args.allow_address_mismatch
            else "address mismatches allowed by explicit override"
        ),
        "split_policy": (
            "deterministic, logic-family-aware assignment with identical TASK "
            "inputs grouped into one split"
        ),
        "seed": args.seed,
    }
    report_path = args.output_dir / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
