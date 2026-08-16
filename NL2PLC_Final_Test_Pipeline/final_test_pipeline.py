#!/usr/bin/env python3
"""Run the frozen planner -> ST -> Rust system on the released Excel test set."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import copy
from pathlib import Path
from typing import Any, Iterable

from openpyxl import load_workbook
from openpyxl.styles import Alignment

from nl2plc_common import normalize_legacy_plc_addresses


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_XLSX = PACKAGE_ROOT / "data" / "Test-Dataset-AI4Industrial-Automation.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the released workbook, run the two LoRA stages and Rust "
            "LoRA, validate outputs, and rebuild the submission workbook."
        )
    )
    parser.add_argument("--test-xlsx", type=Path, default=DEFAULT_TEST_XLSX)
    parser.add_argument("--sheet", default="PLC Scenarios")
    parser.add_argument("--planner-adapter", type=Path)
    parser.add_argument("--coder-adapter", type=Path)
    parser.add_argument("--rust-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/final_test"))
    parser.add_argument("--expected-results", type=int, default=30)
    parser.add_argument("--plan-repair-attempts", type=int, default=3, choices=(0, 1, 2, 3))
    parser.add_argument("--code-retry-attempts", type=int, default=4, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--rust-repair-attempts", type=int, default=4, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--repair-temperature", type=float, default=0.20)
    parser.add_argument("--repair-top-p", type=float, default=0.90)
    parser.add_argument("--st-max-seq-length", type=int, default=4096)
    parser.add_argument("--rust-max-seq-length", type=int, default=8192)
    parser.add_argument("--allow-missing-cargo", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only convert Excel TASK rows to final_test_input.jsonl.",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Reuse existing detailed JSONL outputs and rebuild the workbook/report.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a completed ST or Rust detailed JSONL when its IDs match the test set.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit with status 2 unless every ST and Rust row passes validation.",
    )
    return parser.parse_args()


def normalize_header(value: Any) -> str:
    return str(value or "").strip().upper().replace("_", "-").replace(" ", "-")


def find_header_column(sheet: Any, header: str) -> int | None:
    wanted = normalize_header(header)
    for column in range(1, sheet.max_column + 1):
        if normalize_header(sheet.cell(1, column).value) == wanted:
            return column
    return None


def load_test_records(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    workbook = load_workbook(path, read_only=True, data_only=False)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(
            f"sheet {sheet_name!r} not found; available: {', '.join(workbook.sheetnames)}"
        )
    sheet = workbook[sheet_name]
    task_column = find_header_column(sheet, "TASK")
    if task_column is None:
        raise ValueError("released workbook has no TASK header in row 1")

    records: list[dict[str, Any]] = []
    for source_row in range(2, sheet.max_row + 1):
        original = str(sheet.cell(source_row, task_column).value or "").strip()
        if not original:
            continue
        source_id = f"test_{len(records) + 1:04d}"
        records.append(
            {
                "id": source_id,
                "source_row": source_row,
                "instruction": original,
                "original_instruction": original,
            }
        )
    workbook.close()
    if not records:
        raise ValueError("released workbook contains no non-empty TASK rows")
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("id", "")).strip()
        if not source_id:
            raise ValueError(f"{label} contains a row without id")
        if source_id in output:
            raise ValueError(f"{label} contains duplicate id {source_id}")
        output[source_id] = row
    return output


def completed_output(
    path: Path,
    expected_ids: list[str],
    code_keys: tuple[str, ...],
) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_jsonl(path)
        if [str(row.get("id", "")) for row in rows] != expected_ids:
            return False
        return all(
            bool(row.get("valid_generation")) and bool(code_value(row, *code_keys))
            for row in rows
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def require_adapter(path: Path | None, argument: str) -> Path:
    if path is None:
        raise ValueError(f"{argument} is required unless --prepare-only is used")
    resolved = path.resolve()
    if not (resolved / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{argument} must point to a final_adapter directory: {resolved}"
        )
    return resolved


def run_checked(command: list[str]) -> None:
    print("\nRUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PACKAGE_ROOT, check=True)


def code_value(record: dict[str, Any] | None, *keys: str) -> str:
    if not isinstance(record, dict):
        return ""
    for key in keys:
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    return ""


def copy_header_style(source: Any, destination: Any) -> None:
    if source.has_style:
        destination._style = copy(source._style)
        destination.font = copy(source.font)
        destination.fill = copy(source.fill)
        destination.border = copy(source.border)
        destination.number_format = source.number_format
        destination.protection = copy(source.protection)
    destination.alignment = Alignment(vertical="top", wrap_text=True)


def build_submission_workbook(
    template: Path,
    sheet_name: str,
    records: list[dict[str, Any]],
    st_by_id: dict[str, dict[str, Any]],
    rust_by_id: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    workbook = load_workbook(template, data_only=False)
    sheet = workbook[sheet_name]
    task_column = find_header_column(sheet, "TASK")
    if task_column is None:
        raise ValueError("TASK header disappeared from submission template")

    st_column = find_header_column(sheet, "ST-CODE") or 2
    rust_column = find_header_column(sheet, "RUST-CODE") or 3
    for column, expected in ((st_column, "ST-CODE"), (rust_column, "RUST-CODE")):
        current = sheet.cell(1, column).value
        if current not in (None, "") and normalize_header(current) != expected:
            raise ValueError(
                f"cannot place {expected} in column {column}; header is {current!r}"
            )
        sheet.cell(1, column).value = expected
        copy_header_style(sheet.cell(1, task_column), sheet.cell(1, column))

    for record in records:
        source_id = record["id"]
        source_row = int(record["source_row"])
        st_code = code_value(st_by_id.get(source_id), "predicted_st", "code")
        rust_code = code_value(rust_by_id.get(source_id), "rust_code", "code")
        sheet.cell(source_row, st_column).value = st_code
        sheet.cell(source_row, rust_column).value = rust_code
        sheet.cell(source_row, st_column).alignment = Alignment(
            vertical="top", wrap_text=True
        )
        sheet.cell(source_row, rust_column).alignment = Alignment(
            vertical="top", wrap_text=True
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    workbook.close()


def assemble_outputs(
    template: Path,
    sheet_name: str,
    records: list[dict[str, Any]],
    st_detailed: Path,
    rust_detailed: Path,
    output_dir: Path,
    required_result_count: int | None = None,
) -> tuple[dict[str, Any], Path | None]:
    if required_result_count is not None and len(records) != required_result_count:
        raise ValueError(
            f"strict result count requires {required_result_count} TASK rows, "
            f"but the workbook supplied {len(records)}"
        )
    st_rows = read_jsonl(st_detailed)
    rust_rows = read_jsonl(rust_detailed)
    st_by_id = index_unique(st_rows, "ST detailed output")
    rust_by_id = index_unique(rust_rows, "Rust detailed output")

    expected_ids = [record["id"] for record in records]
    unknown_st = sorted(set(st_by_id) - set(expected_ids))
    unknown_rust = sorted(set(rust_by_id) - set(expected_ids))
    if unknown_st or unknown_rust:
        raise ValueError(
            f"prediction IDs do not belong to this test set: ST={unknown_st}, Rust={unknown_rust}"
        )

    final_rows: list[dict[str, Any]] = []
    st_invalid: list[str] = []
    rust_invalid: list[str] = []
    missing_st: list[str] = []
    missing_rust: list[str] = []
    for record in records:
        source_id = record["id"]
        st = st_by_id.get(source_id, {})
        rust = rust_by_id.get(source_id, {})
        st_code = code_value(st, "predicted_st", "code")
        rust_code = code_value(rust, "rust_code", "code")
        st_valid = bool(st.get("valid_generation"))
        rust_valid = bool(rust.get("valid_generation"))
        if not st_code:
            missing_st.append(source_id)
        if not rust_code:
            missing_rust.append(source_id)
        if not st_valid:
            st_invalid.append(source_id)
        if not rust_valid:
            rust_invalid.append(source_id)
        final_rows.append(
            {
                "id": source_id,
                "source_row": record["source_row"],
                "task": record["original_instruction"],
                "normalized_instruction": normalize_legacy_plc_addresses(
                    record["original_instruction"]
                ),
                "st_code": st_code,
                "rust_code": rust_code,
                "st_valid": st_valid,
                "rust_valid": rust_valid,
                "rust_skipped": bool(rust.get("skipped")),
            }
        )

    all_populated = not (missing_st or missing_rust)
    ready = bool(all_populated and not st_invalid and not rust_invalid)
    report = {
        "test_workbook": str(template.resolve()),
        "sheet": sheet_name,
        "examples": len(records),
        "st_nonempty_count": len(records) - len(missing_st),
        "st_valid_count": len(records) - len(st_invalid),
        "rust_nonempty_count": len(records) - len(missing_rust),
        "rust_valid_count": len(records) - len(rust_invalid),
        "required_result_count": required_result_count,
        "all_results_populated": all_populated,
        "missing_st_ids": missing_st,
        "invalid_st_ids": st_invalid,
        "missing_rust_ids": missing_rust,
        "invalid_rust_ids": rust_invalid,
        "ready_to_submit": ready,
    }
    write_jsonl(output_dir / "final_test_predictions.jsonl", final_rows)
    (output_dir / "final_test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not all_populated:
        return report, None
    workbook_name = (
        "FINAL_SUBMISSION.xlsx" if ready else "STRICT_30_RESULTS_NEEDS_REVIEW.xlsx"
    )
    workbook_path = output_dir / workbook_name
    build_submission_workbook(
        template,
        sheet_name,
        records,
        st_by_id,
        rust_by_id,
        workbook_path,
    )
    return report, workbook_path


def main() -> None:
    args = parse_args()
    if args.prepare_only and args.assemble_only:
        raise ValueError("--prepare-only and --assemble-only are mutually exclusive")

    test_xlsx = args.test_xlsx.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_test_records(test_xlsx, args.sheet)
    if args.expected_results <= 0:
        raise ValueError("--expected-results must be positive")
    if len(records) != args.expected_results:
        raise ValueError(
            f"expected exactly {args.expected_results} TASK rows, found {len(records)}"
        )
    input_jsonl = output_dir / "final_test_input.jsonl"
    write_jsonl(input_jsonl, records)
    print(f"Prepared {len(records)} TASK rows -> {input_jsonl}")

    alias_rows = [
        row["id"]
        for row in records
        if normalize_legacy_plc_addresses(row["instruction"]) != row["instruction"]
    ]
    manifest = {
        "examples": len(records),
        "ids": [row["id"] for row in records],
        "source_rows": [row["source_row"] for row in records],
        "legacy_address_rows_normalized_for_models": alias_rows,
        "legacy_address_mapping": {
            "AI": "%IW",
            "AO": "%QW",
            "DI": "%IX",
            "DO": "%QX",
        },
    }
    (output_dir / "final_test_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.prepare_only:
        return

    st_detailed = output_dir / "st_detailed.jsonl"
    rust_detailed = output_dir / "rust_detailed.jsonl"
    expected_ids = [row["id"] for row in records]

    if not args.assemble_only:
        planner = require_adapter(args.planner_adapter, "--planner-adapter")
        coder = require_adapter(args.coder_adapter, "--coder-adapter")
        rust = require_adapter(args.rust_adapter, "--rust-adapter")

        if not (
            args.resume
            and completed_output(st_detailed, expected_ids, ("predicted_st", "code"))
        ):
            run_checked(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "predict_two_adapter.py"),
                    "--mode",
                    "two_adapter",
                    "--planner-adapter",
                    str(planner),
                    "--coder-adapter",
                    str(coder),
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-jsonl",
                    str(st_detailed),
                    "--submission-output",
                    str(output_dir / "st_compact.jsonl"),
                    "--plan-repair-attempts",
                    str(args.plan_repair_attempts),
                    "--code-retry-attempts",
                    str(args.code_retry_attempts),
                    "--max-seq-length",
                    str(args.st_max_seq_length),
                    "--repair-temperature",
                    str(args.repair_temperature),
                    "--repair-top-p",
                    str(args.repair_top_p),
                ]
            )
        else:
            print(f"Reusing completed ST output: {st_detailed}")

        if not (
            args.resume
            and completed_output(rust_detailed, expected_ids, ("rust_code", "code"))
        ):
            rust_command = [
                sys.executable,
                str(PACKAGE_ROOT / "predict_rust_adapter.py"),
                "--rust-adapter",
                str(rust),
                "--input-jsonl",
                str(st_detailed),
                "--output-jsonl",
                str(rust_detailed),
                "--submission-output",
                str(output_dir / "rust_compact.jsonl"),
                "--runtime-root",
                str(PACKAGE_ROOT / "rust_runtime"),
                "--repair-attempts",
                str(args.rust_repair_attempts),
                "--max-seq-length",
                str(args.rust_max_seq_length),
                "--repair-temperature",
                str(args.repair_temperature),
                "--repair-top-p",
                str(args.repair_top_p),
            ]
            if args.allow_missing_cargo:
                rust_command.append("--allow-missing-cargo")
            run_checked(rust_command)
        else:
            print(f"Reusing completed Rust output: {rust_detailed}")

    if not st_detailed.is_file() or not rust_detailed.is_file():
        raise FileNotFoundError(
            "assemble requires st_detailed.jsonl and rust_detailed.jsonl in output-dir"
        )
    report, workbook_path = assemble_outputs(
        test_xlsx,
        args.sheet,
        records,
        st_detailed,
        rust_detailed,
        output_dir,
        required_result_count=args.expected_results,
    )
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    if workbook_path is not None:
        print(f"Workbook: {workbook_path}")
    else:
        print("Workbook: NOT CREATED because one or more code cells are empty")
    if report["ready_to_submit"]:
        print("STATUS: READY TO SUBMIT")
    else:
        print("STATUS: NEEDS REVIEW; do not submit the candidate workbook yet")
        if args.strict_exit:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
