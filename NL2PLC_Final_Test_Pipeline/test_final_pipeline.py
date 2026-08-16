#!/usr/bin/env python3
"""CPU-only tests for released-workbook conversion and final assembly."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook, load_workbook

from final_test_pipeline import (
    assemble_outputs,
    load_test_records,
    write_jsonl,
)
from predict_two_adapter import build_deterministic_fallback_plan
from nl2plc_common import (
    expected_construct_families,
    extract_instruction_numbers,
    extract_plc_addresses,
    normalize_legacy_plc_addresses,
    validate_plan,
)
from predict_rust_adapter import compact_plan_steps, skipped_rust_check


class FinalPipelineTests(unittest.TestCase):
    def make_workbook(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "PLC Scenarios"
        sheet.append(["TASK"])
        sheet.append(["Start %IX0.0 and run %QX0.0"])
        sheet.append(
            [
                "Set RPM\n\nRPM_Set AI:0.5\nDigital_Display AO:0.6\n"
                "Start_Button DI:0.3\nConveyor_Run DO:1.1"
            ]
        )
        workbook.save(path)

    def test_released_workbook_has_30_rows_and_normalizes_aliases(self) -> None:
        released = Path(__file__).resolve().parent / "data" / "Test-Dataset-AI4Industrial-Automation.xlsx"
        rows = load_test_records(released, "PLC Scenarios")
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0]["id"], "test_0001")
        self.assertEqual(rows[-1]["source_row"], 31)
        normalized = normalize_legacy_plc_addresses(rows[27]["instruction"])
        self.assertIn("RPM_Set: %IW0.5", normalized)
        self.assertIn("Digital_Display: %QW0.6", normalized)
        self.assertIn("Start_Button: %IX0.3", normalized)
        self.assertIn("Conveyor_Run: %QW1.1", normalized)
        self.assertEqual(
            extract_plc_addresses(normalized),
            ["%IW0.5", "%QW0.6", "%IX0.3", "%IX0.4", "%QW1.1"],
        )

    def test_number_contract_covers_attached_units_and_plus_minus(self) -> None:
        instruction = "Pulse for 500ms, wait 5s, and reject outside +/-5 units."
        self.assertEqual(extract_instruction_numbers(instruction), ["500", "5"])
        self.assertIn(
            "counter",
            expected_construct_families("Vision sensor counts boxes during operation."),
        )

    def test_rust_skip_helpers_are_compact_safe(self) -> None:
        check = skipped_rust_check("missing ST")
        self.assertFalse(check["passed"])
        self.assertFalse(check["compiler_check"]["passed"])
        self.assertEqual(compact_plan_steps(None), [])

    def test_deterministic_plan_fallback_validates_for_all_30_tasks(self) -> None:
        released = Path(__file__).resolve().parent / "data" / "Test-Dataset-AI4Industrial-Automation.xlsx"
        rows = load_test_records(released, "PLC Scenarios")
        for row in rows:
            instruction = normalize_legacy_plc_addresses(row["instruction"])
            fallback = build_deterministic_fallback_plan(instruction)
            check = validate_plan(fallback, instruction)
            self.assertTrue(check["passed"], (row["id"], check["issues"]))

    def test_assembly_preserves_order_and_populates_columns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nl2plc_final_test_") as temp:
            root = Path(temp)
            template = root / "test.xlsx"
            self.make_workbook(template)
            records = load_test_records(template, "PLC Scenarios")
            st_path = root / "st_detailed.jsonl"
            rust_path = root / "rust_detailed.jsonl"
            write_jsonl(
                st_path,
                [
                    {
                        "id": row["id"],
                        "predicted_st": f"PROGRAM P{index}\nEND_PROGRAM",
                        "valid_generation": True,
                    }
                    for index, row in enumerate(records, start=1)
                ],
            )
            write_jsonl(
                rust_path,
                [
                    {
                        "id": row["id"],
                        "rust_code": f"pub struct Program{index};",
                        "valid_generation": True,
                        "skipped": False,
                    }
                    for index, row in enumerate(records, start=1)
                ],
            )
            report, output = assemble_outputs(
                template,
                "PLC Scenarios",
                records,
                st_path,
                rust_path,
                root / "out",
            )
            self.assertTrue(report["ready_to_submit"])
            self.assertEqual(output.name, "FINAL_SUBMISSION.xlsx")
            workbook = load_workbook(output, read_only=True, data_only=False)
            sheet = workbook["PLC Scenarios"]
            self.assertEqual([sheet.cell(1, col).value for col in range(1, 4)], [
                "TASK",
                "ST-CODE",
                "RUST-CODE",
            ])
            self.assertEqual(sheet["B2"].value, "PROGRAM P1\nEND_PROGRAM")
            self.assertEqual(sheet["C3"].value, "pub struct Program2;")
            workbook.close()

    def test_released_workbook_builds_exactly_30_populated_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nl2plc_exact_30_") as temp:
            root = Path(temp)
            template = (
                Path(__file__).resolve().parent
                / "data"
                / "Test-Dataset-AI4Industrial-Automation.xlsx"
            )
            records = load_test_records(template, "PLC Scenarios")
            st_path = root / "st_detailed.jsonl"
            rust_path = root / "rust_detailed.jsonl"
            write_jsonl(
                st_path,
                [
                    {
                        "id": row["id"],
                        "predicted_st": f"PROGRAM Strict{index}\nEND_PROGRAM",
                        "valid_generation": True,
                    }
                    for index, row in enumerate(records, start=1)
                ],
            )
            write_jsonl(
                rust_path,
                [
                    {
                        "id": row["id"],
                        "rust_code": f"pub struct Program{index};",
                        "valid_generation": True,
                    }
                    for index, row in enumerate(records, start=1)
                ],
            )
            report, output = assemble_outputs(
                template,
                "PLC Scenarios",
                records,
                st_path,
                rust_path,
                root / "out",
                required_result_count=30,
            )
            self.assertTrue(report["all_results_populated"])
            self.assertEqual(report["st_nonempty_count"], 30)
            self.assertEqual(report["rust_nonempty_count"], 30)
            self.assertIsNotNone(output)
            workbook = load_workbook(output, read_only=True, data_only=False)
            sheet = workbook["PLC Scenarios"]
            self.assertEqual(
                sum(bool(sheet.cell(row, 2).value) for row in range(2, 32)), 30
            )
            self.assertEqual(
                sum(bool(sheet.cell(row, 3).value) for row in range(2, 32)), 30
            )
            workbook.close()

    def test_strict_assembly_never_creates_workbook_with_blank_code(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nl2plc_strict_30_") as temp:
            root = Path(temp)
            template = root / "test.xlsx"
            self.make_workbook(template)
            records = load_test_records(template, "PLC Scenarios")
            st_path = root / "st_detailed.jsonl"
            rust_path = root / "rust_detailed.jsonl"
            write_jsonl(
                st_path,
                [
                    {
                        "id": row["id"],
                        "predicted_st": "" if index == 2 else "PROGRAM P\nEND_PROGRAM",
                        "valid_generation": index != 2,
                    }
                    for index, row in enumerate(records, start=1)
                ],
            )
            write_jsonl(
                rust_path,
                [
                    {
                        "id": row["id"],
                        "rust_code": "" if index == 2 else "pub struct Program;",
                        "valid_generation": index != 2,
                    }
                    for index, row in enumerate(records, start=1)
                ],
            )
            report, output = assemble_outputs(
                template,
                "PLC Scenarios",
                records,
                st_path,
                rust_path,
                root / "out",
                required_result_count=2,
            )
            self.assertFalse(report["all_results_populated"])
            self.assertIsNone(output)
            self.assertFalse((root / "out" / "FINAL_SUBMISSION.xlsx").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
