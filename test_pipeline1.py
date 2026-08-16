#!/usr/bin/env python3
"""Fast local checks that do not download or load an LLM."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from nl2plc_common import (
    extract_plc_addresses,
    validate_address_contract,
    validate_st_structure,
)


VALID_ST = """PROGRAM Demo
  VAR
    sensor AT %IX0.10 : BOOL;
    motor AT %QX1.9 : BOOL;
  END_VAR
  motor := sensor;
END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#20ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : Demo;
  END_RESOURCE
END_CONFIGURATION"""


class CommonValidationTests(unittest.TestCase):
    def test_unusual_addresses_are_preserved(self) -> None:
        text = "Read %IX0.10 and write %QX1.9; do not change %IW1.1."
        self.assertEqual(
            extract_plc_addresses(text),
            ["%IX0.10", "%QX1.9", "%IW1.1"],
        )

    def test_exact_address_contract_passes(self) -> None:
        task = "Drive motor (%QX1.9) from sensor (%IX0.10)."
        check = validate_address_contract(task, VALID_ST)
        self.assertTrue(check["passed"])
        self.assertEqual(check["missing_st_addresses"], [])
        self.assertEqual(check["extra_st_addresses"], [])

    def test_invented_address_is_rejected(self) -> None:
        task = "Drive motor (%QX1.9) from sensor (%IX0.10)."
        code = VALID_ST.replace(
            "motor := sensor;",
            "extra AT %MX9.9 : BOOL;\n  motor := sensor;",
        )
        check = validate_address_contract(task, code)
        self.assertFalse(check["passed"])
        self.assertEqual(check["extra_st_addresses"], ["%MX9.9"])

    def test_program_configuration_mismatch_is_rejected(self) -> None:
        code = VALID_ST.replace(": Demo;", ": OtherProgram;")
        check = validate_st_structure(code)
        self.assertFalse(check["passed"])
        self.assertIn("program_configuration_mismatch", check["issues"])

    def test_valid_structure_passes(self) -> None:
        self.assertTrue(validate_st_structure(VALID_ST)["passed"])


class BuiltDatasetTests(unittest.TestCase):
    def test_built_dataset_if_present(self) -> None:
        data_dir = Path("data/nl2plc_st_sft")
        if not data_dir.exists():
            self.skipTest("Run build_nl2plc_sft_dataset.py first.")

        expected = {"train": 213, "validation": 31, "test": 29}
        group_owners: dict[str, str] = {}
        for split_name, expected_count in expected.items():
            path = data_dir / f"{split_name}.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), expected_count)
            for row in rows:
                self.assertEqual(row["messages"][-1]["role"], "assistant")
                group_id = row["input_group_id"]
                owner = group_owners.setdefault(group_id, split_name)
                self.assertEqual(owner, split_name)

        report = json.loads(
            (data_dir / "dataset_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["source_rows"], 352)
        self.assertEqual(report["strict_sft_rows"], 273)
        self.assertEqual(report["excluded_rows"], 79)


if __name__ == "__main__":
    unittest.main(verbosity=2)
