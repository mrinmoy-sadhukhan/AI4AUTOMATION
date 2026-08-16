#!/usr/bin/env python3
"""CPU-only tests; no model download or Ollama call is required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_two_stage_sft import make_records, process_split
from nl2plc_validation import (
    parse_json_object,
    validate_address_contract,
    validate_plan,
    validate_st_structure,
)
from two_stage_prompts import make_code_prompt, make_plan_prompt


ALGORITHM = {
    "program_name": "ConveyorStart",
    "summary": "Delay conveyor start by three seconds.",
    "variables": [
        {
            "name": "sensor",
            "role": "input",
            "address": "%IX0.0",
            "data_type": "BOOL",
            "purpose": "Start sensor.",
        },
        {
            "name": "motor",
            "role": "output",
            "address": "%QX0.0",
            "data_type": "BOOL",
            "purpose": "Conveyor motor.",
        },
    ],
    "function_blocks": [
        {
            "instance": "TON0",
            "type": "TON",
            "purpose": "Three-second on delay.",
            "parameters": [
                {"name": "IN", "value": "sensor"},
                {"name": "PT", "value": "T#3s"},
            ],
        }
    ],
    "logic_steps": [
        "Run TON0 from sensor with T#3s.",
        "Assign TON0.Q to motor.",
    ],
    "timing_and_thresholds": [
        {"value": "T#3s", "purpose": "Start delay."}
    ],
    "priority_rules": [],
    "reset_behavior": "TON0 and motor reset when sensor is false.",
    "scan_cycle_behavior": "Evaluate timer before assigning motor.",
    "configuration": {
        "configuration_name": "Config0",
        "resource_name": "Res0",
        "resource_target": "PLC",
        "task_name": "task0",
        "task_interval": "T#20ms",
        "task_priority": "0",
        "program_instance": "instance0",
    },
}

ST_CODE = """PROGRAM ConveyorStart
  VAR
    sensor AT %IX0.0 : BOOL;
    motor AT %QX0.0 : BOOL;
    TON0 : TON;
  END_VAR

  TON0(IN := sensor, PT := T#3s);
  motor := TON0.Q;
END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#20ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : ConveyorStart;
  END_RESOURCE
END_CONFIGURATION"""

SOURCE = {
    "id": "row_demo",
    "source_row": 1,
    "split": "train",
    "logic_family": "timer",
    "instruction": (
        "Start conveyor motor (%QX0.0) 3 seconds after sensor "
        "(%IX0.0) is active."
    ),
    "addresses": ["%QX0.0", "%IX0.0"],
    "algorithm": ALGORITHM,
    "gold_st": ST_CODE,
    "verification": {"passed": True},
}


class ValidationTests(unittest.TestCase):
    def test_plan_contract(self) -> None:
        check = validate_plan(ALGORITHM, ["%QX0.0", "%IX0.0"])
        self.assertTrue(check["passed"], check)

    def test_plan_rejects_invented_address(self) -> None:
        modified = {**ALGORITHM, "summary": "Also use %MX9.9."}
        check = validate_plan(modified, ["%QX0.0", "%IX0.0"])
        self.assertFalse(check["passed"])
        self.assertEqual(check["extra_addresses"], ["%MX9.9"])

    def test_json_parser_accepts_fence(self) -> None:
        parsed = parse_json_object('```json\n{"ok": true}\n```')
        self.assertEqual(parsed, {"ok": True})

    def test_gold_contract_and_structure(self) -> None:
        self.assertTrue(
            validate_address_contract(SOURCE["instruction"], ST_CODE)["passed"]
        )
        self.assertTrue(validate_st_structure(ST_CODE)["passed"])


class PromptAndDatasetTests(unittest.TestCase):
    def test_control_modes_are_explicit(self) -> None:
        plan = make_plan_prompt(SOURCE["instruction"], SOURCE["addresses"])
        code = make_code_prompt(
            SOURCE["instruction"],
            SOURCE["addresses"],
            ALGORITHM,
        )
        self.assertIn("TASK: PLAN", plan[-1]["content"])
        self.assertIn("TASK: CODE", code[-1]["content"])
        self.assertIn(
            "structured implementation algorithm",
            code[-1]["content"].casefold(),
        )

    def test_builds_three_targets(self) -> None:
        records = make_records(SOURCE)
        self.assertEqual(set(records), {"PLAN", "CODE", "DIRECT"})
        self.assertEqual(records["PLAN"]["messages"][-1]["role"], "assistant")
        self.assertEqual(records["CODE"]["messages"][-1]["content"], ST_CODE)

    def test_manual_exclusion(self) -> None:
        accepted, rejected = process_split(
            [SOURCE],
            {"row_demo": "human review failed"},
            {},
            False,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["id"], "row_demo")
        self.assertIn("manual exclusion", rejected[0]["reasons"][0])

    def test_nemotron_review_can_be_required(self) -> None:
        accepted, rejected = process_split(
            [SOURCE],
            {},
            {"row_demo": {"id": "row_demo", "verdict": "pass"}},
            True,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
