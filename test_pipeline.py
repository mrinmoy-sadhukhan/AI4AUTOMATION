#!/usr/bin/env python3
"""Local tests that do not call Ollama."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import generate_gold_algorithm_database as pipeline


INSTRUCTION = (
    "Activate conveyor (%QX10.4) if total weight (%IW3.7) exceeds "
    "500 units, stop if below 100 units for 3 seconds."
)

GOLD_ST = """PROGRAM WeightThresholdControl
  VAR
    weight AT %IW3.7 : INT;
    conveyor AT %QX10.4 : BOOL;
    TON0 : TON;
  END_VAR

  TON0(IN := weight < 100, PT := T#3s);
  conveyor := weight > 500 AND NOT TON0.Q;
END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#20ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : WeightThresholdControl;
  END_RESOURCE
END_CONFIGURATION"""

ALGORITHM = {
    "program_name": "WeightThresholdControl",
    "summary": (
        "Control the conveyor from high and low weight thresholds with a "
        "three-second low-weight timer."
    ),
    "variables": [
        {
            "name": "weight",
            "role": "input",
            "address": "%IW3.7",
            "data_type": "INT",
            "purpose": "Measured total weight",
        },
        {
            "name": "conveyor",
            "role": "output",
            "address": "%QX10.4",
            "data_type": "BOOL",
            "purpose": "Conveyor command",
        },
    ],
    "function_blocks": [
        {
            "instance": "TON0",
            "type": "TON",
            "purpose": "Detect weight below 100 for 3 seconds",
            "parameters": [
                {"name": "IN", "value": "weight < 100"},
                {"name": "PT", "value": "T#3s"},
            ],
        }
    ],
    "logic_steps": [
        "Run TON0 while weight is below 100 units.",
        "Set conveyor when weight exceeds 500 and TON0.Q is false.",
    ],
    "timing_and_thresholds": [
        {"value": "500 units", "purpose": "Start threshold"},
        {"value": "100 units", "purpose": "Low-weight condition"},
        {"value": "T#3s", "purpose": "Low-weight delay"},
    ],
    "priority_rules": [],
    "reset_behavior": "TON0 resets when weight is not below 100.",
    "scan_cycle_behavior": (
        "Evaluate the timer and then assign the conveyor output each scan."
    ),
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


def source_record() -> dict:
    return {
        "id": "row_0104",
        "source_row": 104,
        "logic_family": "timer",
        "addresses": ["%QX10.4", "%IW3.7"],
        "split": "train",
        "messages": [
            {"role": "system", "content": "Generate ST."},
            {
                "role": "user",
                "content": (
                    f"Human PLC instruction:\n{INSTRUCTION}\n\n"
                    "Allowed direct PLC addresses (literal, immutable "
                    "contract):\n- %QX10.4\n- %IW3.7\n\n"
                    "Generate the complete IEC 61131-3 Structured Text source."
                ),
            },
            {"role": "assistant", "content": GOLD_ST},
        ],
    }


class PipelineTests(unittest.TestCase):
    def test_extract_instruction(self) -> None:
        instruction, gold = pipeline.get_source_parts(source_record())
        self.assertEqual(instruction, INSTRUCTION)
        self.assertEqual(gold, GOLD_ST)

    def test_gold_contract_passes(self) -> None:
        issues = pipeline.validate_gold_source(
            INSTRUCTION,
            GOLD_ST,
            ["%QX10.4", "%IW3.7"],
        )
        self.assertEqual(issues, [])

    def test_valid_algorithm_passes(self) -> None:
        schema_issues = pipeline.validate_schema_shape(
            ALGORITHM,
            pipeline.ALGORITHM_SCHEMA,
        )
        validation = pipeline.validate_algorithm(
            ALGORITHM,
            INSTRUCTION,
            GOLD_ST,
            ["%QX10.4", "%IW3.7"],
        )
        self.assertEqual(schema_issues, [])
        self.assertTrue(validation["passed"], validation["issues"])

    def test_missing_address_is_rejected(self) -> None:
        broken = json.loads(json.dumps(ALGORITHM))
        broken["variables"][0]["address"] = ""
        validation = pipeline.validate_algorithm(
            broken,
            INSTRUCTION,
            GOLD_ST,
            ["%QX10.4", "%IW3.7"],
        )
        self.assertFalse(validation["passed"])
        self.assertTrue(
            any(
                issue.startswith("algorithm_missing_addresses")
                for issue in validation["issues"]
            )
        )

    def test_missing_threshold_is_rejected(self) -> None:
        broken = json.loads(json.dumps(ALGORITHM))
        broken_text = json.dumps(broken).replace("500", "five hundred")
        broken = json.loads(broken_text)
        validation = pipeline.validate_algorithm(
            broken,
            INSTRUCTION,
            GOLD_ST,
            ["%QX10.4", "%IW3.7"],
        )
        self.assertFalse(validation["passed"])
        self.assertTrue(
            any(
                issue.startswith("instruction_numbers_missing")
                for issue in validation["issues"]
            )
        )

    def test_database_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            connection = pipeline.connect_database(
                output_dir / "gold_algorithms.sqlite3"
            )
            try:
                imported, examples = pipeline.import_sources(
                    connection,
                    [source_record()],
                )
                self.assertEqual(imported, 1)
                run_id = pipeline.insert_run(
                    connection,
                    example_id="row_0104",
                    attempt_number=1,
                    model="fake-teacher",
                    started_at=pipeline.utc_now(),
                    status="accepted",
                    validation={
                        "passed": True,
                        "schema_issues": [],
                        "semantic": pipeline.validate_algorithm(
                            ALGORITHM,
                            INSTRUCTION,
                            GOLD_ST,
                            ["%QX10.4", "%IW3.7"],
                        ),
                    },
                    algorithm=ALGORITHM,
                )
                pipeline.accept_algorithm(
                    connection,
                    examples[0]["example_id"],
                    run_id,
                    ALGORITHM,
                )
                report = pipeline.export_database(connection, output_dir)
            finally:
                connection.close()

            self.assertEqual(report["accepted_algorithms"], 1)
            self.assertEqual(report["sft_records"], 3)
            rows = pipeline.read_jsonl(
                output_dir / "exports" / "sft_multitask.jsonl"
            )
            self.assertEqual(
                {row["task_type"] for row in rows},
                {"PLAN", "CODE", "DIRECT"},
            )
            code_record = next(
                row for row in rows if row["task_type"] == "CODE"
            )
            self.assertEqual(
                code_record["messages"][-1]["content"],
                GOLD_ST,
            )

    def test_process_example_accepts_verified_teacher_output(self) -> None:
        class FakeClient:
            model = "fake-teacher"

            def chat(self, user_prompt: str):
                self.last_prompt = user_prompt
                return ALGORITHM, {
                    "request": {"model": self.model},
                    "raw_content": json.dumps(ALGORITHM),
                    "thinking": "",
                    "prompt_eval_count": 100,
                    "eval_count": 200,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            connection = pipeline.connect_database(
                Path(temp_dir) / "gold_algorithms.sqlite3"
            )
            try:
                _, examples = pipeline.import_sources(
                    connection,
                    [source_record()],
                )
                client = FakeClient()
                status = pipeline.process_example(
                    connection,
                    client,
                    examples[0],
                    max_attempts=1,
                    retry_delay_seconds=0,
                )
                self.assertEqual(status, "accepted")
                self.assertTrue(
                    pipeline.is_accepted(connection, "row_0104")
                )
                self.assertIn("VERIFIED GOLD ST", client.last_prompt)
                self.assertIn(GOLD_ST, client.last_prompt)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
