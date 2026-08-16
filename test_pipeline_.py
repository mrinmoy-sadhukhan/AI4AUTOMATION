#!/usr/bin/env python3
"""CPU-only contract tests for the Rust branch (cargo is optional)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from prepare_rust_sft import make_code_sft, make_repair_sft
from rust_common import (
    extract_instruction,
    extract_plan,
    extract_st,
    validate_rust_candidate,
)
from rust_prompts import make_rust_code_messages, make_repair_rust_messages
from train_rust_lora import extract_messages


INSTRUCTION = (
    "Start a conveyor motor (%QX0.0) 3 seconds after sensor input "
    "(%IX0.0) is activated."
)
PLAN = {
    "schema_version": "nl2plc-plan-v2",
    "objective": "Delay motor start by three seconds.",
    "io_contract": [
        {
            "address": "%QX0.0",
            "data_type": "BOOL",
            "role": "output",
            "purpose": "motor",
        },
        {
            "address": "%IX0.0",
            "data_type": "BOOL",
            "role": "input",
            "purpose": "sensor",
        },
    ],
    "literal_contract": ["3"],
    "required_blocks": [{"type": "TON", "purpose": "delay"}],
    "ordered_logic": [
        {"step": 1, "behavior": "Read the sensor."},
        {"step": 2, "behavior": "Delay and update the motor."},
    ],
    "priority_and_safety": [],
    "state_and_memory": "timer state persists between scans",
    "reset_behavior": "sensor false resets the timer and motor",
    "scan_cycle_behavior": "",
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
GOOD_RUST = """use std::time::Duration;

use crate::function_blocks::OnDelay;
use crate::plc_io::{PlcIo, PlcResult};

#[derive(Default)]
pub struct Program {
    start_delay: OnDelay,
}

impl Program {
    pub fn scan<I: PlcIo>(
        &mut self,
        io: &mut I,
        dt: Duration,
    ) -> PlcResult<()> {
        let sensor = io.read_discrete_input("%IX0.0")?;
        let motor = self.start_delay.update(
            sensor,
            Duration::from_secs(3),
            dt,
        );
        io.write_coil("%QX0.0", motor)?;
        Ok(())
    }
}"""


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def fake_audit() -> dict:
    failed_code = GOOD_RUST.replace("read_discrete_input", "read_input_register")
    failed_validation = {
        "passed": False,
        "issues": ["%IX0.0 must use read_discrete_input"],
    }
    return {
        "id": "row_0001",
        "source_row": 2,
        "split": "train",
        "logic_family": "timer",
        "instruction": INSTRUCTION,
        "addresses": ["%QX0.0", "%IX0.0"],
        "canonical_plan": PLAN,
        "gold_st": ST_CODE,
        "accepted": True,
        "final_rust": GOOD_RUST,
        "history": [
            {
                "attempt": 0,
                "accepted": False,
                "rust_code": failed_code,
                "validation": failed_validation,
                "review": None,
            },
            {
                "attempt": 1,
                "accepted": True,
                "rust_code": GOOD_RUST,
                "validation": {"passed": True, "issues": []},
                "review": None,
            },
        ],
    }


def main() -> None:
    package_root = Path(__file__).resolve().parent
    runtime_root = package_root / "rust_runtime"

    check = validate_rust_candidate(
        INSTRUCTION,
        GOOD_RUST,
        runtime_root,
        allow_missing_cargo=True,
    )
    require(check["passed"], str(check))
    print("[PASS] valid Rust contract")

    wrong_api = GOOD_RUST.replace("read_discrete_input", "read_input_register")
    check = validate_rust_candidate(
        INSTRUCTION,
        wrong_api,
        runtime_root,
        allow_missing_cargo=True,
    )
    require(not check["passed"], "wrong %IX API must fail")
    require(
        not check["io_contract_check"]["passed"],
        "I/O API validator did not fail",
    )
    print("[PASS] wrong I/O API rejection")

    invented = GOOD_RUST.replace(
        "Ok(())", 'let _ = io.read_discrete_input("%IX9.9")?;\n        Ok(())'
    )
    check = validate_rust_candidate(
        INSTRUCTION,
        invented,
        runtime_root,
        allow_missing_cargo=True,
    )
    require(not check["address_check"]["passed"], "invented address must fail")
    print("[PASS] immutable address rejection")

    with_main = GOOD_RUST + "\nfn main() {}\n"
    check = validate_rust_candidate(
        INSTRUCTION,
        with_main,
        runtime_root,
        allow_missing_cargo=True,
    )
    require(not check["structure_check"]["passed"], "fn main must fail")
    print("[PASS] fixed-runtime boundary")

    record = {
        "id": "row_0001",
        "instruction": INSTRUCTION,
        "canonical_plan": PLAN,
        "gold_st": ST_CODE,
    }
    require(extract_instruction(record) == INSTRUCTION, "instruction extraction")
    require(extract_plan(record) == PLAN, "plan extraction")
    require(extract_st(record) == ST_CODE, "ST extraction")
    print("[PASS] canonical-master field extraction")

    prediction = {
        "id": "row_0001",
        "instruction": INSTRUCTION,
        "generated_plan": PLAN,
        "predicted_st": ST_CODE,
    }
    require(extract_plan(prediction) == PLAN, "prediction plan extraction")
    require(extract_st(prediction) == ST_CODE, "prediction ST extraction")
    print("[PASS] two-adapter prediction field extraction")

    code_messages = make_rust_code_messages(
        INSTRUCTION, ["%QX0.0", "%IX0.0"], PLAN, ST_CODE
    )
    repair_messages = make_repair_rust_messages(
        INSTRUCTION,
        ["%QX0.0", "%IX0.0"],
        PLAN,
        ST_CODE,
        wrong_api,
        ["wrong I/O API"],
    )
    require("TASK: RUST_CODE" in code_messages[-1]["content"], "code prompt")
    require("TASK: REPAIR_RUST" in repair_messages[-1]["content"], "repair prompt")
    print("[PASS] generation/repair prompt identity")

    audit = fake_audit()
    code_sft = make_code_sft(audit)
    repairs = make_repair_sft(audit, 1)
    require(code_sft["task_type"] == "RUST_CODE", "RUST_CODE SFT")
    require(len(repairs) == 1, "one real repair row expected")
    require(repairs[0]["task_type"] == "REPAIR_RUST", "REPAIR_RUST SFT")
    extract_messages(code_sft)
    extract_messages(repairs[0])
    print("[PASS] CODE/REPAIR SFT construction")

    if shutil.which("cargo"):
        with tempfile.TemporaryDirectory(prefix="nl2plc_runtime_test_") as name:
            copied = Path(name) / "runtime"
            shutil.copytree(runtime_root, copied)
            (copied / "src" / "program.rs").write_text(
                GOOD_RUST + "\n", encoding="utf-8"
            )
            check = validate_rust_candidate(
                INSTRUCTION, GOOD_RUST, copied, allow_missing_cargo=False
            )
            require(check["compiler_check"]["passed"] is True, str(check))
        print("[PASS] cargo check")
    else:
        print("[SKIP] cargo check (cargo not installed in this environment)")

    print("All Rust pipeline tests passed.")


if __name__ == "__main__":
    main()

