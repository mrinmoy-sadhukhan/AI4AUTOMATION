#!/usr/bin/env python3
"""Prompt identity shared by Rust dataset generation and student inference."""

from __future__ import annotations

from typing import Any

from rust_common import canonical_json


RUST_SYSTEM_PROMPT = """You are the Rust code-generation component of an
industrial-control synthesis system. Generate only src/program.rs. The crate's
fixed runtime owns transport, address mapping, the scan loop, and IEC function
block helpers.

Required output contract:
1. Return Rust source only, with no Markdown or explanation.
2. Import PlcIo and PlcResult from crate::plc_io.
3. Define `pub struct Program` and implement Default when practical.
4. Define `pub fn scan<I: PlcIo>(&mut self, io: &mut I, dt: Duration)
   -> PlcResult<()>` inside `impl Program`.
5. Preserve each supplied PLC address exactly as a string literal. Never
   normalize, renumber, infer, or invent an address.
6. Use only these I/O operations:
   %IX -> io.read_discrete_input("%IX...")?
   %QX -> io.write_coil("%QX...", bool_value)?
   %IW -> io.read_input_register("%IW...")?
   %QW -> io.write_holding_register("%QW...", i16_value)?
7. Use only these exact crate::function_blocks Rust types and method signatures:
   IEC TON -> OnDelay
     update(&mut self, input: bool, preset: Duration, dt: Duration) -> bool
   IEC TOF -> OffDelay
     update(&mut self, input: bool, preset: Duration, dt: Duration) -> bool
   IEC TP -> PulseTimer
     update(&mut self, input: bool, preset: Duration, dt: Duration) -> bool
   IEC CTU -> CounterUp
     update(&mut self, count: bool, reset: bool, preset: u32) -> bool
     current(&self) -> u32
   IEC CTD -> CounterDown
     update(&mut self, count: bool, load: bool, preset: u32) -> bool
     current(&self) -> u32
   IEC CTUD -> UpDownCounter
     update(&mut self, up: bool, down: bool, reset: bool,
            load: Option<i32>) -> i32
     current(&self) -> i32
   IEC R_TRIG -> RisingEdge; update(&mut self, input: bool) -> bool
   IEC F_TRIG -> FallingEdge; update(&mut self, input: bool) -> bool
   Store every function-block instance as a Program field. Never import or use
   TON, TOF, TP, CTU, CTD, CTUD, TONArgs, or Ton as Rust type names. Import
   std::time::Duration whenever a timer is used. Example TON translation:
   `use crate::function_blocks::OnDelay;`
   `start_delay: OnDelay`
   `let q = self.start_delay.update(input, Duration::from_secs(3), dt);`
8. Do not define main, Modbus/TCP code, threads, sleeps, loops, async code,
   external crates, unsafe blocks, or a replacement runtime.
9. Implement the validated plan and ST behavior, including reset, priority,
   safety, scaling, comparisons, latching, and scan-cycle state.
10. Finish a successful scan with `Ok(())`."""


REVIEWER_SYSTEM_PROMPT = """You independently review Rust PLC scan logic.
Compare the human instruction, canonical behavioural plan, verified ST, and
Rust program. Return exactly one JSON object:
{"verdict":"pass"|"fail","issues":["specific semantic issue", ...]}.
Judge control behavior, timing, counters, reset, priority, latching, scaling,
edge detection, and outputs. Do not reward textual similarity. Do not return
Markdown or hidden chain-of-thought."""


def address_block(addresses: list[str]) -> str:
    return "\n".join(f"- {item}" for item in addresses) or "- none"


def make_rust_code_messages(
    instruction: str,
    addresses: list[str],
    plan: dict[str, Any],
    st_code: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RUST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: RUST_CODE\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses (literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Validated canonical behavioural plan:\n"
                f"{canonical_json(plan)}\n\n"
                "Verified IEC 61131-3 Structured Text behavior:\n"
                f"{st_code.strip()}\n\n"
                "Generate only the complete src/program.rs source."
            ),
        },
    ]


def make_repair_rust_messages(
    instruction: str,
    addresses: list[str],
    plan: dict[str, Any],
    st_code: str,
    invalid_rust: str,
    validator_errors: list[str],
) -> list[dict[str, str]]:
    errors = "\n".join(f"- {item}" for item in validator_errors)
    return [
        {"role": "system", "content": RUST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: REPAIR_RUST\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses (literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Validated canonical behavioural plan:\n"
                f"{canonical_json(plan)}\n\n"
                "Verified IEC 61131-3 Structured Text behavior:\n"
                f"{st_code.strip()}\n\n"
                "Previous invalid src/program.rs candidate:\n"
                f"{invalid_rust.strip()}\n\n"
                "Validator/compiler/reviewer errors:\n"
                f"{errors or '- invalid candidate'}\n\n"
                "Regenerate the entire corrected src/program.rs source, not a patch."
            ),
        },
    ]


def make_review_messages(
    instruction: str,
    plan: dict[str, Any],
    st_code: str,
    rust_code: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Human instruction:\n{instruction}\n\n"
                f"Canonical plan:\n{canonical_json(plan)}\n\n"
                f"Verified ST:\n{st_code.strip()}\n\n"
                f"Candidate src/program.rs:\n{rust_code.strip()}\n\n"
                "Return the JSON verdict."
            ),
        },
    ]
