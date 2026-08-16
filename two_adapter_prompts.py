#!/usr/bin/env python3
"""Prompt identity shared by dataset construction and two-adapter inference."""

from __future__ import annotations

from typing import Any

from nl2plc_common_ import PLAN_SCHEMA_VERSION, canonical_json


PLANNER_SYSTEM_PROMPT = f"""You are the planning component of an NL2PLC system.
Convert the human instruction into one canonical behavioural plan for IEC
61131-3 Structured Text.

Return only one valid JSON object using schema {PLAN_SCHEMA_VERSION}. Use exactly
these top-level keys:
schema_version, objective, io_contract, literal_contract, required_blocks,
ordered_logic, priority_and_safety, state_and_memory, reset_behavior,
scan_cycle_behavior.

The plan must contain only facts inferable from the instruction. Preserve every
supplied PLC address and every numerical value exactly. Do not invent program,
task, resource, local-variable, or function-block instance names. Do not output
ST code, Markdown, explanations, or hidden chain-of-thought."""


CODER_SYSTEM_PROMPT = """You are the code-generation component of an NL2PLC
system. Generate IEC 61131-3 Structured Text from the human instruction and the
validated canonical behavioural plan.

Return only the complete ST source without Markdown or explanation. Include a
complete PROGRAM ... END_PROGRAM block and a matching CONFIGURATION ...
END_CONFIGURATION block. Use every allowed direct PLC address exactly once in
declarations and never invent, normalize, renumber, or repair an address.
Implement all logic, values, priorities, state, reset behaviour, timers,
counters, interlocks, and dependencies in the plan.

Unless the instruction explicitly says otherwise, use:
CONFIGURATION Config0, RESOURCE Res0 ON PLC,
TASK task0(INTERVAL := T#20ms, PRIORITY := 0), and program instance instance0."""


def address_block(addresses: list[str]) -> str:
    if not addresses:
        return "- No direct PLC addresses were supplied."
    return "\n".join(f"- {address}" for address in addresses)


def plan_schema_example() -> str:
    return canonical_json(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "objective": "Concise behavioural objective.",
            "io_contract": [
                {
                    "address": "%IX0.0",
                    "data_type": "BOOL",
                    "role": "input",
                    "purpose": "Input purpose.",
                },
                {
                    "address": "%QX0.0",
                    "data_type": "BOOL",
                    "role": "output",
                    "purpose": "Output purpose.",
                },
            ],
            "literal_contract": ["3"],
            "required_blocks": [
                {"type": "TON", "purpose": "Enforce the requested delay."}
            ],
            "ordered_logic": [
                {"step": 1, "behavior": "Read the input."},
                {
                    "step": 2,
                    "behavior": "Delay for the protected value, then set the output.",
                },
            ],
            "priority_and_safety": [],
            "state_and_memory": "",
            "reset_behavior": "Input false resets timer and output.",
            "scan_cycle_behavior": "",
        }
    )


def make_plan_messages(
    instruction: str,
    addresses: list[str],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: PLAN\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses (literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Canonical schema example (shape only; do not copy its values):\n"
                f"{plan_schema_example()}\n\n"
                "Return the complete canonical JSON plan."
            ),
        },
    ]


def make_repair_plan_messages(
    instruction: str,
    addresses: list[str],
    invalid_plan: str | dict[str, Any],
    validator_errors: list[str],
) -> list[dict[str, str]]:
    invalid_text = (
        canonical_json(invalid_plan)
        if isinstance(invalid_plan, dict)
        else str(invalid_plan).strip()
    )
    errors = "\n".join(f"- {item}" for item in validator_errors)
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: REPAIR_PLAN\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses (literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Invalid previous plan:\n"
                f"{invalid_text}\n\n"
                "Deterministic validator errors:\n"
                f"{errors or '- invalid plan'}\n\n"
                "Return one complete corrected canonical JSON plan. Do not "
                "return a patch or explanation."
            ),
        },
    ]


def make_code_messages(
    instruction: str,
    addresses: list[str],
    plan: dict[str, Any],
    *,
    previous_code: str | None = None,
    validator_errors: list[str] | None = None,
) -> list[dict[str, str]]:
    repair = ""
    task = "CODE"
    if previous_code is not None:
        task = "CODE_RETRY"
        errors = "\n".join(f"- {item}" for item in validator_errors or [])
        repair = (
            "\n\nPrevious invalid ST candidate:\n"
            f"{previous_code}\n\n"
            "Deterministic validator errors:\n"
            f"{errors or '- invalid ST structure or address contract'}\n\n"
            "Regenerate the entire corrected ST source, not a patch."
        )
    return [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"TASK: {task}\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses (literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Validated canonical behavioural plan:\n"
                f"{canonical_json(plan)}\n\n"
                "Generate the complete IEC 61131-3 Structured Text source."
                f"{repair}"
            ),
        },
    ]
