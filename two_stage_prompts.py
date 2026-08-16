#!/usr/bin/env python3
"""Canonical prompts shared by dataset building and two-stage inference."""

from __future__ import annotations

import json
from typing import Any


PLAN_SYSTEM_PROMPT = """You convert a human PLC instruction into a concise,
structured implementation algorithm for IEC 61131-3 Structured Text.

Return only one valid JSON object. Preserve every supplied PLC address, value,
condition, priority, timer, counter, interlock, reset rule, and configuration
requirement exactly. Do not return ST code, Markdown, or hidden chain-of-thought."""


CODE_SYSTEM_PROMPT = """You generate IEC 61131-3 Structured Text from a human
PLC instruction and a structured implementation algorithm.

Return only the complete ST source without Markdown or explanation. Include a
complete PROGRAM ... END_PROGRAM block and a matching CONFIGURATION ...
END_CONFIGURATION block. Use exactly the allowed direct PLC addresses; never
invent, normalize, renumber, or repair an address. Implement the supplied
algorithm faithfully."""


DIRECT_SYSTEM_PROMPT = """You generate IEC 61131-3 Structured Text from a human
PLC instruction.

Return only the complete ST source without Markdown or explanation. Include a
complete PROGRAM ... END_PROGRAM block and a matching CONFIGURATION ...
END_CONFIGURATION block. Use exactly the allowed direct PLC addresses; never
invent, normalize, renumber, or repair an address."""


def canonical_algorithm(algorithm: dict[str, Any]) -> str:
    """Use one stable representation in CODE training and inference."""
    return json.dumps(
        algorithm,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def address_block(addresses: list[str]) -> str:
    if not addresses:
        return "- No direct PLC addresses were supplied."
    return "\n".join(f"- {address}" for address in addresses)


def make_plan_prompt(
    instruction: str,
    addresses: list[str],
    *,
    correction: str | None = None,
) -> list[dict[str, str]]:
    suffix = ""
    if correction:
        suffix = (
            "\n\nThe previous PLAN output was invalid:\n"
            f"{correction}\n"
            "Generate a corrected complete JSON plan from the original instruction."
        )
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: PLAN\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses "
                "(literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Return the structured implementation algorithm."
                f"{suffix}"
            ),
        },
    ]


def make_code_prompt(
    instruction: str,
    addresses: list[str],
    algorithm: dict[str, Any],
    *,
    correction: str | None = None,
) -> list[dict[str, str]]:
    suffix = ""
    if correction:
        suffix = (
            "\n\nThe previous ST candidate failed validation:\n"
            f"{correction}\n"
            "Generate a corrected complete ST source."
        )
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: CODE\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses "
                "(literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Structured implementation algorithm:\n"
                f"{canonical_algorithm(algorithm)}\n\n"
                "Generate the complete IEC 61131-3 Structured Text source."
                f"{suffix}"
            ),
        },
    ]


def make_direct_prompt(
    instruction: str,
    addresses: list[str],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: DIRECT\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses "
                "(literal, immutable contract):\n"
                f"{address_block(addresses)}\n\n"
                "Generate the complete IEC 61131-3 Structured Text source."
            ),
        },
    ]

