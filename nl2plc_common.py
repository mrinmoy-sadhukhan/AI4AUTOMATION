#!/usr/bin/env python3
"""Shared parsing and validation helpers for the NL2PLC ST pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


PLC_ADDRESS_RE = re.compile(
    r"%[IQM](?:X|B|W|D|L)?\d+(?:\.\d+)*",
    flags=re.IGNORECASE,
)

PROGRAM_DECL_RE = re.compile(
    r"(?im)^\s*PROGRAM\s+([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
PROGRAM_INSTANCE_RE = re.compile(
    r"(?im)^\s*PROGRAM\s+[A-Za-z_][A-Za-z0-9_]*\s+WITH\s+"
    r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*;"
)
ST_BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", flags=re.DOTALL)
ST_LINE_COMMENT_RE = re.compile(r"//[^\r\n]*")

SYSTEM_PROMPT = """You generate IEC 61131-3 Structured Text (ST) from a human PLC instruction.

Requirements:
1. Return only the complete ST source. Do not use Markdown fences or explanations.
2. Include a complete PROGRAM ... END_PROGRAM block and a matching
   CONFIGURATION ... END_CONFIGURATION block.
3. Copy every direct PLC address exactly as supplied in the user message.
4. Use only the supplied PLC addresses. Never invent, normalize, renumber, or
   repair an address, even if its notation looks unusual.
5. Preserve the requested timers, counters, Boolean logic, safety behavior,
   constants, and control/data dependencies.
6. Ensure the configured program instance refers to the declared PROGRAM name."""


def as_text(value: Any) -> str:
    """Convert a spreadsheet cell to text without altering its internal content."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_plc_addresses(text: str) -> list[str]:
    """Return literal PLC address strings in first-appearance order."""
    return unique_in_order(match.group(0) for match in PLC_ADDRESS_RE.finditer(text))


def validate_address_contract(task: str, st_code: str) -> dict[str, Any]:
    """Check that ST uses exactly the literal address set stated in the task."""
    task_addresses = extract_plc_addresses(task)
    st_addresses = extract_plc_addresses(st_code)
    task_set = set(task_addresses)
    st_set = set(st_addresses)
    return {
        "task_addresses": task_addresses,
        "st_addresses": st_addresses,
        "missing_st_addresses": [
            address for address in task_addresses if address not in st_set
        ],
        "extra_st_addresses": [
            address for address in st_addresses if address not in task_set
        ],
        "passed": task_set == st_set,
    }


def parse_program_names(st_code: str) -> tuple[str | None, list[str]]:
    declaration = PROGRAM_DECL_RE.search(st_code)
    declared_name = declaration.group(1) if declaration else None
    configured_names = PROGRAM_INSTANCE_RE.findall(st_code)
    return declared_name, configured_names


def strip_st_comments(st_code: str) -> str:
    without_blocks = ST_BLOCK_COMMENT_RE.sub(" ", st_code)
    return ST_LINE_COMMENT_RE.sub(" ", without_blocks)


def validate_st_structure(st_code: str) -> dict[str, Any]:
    """Apply conservative static checks that do not require a PLC compiler."""
    upper = st_code.upper()
    declared_name, configured_names = parse_program_names(st_code)
    issues: list[str] = []

    required_tokens = {
        "missing_program": declared_name is None,
        "missing_var": re.search(r"(?im)^\s*VAR(?:_INPUT|_OUTPUT|_IN_OUT)?\b", st_code)
        is None,
        "missing_end_var": "END_VAR" not in upper,
        "missing_end_program": "END_PROGRAM" not in upper,
        "missing_configuration": re.search(
            r"(?im)^\s*CONFIGURATION\s+[A-Za-z_][A-Za-z0-9_]*", st_code
        )
        is None,
        "missing_end_configuration": "END_CONFIGURATION" not in upper,
        "missing_resource": re.search(r"(?im)^\s*RESOURCE\b", st_code) is None,
        "missing_task": re.search(r"(?im)^\s*TASK\b", st_code) is None,
        "missing_program_instance": not configured_names,
    }
    issues.extend(name for name, failed in required_tokens.items() if failed)

    if (
        declared_name is not None
        and configured_names
        and any(name != declared_name for name in configured_names)
    ):
        issues.append("program_configuration_mismatch")

    if "```" in st_code:
        issues.append("markdown_fence_present")

    executable = strip_st_comments(st_code)
    block_pairs = {
        "if_block_unbalanced": (r"\bIF\b", r"\bEND_IF\b"),
        "for_block_unbalanced": (r"\bFOR\b", r"\bEND_FOR\b"),
        "while_block_unbalanced": (r"\bWHILE\b", r"\bEND_WHILE\b"),
        "case_block_unbalanced": (r"\bCASE\b", r"\bEND_CASE\b"),
        "repeat_block_unbalanced": (r"\bREPEAT\b", r"\bUNTIL\b"),
    }
    for issue_name, (opening_pattern, closing_pattern) in block_pairs.items():
        opening_count = len(
            re.findall(opening_pattern, executable, flags=re.IGNORECASE)
        )
        closing_count = len(
            re.findall(closing_pattern, executable, flags=re.IGNORECASE)
        )
        if opening_count != closing_count:
            issues.append(issue_name)

    return {
        "declared_program": declared_name,
        "configured_programs": configured_names,
        "issues": issues,
        "passed": not issues,
    }


def classify_logic_family(task: str, st_code: str) -> str:
    # The CONFIGURATION block contains TASK(INTERVAL := T#20ms) in every row.
    # Classify only from the natural-language request and executable PROGRAM
    # block so that this scheduler period does not turn all rows into "timer".
    program_body = st_code.split("END_PROGRAM", maxsplit=1)[0]
    combined = f"{task}\n{program_body}".upper()
    if re.search(r"\b(CTU|CTD|CTUD|COUNTER|COUNT)\b", combined):
        return "counter"
    if re.search(
        r"\b(EMERGENCY|E[- ]?STOP|ESTOP|INTERLOCK|SHUTDOWN|SAFETY|FAULT)\b",
        combined,
    ):
        return "safety"
    if re.search(r"\b(TON|TOF|TP|TIMER|DELAY|PULSE)\b|T#\d", combined):
        return "timer"
    if re.search(
        r"%[IQM](?:W|D|L)\d|\b(REAL|LREAL|INT|DINT|WORD|SCALE|ANALOG|"
        r"TEMPERATURE|PRESSURE|LEVEL|FLOW|SPEED|RPM|PID)\b",
        combined,
    ):
        return "analog_process"
    return "direct_logic"


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def exact_input_group_key(task: str) -> str:
    """Group identical user inputs so they cannot cross data splits."""
    normalized = normalize_space(task).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def record_fingerprint(task: str, st_code: str) -> str:
    payload = f"{normalize_space(task)}\n{normalize_space(st_code)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_user_prompt(task: str, addresses: list[str]) -> str:
    address_lines = "\n".join(f"- {address}" for address in addresses)
    if not address_lines:
        address_lines = "- No direct PLC addresses were supplied."
    return (
        "Human PLC instruction:\n"
        f"{task}\n\n"
        "Allowed direct PLC addresses (literal, immutable contract):\n"
        f"{address_lines}\n\n"
        "Generate the complete IEC 61131-3 Structured Text source."
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows
