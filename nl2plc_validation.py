#!/usr/bin/env python3
"""Conservative parsing and validation shared by inference and tests."""

from __future__ import annotations

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

PLAN_REQUIRED_KEYS = {
    "program_name",
    "summary",
    "variables",
    "function_blocks",
    "logic_steps",
    "timing_and_thresholds",
    "priority_rules",
    "reset_behavior",
    "scan_cycle_behavior",
    "configuration",
}


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def extract_plc_addresses(text: str) -> list[str]:
    return unique_in_order(match.group(0) for match in PLC_ADDRESS_RE.finditer(text))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_outer_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json|st|structured-text|iecst)?\s*\n?(.*?)\n?```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else stripped


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_outer_markdown_fence(text)
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append("top-level JSON is not an object")
    raise ValueError("; ".join(errors) or "no JSON object found")


def validate_plan(
    plan: dict[str, Any],
    expected_addresses: list[str],
) -> dict[str, Any]:
    issues: list[str] = []
    missing_keys = sorted(PLAN_REQUIRED_KEYS - set(plan))
    extra_addresses = []
    missing_addresses = []
    if missing_keys:
        issues.append("missing keys: " + ", ".join(missing_keys))

    serialized = json.dumps(plan, ensure_ascii=False)
    plan_addresses = extract_plc_addresses(serialized)
    expected_set = set(expected_addresses)
    plan_set = set(plan_addresses)
    missing_addresses = [
        address for address in expected_addresses if address not in plan_set
    ]
    extra_addresses = [
        address for address in plan_addresses if address not in expected_set
    ]
    if missing_addresses:
        issues.append("missing addresses: " + ", ".join(missing_addresses))
    if extra_addresses:
        issues.append("invented addresses: " + ", ".join(extra_addresses))

    variables = plan.get("variables")
    if not isinstance(variables, list):
        issues.append("variables must be an array")
    logic_steps = plan.get("logic_steps")
    if not isinstance(logic_steps, list) or not logic_steps:
        issues.append("logic_steps must be a non-empty array")
    configuration = plan.get("configuration")
    if not isinstance(configuration, dict):
        issues.append("configuration must be an object")

    return {
        "passed": not issues,
        "issues": issues,
        "plan_addresses": plan_addresses,
        "missing_addresses": missing_addresses,
        "extra_addresses": extra_addresses,
    }


def validate_address_contract(
    instruction: str,
    st_code: str,
) -> dict[str, Any]:
    expected = extract_plc_addresses(instruction)
    actual = extract_plc_addresses(st_code)
    expected_set = set(expected)
    actual_set = set(actual)
    return {
        "passed": expected_set == actual_set,
        "expected_addresses": expected,
        "actual_addresses": actual,
        "missing_addresses": [item for item in expected if item not in actual_set],
        "extra_addresses": [item for item in actual if item not in expected_set],
    }


def strip_st_comments(st_code: str) -> str:
    return ST_LINE_COMMENT_RE.sub(" ", ST_BLOCK_COMMENT_RE.sub(" ", st_code))


def validate_st_structure(st_code: str) -> dict[str, Any]:
    upper = st_code.upper()
    declaration = PROGRAM_DECL_RE.search(st_code)
    declared_name = declaration.group(1) if declaration else None
    configured_names = PROGRAM_INSTANCE_RE.findall(st_code)
    issues: list[str] = []
    checks = {
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
    issues.extend(name for name, failed in checks.items() if failed)
    if (
        declared_name is not None
        and configured_names
        and any(name != declared_name for name in configured_names)
    ):
        issues.append("program_configuration_mismatch")
    if "```" in st_code:
        issues.append("markdown_fence_present")

    executable = strip_st_comments(st_code)
    pairs = {
        "if_block_unbalanced": (r"\bIF\b", r"\bEND_IF\b"),
        "for_block_unbalanced": (r"\bFOR\b", r"\bEND_FOR\b"),
        "while_block_unbalanced": (r"\bWHILE\b", r"\bEND_WHILE\b"),
        "case_block_unbalanced": (r"\bCASE\b", r"\bEND_CASE\b"),
        "repeat_block_unbalanced": (r"\bREPEAT\b", r"\bUNTIL\b"),
    }
    for issue, (opening, closing) in pairs.items():
        if len(re.findall(opening, executable, re.IGNORECASE)) != len(
            re.findall(closing, executable, re.IGNORECASE)
        ):
            issues.append(issue)
    return {
        "passed": not issues,
        "issues": issues,
        "declared_program": declared_name,
        "configured_programs": configured_names,
    }


def extract_instruction(record: dict[str, Any]) -> str:
    if record.get("instruction"):
        return str(record["instruction"]).strip()
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{record.get('id')}: missing instruction/messages")
    users = [item for item in messages if item.get("role") == "user"]
    if not users:
        raise ValueError(f"{record.get('id')}: missing user message")
    content = str(users[-1].get("content", "")).strip()
    marker = "Human PLC instruction:"
    if marker in content:
        content = content.split(marker, maxsplit=1)[1]
        for end_marker in (
            "\n\nAllowed direct PLC addresses",
            "\nAllowed direct PLC addresses",
            "\n\nGenerate the complete",
        ):
            if end_marker in content:
                content = content.split(end_marker, maxsplit=1)[0]
                break
    return content.strip()


def extract_reference_st(record: dict[str, Any]) -> str | None:
    if record.get("gold_st"):
        return str(record["gold_st"]).strip()
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    assistants = [item for item in messages if item.get("role") == "assistant"]
    return str(assistants[-1].get("content", "")).strip() if assistants else None

