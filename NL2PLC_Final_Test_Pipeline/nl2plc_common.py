#!/usr/bin/env python3
"""Shared canonical-plan, JSONL, and conservative ST validation utilities."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


PLAN_SCHEMA_VERSION = "nl2plc-plan-v2"
PLAN_REQUIRED_KEYS = {
    "schema_version",
    "objective",
    "io_contract",
    "literal_contract",
    "required_blocks",
    "ordered_logic",
    "priority_and_safety",
    "state_and_memory",
    "reset_behavior",
    "scan_cycle_behavior",
}
PLAN_FORBIDDEN_KEYS = {
    "program_name",
    "configuration",
    "configuration_name",
    "resource_name",
    "resource_target",
    "task_name",
    "task_interval",
    "task_priority",
    "program_instance",
    "instance",
}
ALLOWED_BLOCK_TYPES = {
    "TON",
    "TOF",
    "TP",
    "CTU",
    "CTD",
    "CTUD",
    "R_TRIG",
    "F_TRIG",
    "RTC",
    "PID",
}

PLC_ADDRESS_RE = re.compile(
    r"%[IQM](?:X|B|W|D|L)?\d+(?:\.\d+)*",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])([-+]?\d+(?:\.\d+)?)(?:st|nd|rd|th)?"
    r"(?:(?=\s*(?:ms|milliseconds?|s|secs?|seconds?|mins?|minutes?|"
    r"hours?|psi|ppm|%|°?\s*[CF])\b)|(?![A-Za-z_]))",
    flags=re.IGNORECASE,
)
LEGACY_ADDRESS_RE = re.compile(
    r"(?im)^(?P<indent>[ \t]*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]+"
    r"(?P<family>AI|AO|DI|DO)[ \t]*:[ \t]*"
    r"(?P<address>\d+(?:\.\d+)*)[ \t]*$"
)
LEGACY_ADDRESS_FAMILIES = {
    "AI": "%IW",
    "AO": "%QW",
    "DI": "%IX",
    "DO": "%QX",
}
PROGRAM_DECL_RE = re.compile(
    r"(?im)^\s*PROGRAM\s+([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
PROGRAM_INSTANCE_RE = re.compile(
    r"(?im)^\s*PROGRAM\s+[A-Za-z_][A-Za-z0-9_]*\s+WITH\s+"
    r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*;"
)
ST_BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", flags=re.DOTALL)
ST_LINE_COMMENT_RE = re.compile(r"//[^\r\n]*")


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


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


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


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


def extract_plc_addresses(text: str) -> list[str]:
    return unique_in_order(match.group(0) for match in PLC_ADDRESS_RE.finditer(text))


def extract_instruction_numbers(instruction: str) -> list[str]:
    """Numbers outside direct PLC addresses form an immutable value contract."""
    without_addresses = PLC_ADDRESS_RE.sub(" ", instruction)
    # Treat +/-5 and ±5 as the magnitude 5 instead of accidentally protecting -5.
    without_addresses = without_addresses.replace("+/-", " ").replace("±", " ")
    return unique_in_order(match.group(1) for match in NUMBER_RE.finditer(without_addresses))


def normalize_legacy_plc_addresses(instruction: str) -> str:
    """Convert released-test AI/AO/DI/DO aliases into IEC direct addresses."""

    def replace(match: re.Match[str]) -> str:
        family = LEGACY_ADDRESS_FAMILIES[match.group("family").upper()]
        return (
            f"{match.group('indent')}{match.group('name')}: "
            f"{family}{match.group('address')}"
        )

    return LEGACY_ADDRESS_RE.sub(replace, str(instruction))


def role_from_address(address: str) -> str:
    upper = address.upper()
    if upper.startswith("%I"):
        return "input"
    if upper.startswith("%Q"):
        return "output"
    return "memory"


def extract_instruction(record: dict[str, Any]) -> str:
    if record.get("instruction"):
        return normalize_legacy_plc_addresses(str(record["instruction"]).strip())
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
    return normalize_legacy_plc_addresses(content.strip())


def extract_reference_st(record: dict[str, Any]) -> str | None:
    if record.get("gold_st"):
        return str(record["gold_st"]).strip()
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    assistants = [item for item in messages if item.get("role") == "assistant"]
    return str(assistants[-1].get("content", "")).strip() if assistants else None


def _clean_block_purpose(text: str, instance_names: list[str]) -> str:
    output = text
    for name in sorted(instance_names, key=len, reverse=True):
        if name:
            member_replacements = {
                "Q": "the done output of the required block",
                "QU": "the up-count output of the required block",
                "QD": "the down-count output of the required block",
                "CV": "the current count of the required block",
                "ET": "the elapsed time of the required block",
            }
            for member, replacement in member_replacements.items():
                output = re.sub(
                    rf"\b{re.escape(name)}\.{member}\b",
                    replacement,
                    output,
                    flags=re.IGNORECASE,
                )
            output = re.sub(
                rf"\b{re.escape(name)}\b",
                "the required block",
                output,
                flags=re.IGNORECASE,
            )
    output = re.sub(r"\bthe\s+the\b", "the", output, flags=re.IGNORECASE)
    return " ".join(output.split())


def canonicalize_algorithm(
    instruction: str,
    algorithm: dict[str, Any],
) -> dict[str, Any]:
    """
    Project the gold-derived algorithm into an instruction-learnable plan.

    Exact program, task, resource, block-instance, and local-variable names are
    deliberately removed. The coder adapter owns those implementation choices.
    """
    addresses = extract_plc_addresses(instruction)
    variables = algorithm.get("variables", [])
    by_address = {
        str(item.get("address")): item
        for item in variables
        if isinstance(item, dict) and item.get("address")
    }
    io_contract: list[dict[str, str]] = []
    for address in addresses:
        source = by_address.get(address, {})
        io_contract.append(
            {
                "address": address,
                "data_type": str(source.get("data_type", "BOOL")).upper(),
                "role": role_from_address(address),
                "purpose": str(source.get("purpose", "")).strip()
                or f"PLC {role_from_address(address)} at {address}",
            }
        )

    raw_blocks = [
        item
        for item in algorithm.get("function_blocks", [])
        if isinstance(item, dict)
    ]
    instance_names = [str(item.get("instance", "")) for item in raw_blocks]
    required_blocks: list[dict[str, str]] = []
    seen_blocks: set[tuple[str, str]] = set()
    for item in raw_blocks:
        block_type = str(item.get("type", "")).upper().strip()
        purpose = _clean_block_purpose(str(item.get("purpose", "")), instance_names)
        if not block_type:
            continue
        key = (block_type, purpose)
        if key not in seen_blocks:
            seen_blocks.add(key)
            required_blocks.append({"type": block_type, "purpose": purpose})

    raw_steps = algorithm.get("logic_steps", [])
    ordered_logic = []
    for index, step in enumerate(raw_steps if isinstance(raw_steps, list) else []):
        cleaned = _clean_block_purpose(str(step), instance_names)
        if cleaned:
            ordered_logic.append({"step": index + 1, "behavior": cleaned})
    if not ordered_logic:
        ordered_logic = [
            {
                "step": 1,
                "behavior": str(algorithm.get("summary", instruction)).strip(),
            }
        ]

    priority = algorithm.get("priority_rules", [])
    if not isinstance(priority, list):
        priority = [str(priority)] if priority else []

    scan_text = _clean_block_purpose(
        str(algorithm.get("scan_cycle_behavior", "")).strip(),
        instance_names,
    )
    state_text = ""
    if re.search(
        r"\b(retain|latched|latch|previous|prior|memory|state|edge)\b",
        scan_text,
        flags=re.IGNORECASE,
    ):
        state_text = scan_text
    scan_requirement = ""
    if re.search(
        r"\b(scan cycle|each scan|every scan|cyclic task|task interval)\b",
        instruction,
        re.IGNORECASE,
    ):
        scan_requirement = scan_text

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "objective": _clean_block_purpose(
            str(algorithm.get("summary", instruction)).strip(),
            instance_names,
        ),
        "io_contract": io_contract,
        "literal_contract": extract_instruction_numbers(instruction),
        "required_blocks": required_blocks,
        "ordered_logic": ordered_logic,
        "priority_and_safety": [
            _clean_block_purpose(str(item).strip(), instance_names)
            for item in priority
            if str(item).strip()
        ],
        "state_and_memory": state_text,
        "reset_behavior": _clean_block_purpose(
            str(algorithm.get("reset_behavior", "")).strip(),
            instance_names,
        ),
        "scan_cycle_behavior": scan_requirement,
    }


def canonical_plan_for_record(record: dict[str, Any]) -> dict[str, Any]:
    instruction = extract_instruction(record)
    if isinstance(record.get("canonical_plan"), dict):
        return copy.deepcopy(record["canonical_plan"])
    algorithm = record.get("algorithm")
    if not isinstance(algorithm, dict):
        raise ValueError(f"{record.get('id')}: missing canonical_plan/algorithm")
    return canonicalize_algorithm(instruction, algorithm)


def _nested_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_nested_keys(item))
    return keys


def expected_construct_families(instruction: str) -> list[str]:
    lower = instruction.lower()
    output: list[str] = []
    without_addresses = PLC_ADDRESS_RE.sub(" ", lower)
    has_duration = bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*(?:ms|millisecond|milliseconds|s|sec|secs|"
            r"second|seconds|min|mins|minute|minutes|hour|hours)\b",
            without_addresses,
        )
    )
    if has_duration:
        output.append("timer")
    if "rising edge" in lower or "positive edge" in lower:
        output.append("rising_edge")
    if "falling edge" in lower or "negative edge" in lower:
        output.append("falling_edge")
    if re.search(
        r"\b(counter|count(?:s|ed|ing)?|increments?|decrements?)\b",
        lower,
    ):
        output.append("counter")
    return output


def validate_plan(
    plan: dict[str, Any],
    instruction: str,
) -> dict[str, Any]:
    issues: list[str] = []
    missing_keys = sorted(PLAN_REQUIRED_KEYS - set(plan))
    extra_keys = sorted(set(plan) - PLAN_REQUIRED_KEYS)
    if missing_keys:
        issues.append("missing top-level keys: " + ", ".join(missing_keys))
    if extra_keys:
        issues.append("unexpected top-level keys: " + ", ".join(extra_keys))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        issues.append(f"schema_version must be {PLAN_SCHEMA_VERSION!r}")

    forbidden = sorted(set(_nested_keys(plan)) & PLAN_FORBIDDEN_KEYS)
    if forbidden:
        issues.append(
            "implementation-specific keys are forbidden: " + ", ".join(forbidden)
        )

    expected_addresses = extract_plc_addresses(instruction)
    serialized = json.dumps(plan, ensure_ascii=False)
    actual_addresses = extract_plc_addresses(serialized)
    missing_addresses = [
        item for item in expected_addresses if item not in set(actual_addresses)
    ]
    extra_addresses = [
        item for item in actual_addresses if item not in set(expected_addresses)
    ]
    if missing_addresses:
        issues.append("missing addresses: " + ", ".join(missing_addresses))
    if extra_addresses:
        issues.append("invented addresses: " + ", ".join(extra_addresses))

    io_contract = plan.get("io_contract")
    io_by_address: dict[str, dict[str, Any]] = {}
    if not isinstance(io_contract, list):
        issues.append("io_contract must be an array")
    else:
        for index, item in enumerate(io_contract):
            if not isinstance(item, dict):
                issues.append(f"io_contract[{index}] must be an object")
                continue
            address = str(item.get("address", ""))
            if address in io_by_address:
                issues.append(f"duplicate io_contract address: {address}")
            io_by_address[address] = item
            if address in expected_addresses:
                expected_role = role_from_address(address)
                if item.get("role") != expected_role:
                    issues.append(
                        f"{address} role must be {expected_role}, "
                        f"not {item.get('role')!r}"
                    )
            if not str(item.get("data_type", "")).strip():
                issues.append(f"{address or index} is missing data_type")
        if set(io_by_address) != set(expected_addresses):
            issues.append("io_contract must contain each allowed address exactly once")
            io_missing = [
                item for item in expected_addresses if item not in set(io_by_address)
            ]
            io_extra = [
                item for item in io_by_address if item not in set(expected_addresses)
            ]
            missing_addresses = unique_in_order(
                [*missing_addresses, *io_missing]
            )
            extra_addresses = unique_in_order([*extra_addresses, *io_extra])
            if io_missing:
                issues.append(
                    "io_contract missing addresses: " + ", ".join(io_missing)
                )
            if io_extra:
                issues.append(
                    "io_contract invented addresses: " + ", ".join(io_extra)
                )

    expected_literals = extract_instruction_numbers(instruction)
    literal_contract = plan.get("literal_contract")
    if not isinstance(literal_contract, list) or any(
        not isinstance(item, str) for item in literal_contract
    ):
        issues.append("literal_contract must be an array of strings")
        actual_literals: list[str] = []
    else:
        actual_literals = unique_in_order(str(item) for item in literal_contract)
        if len(actual_literals) != len(literal_contract):
            issues.append("literal_contract contains duplicates")
    missing_literals = [
        item for item in expected_literals if item not in set(actual_literals)
    ]
    extra_literals = [
        item for item in actual_literals if item not in set(expected_literals)
    ]
    if missing_literals:
        issues.append("missing protected literals: " + ", ".join(missing_literals))
    if extra_literals:
        issues.append("invented protected literals: " + ", ".join(extra_literals))

    blocks = plan.get("required_blocks")
    block_types: list[str] = []
    if not isinstance(blocks, list):
        issues.append("required_blocks must be an array")
    else:
        for index, item in enumerate(blocks):
            if not isinstance(item, dict):
                issues.append(f"required_blocks[{index}] must be an object")
                continue
            block_type = str(item.get("type", "")).upper()
            block_types.append(block_type)
            if block_type not in ALLOWED_BLOCK_TYPES:
                issues.append(f"unsupported required block type: {block_type!r}")
            if not str(item.get("purpose", "")).strip():
                issues.append(f"required_blocks[{index}] is missing purpose")

    construct_hints = expected_construct_families(instruction)
    if "timer" in construct_hints and not set(block_types) & {"TON", "TOF", "TP"}:
        issues.append("instruction contains a duration but plan has no timer block")
    if "rising_edge" in construct_hints and "R_TRIG" not in block_types:
        issues.append("instruction requires a rising-edge block")
    if "falling_edge" in construct_hints and "F_TRIG" not in block_types:
        issues.append("instruction requires a falling-edge block")
    if "counter" in construct_hints and not set(block_types) & {"CTU", "CTD", "CTUD"}:
        issues.append("instruction requires a counter block")

    logic = plan.get("ordered_logic")
    if not isinstance(logic, list) or not logic:
        issues.append("ordered_logic must be a non-empty array")
    else:
        for index, item in enumerate(logic):
            if not isinstance(item, dict):
                issues.append(f"ordered_logic[{index}] must be an object")
                continue
            if item.get("step") != index + 1:
                issues.append("ordered_logic step numbers must be consecutive from 1")
            if not str(item.get("behavior", "")).strip():
                issues.append(f"ordered_logic[{index}] is missing behavior")

    for key in ("priority_and_safety",):
        value = plan.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            issues.append(f"{key} must be an array of strings")
    for key in (
        "objective",
        "state_and_memory",
        "reset_behavior",
        "scan_cycle_behavior",
    ):
        if not isinstance(plan.get(key), str):
            issues.append(f"{key} must be a string")

    return {
        "passed": not issues,
        "issues": unique_in_order(issues),
        "expected_addresses": expected_addresses,
        "plan_addresses": actual_addresses,
        "missing_addresses": missing_addresses,
        "extra_addresses": extra_addresses,
        "expected_literals": expected_literals,
        "plan_literals": actual_literals,
        "missing_literals": missing_literals,
        "extra_literals": extra_literals,
        "construct_hints": construct_hints,
        "block_types": block_types,
    }


def validate_address_contract(instruction: str, st_code: str) -> dict[str, Any]:
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
        "issues": unique_in_order(issues),
        "declared_program": declared_name,
        "configured_programs": configured_names,
    }


def summarize_issues(*checks: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for check in checks:
        for item in check.get("issues", []) or []:
            output.append(str(item))
        for key in (
            "missing_addresses",
            "extra_addresses",
            "missing_literals",
            "extra_literals",
        ):
            values = check.get(key)
            if values:
                output.append(f"{key}: {', '.join(str(item) for item in values)}")
    return unique_in_order(output)
