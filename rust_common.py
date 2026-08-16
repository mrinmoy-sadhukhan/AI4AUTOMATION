#!/usr/bin/env python3
"""Shared JSONL, Rust-contract, and compiler utilities for NL2PLC Task B."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PLC_ADDRESS_RE = re.compile(
    r"%[IQM](?:X|B|W|D|L)?\d+(?:\.\d+)*",
    flags=re.IGNORECASE,
)
RUST_FENCE_RE = re.compile(
    r"^\s*```(?:rust)?\s*\n?(.*?)\n?```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
RUST_REQUIRED_METHODS = {
    "%IX": "read_discrete_input",
    "%QX": "write_coil",
    "%IW": "read_input_register",
    "%QW": "write_holding_register",
}


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
    stripped = str(text).strip()
    match = RUST_FENCE_RE.fullmatch(stripped)
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


def extract_instruction(record: dict[str, Any]) -> str:
    for key in ("instruction", "task", "TASK"):
        if str(record.get(key, "")).strip():
            return str(record[key]).strip()
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
            "\n\nValidated canonical behavioural plan",
            "\n\nGenerate the complete",
        ):
            if end_marker in content:
                content = content.split(end_marker, maxsplit=1)[0]
                break
    if not content.strip():
        raise ValueError(f"{record.get('id')}: empty instruction")
    return content.strip()


def extract_plan(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("generated_plan", "canonical_plan", "plan"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError(
        f"{record.get('id')}: missing generated_plan/canonical_plan object"
    )


def extract_st(record: dict[str, Any]) -> str:
    for key in ("predicted_st", "gold_st", "st_code", "generated_st"):
        value = record.get(key)
        if str(value or "").strip():
            return strip_outer_markdown_fence(str(value))
    code = str(record.get("code", "")).strip()
    if re.search(r"(?im)^\s*PROGRAM\s+", code):
        return strip_outer_markdown_fence(code)
    raise ValueError(f"{record.get('id')}: missing ST code")


def address_family(address: str) -> str | None:
    upper = address.upper()
    for family in RUST_REQUIRED_METHODS:
        if upper.startswith(family):
            return family
    return None


def required_api_for_address(address: str) -> str | None:
    family = address_family(address)
    return RUST_REQUIRED_METHODS.get(family or "")


def _method_uses_address(code: str, method: str, address: str) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(method)}\s*\(\s*\"{re.escape(address)}\"",
            code,
        )
    )


def validate_rust_address_contract(
    instruction: str,
    rust_code: str,
) -> dict[str, Any]:
    expected = extract_plc_addresses(instruction)
    actual = extract_plc_addresses(rust_code)
    expected_set = set(expected)
    actual_set = set(actual)
    missing = [item for item in expected if item not in actual_set]
    extra = [item for item in actual if item not in expected_set]
    changed_case = [
        actual_item
        for actual_item in actual
        if actual_item.upper() in {item.upper() for item in expected}
        and actual_item not in expected_set
    ]
    issues: list[str] = []
    if missing:
        issues.append("missing PLC addresses: " + ", ".join(missing))
    if extra:
        issues.append("invented PLC addresses: " + ", ".join(extra))
    if changed_case:
        issues.append(
            "PLC address spelling/case changed: " + ", ".join(changed_case)
        )
    return {
        "passed": not issues,
        "issues": issues,
        "expected_addresses": expected,
        "actual_addresses": actual,
        "missing_addresses": missing,
        "extra_addresses": extra,
        "changed_case_addresses": changed_case,
    }


def validate_rust_io_contract(
    instruction: str,
    rust_code: str,
) -> dict[str, Any]:
    expected = extract_plc_addresses(instruction)
    issues: list[str] = []
    checks: list[dict[str, Any]] = []
    all_methods = sorted(set(RUST_REQUIRED_METHODS.values()))
    for address in expected:
        required = required_api_for_address(address)
        if required is None:
            issues.append(f"unsupported PLC address family: {address}")
            continue
        found_correct = _method_uses_address(rust_code, required, address)
        wrong = [
            method
            for method in all_methods
            if method != required and _method_uses_address(rust_code, method, address)
        ]
        if not found_correct:
            issues.append(f"{address} must use {required}(\"{address}\", ...)")
        if wrong:
            issues.append(
                f"{address} uses wrong PLC I/O API(s): {', '.join(wrong)}"
            )
        checks.append(
            {
                "address": address,
                "required_method": required,
                "correct_method_found": found_correct,
                "wrong_methods_found": wrong,
            }
        )

    forbidden_numeric_apis = sorted(
        set(
            re.findall(
                r"\b(read_coils|read_discrete_inputs|read_holding_registers|"
                r"read_input_registers|write_single_coil|write_single_register)\s*\(",
                rust_code,
            )
        )
    )
    if forbidden_numeric_apis:
        issues.append(
            "raw numeric Modbus APIs are forbidden in program.rs: "
            + ", ".join(forbidden_numeric_apis)
        )
    return {
        "passed": not issues,
        "issues": unique_in_order(issues),
        "checks": checks,
        "forbidden_numeric_apis": forbidden_numeric_apis,
    }


def validate_rust_structure(rust_code: str) -> dict[str, Any]:
    issues: list[str] = []
    required_patterns = {
        "missing_plcio_import_or_reference": r"\bPlcIo\b",
        "missing_plcresult_import_or_reference": r"\bPlcResult\b",
        "missing_program_struct": r"(?m)^\s*pub\s+struct\s+Program\b",
        "missing_program_impl": r"(?m)^\s*impl\s+Program\b",
        "missing_scan_method": r"\bpub\s+fn\s+scan(?:\s*<[^>]+>)?\s*\(",
        "missing_mutable_self": r"&\s*mut\s+self",
        "missing_result_return": r"PlcResult\s*<\s*\(\s*\)\s*>",
    }
    for issue, pattern in required_patterns.items():
        if re.search(pattern, rust_code) is None:
            issues.append(issue)
    if "```" in rust_code:
        issues.append("markdown_fence_present")
    if re.search(r"(?m)^\s*(?:pub\s+)?fn\s+main\s*\(", rust_code):
        issues.append("program_rs_must_not_define_main")
    if re.search(r"\bunsafe\s*\{", rust_code):
        issues.append("unsafe_block_forbidden")
    if re.search(r"\bextern\s+crate\b", rust_code):
        issues.append("external_crate_forbidden")
    if re.search(r"\bmodbus\b|\btokio\b", rust_code, flags=re.IGNORECASE):
        issues.append("transport_or_async_runtime_must_not_be_in_program_rs")
    return {"passed": not issues, "issues": unique_in_order(issues)}


def summarize_issues(*checks: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for check in checks:
        output.extend(str(item) for item in check.get("issues", []) or [])
    return unique_in_order(output)


def cargo_check_program(
    rust_code: str,
    runtime_root: Path,
    *,
    timeout: int = 120,
    allow_missing_cargo: bool = False,
) -> dict[str, Any]:
    cargo = shutil.which("cargo")
    if cargo is None:
        if allow_missing_cargo:
            return {
                "available": False,
                "passed": None,
                "returncode": None,
                "diagnostics": "cargo executable not found; compilation not attempted",
            }
        return {
            "available": False,
            "passed": False,
            "returncode": None,
            "diagnostics": "cargo executable not found",
        }
    required = [
        runtime_root / "Cargo.toml",
        runtime_root / "src" / "lib.rs",
        runtime_root / "src" / "main.rs",
        runtime_root / "src" / "plc_io.rs",
        runtime_root / "src" / "function_blocks.rs",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "available": True,
            "passed": False,
            "returncode": None,
            "diagnostics": "runtime files missing: " + ", ".join(missing),
        }
    try:
        with tempfile.TemporaryDirectory(prefix="nl2plc_rust_check_") as temp_name:
            temp_root = Path(temp_name) / "crate"
            shutil.copytree(runtime_root, temp_root)
            (temp_root / "src" / "program.rs").write_text(
                rust_code.rstrip() + "\n",
                encoding="utf-8",
            )
            process = subprocess.run(
                [cargo, "check", "--quiet", "--offline"],
                cwd=temp_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "passed": False,
            "returncode": None,
            "diagnostics": f"cargo check timed out after {exc.timeout} seconds",
        }
    diagnostics = (process.stderr or process.stdout or "").strip()
    if len(diagnostics) > 12000:
        diagnostics = diagnostics[-12000:]
    return {
        "available": True,
        "passed": process.returncode == 0,
        "returncode": process.returncode,
        "diagnostics": diagnostics,
    }


def validate_rust_candidate(
    instruction: str,
    rust_code: str,
    runtime_root: Path,
    *,
    cargo_timeout: int = 120,
    allow_missing_cargo: bool = False,
) -> dict[str, Any]:
    structure = validate_rust_structure(rust_code)
    address = validate_rust_address_contract(instruction, rust_code)
    io_contract = validate_rust_io_contract(instruction, rust_code)
    static_passed = bool(
        structure["passed"] and address["passed"] and io_contract["passed"]
    )
    if static_passed:
        compiler = cargo_check_program(
            rust_code,
            runtime_root,
            timeout=cargo_timeout,
            allow_missing_cargo=allow_missing_cargo,
        )
    else:
        compiler = {
            "available": shutil.which("cargo") is not None,
            "passed": False,
            "returncode": None,
            "diagnostics": "cargo check skipped because static validation failed",
        }
    compiler_ok = compiler["passed"] is True or (
        allow_missing_cargo and compiler["passed"] is None
    )
    return {
        "passed": bool(static_passed and compiler_ok),
        "static_passed": static_passed,
        "structure_check": structure,
        "address_check": address,
        "io_contract_check": io_contract,
        "compiler_check": compiler,
        "issues": summarize_issues(structure, address, io_contract)
        + ([compiler["diagnostics"]] if not compiler_ok else []),
    }


def normalized(text: str) -> str:
    return " ".join(text.split())


def sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        normalized(left),
        normalized(right),
        autojunk=False,
    ).ratio()
