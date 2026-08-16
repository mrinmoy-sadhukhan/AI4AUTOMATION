#!/usr/bin/env python3
"""Build a verified algorithm database from NL2PLC gold JSONL examples.

The source ST program is treated as immutable gold.  The Ollama teacher is used
only to extract a compact structured algorithm from the human instruction and
the gold ST.  Every teacher response is audited before it can enter the
accepted database or the derived PLAN/CODE/DIRECT SFT dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = (
    "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
)

PLC_ADDRESS_RE = re.compile(
    r"%[IQM](?:X|B|W|D|L)?\d+(?:\.\d+)*",
    flags=re.IGNORECASE,
)
PROGRAM_DECL_RE = re.compile(
    r"(?im)^\s*PROGRAM\s+([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
PROGRAM_INSTANCE_RE = re.compile(
    r"(?im)^\s*PROGRAM\s+([A-Za-z_][A-Za-z0-9_]*)\s+WITH\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*;"
)
CONFIG_RE = re.compile(
    r"(?im)^\s*CONFIGURATION\s+([A-Za-z_][A-Za-z0-9_]*)"
)
RESOURCE_RE = re.compile(
    r"(?im)^\s*RESOURCE\s+([A-Za-z_][A-Za-z0-9_]*)\s+ON\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
TASK_RE = re.compile(
    r"(?im)^\s*TASK\s+([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\(\s*INTERVAL\s*:=\s*([^,\)]+)\s*,\s*PRIORITY\s*:=\s*([^\)]+)\)\s*;"
)
ADDRESSED_DECL_RE = re.compile(
    r"(?im)^\s*([A-Za-z_][A-Za-z0-9_]*)\s+AT\s+"
    r"(%[IQM](?:X|B|W|D|L)?\d+(?:\.\d+)*)\s*:\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*;"
)
PLAIN_DECL_RE = re.compile(
    r"(?im)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?::=.+)?;"
)
ST_BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", flags=re.DOTALL)
ST_LINE_COMMENT_RE = re.compile(r"//[^\r\n]*")

FUNCTION_BLOCK_TYPES = {
    "TON",
    "TOF",
    "TP",
    "CTU",
    "CTD",
    "CTUD",
    "R_TRIG",
    "F_TRIG",
    "RTC",
    "RS",
    "SR",
    "PID",
    "PID_FIXCYCLE",
}


ALGORITHM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "program_name": {"type": "string"},
        "summary": {"type": "string"},
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["input", "output", "in_out", "internal"],
                    },
                    "address": {"type": "string"},
                    "data_type": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": [
                    "name",
                    "role",
                    "address",
                    "data_type",
                    "purpose",
                ],
            },
        },
        "function_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "instance": {"type": "string"},
                    "type": {"type": "string"},
                    "purpose": {"type": "string"},
                    "parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["name", "value"],
                        },
                    },
                },
                "required": ["instance", "type", "purpose", "parameters"],
            },
        },
        "logic_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "timing_and_thresholds": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "value": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["value", "purpose"],
            },
        },
        "priority_rules": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reset_behavior": {"type": "string"},
        "scan_cycle_behavior": {"type": "string"},
        "configuration": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "configuration_name": {"type": "string"},
                "resource_name": {"type": "string"},
                "resource_target": {"type": "string"},
                "task_name": {"type": "string"},
                "task_interval": {"type": "string"},
                "task_priority": {"type": "string"},
                "program_instance": {"type": "string"},
            },
            "required": [
                "configuration_name",
                "resource_name",
                "resource_target",
                "task_name",
                "task_interval",
                "task_priority",
                "program_instance",
            ],
        },
    },
    "required": [
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
    ],
}


TEACHER_SYSTEM_PROMPT = """You are building supervised data for IEC 61131-3
Structured Text generation.

The supplied ST source is verified gold. Do not rewrite it, repair it, or
generate alternative code. Extract a concise, implementation-level algorithm
that faithfully explains how that exact gold source implements the human
instruction.

Rules:
1. Return only one JSON object matching the supplied schema.
2. Copy all direct PLC addresses character-for-character.
3. Do not invent an address, variable, function block, condition, threshold,
   duration, reset rule, priority rule, or configuration value.
4. Include every addressed variable and every declared timer/counter/edge
   function-block instance.
5. logic_steps must be ordered and operational, but must not contain hidden
   chain-of-thought or commentary.
6. Use an empty address string for internal variables.
7. Describe scan-cycle and reset behavior only when it is supported by the
   source. Use an empty string when no separate behavior exists."""


PLAN_SYSTEM_PROMPT = """You convert a human PLC instruction into a concise,
structured implementation algorithm for IEC 61131-3 Structured Text.
Return only valid JSON. Preserve every supplied PLC address, value, condition,
priority, timer, counter, interlock, and reset rule exactly. Do not return ST
code and do not provide hidden chain-of-thought."""


CODE_SYSTEM_PROMPT = """You generate IEC 61131-3 Structured Text from a human
PLC instruction and a verified structured algorithm.

Return only the complete ST source without Markdown or explanation. Include a
complete PROGRAM ... END_PROGRAM block and matching CONFIGURATION ...
END_CONFIGURATION block. Use exactly the allowed direct PLC addresses; never
invent, normalize, renumber, or repair an address."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def extract_addresses(text: str) -> list[str]:
    return unique_in_order(
        [match.group(0) for match in PLC_ADDRESS_RE.finditer(text)]
    )


def strip_st_comments(text: str) -> str:
    return ST_LINE_COMMENT_RE.sub(" ", ST_BLOCK_COMMENT_RE.sub(" ", text))


def extract_human_instruction(user_content: str) -> str:
    marker = "Human PLC instruction:"
    if marker not in user_content:
        return user_content.strip()
    extracted = user_content.split(marker, maxsplit=1)[1]
    for end_marker in (
        "\n\nAllowed direct PLC addresses",
        "\nAllowed direct PLC addresses",
        "\n\nGenerate the complete",
    ):
        if end_marker in extracted:
            extracted = extracted.split(end_marker, maxsplit=1)[0]
            break
    return extracted.strip()


def get_source_parts(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages array")

    user_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    assistant_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not user_messages or not assistant_messages:
        raise ValueError("record must contain user and assistant messages")

    user_content = str(user_messages[-1].get("content", "")).strip()
    gold_st = str(assistant_messages[-1].get("content", "")).strip()
    if not user_content or not gold_st:
        raise ValueError("empty user instruction or gold ST")
    return extract_human_instruction(user_content), gold_st


def parse_gold_st(gold_st: str) -> dict[str, Any]:
    executable = strip_st_comments(gold_st)
    program_match = PROGRAM_DECL_RE.search(executable)
    config_match = CONFIG_RE.search(executable)
    resource_match = RESOURCE_RE.search(executable)
    task_match = TASK_RE.search(executable)
    instance_match = PROGRAM_INSTANCE_RE.search(executable)

    addressed_variables = [
        {
            "name": name,
            "address": address,
            "data_type": data_type,
        }
        for name, address, data_type in ADDRESSED_DECL_RE.findall(executable)
    ]

    function_blocks: list[dict[str, str]] = []
    for name, data_type in PLAIN_DECL_RE.findall(executable):
        if data_type.upper() in FUNCTION_BLOCK_TYPES:
            function_blocks.append(
                {"instance": name, "type": data_type.upper()}
            )

    return {
        "program_name": program_match.group(1) if program_match else "",
        "configuration_name": config_match.group(1) if config_match else "",
        "resource_name": resource_match.group(1) if resource_match else "",
        "resource_target": resource_match.group(2) if resource_match else "",
        "task_name": task_match.group(1) if task_match else "",
        "task_interval": task_match.group(2).strip() if task_match else "",
        "task_priority": task_match.group(3).strip() if task_match else "",
        "program_instance": instance_match.group(1) if instance_match else "",
        "configured_task": instance_match.group(2) if instance_match else "",
        "configured_program": instance_match.group(3) if instance_match else "",
        "addressed_variables": addressed_variables,
        "function_blocks": function_blocks,
    }


def validate_gold_source(
    instruction: str,
    gold_st: str,
    expected_addresses: list[str],
) -> list[str]:
    issues: list[str] = []
    instruction_addresses = extract_addresses(instruction)
    gold_addresses = extract_addresses(gold_st)
    if set(instruction_addresses) != set(expected_addresses):
        issues.append(
            "source_record_addresses_do_not_match_instruction:"
            f" record={expected_addresses}, instruction={instruction_addresses}"
        )
    if set(gold_addresses) != set(expected_addresses):
        issues.append(
            "gold_address_contract_failed:"
            f" expected={expected_addresses}, gold={gold_addresses}"
        )

    upper = strip_st_comments(gold_st).upper()
    for token in (
        "PROGRAM ",
        "END_PROGRAM",
        "CONFIGURATION ",
        "END_CONFIGURATION",
    ):
        if token not in upper:
            issues.append(f"gold_missing_{token.strip().lower()}")

    parsed = parse_gold_st(gold_st)
    if not parsed["program_name"]:
        issues.append("gold_program_name_not_found")
    if (
        parsed["configured_program"]
        and parsed["configured_program"] != parsed["program_name"]
    ):
        issues.append("gold_program_configuration_mismatch")
    return issues


def extract_contract_numbers(instruction: str) -> list[str]:
    without_addresses = PLC_ADDRESS_RE.sub(" ", instruction)
    return unique_in_order(
        re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", without_addresses)
    )


def validate_algorithm(
    algorithm: dict[str, Any],
    instruction: str,
    gold_st: str,
    expected_addresses: list[str],
) -> dict[str, Any]:
    issues: list[str] = []
    parsed_gold = parse_gold_st(gold_st)

    algorithm_text = stable_json(algorithm)
    algorithm_addresses = extract_addresses(algorithm_text)
    if set(algorithm_addresses) != set(expected_addresses):
        missing = [
            address
            for address in expected_addresses
            if address not in set(algorithm_addresses)
        ]
        extra = [
            address
            for address in algorithm_addresses
            if address not in set(expected_addresses)
        ]
        if missing:
            issues.append(f"algorithm_missing_addresses:{missing}")
        if extra:
            issues.append(f"algorithm_extra_addresses:{extra}")

    if str(algorithm.get("program_name", "")).casefold() != str(
        parsed_gold["program_name"]
    ).casefold():
        issues.append(
            "program_name_mismatch:"
            f" expected={parsed_gold['program_name']},"
            f" got={algorithm.get('program_name', '')}"
        )

    variables = algorithm.get("variables")
    if not isinstance(variables, list):
        variables = []
        issues.append("variables_not_an_array")

    variable_lookup: dict[str, list[dict[str, Any]]] = {}
    for variable in variables:
        if not isinstance(variable, dict):
            issues.append("variable_not_an_object")
            continue
        address = str(variable.get("address", ""))
        if address:
            variable_lookup.setdefault(address, []).append(variable)

    for gold_variable in parsed_gold["addressed_variables"]:
        candidates = variable_lookup.get(gold_variable["address"], [])
        if not candidates:
            issues.append(
                f"missing_addressed_variable:{gold_variable['address']}"
            )
            continue
        if not any(
            str(candidate.get("name", "")).casefold()
            == gold_variable["name"].casefold()
            and str(candidate.get("data_type", "")).casefold()
            == gold_variable["data_type"].casefold()
            for candidate in candidates
        ):
            issues.append(
                "addressed_variable_mismatch:"
                f" expected={gold_variable}, candidates={candidates}"
            )

    function_blocks = algorithm.get("function_blocks")
    if not isinstance(function_blocks, list):
        function_blocks = []
        issues.append("function_blocks_not_an_array")
    algorithm_fb = {
        (
            str(item.get("instance", "")).casefold(),
            str(item.get("type", "")).upper(),
        )
        for item in function_blocks
        if isinstance(item, dict)
    }
    gold_fb = {
        (item["instance"].casefold(), item["type"].upper())
        for item in parsed_gold["function_blocks"]
    }
    for missing_fb in sorted(gold_fb - algorithm_fb):
        issues.append(f"missing_function_block:{missing_fb}")
    for extra_fb in sorted(algorithm_fb - gold_fb):
        issues.append(f"invented_function_block:{extra_fb}")

    serialized_casefold = algorithm_text.casefold()
    missing_numbers = [
        number
        for number in extract_contract_numbers(instruction)
        if re.search(
            rf"(?<!\d){re.escape(number)}(?!\d)",
            serialized_casefold,
        )
        is None
    ]
    if missing_numbers:
        issues.append(f"instruction_numbers_missing:{missing_numbers}")

    logic_steps = algorithm.get("logic_steps")
    if not isinstance(logic_steps, list) or not logic_steps:
        issues.append("logic_steps_empty")

    config = algorithm.get("configuration")
    if not isinstance(config, dict):
        config = {}
        issues.append("configuration_not_an_object")
    config_expectations = {
        "configuration_name": parsed_gold["configuration_name"],
        "resource_name": parsed_gold["resource_name"],
        "resource_target": parsed_gold["resource_target"],
        "task_name": parsed_gold["task_name"],
        "task_interval": parsed_gold["task_interval"],
        "task_priority": parsed_gold["task_priority"],
        "program_instance": parsed_gold["program_instance"],
    }
    for key, expected in config_expectations.items():
        actual = str(config.get(key, ""))
        if expected and actual.casefold() != str(expected).casefold():
            issues.append(
                f"configuration_mismatch:{key}:"
                f" expected={expected}, got={actual}"
            )

    return {
        "passed": not issues,
        "issues": issues,
        "expected_addresses": expected_addresses,
        "algorithm_addresses": algorithm_addresses,
        "gold_parse": parsed_gold,
        "contract_numbers": extract_contract_numbers(instruction),
    }


def validate_schema_shape(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> list[str]:
    """Small local validator for the subset of JSON Schema used here."""
    issues: list[str] = []
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
    }.get(expected_type, True)
    if not type_ok:
        return [f"{path}:expected_{expected_type}"]

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                issues.append(f"{path}.{key}:missing")
        if schema.get("additionalProperties") is False:
            allowed = set(schema.get("properties", {}))
            for key in value:
                if key not in allowed:
                    issues.append(f"{path}.{key}:unexpected")
        for key, child_schema in schema.get("properties", {}).items():
            if key in value:
                issues.extend(
                    validate_schema_shape(
                        value[key],
                        child_schema,
                        f"{path}.{key}",
                    )
                )
    elif isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            issues.extend(
                validate_schema_shape(
                    item,
                    schema["items"],
                    f"{path}[{index}]",
                )
            )
    elif isinstance(value, str) and "enum" in schema:
        if value not in schema["enum"]:
            issues.append(f"{path}:not_in_enum:{value}")
    return issues


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher response is not a JSON object")
    return value


def build_teacher_prompt(
    instruction: str,
    gold_st: str,
    expected_addresses: list[str],
    previous_candidate: dict[str, Any] | None = None,
    previous_issues: list[str] | None = None,
) -> str:
    address_lines = "\n".join(f"- {item}" for item in expected_addresses)
    prompt = f"""Extract the structured algorithm for this verified gold pair.

HUMAN INSTRUCTION
{instruction}

IMMUTABLE DIRECT ADDRESSES
{address_lines or "- none"}

VERIFIED GOLD ST
{gold_st}

JSON SCHEMA
{json.dumps(ALGORITHM_SCHEMA, ensure_ascii=False)}
"""
    if previous_candidate is not None and previous_issues:
        prompt += f"""

The previous algorithm candidate was rejected by deterministic validation.
Correct only the structured algorithm; never alter the gold ST.

PREVIOUS CANDIDATE
{json.dumps(previous_candidate, ensure_ascii=False)}

VALIDATION ISSUES
{json.dumps(previous_issues, ensure_ascii=False)}
"""
    return prompt


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        num_ctx: int,
        num_predict: int,
        keep_alive: str,
        think: bool | str,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.think = think

    def chat(self, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "format": ALGORITHM_SCHEMA,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {body[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.url}. Start Ollama and confirm "
                f"the model is installed. Details: {exc}"
            ) from exc

        message = result.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"Ollama response has no message: {result}")
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError("Ollama returned an empty message.content")
        algorithm = parse_json_object(content)
        audit = {
            "request": payload,
            "raw_content": content,
            "thinking": str(message.get("thinking", "")),
            "created_at": result.get("created_at"),
            "done_reason": result.get("done_reason"),
            "total_duration_ns": result.get("total_duration"),
            "load_duration_ns": result.get("load_duration"),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
            "eval_count": result.get("eval_count"),
            "eval_duration_ns": result.get("eval_duration"),
        }
        return algorithm, audit


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_examples (
            example_id TEXT PRIMARY KEY,
            source_row INTEGER,
            split TEXT NOT NULL,
            logic_family TEXT NOT NULL,
            instruction TEXT NOT NULL,
            gold_st TEXT NOT NULL,
            addresses_json TEXT NOT NULL,
            source_json TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            imported_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS teacher_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id TEXT NOT NULL REFERENCES source_examples(example_id),
            attempt_number INTEGER NOT NULL,
            model TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            status TEXT NOT NULL,
            request_json TEXT,
            raw_content TEXT,
            thinking TEXT,
            algorithm_json TEXT,
            validation_json TEXT NOT NULL,
            error TEXT,
            total_duration_ns INTEGER,
            load_duration_ns INTEGER,
            prompt_eval_count INTEGER,
            prompt_eval_duration_ns INTEGER,
            eval_count INTEGER,
            eval_duration_ns INTEGER
        );

        CREATE TABLE IF NOT EXISTS accepted_algorithms (
            example_id TEXT PRIMARY KEY REFERENCES source_examples(example_id),
            run_id INTEGER NOT NULL REFERENCES teacher_runs(run_id),
            algorithm_json TEXT NOT NULL,
            accepted_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_teacher_runs_example
        ON teacher_runs(example_id, run_id);
        """
    )
    return connection


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(value)
    return rows


def import_sources(
    connection: sqlite3.Connection,
    records: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    imported = 0
    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        example_id = str(record.get("id") or f"row_{index:04d}")
        if example_id in seen_ids:
            raise ValueError(f"duplicate example id: {example_id}")
        seen_ids.add(example_id)

        instruction, gold_st = get_source_parts(record)
        addresses_value = record.get("addresses")
        if isinstance(addresses_value, list):
            addresses = unique_in_order([str(item) for item in addresses_value])
        else:
            addresses = extract_addresses(instruction)
        split = str(record.get("split") or "train")
        source_json = stable_json(record)
        source_hash = sha256_text(
            f"{instruction}\n---GOLD-ST---\n{gold_st}\n"
            f"---ADDRESSES---\n{stable_json(addresses)}"
        )

        existing = connection.execute(
            "SELECT source_hash FROM source_examples WHERE example_id = ?",
            (example_id,),
        ).fetchone()
        if existing is not None and existing["source_hash"] != source_hash:
            raise ValueError(
                f"{example_id}: source changed after it was imported. "
                "Use a new database or a new example id."
            )

        connection.execute(
            """
            INSERT OR IGNORE INTO source_examples (
                example_id, source_row, split, logic_family, instruction,
                gold_st, addresses_json, source_json, source_hash, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                example_id,
                record.get("source_row"),
                split,
                str(record.get("logic_family") or "unknown"),
                instruction,
                gold_st,
                stable_json(addresses),
                source_json,
                source_hash,
                utc_now(),
            ),
        )
        if existing is None:
            imported += 1
        prepared.append(
            {
                "example_id": example_id,
                "source_row": record.get("source_row"),
                "split": split,
                "logic_family": str(
                    record.get("logic_family") or "unknown"
                ),
                "instruction": instruction,
                "gold_st": gold_st,
                "addresses": addresses,
                "source_record": record,
            }
        )
    connection.commit()
    return imported, prepared


def insert_run(
    connection: sqlite3.Connection,
    *,
    example_id: str,
    attempt_number: int,
    model: str,
    started_at: str,
    status: str,
    validation: dict[str, Any],
    audit: dict[str, Any] | None = None,
    algorithm: dict[str, Any] | None = None,
    error: str | None = None,
) -> int:
    audit = audit or {}
    cursor = connection.execute(
        """
        INSERT INTO teacher_runs (
            example_id, attempt_number, model, started_at, finished_at,
            status, request_json, raw_content, thinking, algorithm_json,
            validation_json, error, total_duration_ns, load_duration_ns,
            prompt_eval_count, prompt_eval_duration_ns, eval_count,
            eval_duration_ns
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            example_id,
            attempt_number,
            model,
            started_at,
            utc_now(),
            status,
            stable_json(audit.get("request"))
            if audit.get("request") is not None
            else None,
            audit.get("raw_content"),
            audit.get("thinking"),
            stable_json(algorithm) if algorithm is not None else None,
            stable_json(validation),
            error,
            audit.get("total_duration_ns"),
            audit.get("load_duration_ns"),
            audit.get("prompt_eval_count"),
            audit.get("prompt_eval_duration_ns"),
            audit.get("eval_count"),
            audit.get("eval_duration_ns"),
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def accept_algorithm(
    connection: sqlite3.Connection,
    example_id: str,
    run_id: int,
    algorithm: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO accepted_algorithms (
            example_id, run_id, algorithm_json, accepted_at
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(example_id) DO UPDATE SET
            run_id = excluded.run_id,
            algorithm_json = excluded.algorithm_json,
            accepted_at = excluded.accepted_at
        """,
        (
            example_id,
            run_id,
            stable_json(algorithm),
            utc_now(),
        ),
    )
    connection.commit()


def is_accepted(connection: sqlite3.Connection, example_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM accepted_algorithms WHERE example_id = ?",
        (example_id,),
    ).fetchone()
    return row is not None


def prior_attempt_count(
    connection: sqlite3.Connection,
    example_id: str,
) -> int:
    row = connection.execute(
        "SELECT COUNT(*) AS count FROM teacher_runs WHERE example_id = ?",
        (example_id,),
    ).fetchone()
    return int(row["count"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_plan_messages(
    instruction: str,
    addresses: list[str],
    algorithm: dict[str, Any],
) -> list[dict[str, str]]:
    address_lines = "\n".join(f"- {item}" for item in addresses) or "- none"
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: PLAN\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses "
                "(literal, immutable contract):\n"
                f"{address_lines}\n\n"
                "Return the structured implementation algorithm."
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                algorithm,
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


def make_code_messages(
    instruction: str,
    addresses: list[str],
    algorithm: dict[str, Any],
    gold_st: str,
) -> list[dict[str, str]]:
    address_lines = "\n".join(f"- {item}" for item in addresses) or "- none"
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TASK: CODE\n\n"
                f"Human PLC instruction:\n{instruction}\n\n"
                "Allowed direct PLC addresses "
                "(literal, immutable contract):\n"
                f"{address_lines}\n\n"
                "Verified structured algorithm:\n"
                f"{json.dumps(algorithm, ensure_ascii=False, indent=2)}\n\n"
                "Generate the complete IEC 61131-3 Structured Text source."
            ),
        },
        {"role": "assistant", "content": gold_st},
    ]


def make_direct_messages(source_record: dict[str, Any]) -> list[dict[str, str]]:
    messages = source_record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("source record has no messages")
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in messages
    ]


def export_database(
    connection: sqlite3.Connection,
    output_dir: Path,
) -> dict[str, Any]:
    accepted_rows = connection.execute(
        """
        SELECT
            s.*, a.algorithm_json, a.accepted_at, a.run_id,
            r.model, r.validation_json, r.prompt_eval_count, r.eval_count,
            r.total_duration_ns
        FROM source_examples AS s
        JOIN accepted_algorithms AS a USING (example_id)
        JOIN teacher_runs AS r ON r.run_id = a.run_id
        ORDER BY s.example_id
        """
    ).fetchall()

    enriched: list[dict[str, Any]] = []
    multitask: list[dict[str, Any]] = []
    task_rows: dict[str, list[dict[str, Any]]] = {
        "PLAN": [],
        "CODE": [],
        "DIRECT": [],
    }
    for row in accepted_rows:
        addresses = json.loads(row["addresses_json"])
        source_record = json.loads(row["source_json"])
        algorithm = json.loads(row["algorithm_json"])
        validation = json.loads(row["validation_json"])
        enriched_row = {
            "id": row["example_id"],
            "source_row": row["source_row"],
            "split": row["split"],
            "logic_family": row["logic_family"],
            "instruction": row["instruction"],
            "addresses": addresses,
            "algorithm": algorithm,
            "gold_st": row["gold_st"],
            "verification": validation,
            "teacher": {
                "model": row["model"],
                "run_id": row["run_id"],
                "accepted_at": row["accepted_at"],
                "prompt_tokens": row["prompt_eval_count"],
                "generated_tokens": row["eval_count"],
                "total_duration_ns": row["total_duration_ns"],
            },
        }
        enriched.append(enriched_row)

        common = {
            "source_id": row["example_id"],
            "source_row": row["source_row"],
            "split": row["split"],
            "logic_family": row["logic_family"],
            "addresses": addresses,
        }
        records = {
            "PLAN": {
                **common,
                "id": f"{row['example_id']}__plan",
                "task_type": "PLAN",
                "messages": make_plan_messages(
                    row["instruction"],
                    addresses,
                    algorithm,
                ),
            },
            "CODE": {
                **common,
                "id": f"{row['example_id']}__code",
                "task_type": "CODE",
                "messages": make_code_messages(
                    row["instruction"],
                    addresses,
                    algorithm,
                    row["gold_st"],
                ),
            },
            "DIRECT": {
                **common,
                "id": f"{row['example_id']}__direct",
                "task_type": "DIRECT",
                "messages": make_direct_messages(source_record),
            },
        }
        for task_type in ("PLAN", "CODE", "DIRECT"):
            task_rows[task_type].append(records[task_type])
            multitask.append(records[task_type])

    latest_rejected_rows = connection.execute(
        """
        SELECT s.*, r.*
        FROM source_examples AS s
        JOIN teacher_runs AS r ON r.run_id = (
            SELECT MAX(r2.run_id)
            FROM teacher_runs AS r2
            WHERE r2.example_id = s.example_id
        )
        LEFT JOIN accepted_algorithms AS a USING (example_id)
        WHERE a.example_id IS NULL
        ORDER BY s.example_id
        """
    ).fetchall()
    rejected = [
        {
            "id": row["example_id"],
            "source_row": row["source_row"],
            "split": row["split"],
            "instruction": row["instruction"],
            "addresses": json.loads(row["addresses_json"]),
            "gold_st": row["gold_st"],
            "latest_status": row["status"],
            "latest_attempt": row["attempt_number"],
            "algorithm": json.loads(row["algorithm_json"])
            if row["algorithm_json"]
            else None,
            "validation": json.loads(row["validation_json"]),
            "error": row["error"],
        }
        for row in latest_rejected_rows
    ]

    exports_dir = output_dir / "exports"
    write_jsonl(exports_dir / "enriched_master.jsonl", enriched)
    write_jsonl(exports_dir / "rejected_algorithms.jsonl", rejected)
    write_jsonl(exports_dir / "sft_multitask.jsonl", multitask)
    for task_type, rows in task_rows.items():
        write_jsonl(
            exports_dir / f"sft_{task_type.lower()}.jsonl",
            rows,
        )

    total_sources = int(
        connection.execute(
            "SELECT COUNT(*) AS count FROM source_examples"
        ).fetchone()["count"]
    )
    total_runs = int(
        connection.execute(
            "SELECT COUNT(*) AS count FROM teacher_runs"
        ).fetchone()["count"]
    )
    logic_families = Counter(row["logic_family"] for row in accepted_rows)
    report = {
        "generated_at": utc_now(),
        "source_examples": total_sources,
        "accepted_algorithms": len(accepted_rows),
        "unaccepted_examples": total_sources - len(accepted_rows),
        "teacher_runs": total_runs,
        "sft_records": len(multitask),
        "sft_record_counts": {
            key: len(value) for key, value in task_rows.items()
        },
        "accepted_logic_families": dict(sorted(logic_families.items())),
        "files": {
            "database": str(output_dir / "gold_algorithms.sqlite3"),
            "enriched": str(exports_dir / "enriched_master.jsonl"),
            "rejected": str(exports_dir / "rejected_algorithms.jsonl"),
            "multitask_sft": str(exports_dir / "sft_multitask.jsonl"),
        },
    }
    (exports_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def process_example(
    connection: sqlite3.Connection,
    client: OllamaClient,
    example: dict[str, Any],
    max_attempts: int,
    retry_delay_seconds: float,
) -> str:
    example_id = example["example_id"]
    source_issues = validate_gold_source(
        example["instruction"],
        example["gold_st"],
        example["addresses"],
    )
    if source_issues:
        attempt_number = prior_attempt_count(connection, example_id) + 1
        insert_run(
            connection,
            example_id=example_id,
            attempt_number=attempt_number,
            model=client.model,
            started_at=utc_now(),
            status="source_rejected",
            validation={"passed": False, "issues": source_issues},
            error="Gold source contract failed; teacher was not called.",
        )
        return "source_rejected"

    previous_candidate: dict[str, Any] | None = None
    previous_issues: list[str] | None = None
    for local_attempt in range(1, max_attempts + 1):
        attempt_number = prior_attempt_count(connection, example_id) + 1
        started_at = utc_now()
        prompt = build_teacher_prompt(
            example["instruction"],
            example["gold_st"],
            example["addresses"],
            previous_candidate,
            previous_issues,
        )
        try:
            algorithm, audit = client.chat(prompt)
            schema_issues = validate_schema_shape(
                algorithm,
                ALGORITHM_SCHEMA,
            )
            semantic_validation = validate_algorithm(
                algorithm,
                example["instruction"],
                example["gold_st"],
                example["addresses"],
            )
            validation = {
                "passed": not schema_issues
                and semantic_validation["passed"],
                "schema_issues": schema_issues,
                "semantic": semantic_validation,
            }
            status = "accepted" if validation["passed"] else "rejected"
            run_id = insert_run(
                connection,
                example_id=example_id,
                attempt_number=attempt_number,
                model=client.model,
                started_at=started_at,
                status=status,
                validation=validation,
                audit=audit,
                algorithm=algorithm,
            )
            if validation["passed"]:
                accept_algorithm(
                    connection,
                    example_id,
                    run_id,
                    algorithm,
                )
                return "accepted"

            previous_candidate = algorithm
            previous_issues = [
                *schema_issues,
                *semantic_validation["issues"],
            ]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            insert_run(
                connection,
                example_id=example_id,
                attempt_number=attempt_number,
                model=client.model,
                started_at=started_at,
                status="error",
                validation={"passed": False, "issues": [error]},
                error=error,
            )
            previous_candidate = None
            previous_issues = [error]

        if local_attempt < max_attempts and retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)
    return "unaccepted"


def parse_think(value: str) -> bool | str:
    normalized = value.casefold()
    if normalized == "off":
        return False
    if normalized == "on":
        return True
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and verify structured algorithms from immutable gold "
            "NL2PLC instruction/ST pairs using a local Ollama teacher."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gold_algorithm_database"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument(
        "--think",
        choices=["off", "on", "low", "medium", "high", "max"],
        default="off",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Teacher/repair attempts per example in this run.",
    )
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected rows; use --limit 1 first.",
    )
    parser.add_argument(
        "--id",
        action="append",
        dest="ids",
        help="Process only this example id; may be repeated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate rows already accepted in this database.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Rebuild JSONL/report exports without calling Ollama.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.num_ctx <= 0 or args.num_predict <= 0:
        raise ValueError("--num-ctx and --num-predict must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / "gold_algorithms.sqlite3"
    connection = connect_database(database_path)
    try:
        records = read_jsonl(args.input_jsonl)
        imported, examples = import_sources(connection, records)
        print(
            f"Loaded {len(examples)} gold rows "
            f"({imported} newly imported) into {database_path}"
        )

        if args.export_only:
            report = export_database(connection, args.output_dir)
            print(json.dumps(report, indent=2))
            return 0

        if args.ids:
            requested = set(args.ids)
            examples = [
                item for item in examples if item["example_id"] in requested
            ]
            missing = requested - {
                item["example_id"] for item in examples
            }
            if missing:
                raise ValueError(f"Unknown --id values: {sorted(missing)}")
        if not args.force:
            examples = [
                item
                for item in examples
                if not is_accepted(connection, item["example_id"])
            ]
        if args.limit is not None:
            examples = examples[: args.limit]

        if not examples:
            print("No rows require teacher generation; rebuilding exports.")
            report = export_database(connection, args.output_dir)
            print(json.dumps(report, indent=2))
            return 0

        client = OllamaClient(
            base_url=args.ollama_url,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            keep_alive=args.keep_alive,
            think=parse_think(args.think),
        )

        run_counts: Counter[str] = Counter()
        for index, example in enumerate(examples, start=1):
            status = process_example(
                connection,
                client,
                example,
                max_attempts=args.attempts,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            run_counts[status] += 1
            print(
                f"[{index}/{len(examples)}] "
                f"{example['example_id']} -> {status}",
                flush=True,
            )

        report = export_database(connection, args.output_dir)
        print("Run status:", dict(run_counts))
        print(json.dumps(report, indent=2))
        return 0 if run_counts["accepted"] == len(examples) else 2
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed rows are already committed.", file=sys.stderr)
        raise SystemExit(130)
