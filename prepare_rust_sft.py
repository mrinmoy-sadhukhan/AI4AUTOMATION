#!/usr/bin/env python3
"""Generate, compile, review, and package verified Rust LoRA supervision."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from rust_common import (
    extract_instruction,
    extract_plan,
    extract_plc_addresses,
    extract_st,
    parse_json_object,
    read_jsonl,
    strip_outer_markdown_fence,
    validate_rust_candidate,
    write_jsonl,
)
from rust_prompts import (
    make_repair_rust_messages,
    make_review_messages,
    make_rust_code_messages,
)


DEFAULT_TEACHER = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
DEFAULT_REVIEWER = "nemotron-cascade-2:30b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-master", type=Path, required=True)
    parser.add_argument("--validation-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--reviewer-model", default=None)
    parser.add_argument("--require-reviewer-pass", action="store_true")
    parser.add_argument("--teacher-num-ctx", type=int, default=16384)
    parser.add_argument("--teacher-max-tokens", type=int, default=4096)
    parser.add_argument("--reviewer-num-ctx", type=int, default=16384)
    parser.add_argument("--reviewer-max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cargo-timeout", type=int, default=120)
    parser.add_argument(
        "--max-repair-attempts", type=int, default=2, choices=(0, 1, 2, 3)
    )
    parser.add_argument(
        "--repairs-per-source", type=int, default=1, choices=(0, 1, 2, 3)
    )
    parser.add_argument("--allow-missing-cargo", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class OllamaClient:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.endpoint = base_url.rstrip("/") + "/api/chat"
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        num_ctx: int,
        num_predict: int,
        temperature: float,
        seed: int,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
                "seed": seed,
            },
        }
        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.endpoint}: {exc}") from exc
        message = result.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"Ollama response has no message object: {result}")
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response")
        return {
            "content": content,
            "thinking": str(message.get("thinking", "")),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
            "total_duration": result.get("total_duration"),
            "wall_seconds": time.monotonic() - started,
        }


def review_candidate(
    client: OllamaClient,
    args: argparse.Namespace,
    instruction: str,
    plan: dict[str, Any],
    st_code: str,
    rust_code: str,
    seed: int,
) -> dict[str, Any] | None:
    if not args.reviewer_model:
        return None
    response = client.chat(
        args.reviewer_model,
        make_review_messages(instruction, plan, st_code, rust_code),
        num_ctx=args.reviewer_num_ctx,
        num_predict=args.reviewer_max_tokens,
        temperature=0.0,
        seed=seed,
        json_mode=True,
    )
    try:
        parsed = parse_json_object(response["content"])
        verdict = str(parsed.get("verdict", "")).strip().lower()
        raw_issues = parsed.get("issues", [])
        issues = (
            [str(item).strip() for item in raw_issues if str(item).strip()]
            if isinstance(raw_issues, list)
            else [str(raw_issues).strip()]
        )
        passed = verdict == "pass" and not issues
        if verdict not in {"pass", "fail"}:
            issues.append("reviewer verdict must be pass or fail")
            passed = False
    except ValueError as exc:
        parsed = None
        verdict = "invalid_json"
        issues = [f"reviewer returned invalid JSON: {exc}"]
        passed = False
    return {
        "passed": passed,
        "verdict": verdict,
        "issues": issues,
        "parsed": parsed,
        "raw_output": response["content"],
        "usage": {key: value for key, value in response.items() if key != "content"},
    }


def bounded_errors(errors: list[str], max_chars: int = 8000) -> list[str]:
    output: list[str] = []
    used = 0
    for error in errors:
        text = " ".join(str(error).split())
        if not text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        output.append(text)
        used += len(text)
    return output or ["candidate failed validation"]


def validate_source(record: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    instruction = extract_instruction(record)
    plan = extract_plan(record)
    st_code = extract_st(record)
    expected = extract_plc_addresses(instruction)
    plan_addresses = extract_plc_addresses(json.dumps(plan, ensure_ascii=False))
    st_addresses = extract_plc_addresses(st_code)
    if set(plan_addresses) != set(expected):
        raise ValueError(
            f"{record.get('id')}: canonical plan address contract is invalid"
        )
    if set(st_addresses) != set(expected):
        raise ValueError(f"{record.get('id')}: gold ST address contract is invalid")
    return instruction, plan, st_code


def generate_record(
    client: OllamaClient,
    args: argparse.Namespace,
    runtime_root: Path,
    record: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    source_id = str(record.get("id", f"row_{index:04d}"))
    instruction, plan, st_code = validate_source(record)
    addresses = extract_plc_addresses(instruction)
    history: list[dict[str, Any]] = []
    messages = make_rust_code_messages(instruction, addresses, plan, st_code)
    previous_code = ""
    previous_errors: list[str] = []
    accepted = False
    final_code = ""
    final_check: dict[str, Any] | None = None
    final_review: dict[str, Any] | None = None

    for attempt in range(args.max_repair_attempts + 1):
        task = "RUST_CODE" if attempt == 0 else "REPAIR_RUST"
        response = client.chat(
            args.teacher_model,
            messages,
            num_ctx=args.teacher_num_ctx,
            num_predict=args.teacher_max_tokens,
            temperature=args.temperature,
            seed=args.seed + index + attempt,
        )
        code = strip_outer_markdown_fence(response["content"])
        check = validate_rust_candidate(
            instruction,
            code,
            runtime_root,
            cargo_timeout=args.cargo_timeout,
            allow_missing_cargo=args.allow_missing_cargo,
        )
        review = None
        if check["passed"] and args.reviewer_model:
            review = review_candidate(
                client,
                args,
                instruction,
                plan,
                st_code,
                code,
                args.seed + index + attempt,
            )
        review_passed = review is None or bool(review["passed"])
        accepted = bool(
            check["passed"]
            and (review_passed or not args.require_reviewer_pass)
        )
        errors = list(check.get("issues", []))
        if review is not None and not review["passed"]:
            errors.extend(f"semantic reviewer: {item}" for item in review["issues"])
        history.append(
            {
                "attempt": attempt,
                "task_type": task,
                "rust_code": code,
                "validation": check,
                "review": review,
                "accepted": accepted,
                "teacher_usage": {
                    key: value for key, value in response.items() if key != "content"
                },
            }
        )
        final_code = code
        final_check = check
        final_review = review
        if accepted:
            break
        previous_code = code
        previous_errors = bounded_errors(errors)
        messages = make_repair_rust_messages(
            instruction,
            addresses,
            plan,
            st_code,
            previous_code,
            previous_errors,
        )

    return {
        "id": source_id,
        "source_row": record.get("source_row"),
        "split": record.get("split"),
        "logic_family": record.get("logic_family"),
        "instruction": instruction,
        "addresses": addresses,
        "canonical_plan": plan,
        "gold_st": st_code,
        "accepted": accepted,
        "final_rust": final_code,
        "final_validation": final_check,
        "final_review": final_review,
        "attempts_used": max(0, len(history) - 1),
        "history": history,
    }


def make_code_sft(audit: dict[str, Any]) -> dict[str, Any]:
    messages = make_rust_code_messages(
        audit["instruction"],
        audit["addresses"],
        audit["canonical_plan"],
        audit["gold_st"],
    )
    messages.append({"role": "assistant", "content": audit["final_rust"]})
    return {
        "id": f"{audit['id']}__rust_code",
        "source_id": audit["id"],
        "source_row": audit.get("source_row"),
        "split": audit.get("split"),
        "logic_family": audit.get("logic_family"),
        "task_type": "RUST_CODE",
        "addresses": audit["addresses"],
        "messages": messages,
    }


def make_repair_sft(audit: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not audit["accepted"]:
        return []
    failed = [item for item in audit["history"] if not item["accepted"]]
    output: list[dict[str, Any]] = []
    for item in failed[-limit:]:
        errors = list(item["validation"].get("issues", []))
        review = item.get("review")
        if isinstance(review, dict) and not review.get("passed"):
            errors.extend(
                f"semantic reviewer: {issue}" for issue in review.get("issues", [])
            )
        messages = make_repair_rust_messages(
            audit["instruction"],
            audit["addresses"],
            audit["canonical_plan"],
            audit["gold_st"],
            item["rust_code"],
            bounded_errors(errors),
        )
        messages.append({"role": "assistant", "content": audit["final_rust"]})
        output.append(
            {
                "id": f"{audit['id']}__rust_repair__{item['attempt']}",
                "source_id": audit["id"],
                "source_row": audit.get("source_row"),
                "split": audit.get("split"),
                "logic_family": audit.get("logic_family"),
                "task_type": "REPAIR_RUST",
                "addresses": audit["addresses"],
                "messages": messages,
            }
        )
    return output


def process_split(
    split_name: str,
    records: list[dict[str, Any]],
    client: OllamaClient,
    args: argparse.Namespace,
    runtime_root: Path,
) -> list[dict[str, Any]]:
    audit_path = args.output_dir / f"{split_name}_generation_audit.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and audit_path.is_file():
        existing = {str(row["id"]): row for row in read_jsonl(audit_path)}
    output: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        source_id = str(record.get("id", f"row_{index:04d}"))
        cached = existing.get(source_id)
        if cached is not None and (cached.get("accepted") or not args.retry_rejected):
            audit = cached
            state = "resumed"
        else:
            try:
                audit = generate_record(client, args, runtime_root, record, index)
            except Exception as exc:
                audit = {
                    "id": source_id,
                    "source_row": record.get("source_row"),
                    "split": record.get("split", split_name),
                    "logic_family": record.get("logic_family"),
                    "accepted": False,
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                    "history": [],
                }
            state = "accepted" if audit.get("accepted") else "rejected"
        output.append(audit)
        write_jsonl(audit_path, output)
        print(
            f"[{split_name.upper()} {index}/{len(records)}] {source_id} "
            f"-> {state}; attempts={len(audit.get('history', []))}"
        )
    return output


def main() -> None:
    args = parse_args()
    for path in (args.train_master, args.validation_master):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.require_reviewer_pass and not args.reviewer_model:
        raise ValueError("--require-reviewer-pass requires --reviewer-model")
    if args.runtime_root is None:
        args.runtime_root = Path(__file__).resolve().parent / "rust_runtime"
    if not args.runtime_root.is_dir():
        raise FileNotFoundError(args.runtime_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(args.train_master)
    validation = read_jsonl(args.validation_master)
    if args.max_examples is not None:
        train = train[: args.max_examples]
        validation = validation[: args.max_examples]
    train_ids = {str(row.get("id")) for row in train}
    validation_ids = {str(row.get("id")) for row in validation}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(f"train/validation source leakage: {overlap[:10]}")

    client = OllamaClient(args.ollama_url, args.timeout)
    train_audit = process_split(
        "train", train, client, args, args.runtime_root
    )
    validation_audit = process_split(
        "validation", validation, client, args, args.runtime_root
    )
    accepted_train = [row for row in train_audit if row.get("accepted")]
    accepted_validation = [row for row in validation_audit if row.get("accepted")]
    rejected_train = [row for row in train_audit if not row.get("accepted")]
    rejected_validation = [
        row for row in validation_audit if not row.get("accepted")
    ]
    if not accepted_train:
        raise ValueError("no accepted Rust training records")
    if not accepted_validation:
        raise ValueError("no accepted Rust validation records")

    outputs = {
        "train_rust_coder": [make_code_sft(row) for row in accepted_train],
        "train_rust_repair": [
            repair
            for row in accepted_train
            for repair in make_repair_sft(row, args.repairs_per_source)
        ],
        "validation_rust_coder": [
            make_code_sft(row) for row in accepted_validation
        ],
        "validation_rust_repair": [
            repair
            for row in accepted_validation
            for repair in make_repair_sft(row, args.repairs_per_source)
        ],
        "verified_train_rust_master": accepted_train,
        "verified_validation_rust_master": accepted_validation,
        "rejected_train_rust": rejected_train,
        "rejected_validation_rust": rejected_validation,
    }
    for name, rows in outputs.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    family_counts = Counter(
        str(row.get("logic_family", "unknown")) for row in accepted_train
    )
    repaired = [
        row
        for row in [*accepted_train, *accepted_validation]
        if row.get("attempts_used", 0) > 0
    ]
    report = {
        "teacher_model": args.teacher_model,
        "reviewer_model": args.reviewer_model,
        "reviewer_required": args.require_reviewer_pass,
        "cargo_required": not args.allow_missing_cargo,
        "train_sources": len(train),
        "validation_sources": len(validation),
        "accepted_train_sources": len(accepted_train),
        "accepted_validation_sources": len(accepted_validation),
        "rejected_train_sources": len(rejected_train),
        "rejected_validation_sources": len(rejected_validation),
        "sources_recovered_by_repair": len(repaired),
        "accepted_train_by_logic_family": dict(sorted(family_counts.items())),
        "files": {
            name: {"path": str(args.output_dir / f"{name}.jsonl"), "rows": len(rows)}
            for name, rows in outputs.items()
        },
    }
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

