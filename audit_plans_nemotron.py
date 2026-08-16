#!/usr/bin/env python3
"""Optional independent semantic review of gold-derived plans with Nemotron."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from nl2plc_validation import read_jsonl, write_jsonl
from two_stage_prompts import canonical_algorithm


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "review", "fail"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "instruction_gold_conflict",
                            "missing_behavior",
                            "invented_behavior",
                            "wrong_priority",
                            "wrong_timer_counter",
                            "wrong_reset_scan",
                            "wrong_address_or_value",
                            "other",
                        ],
                    },
                    "detail": {"type": "string"},
                },
                "required": ["category", "detail"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["verdict", "issues", "summary"],
}


SYSTEM_PROMPT = """You are an independent semantic auditor for an NL2PLC
training dataset. Compare the human instruction, immutable gold ST, and candidate
structured algorithm.

Judge whether the algorithm faithfully describes what the exact gold ST does and
whether the instruction and gold ST are mutually consistent. Check priorities,
reset/scan-cycle behavior, timers, counters, edges, latches, thresholds, values,
and all direct addresses. Do not repair or rewrite the gold ST. Return only the
requested JSON review. Use pass only when no material issue exists; use review
when human/compile inspection is needed; use fail for a clear contradiction."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-enriched", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="nemotron-3-nano:30b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama returned a non-object response")
    return parsed


def review_one(
    row: dict[str, Any],
    *,
    model: str,
    ollama_url: str,
    num_ctx: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    prompt = (
        f"EXAMPLE ID\n{row['id']}\n\n"
        f"HUMAN INSTRUCTION\n{row['instruction']}\n\n"
        f"IMMUTABLE GOLD ST\n{row['gold_st']}\n\n"
        "CANDIDATE STRUCTURED ALGORITHM\n"
        f"{canonical_algorithm(row['algorithm'])}"
    )
    payload = {
        "model": model,
        "stream": False,
        "think": True,
        "format": REVIEW_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
        },
    }
    response = post_json(
        ollama_url.rstrip("/") + "/api/chat",
        payload,
        timeout,
    )
    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama response has no message object")
    content = str(message.get("content", ""))
    review = json.loads(content)
    if not isinstance(review, dict):
        raise ValueError("review is not an object")
    if review.get("verdict") not in {"pass", "review", "fail"}:
        raise ValueError("invalid review verdict")
    return {
        "id": str(row["id"]),
        "verdict": review["verdict"],
        "issues": review.get("issues", []),
        "summary": str(review.get("summary", "")),
        "reviewer_model": model,
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "total_duration_ns": response.get("total_duration"),
    }


def main() -> None:
    args = parse_args()
    if args.num_ctx <= 0 or args.retries <= 0 or args.timeout <= 0:
        raise ValueError("num-ctx, retries, and timeout must be positive")
    rows = read_jsonl(args.input_enriched)
    if args.limit is not None:
        rows = rows[: args.limit]
    prior: list[dict[str, Any]] = []
    if args.output_jsonl.exists() and not args.no_resume:
        prior = read_jsonl(args.output_jsonl)
    by_id = {str(row["id"]): row for row in prior}

    for index, row in enumerate(rows, start=1):
        source_id = str(row["id"])
        if source_id in by_id:
            print(f"[{index}/{len(rows)}] {source_id} -> cached")
            continue
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                review = review_one(
                    row,
                    model=args.model,
                    ollama_url=args.ollama_url,
                    num_ctx=args.num_ctx,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 8))
                continue
            by_id[source_id] = review
            break
        else:
            by_id[source_id] = {
                "id": source_id,
                "verdict": "review",
                "issues": [{"category": "other", "detail": last_error}],
                "summary": "Automated review failed; inspect manually.",
                "reviewer_model": args.model,
            }
        ordered = [by_id[str(item["id"])] for item in rows if str(item["id"]) in by_id]
        write_jsonl(args.output_jsonl, ordered)
        print(
            f"[{index}/{len(rows)}] {source_id} "
            f"-> {by_id[source_id]['verdict']}"
        )


if __name__ == "__main__":
    main()

