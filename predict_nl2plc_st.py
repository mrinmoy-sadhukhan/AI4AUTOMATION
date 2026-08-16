#!/usr/bin/env python3
"""Deterministic ST generation plus address/structure validation."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from nl2plc_common import (
    read_jsonl,
    validate_address_contract,
    validate_st_structure,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Final adapter, merged model directory, or Hugging Face model id.",
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def strip_outer_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:st|structured-text|iecst)?\s*\n?(.*?)\n?```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else stripped


def normalized_code(text: str) -> str:
    return " ".join(text.split())


def split_prompt_reference(
    record: dict[str, Any],
) -> tuple[list[dict[str, str]], str | None]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {record.get('id')} has no messages.")
    if messages[-1].get("role") == "assistant":
        prompt = messages[:-1]
        reference = str(messages[-1].get("content", ""))
    else:
        prompt = messages
        reference = None
    return (
        [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in prompt
        ],
        reference,
    )


def task_from_prompt(messages: list[dict[str, str]]) -> str:
    user_contents = [
        message["content"] for message in messages if message["role"] == "user"
    ]
    if not user_contents:
        return ""
    content = user_contents[-1]
    match = re.search(
        r"Human PLC instruction:\n(.*?)\n\nAllowed direct PLC addresses",
        content,
        flags=re.DOTALL,
    )
    return match.group(1) if match else content


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.max_new_tokens <= 0 or args.max_seq_length <= 0:
        raise ValueError("Token limits must be positive.")

    # Unsloth must be imported before Transformers.
    from unsloth import FastLanguageModel

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Unsloth inference.")
    torch.manual_seed(args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_jsonl(args.input_jsonl)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prompt_messages, reference = split_prompt_reference(record)
        task = task_from_prompt(prompt_messages)
        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        attention_mask = torch.ones_like(inputs)
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw_prediction = tokenizer.decode(
            output[0, inputs.shape[-1] :],
            skip_special_tokens=True,
        )
        predicted_st = strip_outer_markdown_fence(raw_prediction)
        address_check = validate_address_contract(task, predicted_st)
        structure_check = validate_st_structure(predicted_st)
        result: dict[str, Any] = {
            "id": record.get("id", f"example_{index:04d}"),
            "source_row": record.get("source_row"),
            "predicted_st": predicted_st,
            "raw_prediction": raw_prediction,
            "address_check": address_check,
            "structure_check": structure_check,
            "valid_generation": (
                address_check["passed"] and structure_check["passed"]
            ),
        }
        if reference is not None:
            result["reference_st"] = reference
            result["exact_match"] = normalized_code(predicted_st) == normalized_code(
                reference
            )
            result["sequence_similarity"] = SequenceMatcher(
                None,
                normalized_code(reference),
                normalized_code(predicted_st),
                autojunk=False,
            ).ratio()
        predictions.append(result)
        print(
            f"[{index}/{len(records)}] {result['id']} "
            f"address={address_check['passed']} "
            f"structure={structure_check['passed']}"
        )

    write_jsonl(args.output_jsonl, predictions)
    comparable = [row for row in predictions if "reference_st" in row]
    metrics = {
        "examples": len(predictions),
        "address_contract_pass_rate": (
            mean(row["address_check"]["passed"] for row in predictions)
            if predictions
            else None
        ),
        "static_structure_pass_rate": (
            mean(row["structure_check"]["passed"] for row in predictions)
            if predictions
            else None
        ),
        "valid_generation_rate": (
            mean(row["valid_generation"] for row in predictions)
            if predictions
            else None
        ),
        "exact_match_rate": (
            mean(row["exact_match"] for row in comparable) if comparable else None
        ),
        "mean_sequence_similarity": (
            mean(row["sequence_similarity"] for row in comparable)
            if comparable
            else None
        ),
    }
    metrics_path = args.output_jsonl.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
