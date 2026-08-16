#!/usr/bin/env python3
"""QLoRA SFT for NL-to-IEC-61131-3-ST with Unsloth."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-Coder-7B-Instruct on strict NL2PLC ST data."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Unsloth/Hugging Face model id or local model directory.",
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--drop-overlength",
        action="store_true",
        help=(
            "Drop examples longer than --max-seq-length. The default is to stop "
            "instead of silently losing part of an ST program."
        ),
    )
    parser.add_argument(
        "--merge-16bit",
        action="store_true",
        help="Also save a merged 16-bit Hugging Face model after training.",
    )
    parser.add_argument(
        "--save-gguf",
        action="store_true",
        help="Also export a q4_k_m GGUF after training (slow and disk intensive).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
    }
    bad = {name: value for name, value in positive_values.items() if value <= 0}
    if bad:
        raise ValueError(f"These values must be positive: {bad}")
    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps.")
    for path in (args.train_jsonl, args.validation_jsonl):
        if not path.is_file():
            raise FileNotFoundError(path)


def supported_kwargs(callable_object: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return candidates
    return {key: value for key, value in candidates.items() if key in parameters}


def extract_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Record {example.get('id')} has invalid messages.")
    if messages[-1].get("role") != "assistant":
        raise ValueError(
            f"Record {example.get('id')} must end with an assistant ST answer."
        )
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def prepare_dataset(raw_dataset: Any) -> Any:
    def convert(example: dict[str, Any]) -> dict[str, Any]:
        messages = extract_messages(example)
        return {
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }

    return raw_dataset.map(
        convert,
        remove_columns=raw_dataset.column_names,
        desc="Build prompt/completion conversations",
    )


def token_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(token_ids, "shape"):
        return int(token_ids.shape[-1])
    return len(token_ids)


def enforce_length_contract(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
    drop_overlength: bool,
    split_name: str,
) -> tuple[Any, dict[str, Any]]:
    lengths: list[int] = []
    overlength_indices: list[int] = []
    overlength_ids: list[str] = []
    for index, example in enumerate(dataset):
        length = token_length(tokenizer, extract_messages(example))
        lengths.append(length)
        if length > max_seq_length:
            overlength_indices.append(index)
            overlength_ids.append(str(example.get("id", index)))

    if overlength_indices and not drop_overlength:
        preview = ", ".join(overlength_ids[:10])
        raise ValueError(
            f"{split_name} has {len(overlength_indices)} examples longer than "
            f"{max_seq_length} tokens ({preview}). Increase --max-seq-length or "
            "explicitly use --drop-overlength."
        )

    if overlength_indices:
        blocked = set(overlength_indices)
        dataset = dataset.select(
            [index for index in range(len(dataset)) if index not in blocked]
        )
        kept_lengths = [
            length for index, length in enumerate(lengths) if index not in blocked
        ]
    else:
        kept_lengths = lengths

    sorted_lengths = sorted(kept_lengths)
    p95_index = max(0, math.ceil(0.95 * len(sorted_lengths)) - 1)
    stats = {
        "examples_before_length_filter": len(lengths),
        "examples_after_length_filter": len(kept_lengths),
        "dropped_overlength": len(overlength_indices),
        "min_tokens": min(sorted_lengths) if sorted_lengths else None,
        "mean_tokens": (
            sum(sorted_lengths) / len(sorted_lengths) if sorted_lengths else None
        ),
        "p95_tokens": sorted_lengths[p95_index] if sorted_lengths else None,
        "max_tokens": max(sorted_lengths) if sorted_lengths else None,
    }
    return dataset, stats


def build_sft_config(SFTConfig: Any, args: argparse.Namespace, use_bf16: bool) -> Any:
    candidates: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "packing": False,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "eos_token": "<|im_end|>",
        "dataset_num_proc": args.dataset_num_proc,
        "max_length": args.max_seq_length,
        "max_seq_length": args.max_seq_length,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": True,
    }
    config_kwargs = supported_kwargs(SFTConfig, candidates)
    if "eval_strategy" not in config_kwargs and "evaluation_strategy" not in config_kwargs:
        raise RuntimeError(
            "Installed TRL/Transformers does not expose a supported evaluation "
            "strategy argument. Upgrade Unsloth and TRL."
        )
    return SFTConfig(**config_kwargs)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import Unsloth first so its patches are applied before Transformers/TRL.
    import unsloth
    from unsloth import FastLanguageModel, is_bfloat16_supported

    import datasets
    import torch
    import transformers
    import trl
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Unsloth QLoRA configuration.")

    use_bf16 = bool(is_bfloat16_supported())
    print("[VERSIONS]")
    print("  Python:", platform.python_version())
    print("  PyTorch:", torch.__version__)
    print("  Transformers:", transformers.__version__)
    print("  Datasets:", datasets.__version__)
    print("  TRL:", trl.__version__)
    print("  Unsloth:", getattr(unsloth, "__version__", "unknown"))
    print("[GPU]", torch.cuda.get_device_name(0))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_jsonl),
            "validation": str(args.validation_jsonl),
        },
    )
    train_raw, train_stats = enforce_length_contract(
        dataset["train"],
        tokenizer,
        args.max_seq_length,
        args.drop_overlength,
        "train",
    )
    validation_raw, validation_stats = enforce_length_contract(
        dataset["validation"],
        tokenizer,
        args.max_seq_length,
        args.drop_overlength,
        "validation",
    )
    train_dataset = prepare_dataset(train_raw)
    validation_dataset = prepare_dataset(validation_raw)
    print("[TOKEN LENGTHS]")
    print(json.dumps({"train": train_stats, "validation": validation_stats}, indent=2))

    training_args = build_sft_config(SFTConfig, args, use_bf16)
    trainer_candidates = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }
    trainer_kwargs = supported_kwargs(SFTTrainer.__init__, trainer_candidates)
    # Prefer the current "processing_class" name when both are accepted through
    # a permissive **kwargs signature.
    if "processing_class" in trainer_kwargs:
        trainer_kwargs.pop("tokenizer", None)
    trainer = SFTTrainer(**trainer_kwargs)

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
        if args.resume_from_checkpoint
        else None
    )
    eval_metrics = trainer.evaluate()

    final_adapter = args.output_dir / "final_adapter"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    trainer.save_state()

    run_summary = {
        "model_name": args.model_name,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "max_seq_length": args.max_seq_length,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "token_length_stats": {
            "train": train_stats,
            "validation": validation_stats,
        },
        "final_adapter": str(final_adapter.resolve()),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    if args.merge_16bit:
        merged_dir = args.output_dir / "merged_16bit"
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    if args.save_gguf:
        gguf_dir = args.output_dir / "gguf_q4_k_m"
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )

    print("[DONE]")
    print(json.dumps(run_summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
