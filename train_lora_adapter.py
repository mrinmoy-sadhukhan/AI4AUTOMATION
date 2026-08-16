#!/usr/bin/env python3
"""Train either the planner LoRA or coder LoRA from the same clean 7B base."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
ROLE_TASKS = {
    "planner": {"PLAN", "REPAIR_PLAN"},
    "coder": {"CODE"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-role", choices=tuple(ROLE_TASKS), required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--drop-overlength",
        action="store_true",
        help="Drop complete conversations over the limit; default is to stop.",
    )
    parser.add_argument("--merge-16bit", action="store_true")
    parser.add_argument("--save-gguf", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
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
        "early_stopping_patience": args.early_stopping_patience,
    }
    bad = {key: value for key, value in positive.items() if value <= 0}
    if bad:
        raise ValueError(f"arguments must be positive: {bad}")
    if args.save_steps % args.eval_steps:
        raise ValueError("--save-steps must be a multiple of --eval-steps")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    for path in [*args.train_jsonl, args.validation_jsonl]:
        if not path.is_file():
            raise FileNotFoundError(path)


def supported_kwargs(callable_object: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    dataclass_fields = getattr(callable_object, "__dataclass_fields__", None)
    if dataclass_fields:
        return {
            key: value for key, value in candidates.items() if key in dataclass_fields
        }
    parameters = inspect.signature(callable_object).parameters
    if any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    ):
        return candidates
    return {key: value for key, value in candidates.items() if key in parameters}


def extract_messages(
    example: dict[str, Any],
    adapter_role: str,
) -> list[dict[str, str]]:
    task_type = str(example.get("task_type", ""))
    allowed = ROLE_TASKS[adapter_role]
    if task_type not in allowed:
        raise ValueError(
            f"{example.get('id')}: task_type={task_type!r} is not allowed "
            f"for {adapter_role}; expected {sorted(allowed)}"
        )
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"{example.get('id')}: invalid messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{example.get('id')}: final message must be assistant")
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in messages
    ]


def prepare_dataset(raw_dataset: Any, adapter_role: str) -> Any:
    def convert(example: dict[str, Any]) -> dict[str, Any]:
        messages = extract_messages(example, adapter_role)
        return {
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }

    return raw_dataset.map(
        convert,
        remove_columns=raw_dataset.column_names,
        desc=f"Build {adapter_role} prompt/completion conversations",
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
    adapter_role: str,
) -> tuple[Any, dict[str, Any]]:
    lengths: list[int] = []
    overlength: list[int] = []
    task_counts: Counter[str] = Counter()
    ids_seen: set[str] = set()
    duplicate_ids: list[str] = []
    for index, example in enumerate(dataset):
        example_id = str(example.get("id", index))
        if example_id in ids_seen:
            duplicate_ids.append(example_id)
        ids_seen.add(example_id)
        task_counts[str(example.get("task_type"))] += 1
        length = token_length(
            tokenizer,
            extract_messages(example, adapter_role),
        )
        lengths.append(length)
        if length > max_seq_length:
            overlength.append(index)
    if duplicate_ids:
        raise ValueError(f"{split_name}: duplicate ids: {duplicate_ids[:10]}")
    if overlength and not drop_overlength:
        ids = [str(dataset[index].get("id", index)) for index in overlength[:10]]
        raise ValueError(
            f"{split_name}: {len(overlength)} rows exceed {max_seq_length} "
            f"tokens: {ids}. Increase the limit or use --drop-overlength."
        )
    blocked = set(overlength)
    kept_indices = [index for index in range(len(dataset)) if index not in blocked]
    kept_lengths = [
        length for index, length in enumerate(lengths) if index not in blocked
    ]
    if overlength:
        dataset = dataset.select(kept_indices)
    ordered = sorted(kept_lengths)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return dataset, {
        "examples_before": len(lengths),
        "examples_after": len(kept_lengths),
        "dropped_overlength": len(overlength),
        "task_counts": dict(task_counts),
        "min_tokens": min(ordered) if ordered else None,
        "mean_tokens": sum(ordered) / len(ordered) if ordered else None,
        "p95_tokens": ordered[p95_index] if ordered else None,
        "max_tokens": max(ordered) if ordered else None,
    }


def build_sft_config(SFTConfig: Any, args: argparse.Namespace, bf16: bool) -> Any:
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
        "bf16": bf16,
        "fp16": not bf16,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
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
    kwargs = supported_kwargs(SFTConfig, candidates)
    if "eval_strategy" not in kwargs and "evaluation_strategy" not in kwargs:
        raise RuntimeError("No supported evaluation-strategy argument was found.")
    if "completion_only_loss" not in kwargs:
        raise RuntimeError(
            "Installed TRL does not expose completion_only_loss. Upgrade the "
            "Unsloth/TRL environment instead of training on prompt tokens."
        )
    return SFTConfig(**kwargs)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("CUDA is required.")
    bf16 = bool(is_bfloat16_supported())
    print("[VERSIONS]")
    print("  Python:", platform.python_version())
    print("  PyTorch:", torch.__version__)
    print("  Transformers:", transformers.__version__)
    print("  Datasets:", datasets.__version__)
    print("  TRL:", trl.__version__)
    print("  Unsloth:", getattr(unsloth, "__version__", "unknown"))
    print("[GPU]", torch.cuda.get_device_name(0))
    print("[ROLE]", args.adapter_role)

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
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": [str(path) for path in args.train_jsonl],
            "validation": str(args.validation_jsonl),
        },
    )
    train_raw, train_stats = enforce_length_contract(
        dataset["train"],
        tokenizer,
        args.max_seq_length,
        args.drop_overlength,
        "train",
        args.adapter_role,
    )
    validation_raw, validation_stats = enforce_length_contract(
        dataset["validation"],
        tokenizer,
        args.max_seq_length,
        args.drop_overlength,
        "validation",
        args.adapter_role,
    )
    if not len(train_raw) or not len(validation_raw):
        raise ValueError("training and validation datasets must both be non-empty")
    train_dataset = prepare_dataset(train_raw, args.adapter_role)
    validation_dataset = prepare_dataset(validation_raw, args.adapter_role)
    print(
        "[DATASET]\n"
        + json.dumps(
            {"train": train_stats, "validation": validation_stats},
            ensure_ascii=False,
            indent=2,
        )
    )

    training_args = build_sft_config(SFTConfig, args, bf16)
    trainer_candidates = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    }
    trainer_kwargs = supported_kwargs(SFTTrainer.__init__, trainer_candidates)
    if "processing_class" in trainer_kwargs:
        trainer_kwargs.pop("tokenizer", None)
    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint or None
    )
    eval_metrics = trainer.evaluate()

    final_adapter = args.output_dir / "final_adapter"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    trainer.save_state()
    summary = {
        "adapter_role": args.adapter_role,
        "model_name": args.model_name,
        "train_files": [str(path) for path in args.train_jsonl],
        "validation_file": str(args.validation_jsonl),
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
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if args.merge_16bit:
        model.save_pretrained_merged(
            str(args.output_dir / "merged_16bit"),
            tokenizer,
            save_method="merged_16bit",
        )
    if args.save_gguf:
        model.save_pretrained_gguf(
            str(args.output_dir / "gguf_q4_k_m"),
            tokenizer,
            quantization_method="q4_k_m",
        )
    print("[DONE]\n" + json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
