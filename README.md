------------make train and validation split---------------------------------
#python build_nl2plc_sft_dataset.py --excel "Training-Dataset-AI4Industrial-Automation-FIRE-2026.xlsx" --output-dir data/nl2plc_st_sft --train-count 213 --validation-count 31 --test-count 29  

-------------------------Submission---------------------------------
#python generate_gold_algorithm_database.py --input-jsonl examples\gold_example.jsonl --output-dir runs\one_gold_example --limit 1
#python inspect_gold_database.py --database runs\one_gold_example\gold_algorithms.sqlite3 --id row_0104
#python generate_gold_algorithm_database.py --input-jsonl "C:\Users\Mrinmoy\Desktop\NL2PLC_Unsloth_SFT_Qwen25Coder7B\data\nl2plc_st_sft\train.jsonl" --output-dir data\gold_algorithm_database
#python test_pipeline.py
#python generate_gold_algorithm_database.py --input-jsonl "data\nl2plc_st_sft\validation.jsonl" --output-dir "data\validation_algorithm_database"
#python audit_plans_nemotron.py --input-enriched "data\gold_algorithm_database\exports\enriched_master.jsonl" --output-jsonl "data\reviews\train_cascade2_reviews.jsonl" --model "nemotron-cascade-2:30b" --num-ctx 16384 --temperature 0.1 --timeout 600 --limit 3
#python audit_plans_nemotron.py --input-enriched "data\gold_algorithm_database\exports\enriched_master.jsonl" --output-jsonl "data\reviews\train_cascade2_reviews.jsonl" --model "nemotron-cascade-2:30b" --num-ctx 16384 --temperature 0.1 --timeout 600
#python audit_plans_nemotron.py --input-enriched "data\validation_algorithm_database\exports\enriched_master.jsonl" --output-jsonl "data\reviews\validation_cascade2_reviews.jsonl" --model "nemotron-cascade-2:30b" --num-ctx 16384 --temperature 0.1 --timeout 600
#cp review_manifest.example.json review_manifest.json

----------------the previous planning is ok but best works with oracle plan(teacher) but here modified-----------------
#python build_two_adapter_sft.py --train-enriched "data\gold_algorithm_database\exports\enriched_master.jsonl" --validation-enriched "data\validation_algorithm_database\exports\enriched_master.jsonl" --review-manifest "review_manifest.example.json" --nemotron-reviews "data\reviews\train_cascade2_reviews.jsonl" "data\reviews\validation_cascade2_reviews.jsonl" --require-nemotron-pass --repairs-per-source 1 --output-dir "data\two_adapter_sft"
#python train_lora_adapter.py --adapter-role planner --model-name "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" --train-jsonl "data\two_adapter_sft\train_planner.jsonl" --validation-jsonl "data\two_adapter_sft\validation_planner.jsonl" --output-dir "models\qwen25coder7b_planner_lora" --max-seq-length 3072 --epochs 3 --batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 1e-4 --lora-rank 16 --lora-alpha 32 --eval-steps 20 --save-steps 20
#python train_lora_adapter.py --adapter-role coder --model-name "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" --train-jsonl "data\two_adapter_sft\train_coder.jsonl" --validation-jsonl "data\two_adapter_sft\validation_coder.jsonl" --output-dir "models\qwen25coder7b_coder_lora" --max-seq-length 4096 --epochs 3 --batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 5e-5 --lora-rank 32 --lora-alpha 64 --eval-steps 20 --save-steps 20
#python predict_two_adapter.py --mode two_adapter --planner-adapter "models\qwen25coder7b_planner_lora\final_adapter" --coder-adapter "models\qwen25coder7b_coder_lora\final_adapter" --input-jsonl "data\nl2plc_st_sft\validation.jsonl" --output-jsonl "runs\two_adapter_validation.jsonl" --submission-output "runs\two_adapter_validation_compact.jsonl" --plan-repair-attempts 1 --code-retry-attempts 1 --max-seq-length 4096


------------------Now Rust training---------------------------------------------
python prepare_rust_sft.py --train-master "data\two_adapter_sft\canonical_train_master.jsonl" --validation-master "data\two_adapter_sft\canonical_validation_master.jsonl" --output-dir "data\rust_adapter_sft" --teacher-model "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M" --reviewer-model "nemotron-cascade-2:30b" --require-reviewer-pass  --max-repair-attempts 2 --repairs-per-source 1 --resume
python train_rust_lora.py  --train-jsonl "data\rust_adapter_sft\train_rust_coder.jsonl" "data\rust_adapter_sft\train_rust_repair.jsonl" --validation-jsonl "data\rust_adapter_sft\validation_rust_coder.jsonl" --output-dir "models\qwen25coder7b_rust_lora" --max-seq-length 6144 --epochs 3 --batch-size 1 --gradient-accumulation-steps 8 --learning-rate 5e-5 --lora-rank 32 --lora-alpha 64
python predict_rust_adapter.py --rust-adapter "models\qwen25coder7b_rust_lora\final_adapter" --input-jsonl "runs\two_adapter_validation.jsonl" --output-jsonl "runs\rust_adapter_validation.jsonl" --submission-output "runs\rust_adapter_validation_compact.jsonl" --repair-attempts 1

# AI4Industrial Automation final test pipeline

This package runs the frozen three-adapter system on the released workbook:

```text
Excel TASK rows
  -> planner LoRA (+ bounded plan repair)
  -> coder LoRA (+ deterministic ST retry)
  -> Rust LoRA (+ static, I/O API, address, and Cargo checks)
  -> final Excel workbook in the original 30-row order
```

The included input is `data\Test-Dataset-AI4Industrial-Automation.xlsx`.
It contains 30 tasks in `PLC Scenarios!A2:A31` and no official IDs, so the
pipeline assigns `test_0001` through `test_0030` strictly by worksheet order.

## Strict 30-result contract

The runner requires exactly 30 non-empty TASK rows and will never create an
Excel result containing blank ST or Rust cells. It performs up to three plan
repairs, four ST repairs, and four Rust repairs. If all learned plan repairs
fail, a conservative deterministic plan is built only from that test
instruction (never from training answers or gold code) so the ST stage is not
skipped.

The two final conditions are deliberately separate:

- `all_results_populated=true`: all 30 ST and all 30 Rust cells contain code.
- `ready_to_submit=true`: all 30 ST and Rust candidates also pass their
  deterministic validation and every Rust candidate passes Cargo.

If even one code cell is empty, no workbook is created. This prevents a partial
30-row sheet from being mistaken for a complete result.

## Important released-test correction

Test row 29 (pipeline ID `test_0028`) uses legacy address labels rather than
IEC direct-address notation. The model-facing instruction is normalized as:

| Released label | IEC address family |
|---|---|
| `AI:n` | `%IWn` |
| `AO:n` | `%QWn` |
| `DI:n` | `%IXn` |
| `DO:n` | `%QXn` |

The original TASK cell remains unchanged in the submission workbook. Only the
model-facing copy is normalized.

## 1. Put the package beside your existing `models` folder

Recommended layout:

```text
NL2PLC_Unsloth_SFT_Qwen25Coder7B\
├── models\
├── NL2PLC_Final_Test_Pipeline\
└── runs\
```

Activate the same Conda environment used for validation:

```cmd
conda activate ssltest
cd C:\Users\Mrinmoy\Desktop\NL2PLC_Unsloth_SFT_Qwen25Coder7B
```

Install only the added workbook dependency if it is missing:

```cmd
python -m pip install openpyxl
```

Cargo must be visible to this Python environment:

```cmd
python -c "import shutil; print(shutil.which('cargo'))"
```

Do not use `--allow-missing-cargo` for the final competition run.

## 2. Run CPU-only package tests

```cmd
python NL2PLC_Final_Test_Pipeline\test_final_pipeline.py
```

Expected result:

```text
Ran 4 tests
OK
```

## 3. Run the final test set

Use the planner version already selected from validation. The command below
uses planner v2:

```cmd
python NL2PLC_Final_Test_Pipeline\final_test_pipeline.py ^
  --test-xlsx "NL2PLC_Final_Test_Pipeline\data\Test-Dataset-AI4Industrial-Automation.xlsx" ^
  --planner-adapter "models\qwen25coder7b_planner_lora_v2\final_adapter" ^
  --coder-adapter "models\qwen25coder7b_coder_lora\final_adapter" ^
  --rust-adapter "models\qwen25coder7b_rust_lora\final_adapter" ^
  --output-dir "runs\final_test" ^
  --expected-results 30 ^
  --plan-repair-attempts 3 ^
  --code-retry-attempts 4 ^
  --rust-repair-attempts 4 ^
  --repair-temperature 0.20 ^
  --strict-exit
```

If validation selected planner v1, change only the planner adapter path to:

```text
models\qwen25coder7b_planner_lora\final_adapter
```

The pipeline never uses gold ST or gold Rust from the training workbook.

## Outputs

All files are written under `runs\final_test`:

| File | Purpose |
|---|---|
| `final_test_input.jsonl` | 30 ordered model inputs |
| `final_test_manifest.json` | IDs, worksheet rows, and alias-normalization audit |
| `st_detailed.jsonl` | plan/ST candidates and deterministic checks |
| `st_detailed.plans.jsonl` | complete plan histories |
| `st_detailed.metrics.json` | planner and ST pass metrics |
| `rust_detailed.jsonl` | Rust candidates, repairs, static checks, and Cargo results |
| `rust_detailed.metrics.json` | Rust validation metrics |
| `final_test_predictions.jsonl` | merged ST/Rust rows in test order |
| `final_test_report.json` | final readiness gate and failing IDs |

When all 30 ST and Rust outputs pass, the workbook is named:

```text
runs\final_test\FINAL_SUBMISSION.xlsx
```

If any row is missing or invalid, the workbook is deliberately named:

```text
runs\final_test\STRICT_30_RESULTS_NEEDS_REVIEW.xlsx
```

That review workbook is created only when all 30 ST and all 30 Rust cells are
populated. If any code cell is blank, the report is written but no workbook is
created.

Do not submit the candidate workbook until `final_test_report.json` says:

```json
"ready_to_submit": true
```

## Resume after interruption

```cmd
python NL2PLC_Final_Test_Pipeline\final_test_pipeline.py ^
  --test-xlsx "NL2PLC_Final_Test_Pipeline\data\Test-Dataset-AI4Industrial-Automation.xlsx" ^
  --planner-adapter "models\qwen25coder7b_planner_lora_v2\final_adapter" ^
  --coder-adapter "models\qwen25coder7b_coder_lora\final_adapter" ^
  --rust-adapter "models\qwen25coder7b_rust_lora\final_adapter" ^
  --output-dir "runs\final_test" ^
  --resume ^
  --strict-exit
```

`--resume` reuses a stage only when its JSONL contains all 30 expected IDs in
the correct order, every code field is non-empty, and every row is valid. A
partial or failed stage is regenerated instead of being mistaken for a
finished run.

## Rebuild Excel after manual JSONL repair

If you manually repair failed candidates inside the two detailed JSONL files:

```cmd
python NL2PLC_Final_Test_Pipeline\final_test_pipeline.py ^
  --test-xlsx "NL2PLC_Final_Test_Pipeline\data\Test-Dataset-AI4Industrial-Automation.xlsx" ^
  --output-dir "runs\final_test" ^
  --assemble-only ^
  --strict-exit
```

The submission keeps the original workbook and fills only `ST-CODE` and
`RUST-CODE` in columns B and C.
