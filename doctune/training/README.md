# `doctune/training`

This package contains **Phases 3 and 4** of the doctune training pipeline: Supervised
Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO). Both scripts
share a common configuration and dataset layer via `training_utils.py`.

```
alignment_dataset.jsonl   golden_eval.jsonl
         │                        │
         ▼                        ▼
   train_sft.py  ──► ./doctune-sft/  (LoRA adapters)
                              │
                              ▼
   train_dpo.py  ──► ./doctune-dpo-beta<β>-lr<lr>/  (per-sweep adapter)
                              │
                              ▼
                     _log_sweep_summary()  →  best run identified
```

---

## Files

| File | Phase | Role |
|---|---|---|
| `training_utils.py` | Shared | Common CLI args, `TrainingArguments` builder, dataset loader |
| `train_sft.py` | Phase 3 | Supervised fine-tuning with LoRA + SFTTrainer |
| `train_dpo.py` | Phase 4 | DPO alignment with hyperparameter sweep + MLflow tracking |
| `__init__.py` | — | Package marker |

---

## `training_utils.py`

Shared utilities imported by both training scripts. Contains no training logic itself
— only configuration, argument registration, and I/O helpers.

### `add_common_train_args(parser)`

Registers CLI arguments shared by all training entry points onto an
`argparse.ArgumentParser`. Both `train_sft.py` and `train_dpo.py` call this first,
then add their own script-specific flags.

| Flag | Default | Description |
|---|---|---|
| `--model-id` | *(required)* | HuggingFace model identifier (e.g. `meta-llama/Llama-3.1-8B`) |
| `--dataset` | `alignment_dataset.jsonl` | Path to the SFT/DPO training JSONL |
| `--eval-dataset` | `golden_eval.jsonl` | Path to the evaluation JSONL |
| `--output` | `None` | Output directory; derived from run name if omitted |
| `--max-seq-length` | `2048` | Maximum sequence length for trainer truncation |
| `--max-prompt-length` | `1024` | Maximum prompt token length passed to `DPOTrainer` |

---

### `build_training_args(...) → TrainingArguments`

Constructs a `TrainingArguments` instance with project-wide defaults. All callers
pass their script-specific values for the parameters that differ; everything else is
fixed here as a single source of truth.

| Parameter | Default | Description |
|---|---|---|
| `output_dir` | *(required)* | Directory for checkpoints and final model |
| `run_name` | *(required)* | MLflow run name (derived by `derive_run_name`) |
| `epochs` | `1` | Number of training epochs |
| `batch_size` | `4` | Per-device training batch size |
| `grad_accum` | `8` | Gradient accumulation steps |
| `lr` | `2e-4` | Learning rate |
| `remove_unused_columns` | `True` | Drop dataset columns not used by the model (`False` for DPO) |

**Fixed project-wide defaults** (not exposed as parameters):

| Setting | Value | Rationale |
|---|---|---|
| `report_to` | `"mlflow"` | All runs tracked in MLflow automatically |
| `optim` | `"adamw_torch"` | Standard AdamW, no fused variant needed |
| `lr_scheduler_type` | `"cosine"` | Smooth decay with warmup |
| `warmup_ratio` | `0.1` | 10% of steps used for warmup |
| `logging_steps` | `10` | Loss logged every 10 steps |
| `save_strategy` | `"epoch"` | Checkpoint per epoch |
| `eval_strategy` | `"epoch"` | Eval per epoch |
| `bf16` | `True` | BF16 mixed precision (requires Ampere+ GPU) |
| `fp16` | `False` | FP16 disabled in favour of BF16 |
| `max_grad_norm` | `0.3` | Gradient clipping |
| `gradient_checkpointing` | `True` | Reduces VRAM by recomputing activations |

---

### `load_datasets(data_path, eval_path) → tuple[Dataset, Dataset]`

Loads train and eval datasets from JSONL files using HuggingFace `datasets`.

Performs explicit `os.path.isfile` checks on both paths **before** calling
`load_dataset`, raising a descriptive `FileNotFoundError` if either file is missing:

```
FileNotFoundError: Training dataset not found: 'alignment_dataset.jsonl'.
Run the data pipeline first to generate it (see doctune/data/README.md).
```

This replaces the generic HuggingFace stacktrace that would otherwise obscure the
root cause for new users.

| Argument | Description |
|---|---|
| `data_path` | Path to the training JSONL file. |
| `eval_path` | Path to the evaluation JSONL file. |

Returns a `(train_dataset, eval_dataset)` tuple of HuggingFace `Dataset` objects.

---

## `train_sft.py` — Phase 3: Supervised Fine-Tuning

Injects domain-specific knowledge from `alignment_dataset.jsonl` into a foundation
model using LoRA adapters and `trl.SFTTrainer`. Produces a `./doctune-sft/` adapter
directory consumed by DPO in Phase 4.

### `parse_args() → Namespace`

Calls `add_common_train_args()` then adds SFT-specific flags:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `4` | Per-device train batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA scaling factor |
| `--lora-dropout` | `0.05` | LoRA dropout rate |

---

### `main()` — SFT orchestration

| Step | Action |
|---|---|
| 1 | Load tokenizer (right padding) and base model via `model_utils` |
| 2 | Auto-detect LoRA target modules via `detect_lora_target_modules(model)` |
| 3 | Build `LoraConfig` with `r`, `lora_alpha`, `lora_dropout` from CLI args |
| 4 | Load and format datasets; apply `formatting_prompts_func` |
| 5 | Build `TrainingArguments` via `build_training_args` |
| 6 | Initialise `SFTTrainer` and call `.train()` / `.save_model()` |
| 7 | Delete trainer + model and clear GPU cache |

#### `formatting_prompts_func(example) → list[str]`

Inline function defined inside `main()`. Converts batched dataset rows into
conversational strings using the model's chat template:

```python
messages = [
    {"role": "user",      "content": prompt},
    {"role": "assistant", "content": chosen},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

Returns a list of formatted strings, one per row in the batch.

#### LoRA configuration notes

- **`target_modules`**: auto-detected from the model's named modules via
  `detect_lora_target_modules`. Targets attention projection layers
  (`q_proj`, `v_proj`, etc.) automatically.
- **`modules_to_save`**: hardcoded as `["lm_head", "embed_tokens"]`.
  These are **Llama-family specific** — update this list when switching
  to a different model architecture (Mistral, Qwen, Phi, etc.).
- **`bias`**: `"none"` — bias parameters are not trained, keeping adapter
  size minimal.

---

### CLI example — SFT

```bash
python -m doctune.training.train_sft \
    --model-id meta-llama/Llama-3.1-8B \
    --dataset ./data/alignment_dataset.jsonl \
    --eval-dataset ./eval/golden_eval.jsonl \
    --epochs 3 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --output ./doctune-sft
```

---

## `train_dpo.py` — Phase 4: Direct Preference Optimization

Aligns the SFT model using `(prompt, chosen, rejected)` preference tuples to
penalise hallucinations and boundary violations. Runs a **hyperparameter sweep**
over `β` (divergence regularisation) and learning rate, tracking every run in
MLflow and ranking results by reward margin at the end.

### `_SweepResult` (dataclass)

Holds the outcome of a single sweep configuration. Used for ranking and MLflow
tagging after the sweep completes.

| Field | Type | Description |
|---|---|---|
| `run_name` | `str` | Human-readable run identifier (`<base>-beta<β>-lr<lr>`) |
| `beta` | `float` | The β value used |
| `lr` | `float` | The learning rate used |
| `eval_loss` | `float` | Final eval loss; `float("inf")` if no eval log found |
| `rewards_chosen` | `float` | Mean reward margin on chosen responses |
| `rewards_rejected` | `float` | Mean reward margin on rejected responses |
| `reward_margin` | `float` | `rewards_chosen − rewards_rejected` — primary ranking signal |
| `extra` | `dict` | The full last eval log entry for reference |

`__str__` formats as: `<run_name> | β=<β> lr=<lr> | eval_loss=<x> | reward_margin=<y>`

---

### `parse_args() → Namespace`

Calls `add_common_train_args()` then adds DPO-specific flags:

| Flag | Default | Description |
|---|---|---|
| `--sft-adapter` | `./doctune-sft` | Path to the SFT LoRA adapter directory (output of `train_sft.py`) |
| `--betas` | `[0.05, 0.1, 0.25, 0.5]` | β values to sweep (controls KL divergence from reference) |
| `--lrs` | `[5e-6, 1e-6]` | Learning rates to sweep |
| `--batch-size` | `2` | Per-device batch size (lower than SFT due to reference model memory) |
| `--grad-accum` | `16` | Gradient accumulation steps |

---

### `_extract_sweep_result(trainer, run_name, beta, lr) → _SweepResult`

Extracts final eval metrics from a completed `DPOTrainer` instance by scanning
`trainer.state.log_history` for entries containing `"eval_loss"`. Takes the last
such entry as the final epoch result.

Metrics extracted:

| Metric | Key in log history |
|---|---|
| Eval loss | `"eval_loss"` |
| Chosen reward margin | `"eval_rewards/chosen"` |
| Rejected reward margin | `"eval_rewards/rejected"` |

Logs all four metrics to MLflow as a **nested run** under the sweep's parent run.
MLflow errors are caught and logged as warnings — tracking is non-critical.

---

### `_log_sweep_summary(results)`

Ranks all `_SweepResult` instances by `(-reward_margin, eval_loss)` — higher margin
first, lower loss as tiebreak — and prints a ranked banner:

```
============================================================
DPO SWEEP COMPLETE — RESULTS RANKED BY REWARD MARGIN
============================================================
  1. llama-dpo-beta0.1-lr5e-06 | β=0.1 lr=5e-06 | ...  <- BEST
  2. llama-dpo-beta0.25-lr5e-06 | β=0.25 lr=5e-06 | ...
  ...
============================================================
  Best adapter saved to: ./llama-dpo-beta0.1-lr5e-06
============================================================
```

Also tags the outer MLflow run with `dpo_best_run`, `dpo_best_beta`,
`dpo_best_lr`, and `dpo_best_margin` for experiment tracking.

---

### `main()` — DPO sweep orchestration

| Step | Action |
|---|---|
| 1 | Load tokenizer (right padding) and base model |
| 2 | Load SFT adapters with `PeftModel.from_pretrained(..., is_trainable=True)` |
| 3 | Load and format datasets; apply `format_dpo_dataset` |
| 4 | For each `(β, lr)` pair: build `TrainingArguments`, run `DPOTrainer`, extract result, free memory |
| 5 | Call `_log_sweep_summary` to rank and tag best run |
| 6 | Delete model + base model and clear GPU cache |

#### `format_dpo_dataset(example) → dict`

Inline function defined inside `main()`. Formats a single dataset row for
`DPOTrainer`:

```python
{
    "prompt":   tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True),
    "chosen":   f"{chosen}{tokenizer.eos_token}",
    "rejected": f"{rejected}{tokenizer.eos_token}",
}
```

Applied via `dataset.map(format_dpo_dataset)` before the sweep begins (once, not
per-sweep).

#### Sweep memory management

Between sweep iterations:
```python
del trainer
clear_gpu_cache()
```

After the sweep:
```python
del model, base_model
clear_gpu_cache()
```

This prevents OOM errors across a large `(β × lr)` grid.

---

### CLI example — DPO sweep

```bash
python -m doctune.training.train_dpo \
    --model-id meta-llama/Llama-3.1-8B \
    --sft-adapter ./doctune-sft \
    --dataset ./data/alignment_dataset.jsonl \
    --eval-dataset ./eval/golden_eval.jsonl \
    --betas 0.05 0.1 0.25 \
    --lrs 5e-6 1e-6 \
    --batch-size 2 \
    --grad-accum 16 \
    --max-prompt-length 1024
```

---

## Dependency on `doctune.utils.model_utils`

Both training scripts delegate model loading and bookkeeping to shared utilities in
`doctune/utils/model_utils.py`:

| Function | Used by | Description |
|---|---|---|
| `load_tokenizer(model_id, padding_side)` | SFT, DPO | Loads and configures the tokenizer; adds pad token if missing |
| `load_base_model(model_id, tokenizer)` | SFT, DPO | Loads the base model in BF16 with device map auto |
| `derive_run_name(model_id, phase)` | SFT, DPO | Produces a short, slug-safe run name (e.g. `llama-3-1-8b-sft`) |
| `detect_lora_target_modules(model)` | SFT | Finds attention projection layers by scanning module names |
| `clear_gpu_cache()` | SFT, DPO | Calls `torch.cuda.empty_cache()` and `gc.collect()` |
| `format_prompt_for_eval(tokenizer, text)` | eval only | Not used in training |

---

## Notes on MLflow tracking

Both phases report to MLflow automatically via `report_to="mlflow"` in
`TrainingArguments`. No explicit `mlflow.start_run()` is needed for the training
loop itself.

`train_dpo.py` additionally wraps each sweep iteration in a **nested MLflow run**
(inside `_extract_sweep_result`) to track per-β/lr metrics side-by-side, and tags
the outer run with the best configuration after the sweep via `_log_sweep_summary`.

MLflow errors in both functions are caught with `except Exception` (annotated
`# noqa: BLE001`) — tracking is treated as a non-critical observability layer and
must never abort a training run.
