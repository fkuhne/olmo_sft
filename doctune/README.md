# `doctune`

End-to-end pipeline for domain-adapting any HuggingFace causal language model on a
corpus of PDF documents using Supervised Fine-Tuning (SFT) and Direct Preference
Optimization (DPO).

Given a folder of PDF manuals and an API key, doctune produces a fully merged,
deployment-ready model that answers domain questions accurately and refuses
out-of-domain queries gracefully.

---

## Pipeline at a glance

```
Phase 1          Phase 2          Phase 3       Phase 4       Phase 5        Phase 6
─────────────    ─────────────    ──────────    ──────────    ──────────    ──────────
 PDF manuals     Extraction  →    SFT     →     DPO     →    Eval     →    Merge
                 Synthesis        (LoRA)        (LoRA)        (judge)       (deploy)

 ./manuals/      data/            training/     training/     eval/         deploy/
                 extraction/      train_sft     train_dpo     evaluate      merge_model
                 pipeline/                                    golden_eval
                 synthesis/
```

| Phase | Package | Input | Output | GPU? |
|---|---|---|---|---|
| 1 – Extraction | `data/extraction` | PDF files | Markdown chunks (cached) | No |
| 2 – Synthesis | `data/pipeline` + `data/synthesis` | Cached chunks | `alignment_dataset.jsonl` | No |
| 3 – SFT | `training/train_sft` | JSONL + base model | LoRA adapters (`./doctune-sft/`) | Yes |
| 4 – DPO | `training/train_dpo` | JSONL + SFT adapters | LoRA adapters (`./doctune-dpo-*`) | Yes |
| 5 – Eval | `eval/evaluate` | Fine-tuned model | `eval_results.json` | Yes |
| 5a – Golden Set | `eval/generate_golden_eval` | Teacher LLM API | `golden_eval.jsonl` | No |
| 6 – Deploy | `deploy/merge_model` | Base model + best DPO adapter | Standalone merged model | No (CPU) |

---

## Package structure

```
doctune/
├── data/                       Phase 1–2: data curation
│   ├── extraction/             PDF → markdown chunks (Docling)
│   ├── pipeline/               Orchestration, caching, CLI entry point
│   └── synthesis/              Diversity selection, dedup, teacher-model synthesis
│
├── training/                   Phase 3–4: fine-tuning
│   ├── train_sft.py            Supervised fine-tuning with LoRA
│   ├── train_dpo.py            DPO alignment with β/lr sweep + MLflow
│   └── training_utils.py       Shared CLI args, TrainingArguments, dataset loading
│
├── eval/                       Phase 5: evaluation
│   ├── evaluate.py             In-domain accuracy + out-of-domain refusal testing
│   └── generate_golden_eval.py Type-balanced golden eval set generation
│
├── deploy/                     Phase 6: deployment
│   └── merge_model.py          LoRA weight fusion → standalone model
│
└── utils/                      Shared utilities
    ├── model_utils.py          Model loading, tokenizer, LoRA detection, GPU cleanup
    ├── pricing.py              Per-model token pricing and cost estimation
    └── provider_utils.py       Provider detection, client construction, retry logic
```

Each sub-package has its own `README.md` with detailed documentation of every class,
function, constant, and CLI flag:

| README | Covers |
|---|---|
| [`data/README.md`](data/README.md) | Full pipeline overview, two-stage run examples, cache layout, configuration reference |
| [`data/extraction/README.md`](data/extraction/README.md) | `DoclingManualExtractor`, page batching, GPU fallback, CLI |
| [`data/pipeline/README.md`](data/pipeline/README.md) | `DatasetBuilder`, `PipelineCache`, gate ordering |
| [`data/synthesis/README.md`](data/synthesis/README.md) | `LateChunker`, `DiversitySelector`, `ChunkFilter`, `DatasetFilter`, `TeacherModelSynthesizer` |
| [`training/README.md`](training/README.md) | `train_sft.py`, `train_dpo.py`, sweep mechanics, MLflow integration |
| [`eval/README.md`](eval/README.md) | Golden eval generation, contamination guard, LLM-as-judge scoring |
| [`deploy/README.md`](deploy/README.md) | Weight merging, `merge_and_unload()` rationale, output layout |
| [`utils/README.md`](utils/README.md) | Model utilities, pricing, provider dispatch |

---

## Quick start

### Phase 1–2: Data curation (local, no GPU)

```bash
# One-time setup
bash setup/local_setup.sh
source .venv/bin/activate

# Set your API key
export OPENAI_API_KEY="sk-..."

# Place PDFs and run the pipeline
cp ~/my-manuals/*.pdf ./manuals/
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain "home appliances" \
    --model gpt-4o \
    --output ./alignment_dataset.jsonl

# Generate golden eval set (use a different provider to prevent contamination)
python -m doctune.eval.generate_golden_eval \
    --model claude-3-5-sonnet-20241022 \
    --train-model gpt-4o \
    --domain "home appliances" \
    --count 300
```

### Phase 3–6: Training and deployment (GPU pod)

```bash
# One-time setup on RunPod
bash setup/runpod_setup.sh

# Upload datasets
scp alignment_dataset.jsonl golden_eval.jsonl user@<pod-ip>:/workspace/doctune/

# Phase 3: SFT
python -m doctune.training.train_sft \
    --model-id meta-llama/Llama-3.1-8B

# Phase 4: DPO sweep
python -m doctune.training.train_dpo \
    --model-id meta-llama/Llama-3.1-8B \
    --betas 0.05 0.1 0.25

# Phase 5: Evaluate
python -m doctune.eval.evaluate \
    --model-id meta-llama/Llama-3.1-8B \
    --adapter ./doctune-dpo-beta0.1-lr5e-06 \
    --baseline --judge

# Phase 6: Merge for deployment
python -m doctune.deploy.merge_model \
    --model-id meta-llama/Llama-3.1-8B \
    --adapter ./doctune-dpo-beta0.1-lr5e-06
```

---

## Key design principles

### Resumability
Every pipeline stage writes results atomically to a cache. Interrupted runs (rate
limits, crashes, timeouts) resume from the last checkpoint with zero rework.

### Cost efficiency
Three gates fire before any teacher-model API call:
1. **Resume filter** — already-cached chunks are skipped.
2. **Diversity selection** — reduces chunks to the most semantically varied subset.
3. **Chunk deduplication** — drops near-duplicate source material.

### Contamination prevention
The eval golden set enforces **cross-provider generation**: if training data was
synthesised with GPT-4o, the golden eval set must come from Claude (or vice versa).
This is enforced at runtime by the `--train-model` flag.

### Modularity
Each phase is a standalone `python -m` entry point with its own CLI. Phases can be
run independently, on different machines, and at different times. The only coupling
is through JSONL files and adapter directories on disk.

---

## Dependencies

### Core (all phases)
- `openai` — OpenAI API client
- `anthropic` — Anthropic API client
- `pydantic` — Structured output validation
- `huggingface_hub` — Model and tokenizer downloads
- `sentence-transformers` — Embedding models for deduplication
- `docling` — PDF-to-Markdown extraction (IBM DocLayNet)
- `torch` — Tensor operations and GPU acceleration

### Training extras (`pip install -e ".[training]"`)
- `transformers` — Model loading and tokenization
- `peft` — LoRA adapter management
- `trl` — SFTTrainer and DPOTrainer
- `accelerate` — Distributed training and device placement
- `datasets` — HuggingFace dataset loading
- `bitsandbytes` — Quantized training support
- `flash-attn` — Flash Attention 2 (Linux + CUDA only)
- `mlflow` — Experiment tracking

### Development (`pip install -e ".[dev]"`)
- `pytest`, `ruff`, `pylint`

---

## Environment variables

| Variable | Used by | Description |
|---|---|---|
| `OPENAI_API_KEY` | synthesis, eval | OpenAI API key |
| `ANTHROPIC_API_KEY` | synthesis, eval | Anthropic API key |
| `DOMAIN` | eval | Domain string fallback (prefer `--domain` CLI flag) |
| `DOCTUNE_DOCLING_PAGE_BATCH_SIZE` | extraction | Pages per Docling batch (default: 25) |
| `DOCTUNE_DOCLING_RETRY_ATTEMPTS` | extraction | Max retry attempts per failing page range |
| `DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS` | extraction | Retry backoff base in seconds |
| `DOCTUNE_DOCLING_USE_GPU` | extraction | OCR device: `auto`, `cpu`, `cuda`, `cuda:N` |
