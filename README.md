# 🔬 Doctune: Domain Adaptation Pipeline for Small Language Models

> End-to-end blueprint for domain-adapting **any HuggingFace causal LM** on PDF document corpora using SFT + DPO

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-≥3.12-green.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-recommended-blueviolet.svg)](https://docs.astral.sh/uv/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**Doctune** is a turnkey domain-adaptation toolkit that takes raw PDF documentation and
produces a fine-tuned, deployment-ready language model in a single reproducible pipeline.
It bridges the gap between unstructured enterprise documents and production-grade LLMs by
automating every step — from layout-aware extraction through preference alignment — so
teams can ship domain-expert models without deep ML infrastructure expertise.

### Key Features

- 📄 **Document parsing with [Docling](https://github.com/DS4SD/docling)** — layout-aware OCR via IBM DocLayNet with batched page processing, retry/split-on-failure, and GPU auto-detection
- 🧬 **Late chunking for global document awareness** — jina-embeddings-v3 produces document-level contextual embeddings before chunking, preserving cross-section semantics
- 🧹 **Semantic deduplication** — two-stage cosine similarity filtering (chunk-level at 0.82, prompt-level at 0.92) eliminates redundant training data and cuts API costs
- 🎯 **Diversity selection** — greedy farthest-first sampling keeps only the most semantically varied chunks, maximising coverage while minimising synthesis spend
- 🤖 **Teacher-model synthesis** — two-stage prompt structure (focus selection → 3-angle QA) generates high-quality SFT pairs from OpenAI, Anthropic, or local Ollama models
- 🎓 **SFT fine-tuning** — LoRA-based supervised fine-tuning with auto-detected target modules, works with any HuggingFace causal LM
- ⚖️ **DPO preference alignment** — β/learning-rate sweep with automatic ranking by reward margin selects the best adapter configuration
- 📊 **MLflow integration** — every training run (SFT and DPO sweep) is tracked with nested experiment logging for side-by-side comparison
- 🔬 **Golden evaluation with contamination guard** — type-balanced eval sets enforce cross-provider generation to prevent distributional bias
- 🚀 **vLLM serving** — merged models deploy directly to a high-throughput OpenAI-compatible inference server
- ☁️ **RunPod support** — turnkey GPU pod bootstrap scripts for Phases 3–6 (training, eval, merge, serving)
- 🔄 **Full resumability** — every pipeline stage writes atomically to a SHA-256 keyed cache; interrupted runs resume with zero rework
- 🏗️ **Model-agnostic** — works out of the box with LLaMA, Mistral, Phi, Gemma, Qwen, OLMo, and any other HuggingFace causal LM

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Architecture Overview](#-architecture-overview)
- [Repository Structure](#-repository-structure)
- [Documentation Map](#-documentation-map)
- [Data Pipeline](#-data-pipeline)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Deployment](#-deployment)
- [Makefile Targets](#-makefile-targets)
- [Environment Variables](#-environment-variables)
- [Infrastructure](#-infrastructure)
- [Suggested Execution Order](#-suggested-execution-order)
- [Key Dependencies](#-key-dependencies)
- [External References](#-external-references)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## ⚡ Quick Start

### Option A: Free Local Data Generation (Ollama — No API Key, No GPU)

Run everything locally using free, open-weight LLMs via [Ollama](https://ollama.com):

```bash
# 1. Install Ollama (https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh   # Linux
# macOS: download from https://ollama.com/download

# 2. Pull a model (llama3.1:8b is a good default)
ollama pull llama3.1:8b

# 3. Clone the repo and set up
git clone https://github.com/felipekuhne/doctune.git && cd doctune
uv venv .venv && source .venv/bin/activate
uv pip install -e "."

# 4. Generate the dataset — no API key needed!
mkdir -p manuals && cp /path/to/your/*.pdf manuals/
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain "my product" \
    --model llama3.1:8b
```

> **Tip:** For better quality output, use a larger model like `llama3.1:70b` or `qwen2.5:14b` if your machine has enough RAM.

### Option B: Paid API Data Generation (OpenAI / Anthropic — No GPU)

Higher quality synthetic data using cloud APIs:

```bash
# 1. Clone the repo
git clone https://github.com/felipekuhne/doctune.git && cd doctune

# 2. Install uv (if you don't have it already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create a virtual environment and install base dependencies
uv venv .venv && source .venv/bin/activate
uv pip install -e "."       # Base deps only (no training/GPU packages)

# 4. Generate the dataset from your PDFs
export OPENAI_API_KEY="your_key_here"
mkdir -p manuals && cp /path/to/your/*.pdf manuals/
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain "my product" \
    --model gpt-4o

# Or use an Anthropic model instead:
export ANTHROPIC_API_KEY="your_key_here"
python -m doctune.data.pipeline.build_dataset \
    --model claude-3-5-sonnet-20241022

# 5. (Optional) Generate the golden evaluation set
python -m doctune.eval.generate_golden_eval \
    --model claude-3-5-sonnet-20241022 \
    --train-model gpt-4o \
    --domain "my product" \
    --count 300
```

Then transfer the generated `alignment_dataset.jsonl` and `golden_eval.jsonl` to your GPU environment for training.

### Optional: Enable GPU OCR for Docling (Windows + NVIDIA)

By default, many local installs resolve to CPU-only PyTorch wheels. If you want
Docling/RapidOCR OCR stages to use NVIDIA CUDA, install a CUDA-enabled wheel:

```bash
# In your activated .venv
uv pip install --python .venv/Scripts/python.exe --upgrade \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

# Verify GPU visibility
.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`pdf_extractor.py` now uses RapidOCR with the `torch` backend and has device
guards:

- `DOCTUNE_DOCLING_USE_GPU=auto` (default): Use `cuda:0` when available, else CPU.
- `DOCTUNE_DOCLING_USE_GPU=cpu`: Force CPU execution.
- `DOCTUNE_DOCLING_USE_GPU=cuda` or `cuda:0`: Force GPU; falls back to CPU if unavailable.

On low-VRAM GPUs, keep extraction stable by lowering page batch size:

```bash
python -m doctune.data.extraction.extract_dataset --docling-page-batch-size 5
```

### Option C: Full Pipeline (GPU Required)

```bash
# 1. On the GPU pod, install all dependencies
uv pip install -e ".[training]"

# 2. Generate dataset (or upload the .jsonl files generated locally)
export OPENAI_API_KEY="your_key_here"   # or ANTHROPIC_API_KEY
python -m doctune.data.pipeline.build_dataset     # --model claude-3-5-sonnet-20241022 for Anthropic

# 3. Train (SFT → DPO → Evaluate → Merge)
#    Replace <your-model-id> with any HuggingFace model
#    Examples: meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-v0.3, allenai/OLMo-2-0425-1B
python -m doctune.training.train_sft   --model-id <your-model-id>
python -m doctune.training.train_dpo   --model-id <your-model-id>
python -m doctune.eval.evaluate        --model-id <your-model-id>
python -m doctune.deploy.merge_model   --model-id <your-model-id>
```

**Hardware Requirements (Phases 3–6):** ≥ 24 GB VRAM (A100, RTX 3090/4090, or RTX 6000 Ada). Flash Attention 2 is auto-detected — if unavailable, the pipeline falls back to eager attention automatically.

---

## 🏗️ Architecture Overview

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

### End-to-end data flow

```
 📄 PDF Documents (./manuals/)
       │
       ▼
 🔍 Docling Parser (DoclingManualExtractor)
       │   Layout-aware OCR via IBM DocLayNet
       │   Batched page processing + retry/split-on-failure
       ▼
 📦 PipelineCache (.cache/<domain>/chunks_<hash>.json)
       │   SHA-256 keyed — renaming a PDF won't invalidate its cache
       ▼
 🎯 DiversitySelector (optional — jina-embeddings-v3 late chunking)
       │   Greedy farthest-first selection → keeps top N% most varied chunks
       ▼
 🧹 ChunkFilter (cosine dedup at 0.82 — all-MiniLM-L6-v2)
       │   Prevents near-duplicate source material reaching the API
       ▼
 🤖 TeacherModelSynthesizer (OpenAI / Anthropic / Ollama)
       │   Two-stage prompt: focus-selection → 3-angle QA generation
       │   Immediately generates a DPO rejected response per pair
       ▼
 🔍 DatasetFilter (cosine dedup at 0.92 — all-MiniLM-L6-v2)
       │   Drops near-duplicate generated questions across all documents
       ▼
 📊 alignment_dataset.jsonl  {prompt, chosen, rejected, metadata}
       │
       ▼
 🎓 SFT Training (LoRA — auto-detected target modules)
       │   3 epochs, cosine schedule, BF16
       ▼
 ⚖️  DPO Alignment (β/lr sweep — tracked in MLflow)
       │   Best adapter selected by reward margin
       ▼
 🔗 LoRA Weight Merge → 🚀 vLLM Serving
```

### Key design principles

| Principle | How it works |
|---|---|
| **Resumability** | Every pipeline stage writes results atomically to a cache. Interrupted runs (rate limits, crashes, timeouts) resume from the last checkpoint with zero rework. |
| **Cost efficiency** | Three gates fire before any teacher-model API call: resume filter → diversity selection → chunk deduplication. |
| **Contamination prevention** | The eval golden set enforces **cross-provider generation**: if training data was synthesised with GPT-4o, the golden eval set must come from Claude (or vice versa). Enforced at runtime via `--train-model`. |
| **Modularity** | Each phase is a standalone `python -m` entry point with its own CLI. Phases can be run independently, on different machines, and at different times. The only coupling is through JSONL files and adapter directories on disk. |
| **Model-agnostic** | LoRA target modules are auto-detected from the model architecture at runtime, so the pipeline works with **any** HuggingFace causal LM (LLaMA, Mistral, Phi, Gemma, Qwen, OLMo, etc.). |

---

## 📁 Repository Structure

```
doctune/
├── README.md                       # This file
├── MODEL_CARD.md                   # Model card (base model, intended use, limitations)
├── DATA_CARD.md                    # Dataset card (schema, quality controls, flaw types)
├── sft_plan.md                     # Master Execution Blueprint (Phases 1–6)
├── data_engineering_spec.md        # Data pipeline theoretical framework
├── late_chunking.md                # Late-chunking technique deep-dive
├── Makefile                        # Dev targets (make data, make train-sft, etc.)
├── olmo.ipynb                      # Exploratory notebook
├── pyproject.toml                  # Project metadata & pinned dependencies
├── LICENSE                         # Apache 2.0
├── .gitignore
│
├── setup/                          # Environment bootstrap scripts
│   ├── local_setup.sh              # Local macOS/Linux setup (Phase 2 — no GPU)
│   └── runpod_setup.sh             # RunPod GPU environment setup (Phases 3–6)
│
├── examples/                       # Reference data samples
│   ├── sample_dataset.jsonl        # Example alignment dataset rows
│   └── sample_synthesis_with_metadata.jsonl  # Synthesis output with cost metadata
│
├── manuals/                        # Place your PDF files here (gitignored)
│
└── doctune/                        # Main Python Package
    ├── __init__.py
    │
    ├── data/                       # Phases 1–2: Data curation pipeline
    │   ├── extraction/             # Stage 1: PDF → enriched markdown chunks
    │   │   ├── pdf_extractor.py    #   DoclingManualExtractor (IBM Docling + DocLayNet)
    │   │   └── extract_dataset.py  #   CLI entry point for extraction-only runs
    │   ├── pipeline/               # Stage 2: Orchestration, caching, CLI
    │   │   ├── build_dataset.py    #   DatasetBuilder — full pipeline orchestrator
    │   │   ├── pipeline_cache.py   #   PipelineCache — SHA-256 keyed on-disk cache
    │   │   └── pipeline_utils.py   #   Shared helpers: PDF discovery, CLI args
    │   └── synthesis/              # Stages 2b–2d: Filtering + synthesis
    │       ├── late_chunker.py     #   LateChunker — jina-embeddings-v3 late chunking
    │       ├── diversity_selector.py  # DiversitySelector — greedy farthest-first
    │       ├── deduplicate_dataset.py # ChunkFilter + DatasetFilter (cosine dedup)
    │       ├── teacher_model_synthesis.py  # TeacherModelSynthesizer (OpenAI/Anthropic/Ollama)
    │       └── report_synthesis_spend.py   # CLI: audit token and cost usage
    │
    ├── training/                   # Phases 3–4: SFT and DPO
    │   ├── train_sft.py            #   Supervised fine-tuning with LoRA + SFTTrainer
    │   ├── train_dpo.py            #   DPO alignment with β/lr sweep + MLflow
    │   └── training_utils.py       #   Shared CLI args, TrainingArguments, dataset loading
    │
    ├── eval/                       # Phase 5: Evaluation
    │   ├── evaluate.py             #   In-domain accuracy + out-of-domain refusal testing
    │   └── generate_golden_eval.py #   Type-balanced golden eval set generation
    │
    ├── deploy/                     # Phase 6: Deployment
    │   └── merge_model.py          #   LoRA weight fusion → standalone model
    │
    └── utils/                      # Shared utilities
        ├── model_utils.py          #   Model loading, tokenizer, LoRA detection, GPU cleanup
        ├── pricing.py              #   Per-model token pricing and cost estimation
        └── provider_utils.py       #   Provider detection, client construction, retry logic
```

---

## 📖 Documentation Map

Every sub-package has its own comprehensive README. Use the table below to navigate to the right documentation:

| README | Phase | Covers |
|---|---|---|
| [`doctune/README.md`](doctune/README.md) | All | Package overview, quick start, dependencies, env vars |
| [`doctune/data/README.md`](doctune/data/README.md) | 1–2 | Full pipeline overview, two-stage run examples, cache layout, configuration reference |
| [`doctune/data/extraction/README.md`](doctune/data/extraction/README.md) | 1 | `DoclingManualExtractor`, page batching, GPU fallback, retry/split-on-failure, CLI |
| [`doctune/data/pipeline/README.md`](doctune/data/pipeline/README.md) | 2 | `DatasetBuilder`, `PipelineCache`, gate ordering, data flow diagram, CLI flags |
| [`doctune/data/synthesis/README.md`](doctune/data/synthesis/README.md) | 2b–2d | `LateChunker`, `DiversitySelector`, `ChunkFilter`, `DatasetFilter`, `TeacherModelSynthesizer`, cost reporting |
| [`doctune/training/README.md`](doctune/training/README.md) | 3–4 | `train_sft.py`, `train_dpo.py`, `training_utils.py`, sweep mechanics, MLflow integration |
| [`doctune/eval/README.md`](doctune/eval/README.md) | 5 | Golden eval generation, contamination guard, LLM-as-judge scoring, output formats |
| [`doctune/deploy/README.md`](doctune/deploy/README.md) | 6 | Weight merging, `merge_and_unload()` rationale, output layout, vLLM/Ollama/HF Hub deployment |
| [`doctune/utils/README.md`](doctune/utils/README.md) | All | Model utilities, pricing tables, provider dispatch, retry logic |
| [`MODEL_CARD.md`](MODEL_CARD.md) | — | Model details, intended use, limitations, ethical considerations |
| [`DATA_CARD.md`](DATA_CARD.md) | — | Dataset schema, quality controls, flaw types, statistics |

---

## 📦 Data Pipeline

The data pipeline converts raw PDF documents into a training-ready `alignment_dataset.jsonl` file. It can run **entirely on CPU** — no GPU required.

### Two-stage approach (recommended for large corpora)

```bash
# Stage 1: Extract and cache all chunks (computationally heavy, but no API calls)
python -m doctune.data.extraction.extract_dataset \
    --input-dir ./manuals \
    --domain my_product \
    --docling-page-batch-size 10

# Stage 2: Synthesize from the warmed cache (API calls, but no Docling needed)
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain my_product \
    --model gpt-4o \
    --output ./alignment_dataset.jsonl \
    --diversity-ratio 0.65 \
    --log-level INFO
```

### Single-command approach

```bash
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain my_product \
    --model gpt-4o \
    --output ./alignment_dataset.jsonl
```

### Pipeline gates

Three cost-saving gates fire **before** the teacher-model API is called:

| Gate | Stage | What it does | Default threshold |
|---|---|---|---|
| **Resume filter** | Pre-synthesis | Skips already-cached chunks | — |
| **DiversitySelector** | Pre-synthesis | Keeps the N% most semantically varied chunks per document (jina-embeddings-v3) | 70% (`--diversity-ratio 0.7`) |
| **ChunkFilter** | Pre-synthesis | Drops near-duplicate source chunks (all-MiniLM-L6-v2) | 0.82 (`--chunk-sim-threshold`) |
| **DatasetFilter** | Post-synthesis | Drops near-duplicate generated prompts | 0.92 (`--pair-sim-threshold`) |

### Cache layout

```
.cache/<domain>/
    chunks_<hash>.json          Extracted markdown chunks for one PDF
    synthesis_<hash>.jsonl      Synthesis results (append-only, one line per chunk)
    metadata_<hash>.json        Provenance: original path, chunk count, timestamp
```

The hash is the first 16 characters of the PDF's SHA-256 digest. Renaming a PDF does not invalidate its cache; changing its content does.

### Dataset schema

Every row in `alignment_dataset.jsonl` follows this structure:

```json
{
  "prompt":   "A realistic user question derived from the source text",
  "chosen":   "The factually correct, step-by-step answer grounded in the source",
  "rejected": "A plausible but subtly incorrect answer (for DPO alignment)",
  "metadata": {
    "model": "gpt-4o",
    "input_tokens": 412,
    "output_tokens": 187,
    "cost_usd": 0.0000234
  }
}
```

### Cost auditing

Review your synthesis spend at any time:

```bash
python -m doctune.data.synthesis.report_synthesis_spend --input .cache/my_domain
```

---

## 🔧 Training

### Phase 3: Supervised Fine-Tuning (SFT)

Injects domain-specific knowledge into a foundation model using LoRA adapters:

```bash
python -m doctune.training.train_sft \
    --model-id meta-llama/Llama-3.1-8B \
    --dataset ./alignment_dataset.jsonl \
    --eval-dataset ./golden_eval.jsonl \
    --epochs 3 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32
```

### Phase 4: DPO Alignment

Aligns the SFT model using preference pairs with a hyperparameter sweep over β and learning rate:

```bash
python -m doctune.training.train_dpo \
    --model-id meta-llama/Llama-3.1-8B \
    --sft-adapter ./doctune-sft \
    --betas 0.05 0.1 0.25 0.5 \
    --lrs 5e-6 1e-6
```

The sweep ranks all configurations by reward margin and identifies the best adapter:

```
============================================================
DPO SWEEP COMPLETE — RESULTS RANKED BY REWARD MARGIN
============================================================
  1. llama-dpo-beta0.1-lr5e-06 | β=0.1 lr=5e-06 | ...  <- BEST
  2. llama-dpo-beta0.25-lr5e-06 | β=0.25 lr=5e-06 | ...
============================================================
  Best adapter saved to: ./llama-dpo-beta0.1-lr5e-06
============================================================
```

### Training configuration reference

| Parameter | SFT | DPO |
|---|---|---|
| Base Model | User-specified via `--model-id` | Post-SFT adapter |
| LoRA Rank (r) | 16 | — (reuses SFT adapters) |
| LoRA Alpha (α) | 32 | — |
| Target Modules | Auto-detected from model architecture | — |
| Learning Rate | 2e-4 | Sweep: [5e-6, 1e-6] |
| Scheduler | Cosine (warmup 10%) | Cosine (warmup 10%) |
| Epochs | 3 | 1 |
| Effective Batch Size | 32 (4 × 8 accum) | 32 (2 × 16 accum) |
| Precision | BF16 | BF16 |
| DPO β | — | Sweep: [0.05, 0.1, 0.25, 0.5] |
| Experiment Tracking | MLflow | MLflow (nested per-sweep runs) |

### MLflow experiment tracking

Both phases report to [MLflow](https://mlflow.org/) automatically. Launch the MLflow UI with:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

The DPO sweep additionally wraps each `(β, lr)` configuration in a nested MLflow run for side-by-side comparison.

---

## 📈 Evaluation

### Golden evaluation set

Generate a type-balanced golden eval set with contamination prevention:

```bash
python -m doctune.eval.generate_golden_eval \
    --model claude-3-5-sonnet-20241022 \
    --train-model gpt-4o \
    --domain "home appliances" \
    --count 300 \
    --yes
```

The golden set is distributed across three scenario types:

| Type | Share | Description |
|---|---|---|
| **Factual** | 30% | Direct knowledge retrieval (error codes, specifications, button locations) |
| **Procedural** | 40% | Step-by-step how-to — rejected answer contains a wrong-sequence flaw |
| **Edge-case** | 30% | Multi-step diagnostic reasoning under unusual or failure conditions |

**Contamination guard:** The `--train-model` flag enforces that the eval model comes from a different provider family than the training model. This prevents distributional bias from inflating accuracy metrics.

### Model evaluation

Run the fine-tuned model against the golden set and score it:

```bash
python -m doctune.eval.evaluate \
    --model-id meta-llama/Llama-3.1-8B \
    --adapter ./doctune-dpo-beta0.1-lr5e-06 \
    --baseline \
    --judge \
    --judge-model gpt-4o
```

| Flag | Description |
|---|---|
| `--baseline` | Also runs inference on the **unmodified** base model for comparison |
| `--judge` | Enables **LLM-as-judge** scoring with a configurable judge model |

**Two-signal scoring:**
- **Keyword-based refusal detection** — fast heuristic for out-of-domain boundary enforcement
- **LLM-as-judge** — nuanced qualitative scoring across relevance, accuracy, helpfulness (in-domain) and refusal, safety (out-of-domain)

---

## 🚀 Deployment

### Merge LoRA adapters

Fuse adapters into the base model for standalone inference:

```bash
python -m doctune.deploy.merge_model \
    --model-id meta-llama/Llama-3.1-8B \
    --adapter ./doctune-dpo-beta0.1-lr5e-06 \
    --output ./doctune-merged
```

The merged model is a standard HuggingFace checkpoint — no PEFT dependency at runtime:

```
./doctune-merged/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    ...
```

### Serving options

```bash
# vLLM (recommended for production)
python -m vllm.entrypoints.openai.api_server \
    --model ./doctune-merged \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --port 8000

# Push to HuggingFace Hub
huggingface-cli upload <your-org>/<model-name> ./doctune-merged

# Load with Ollama (after creating a Modelfile)
ollama create my-domain-model -f ./Modelfile
```

### Load in Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./doctune-merged")
tokenizer = AutoTokenizer.from_pretrained("./doctune-merged")
```

---

## 🛠️ Makefile Targets

Run `make help` to see all available targets:

| Target | Description | GPU? |
|---|---|---|
| `make install` | Install all dependencies (runtime + training + dev) | — |
| `make local-setup` | Set up local environment for Phase 2 data generation | No |
| `make data` | Generate the training dataset from PDFs in `./manuals/` | No |
| `make spend-report` | Summarize token and USD spend from synthesis cache | No |
| `make train-sft` | Run SFT (requires `MODEL_ID` env var) | Yes |
| `make train-dpo` | Run DPO sweep (requires `MODEL_ID` env var) | Yes |
| `make eval` | Evaluate the fine-tuned model (requires `MODEL_ID`) | Yes |
| `make eval-baseline` | Evaluate with base model baseline comparison | Yes |
| `make merge` | Merge LoRA adapters into standalone model | No (CPU) |
| `make serve` | Launch vLLM inference server on port 8000 | Yes |
| `make lint` | Run ruff linter | No |
| `make clean` | Remove generated artifacts (datasets, checkpoints, mlruns) | No |

**Example usage:**

```bash
export MODEL_ID="meta-llama/Llama-3.1-8B"
make train-sft
make train-dpo
make eval
make merge
make serve
```

---

## 🌐 Environment Variables

| Variable | Used by | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | synthesis, eval | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | synthesis, eval | — | Anthropic API key |
| `DOMAIN` | eval | `"technical documentation"` | Domain string fallback (prefer `--domain` CLI flag) |
| `DOCTUNE_DOCLING_PAGE_BATCH_SIZE` | extraction | `25` | Pages per Docling conversion batch |
| `DOCTUNE_DOCLING_RETRY_ATTEMPTS` | extraction | `3` | Max retry attempts per failing page range |
| `DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS` | extraction | `1.0` | Retry backoff base in seconds |
| `DOCTUNE_DOCLING_USE_GPU` | extraction | `auto` | OCR device: `auto`, `cpu`, `cuda`, `cuda:N` |

---

## 🖥️ Infrastructure

### Local (Phases 1–2 — Data Generation)

* **Script:** [`setup/local_setup.sh`](setup/local_setup.sh) — Sets up a Python virtual environment with base dependencies for PDF extraction and dataset generation.
* **Requirements:** Python ≥ 3.12, macOS or Linux, [uv](https://docs.astral.sh/uv/) (auto-installed by the script).
* **API key:** OpenAI or Anthropic (or none with Ollama).
* **No GPU required.**

```bash
bash setup/local_setup.sh
source .venv/bin/activate
```

### Remote GPU Pod (Phases 3–6 — Training & Deployment)

* **Script:** [`setup/runpod_setup.sh`](setup/runpod_setup.sh) — Initializes the RunPod GPU environment: installs the HF fine-tuning stack, Flash Attention 2, and launches MLflow UI.
* **Hardware:** Minimum 24 GB VRAM (A100, RTX 3090/4090, RTX 6000 Ada).
* **Storage:** ≥ 50 GB Container Disk + 100 GB Volume Disk.
* **Template:** [RunPod](https://www.runpod.io/) PyTorch template recommended.

```bash
bash setup/runpod_setup.sh
```

---

## 🚀 Suggested Execution Order

### Steps 1–3: Data Curation (can run locally — no GPU)

1. **Set up the local environment:**
   ```bash
   bash setup/local_setup.sh   # or: uv venv .venv && source .venv/bin/activate && uv pip install -e "."
   ```
2. **Export your API key:**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   # or: export ANTHROPIC_API_KEY="your_key_here"
   ```
3. **Place your target PDFs in `./manuals/` and run the pipeline:**
   ```bash
   python -m doctune.data.pipeline.build_dataset \
       --input-dir ./manuals \
       --domain "my product" \
       --model gpt-4o
   ```
4. **(Optional) Generate the golden evaluation set:**
   ```bash
   python -m doctune.eval.generate_golden_eval \
       --model claude-3-5-sonnet-20241022 \
       --train-model gpt-4o \
       --domain "my product" \
       --count 300
   ```

### Steps 4–7: Training & Deployment (GPU required)

5. **Connect to the GPU pod and set up:**
   ```bash
   bash setup/runpod_setup.sh
   ```
6. **Transfer datasets to the pod** (if generated locally):
   ```bash
   scp alignment_dataset.jsonl golden_eval.jsonl user@<pod-ip>:/workspace/doctune/
   ```
7. **Run the training pipeline:**
   ```bash
   export MODEL_ID="meta-llama/Llama-3.1-8B"  # or any HuggingFace model

   # Phase 3: SFT
   python -m doctune.training.train_sft --model-id $MODEL_ID

   # Phase 4: DPO sweep
   python -m doctune.training.train_dpo --model-id $MODEL_ID --betas 0.05 0.1 0.25

   # Phase 5: Evaluate
   python -m doctune.eval.evaluate --model-id $MODEL_ID \
       --adapter ./doctune-dpo-beta0.1-lr5e-06 --baseline --judge

   # Phase 6: Merge and deploy
   python -m doctune.deploy.merge_model --model-id $MODEL_ID \
       --adapter ./doctune-dpo-beta0.1-lr5e-06

   # Serve via vLLM
   make serve
   ```

---

## 📚 Key Dependencies

### Core (all phases)

| Package | Purpose |
|---|---|
| [`openai`](https://github.com/openai/openai-python) | OpenAI API client for synthesis and evaluation |
| [`anthropic`](https://github.com/anthropics/anthropic-sdk-python) | Anthropic API client for synthesis and evaluation |
| [`pydantic`](https://docs.pydantic.dev/) | Structured output validation for API responses |
| [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) | Model and tokenizer downloads |
| [`sentence-transformers`](https://www.sbert.net/) | Embedding models for deduplication (all-MiniLM-L6-v2) |
| [`docling`](https://github.com/DS4SD/docling) | Layout-aware PDF-to-Markdown extraction (IBM DocLayNet) |
| [`torch`](https://pytorch.org/) | Tensor operations and GPU acceleration |

### Training extras (`pip install -e ".[training]"`)

| Package | Purpose |
|---|---|
| [`transformers`](https://huggingface.co/docs/transformers) | Model loading and tokenization |
| [`peft`](https://huggingface.co/docs/peft) | LoRA adapter management |
| [`trl`](https://huggingface.co/docs/trl) | SFTTrainer and DPOTrainer |
| [`accelerate`](https://huggingface.co/docs/accelerate) | Distributed training and device placement |
| [`datasets`](https://huggingface.co/docs/datasets) | HuggingFace dataset loading |
| [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) | Quantized training support |
| [`flash-attn`](https://github.com/Dao-AILab/flash-attention) | Flash Attention 2 (Linux + CUDA only) |
| [`mlflow`](https://mlflow.org/) | Experiment tracking and run comparison |

### Development (`pip install -e ".[dev]"`)

| Package | Purpose |
|---|---|
| [`pytest`](https://docs.pytest.org/) | Test framework |
| [`ruff`](https://docs.astral.sh/ruff/) | Fast Python linter |
| [`pylint`](https://pylint.readthedocs.io/) | Code analysis |

---

## 🔗 External References

### Core technologies

- [IBM Docling](https://github.com/DS4SD/docling) — The layout-aware PDF extraction engine underlying Stage 1
- [DocLayNet](https://github.com/DS4SD/DocLayNet) — IBM's document layout analysis dataset used by Docling
- [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) — 8192-context embedding model used for late-chunking diversity selection
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Lightweight embedding model used for cosine deduplication

### Training methodology

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — The parameter-efficient fine-tuning method used in Phases 3–4
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — The alignment method used in Phase 4
- [SFTTrainer (TRL)](https://huggingface.co/docs/trl/sft_trainer) — HuggingFace's supervised fine-tuning trainer
- [DPOTrainer (TRL)](https://huggingface.co/docs/trl/dpo_trainer) — HuggingFace's DPO trainer
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) — Memory-efficient attention for longer sequences

### Infrastructure

- [RunPod](https://www.runpod.io/) — GPU cloud provider recommended for Phases 3–6
- [vLLM](https://docs.vllm.ai/) — High-throughput inference engine for serving the merged model
- [MLflow](https://mlflow.org/docs/latest/) — Experiment tracking for training runs
- [uv](https://docs.astral.sh/uv/) — Fast Python package manager used for dependency resolution
- [Ollama](https://ollama.com/) — Local LLM runtime for free, offline data generation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `uv pip install -e ".[dev]"`
4. Run the linter before committing: `make lint`
5. Submit a pull request

### Code style

- Python ≥ 3.12
- Linted with [ruff](https://docs.astral.sh/ruff/)
- Type hints encouraged
- Docstrings for all public functions

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 📝 Citation

```bibtex
@misc{doctune,
  title={Doctune: Domain Adaptation Pipeline for Small Language Models},
  author={Felipe Kühne},
  year={2026},
  url={https://github.com/fkuhne/doctune}
}
```
