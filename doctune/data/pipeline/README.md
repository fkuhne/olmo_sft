# `doctune/data/pipeline`

This package is **Stage 2** of the doctune data pipeline. It orchestrates the full
journey from cached extraction chunks to a finished, deduplicated `alignment_dataset.jsonl`
file ready for SFT and DPO training.

```
PipelineCache (chunks)
       │
       ▼
DiversitySelector ──► ChunkFilter ──► TeacherModelSynthesizer ──► DatasetFilter
                                                                         │
                                                                         ▼
                                                               alignment_dataset.jsonl
```

It also houses the shared infrastructure — cache, utility functions, and CLI helpers —
used by both this orchestration layer and the standalone extraction script
(`extraction/extract_dataset.py`).

---

## Why a dedicated pipeline package?

- **Separation of concerns**: extraction (Stage 1) is computationally heavy but
  stateless. The pipeline (Stage 2) is API-driven and stateful. Keeping them separate
  lets you warm the extraction cache independently and then run synthesis at any time.
- **Resumability**: the `PipelineCache` persists every synthesized chunk to disk as it
  completes. A killed or rate-limited run can resume from the last committed chunk
  without re-running OCR or re-calling the teacher model.
- **Shared plumbing**: `pipeline_utils.py` and `pipeline_cache.py` are consumed by
  both `build_dataset.py` and `extraction/extract_dataset.py`, ensuring consistent
  caching behaviour and CLI conventions across both entry points.

---

## Files

| File | Role |
|---|---|
| `build_dataset.py` | Full pipeline orchestrator + CLI entry point (Stage 2) |
| `pipeline_utils.py` | Shared helpers: PDF discovery, cached extraction, CLI arg registration, extractor/cache initialisation |
| `pipeline_cache.py` | Persistent, resumable on-disk cache for extraction chunks and synthesis results |
| `__init__.py` | Package namespace (currently empty) |

---

## `build_dataset.py`

### Module-level constant

| Constant | Value | Description |
|---|---|---|
| `_INTER_CHUNK_SLEEP_S` | `1.0` | Seconds to sleep between teacher-model API calls. Prevents burst rate-limit errors on chunks processed back-to-back. |

---

### `_BuildStats` (dataclass)

Internal accumulator that tracks pipeline-wide metrics as the build loop progresses.
Not exported; used only within `DatasetBuilder`.

| Field | Type | Description |
|---|---|---|
| `total_chunks_processed` | `int` | Running count of all chunks that entered the synthesis loop (including skipped ones). |
| `total_pairs_generated` | `int` | Total QA pairs returned by the teacher model across all chunks. |
| `skipped_chunks` | `int` | Chunks that raised an unhandled exception during synthesis. |
| `pdf_count` | `int` | Total number of PDF manuals processed. |

#### `log_summary(unique_pairs)`

Prints the final pipeline summary banner to stdout after all PDFs have been processed.
Receives `unique_pairs` — the count of QA pairs that survived deduplication — from
the caller, since this count lives in `DatasetFilter`.

---

### `DatasetBuilder`

The master orchestrator for the data curation pipeline. Instantiate it with the
desired configuration, then call `build()` once.

#### `__init__(...)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_dir` | `str` | — | Directory containing PDF manuals to process. |
| `output_file` | `str` | — | Path for the output `alignment_dataset.jsonl`. |
| `model` | `str` | — | Teacher model identifier (e.g. `"gpt-4o"`). |
| `provider` | `str\|None` | `None` | API provider; auto-detected from the model name if `None`. |
| `domain` | `str` | — | Subject-matter domain injected into synthesis prompts (e.g. `"home appliances"`). |
| `extractor` | `DoclingManualExtractor\|None` | — | Pre-initialised Docling extractor. Pass `None` to rely entirely on the cache. |
| `cache` | `PipelineCache\|None` | — | Pipeline cache instance. Pass `None` to disable caching. |
| `diversity_ratio` | `float\|None` | `0.7` | Fraction of chunks to retain after diversity selection. `None` disables diversity filtering entirely. |
| `chunk_sim_threshold` | `float` | `0.82` | Cosine similarity ceiling for chunk-level deduplication. |
| `pair_sim_threshold` | `float` | `0.92` | Cosine similarity ceiling for prompt-level deduplication. |

Constructed sub-components:

| Attribute | Type | Description |
|---|---|---|
| `synthesizer` | `TeacherModelSynthesizer` | Calls the teacher model API and parses returned QA tuples. |
| `filter` | `DatasetFilter` | Prompt-level semantic deduplication; accumulates accepted pairs. |
| `chunk_filter` | `ChunkFilter` | Chunk-level semantic deduplication applied before synthesis. |
| `diversity_selector` | `DiversitySelector\|None` | Greedy farthest-first diversity selector; `None` when disabled. |

---

#### `build()` — public entry point

Drives the full pipeline end-to-end. Calls `_process_single_pdf` for each discovered
PDF, then writes the deduplicated dataset to `output_file`. Per-PDF exceptions are
caught, logged, and skipped so a single bad manual does not abort the entire run.

**Pipeline steps:**

1. Discovers all `*.pdf` files in `input_dir`.
2. For each PDF, calls `_process_single_pdf`.
3. After all PDFs, prints the summary banner and saves `alignment_dataset.jsonl`.

---

#### `_process_single_pdf(pdf_path, device_context, stats)` — private

Runs the four-stage sub-pipeline for a single PDF:

| Step | Helper / call | Description |
|---|---|---|
| 1. Extract | `extract_chunks_cached(...)` | Loads chunks from cache or runs live Docling extraction. |
| 2. Resume | `_resume_from_cache(...)` | Re-ingests synthesis results from a prior run, skipping already-completed chunks. |
| 3. Gate | `_select_active_chunks(...)` | Applies chunk dedup and (optionally) diversity selection to determine which chunks proceed to synthesis. |
| 4. Synthesize | `_synthesize_chunk(...)` per chunk | Calls the teacher model; funnels results through `DatasetFilter`; sleeps `_INTER_CHUNK_SLEEP_S` between calls. |

Chunks eliminated by the filtering gates in step 3 are written to the cache as empty
records so that a resumed run skips them without re-evaluating the gates.

---

#### Private helper methods

| Method | Signature | Description |
|---|---|---|
| `_cache_synthesis` | `(pdf_hash, chunk_index, results) → None` | Writes synthesis results for one chunk to the cache. No-ops when caching is disabled. |
| `_resume_from_cache` | `(pdf_hash, enriched_chunks, stats) → set[int]` | Loads all cached synthesis results for this PDF, feeds them back through `DatasetFilter`, increments stats, and returns the set of already-completed chunk indices. |
| `_select_active_chunks` | `(enriched_chunks, completed_indices) → list[tuple[int, str]]` | Filters out completed chunks, runs `ChunkFilter` to remove near-duplicates, then optionally runs `DiversitySelector` to keep only the most semantically varied subset. Returns ordered `(index, chunk)` pairs. |
| `_synthesize_chunk` | `(j, chunk, len_chunks, pdf_hash, stats) → list[dict]` | Calls `TeacherModelSynthesizer.process_chunk`, writes the result to cache immediately, and returns the generated QA-tuple list. On exception, logs, increments `skipped_chunks`, caches an empty record, and returns `[]`. |

---

### CLI flags (`build_dataset.py`)

Inherits all flags from `add_common_cli_args` (see `pipeline_utils.py`) plus:

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-4o` | Teacher model identifier |
| `--provider` | *(auto)* | API provider; auto-detected from model name if omitted |
| `--output` | `alignment_dataset.jsonl` | Path for the output JSONL file |
| `--diversity-ratio` | `0.7` | Fraction of chunks to keep after diversity selection |
| `--no-diversity` | `False` | Disables the diversity selector entirely |
| `--log-level` | `WARNING` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--log-file` | *(none)* | Optional file path to receive log output in addition to stderr |

**Example:**

```bash
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain "home appliances" \
    --model gpt-4o \
    --output ./data/alignment_dataset.jsonl \
    --diversity-ratio 0.65 \
    --log-level INFO
```

> **Note:** `build_dataset.py` does not initialise the Docling extractor by default
> (`init_extractor=False`). It assumes extraction chunks are already present in the
> cache from a prior `extract_dataset.py` run. Pass an extractor explicitly if you
> want both stages in a single command.

---

## `pipeline_utils.py`

Shared utility functions imported by both `build_dataset.py` and
`extraction/extract_dataset.py`. Contains no state.

### Functions

#### `extract_device_context(filename) → str`

Converts a PDF filename into a clean, title-cased source-context label suitable for
injecting into chunk headers.

```python
extract_device_context("product_user_guide.pdf")  # → "Product"
extract_device_context("dryer-installation-manual.pdf")  # → "Dryer Installation"
```

Strips trailing `" Manual"` and `" User Guide"` suffixes after title-casing, so chunk
headers read naturally (e.g. `"Product"` rather than `"Product User Guide"`).

---

#### `discover_pdfs(input_dir) → list[str]`

Returns a sorted list of all `*.pdf` file paths inside `input_dir`. Returns an empty
list if no PDFs are found; callers are responsible for handling the empty case.

---

#### `extract_chunks_cached(pdf_path, device_context, extractor, cache) → list[str]`

The caching gateway for extraction. Behaviour:

1. If `cache` is not `None`, computes the PDF's SHA-256 hash and checks for a cached
   chunk file.
2. On a **cache hit**, loads and returns chunks from disk immediately (no Docling call).
3. On a **cache miss**, calls `extractor.process_manual(pdf_path, device_context)` to
   run live Docling extraction.
4. Saves the freshly extracted chunks to cache for future runs.
5. If both `cache` is `None` and `extractor` is `None`, logs a warning and returns `[]`.

---

#### `add_common_cli_args(parser)`

Registers CLI flags shared by all pipeline entry points onto an
`argparse.ArgumentParser`. These flags are used by both `build_dataset.py` and
`extract_dataset.py`:

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | `./manuals` | Directory containing PDF manuals |
| `--domain` | `technical documentation` | Subject-matter domain (used in prompts and cache subdirectory) |
| `--cache-dir` | `.cache` | Root directory for the pipeline cache |
| `--no-cache` | `False` | Disable caching; forces a full re-run every time |
| `--chunk-sim-threshold` | `0.82` | Cosine similarity ceiling for chunk-level deduplication |
| `--pair-sim-threshold` | `0.92` | Cosine similarity ceiling for prompt-level deduplication |

---

#### `add_extraction_cli_args(parser)`

Registers CLI flags specific to the extraction step only:

| Flag | Default | Description |
|---|---|---|
| `--docling-page-batch-size` | env / `25` | Pages per Docling conversion batch; overrides `DOCTUNE_DOCLING_PAGE_BATCH_SIZE` |

---

#### `init_extractor_and_cache(args, init_extractor=True) → tuple[DoclingManualExtractor|None, PipelineCache|None]`

Builds the optional extractor and cache objects from parsed CLI arguments.

| Parameter | Default | Description |
|---|---|---|
| `args` | — | Parsed `argparse.Namespace` containing flags from `add_common_cli_args`. |
| `init_extractor` | `True` | When `False`, skips Docling initialisation (used by `build_dataset.py`, which assumes the cache is pre-warmed). |

Returns `(extractor, cache)` where either may be `None` depending on flags and
`init_extractor`.

---

## `pipeline_cache.py`

### `PipelineCache`

Manages all on-disk caching for the pipeline. The cache is structured as a flat
directory (`<cache_dir>/<domain>/`) containing per-PDF JSON/JSONL files, keyed by a
16-character SHA-256 prefix of each PDF's binary content.

#### Cache layout

```
.cache/<domain>/
    chunks_<hash>.json        — extracted markdown chunks (list[str])
    synthesis_<hash>.jsonl    — append-only synthesis results (one JSON line per chunk)
    metadata_<hash>.json      — pdf_path, chunk_count, cached_at timestamp
```

#### Construction — `__init__(cache_dir=".cache", domain="technical_documentation")`

Sanitises the domain name to a filesystem-safe slug and creates the cache directory
if it does not exist.

---

#### PDF identity

| Method | Signature | Description |
|---|---|---|
| `get_pdf_hash` | `(pdf_path) → str` | Computes a SHA-256 hash of the PDF's binary content and returns the first 16 hex characters. This is the stable key used for all cache lookups. |

---

#### Chunk caching — extraction results

| Method | Signature | Description |
|---|---|---|
| `has_chunks` | `(pdf_hash) → bool` | Returns `True` if a chunk cache file exists for this PDF hash. |
| `load_chunks` | `(pdf_hash) → list[str]` | Reads and deserialises the cached chunk list from disk. |
| `save_chunks` | `(pdf_hash, chunks, pdf_path) → None` | Serialises the chunk list to `chunks_<hash>.json` and writes a companion `metadata_<hash>.json` with provenance info (original path, chunk count, timestamp). |

---

#### Synthesis caching — per-chunk, append-only

The synthesis cache uses **JSONL** (one JSON object per line) so that each chunk's
result is committed atomically as it completes. A killed process loses at most the
one chunk currently in flight.

| Method | Signature | Description |
|---|---|---|
| `get_completed_chunk_indices` | `(pdf_hash) → set[int]` | Scans the synthesis JSONL and returns the set of chunk indices that have already been written. Used by `_resume_from_cache` to skip completed work. |
| `load_all_synthesis_results` | `(pdf_hash) → list[dict]` | Loads every QA-tuple dict from all JSONL records for a given PDF. Used to re-ingest prior results into `DatasetFilter` on resume. |
| `append_synthesis_result` | `(pdf_hash, chunk_index, results) → None` | Appends a single JSON line `{chunk_index, results, timestamp}` to the synthesis JSONL file. Safe to call after process interruption; subsequent appends simply add more lines. |
| `_iter_synthesis_records` | `(pdf_hash) → Iterator[dict]` | Internal generator that yields parsed records from the JSONL file, skipping blank and malformed lines gracefully. |

---

## Pipeline gate ordering

When all filters are enabled, chunks pass through three successive gates before they
reach the teacher model:

```
All chunks for a PDF
        │
        ▼
[Gate 0] Resume filter
   Skip indices already completed in a previous run
        │
        ▼
[Gate 1] ChunkFilter (chunk-level dedup)
   Discard source chunks that are too similar to an already-seen chunk
   (cosine similarity ≥ chunk_sim_threshold, default 0.82)
        │
        ▼
[Gate 2] DiversitySelector (optional)
   Greedy farthest-first selection to keep the most varied subset
   (retains diversity_ratio fraction, default 70%)
        │
        ▼
TeacherModelSynthesizer  →  DatasetFilter (prompt-level dedup)
   Generates QA pairs        Discards near-duplicate prompts
                             (cosine similarity ≥ pair_sim_threshold, default 0.92)
        │
        ▼
alignment_dataset.jsonl
```

---

## Data flow diagram

```
┌─────────────┐   extract_chunks_cached   ┌───────────────┐
│  PDF files  │ ─────────────────────────►│ PipelineCache │
│  (*.pdf)    │                           │  chunks_*.json│
└─────────────┘                           └──────┬────────┘
                                                 │ list[str]
                                          ┌──────▼────────┐
                                          │  ChunkFilter  │ ← chunk_sim_threshold
                                          └──────┬────────┘
                                                 │ deduped
                                          ┌──────▼────────────┐
                                          │ DiversitySelector │ ← diversity_ratio
                                          │  (optional)       │
                                          └──────┬────────────┘
                                                 │ selected
                                    ┌────────────▼──────────────┐
                                    │  TeacherModelSynthesizer  │
                                    │  (teacher model API)      │
                                    └────────────┬──────────────┘
                                                 │ QA tuples
                                          ┌──────▼────────┐
                                          │ DatasetFilter │ ← pair_sim_threshold
                                          │  (pair dedup) │
                                          └──────┬────────┘
                                                 │ unique pairs
                                    ┌────────────▼──────────────┐
                                    │  alignment_dataset.jsonl  │
                                    └───────────────────────────┘
```
