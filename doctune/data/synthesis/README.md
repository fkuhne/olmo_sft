# `doctune/data/synthesis`

This package contains **Stages 2b–2d** of the doctune data pipeline: the components
that transform raw extraction chunks into a high-quality, deduplicated
`alignment_dataset.jsonl` ready for SFT and DPO fine-tuning.

```
extraction chunks (list[str])
        │
        ▼
 LateChunker ──► DiversitySelector   ← selects the most semantically varied chunks
        │
        ▼
   ChunkFilter                        ← drops near-duplicate source chunks
        │
        ▼
 TeacherModelSynthesizer              ← generates SFT (chosen) + DPO (rejected) pairs
        │
        ▼
  DatasetFilter                       ← drops near-duplicate generated prompts
        │
        ▼
 alignment_dataset.jsonl
```

---

## Why a dedicated synthesis package?

- **Model-agnostic generation**: `TeacherModelSynthesizer` supports OpenAI, Anthropic,
  and Ollama through a unified provider dispatch layer, so the rest of the pipeline
  never needs to know which API is in use.
- **Layered filtering**: two independent deduplication gates operate at different
  granularities (source chunks before synthesis, generated prompts after), maximising
  dataset diversity without wasting API budget on redundant content.
- **Reusable embedding primitives**: `LateChunker` is a standalone module with no
  doctune imports. It can be consumed by any future component (deduplication, retrieval
  indexing) without circular dependencies.

---

## Files

| File | Role |
|---|---|
| `late_chunker.py` | Reusable late-chunking encoder for jina-embeddings-v3. No doctune imports. |
| `diversity_selector.py` | Greedy farthest-first diversity selection using late-chunked embeddings. |
| `deduplicate_dataset.py` | Chunk-level (`ChunkFilter`) and prompt-level (`DatasetFilter`) semantic deduplication. |
| `teacher_model_synthesis.py` | Synthetic SFT and DPO data generation via a teacher LLM. |
| `report_synthesis_spend.py` | CLI utility to audit token and cost usage from synthesis cache files. |
| `__init__.py` | Package API surface. |

---

## `late_chunker.py`

The foundational encoding module. It encodes whole documents through jina-embeddings-v3's
transformer self-attention before pooling per-chunk embeddings from the token-level
hidden states. This means every chunk vector carries **full-document context** — a
"Chapter 5: Safety Valve" chunk is aware of the "XR-7" model name introduced in
Chapter 1, producing embeddings that reflect cross-referential meaning rather than
surface vocabulary alone.

> **Design rule**: This module has **no doctune imports**. Only stdlib, numpy, and torch.
> This keeps the dependency graph strictly layered:
> ```
> late_chunker.py  ←  diversity_selector.py
>                  ←  deduplicate_dataset.py  (future)
> ```

### Module-level constants

| Constant | Value | Description |
|---|---|---|
| `MODEL_ID` | `"jinaai/jina-embeddings-v3"` | HuggingFace model identifier. Public so consumers can reference it without hard-coding strings. |
| `MAX_TOKENS` | `8192` | jina-embeddings-v3 context window size in tokens. Determines when sliding-window encoding is used. |
| `EMBED_DIM` | `1024` | Embedding output dimension. Used to allocate result arrays. |
| `_WINDOW_TOKENS` | `7500` | Tokens per sliding window (internal). |
| `_OVERLAP_TOKENS` | `500` | Overlap between adjacent windows (internal). |
| `_SEPARATOR` | `"\n"` | Separator used when reconstructing the full document from chunks. |

---

### Pure helper functions

These functions have **no model dependency** and can be unit-tested with synthetic data.

#### `normalize(text) → str`

NFKC-normalises text to ensure tokenizer alignment. Folds compatibility characters
(curly quotes, em-dashes, ligatures) into canonical equivalents, consistent with what
BPE tokenizers expect.

---

#### `reconstruct_full_text(norm_chunks) → (str, list[tuple[int, int]])`

Joins normalised chunks with `_SEPARATOR` into a single document string and returns
per-chunk character spans `(start_char, end_char)`. These spans are the bridge between
"which text belongs to which chunk" and the token-level offset mapping used for pooling.

---

#### `char_span_to_token_span(char_start, char_end, offset_mapping) → (int, int)`

Maps a character span to inclusive-start / exclusive-end token indices using the
tokenizer's `offset_mapping`. Handles subword tokens straddling boundaries correctly.
Raises `ValueError` if no tokens fall within the span (caused by truncation or a
normalisation mismatch).

---

#### `pool_chunk(token_embeddings, token_start, token_end) → Tensor`

Mean-pools the token embeddings in the range `[token_start, token_end)` and L2-
normalises the result to produce a single unit-norm chunk embedding.

---

#### `pool_all_spans(token_embeddings, offset_mapping, char_spans) → np.ndarray`

Calls `char_span_to_token_span` + `pool_chunk` for every chunk in a single-window
document. Returns a numpy array of shape `[n_chunks, EMBED_DIM]`. Skips chunks that
raise `ValueError` (logged as warnings) and leaves their row as zeros.

---

#### `assign_chunks_to_windows(char_spans, window_char_ranges) → list[int]`

Assigns each chunk to the sliding window where it sits most centrally (chunk midpoint
closest to window midpoint). Maximises contextual coverage from the transformer's
bidirectional attention. Pure function — no model dependency.

---

### `LateChunker`

The main class. Lazy-loaded: the ~2.2 GB jina-embeddings-v3 model is not downloaded
or moved to GPU until the first call to `encode()`.

#### Construction — `__init__(model_id=MODEL_ID, device=None)`

| Argument | Default | Description |
|---|---|---|
| `model_id` | `MODEL_ID` | HuggingFace model identifier. Override only for testing alternatives. |
| `device` | `None` | `"cuda"`, `"cpu"`, or `None` for auto-detection. |

**Key attribute**: `last_token_count` — stores the token count from the most recent
`encode()` call, allowing consumers (e.g. `DiversitySelector`) to detect sliding-window
usage without re-tokenising.

#### `encode(chunks, *, task="retrieval.passage") → np.ndarray`

Public entry point. Encodes a list of chunk strings via late chunking and returns a
`[n_chunks, 1024]` numpy array, each row L2-normalised.

Automatically chooses between single-pass (≤ `MAX_TOKENS`) and sliding-window
(> `MAX_TOKENS`) encoding. Updates `last_token_count`.

The `task` parameter selects jina-embeddings-v3's LoRA adapter:
- `"retrieval.passage"` — for diversity selection and indexing (default)
- `"text-matching"` — for deduplication similarity

#### `count_tokens(text) → int`

Returns the token count of `text` without truncation. Triggers lazy loading.

#### Private helpers

| Method | Description |
|---|---|
| `_ensure_loaded()` | Downloads tokenizer + model on first call; moves model to device. |
| `_encode_single(full_text, char_spans, *, task)` | Single-pass encoding for short documents. Calls `_get_token_embeddings` then `pool_all_spans`. |
| `_encode_sliding_window(full_text, char_spans, *, task)` | Overlapping-window encoding for long documents. Tokenises without truncation, builds windows, assigns each chunk to its most central window, encodes each window, pools assigned chunks. |
| `_get_token_embeddings(text, *, task)` | Tokenises `text`, pops `offset_mapping` before the forward pass (Jina's custom `forward()` rejects it), runs the model, returns `(token_embeddings, offset_mapping)`. |

---

## `diversity_selector.py`

Uses `LateChunker` embeddings to select the most semantically varied subset of chunks
for a document, reducing API calls to the teacher model without sacrificing coverage.

### `SelectionResult` (dataclass)

Container for the output of a single diversity selection pass.

| Field | Type | Description |
|---|---|---|
| `selected_chunks` | `list[str]` | Ordered list of selected chunk strings. |
| `selected_indices` | `list[int]` | Original indices of selected chunks in the input list. |
| `embeddings` | `np.ndarray` | Late-chunked embeddings for **all** input chunks, shape `[n, 1024]`. Retained for downstream use (e.g. cross-document dedup). |
| `dropped_count` | `int` | Number of chunks not selected. |
| `used_sliding_window` | `bool` | Whether the document exceeded `MAX_TOKENS` and required windowed encoding. |
| `stats` | `dict` | Summary dict: `total_chunks`, `selected_chunks`, `diversity_ratio`, `total_tokens`. |

---

### `DiversitySelector`

#### Construction — `__init__(model_id, diversity_ratio, min_chunks, device)`

| Argument | Default | Description |
|---|---|---|
| `model_id` | `MODEL_ID` | Passed through to `LateChunker`. |
| `diversity_ratio` | `0.7` | Fraction of chunks to keep. `0.7` on 40 chunks → 28 selected. |
| `min_chunks` | `5` | Minimum chunks to select regardless of ratio. Prevents very short documents from being over-reduced. |
| `device` | `None` | Passed through to `LateChunker`. |

The underlying `LateChunker` is lazy — no model is loaded until `select()` is first called.

#### `select(chunks) → SelectionResult`

Public entry point. Encodes the chunk list via `LateChunker.encode()`, then applies
`_greedy_farthest_first` to select the `k` most diverse chunks, where
`k = max(min_chunks, round(len(chunks) * diversity_ratio))`.

---

### `_greedy_farthest_first(embeddings, k) → list[int]` (module-level)

The Gonzalez algorithm for maximum-diversity subset selection. O(n × k) time.

1. **Seed** with the chunk whose L2 norm is largest (deterministic, no randomness needed since embeddings are unit-normed).
2. **Greedily** add the chunk with the maximum minimum distance to all already-selected chunks.
3. Returns indices in **selection order** (most diverse first).

Uses cosine distance (= 1 − dot-product of unit-norm vectors) as the distance metric.

---

## `deduplicate_dataset.py`

Two deduplication filters that operate at different stages of the pipeline, each using
`all-MiniLM-L6-v2` sentence-transformer embeddings.

### Module-level

| Symbol | Description |
|---|---|
| `_DEDUP_MODEL_ID` | `"all-MiniLM-L6-v2"` — embedding model name, centralised as a constant. |
| `_embedding_model` | Module-level singleton; loaded once and shared between both filter classes. |
| `_get_embedding_model()` | Lazy loader for the singleton. Returns the shared `SentenceTransformer` instance. |
| `_REQUIRED_KEYS` | `("prompt", "chosen", "rejected")` — schema keys validated in `DatasetFilter`. |

---

### `DatasetFilter`

Prompt-level deduplication. Applied **after** teacher-model synthesis. Maintains an
in-memory vector store of accepted prompt embeddings and drops any new pair whose
prompt is too similar to an already-accepted one.

#### Construction — `__init__(similarity_threshold=0.92)`

Lower values are more aggressive (keep only very distinct prompts). Default `0.92`
was calibrated empirically for technical QA diversity.

**Key attributes:**

| Attribute | Description |
|---|---|
| `accepted_data` | `list[dict]` — the filtered QA pairs; written to disk by `save_dataset`. |
| `accepted_embeddings` | `torch.Tensor | None` — growing matrix of accepted prompt embeddings, shape `[n, 384]`. |

#### `validate_schema(qa_pair) → bool` (static)

Returns `True` only if `qa_pair` contains all three required keys (`"prompt"`,
`"chosen"`, `"rejected"`) as non-empty strings after stripping whitespace.

#### `process_new_pair(qa_pair) → bool`

Validates schema, encodes the prompt, compares against all accepted embeddings via
cosine similarity. Accepts the first pair unconditionally. Subsequent pairs are
accepted only if `max_similarity ≤ similarity_threshold`. Dropped pairs produce a
structured `DEDUP_DROP` audit log at `INFO` level with the similarity score, threshold,
and the matched prompt — useful for threshold calibration.

Returns `True` if accepted, `False` if dropped.

#### `save_dataset(output_path)`

Serialises `accepted_data` to JSONL and writes to `output_path`. Raises `OSError`
(logged before re-raising) if the write fails — e.g. permission error or full disk.

---

### `ChunkFilter`

Source-chunk deduplication. Applied **before** teacher-model synthesis so duplicate
source material never reaches the API.

Intentionally more permissive than `DatasetFilter` (lower default threshold, `0.82`)
because raw chunk text scores higher surface similarity than generated questions, even
when the underlying content is genuinely distinct.

#### Construction — `__init__(similarity_threshold=0.82)`

#### `is_duplicate(chunk_text) → bool`

Returns `True` if `chunk_text` is too similar to a previously seen chunk (above the
threshold). **Side-effect on `False`**: adds the chunk's embedding to the internal
store so future chunks are compared against it.

#### `log_summary()`

Prints accepted/rejected counts and the configured threshold to stdout. Called at the
end of the pipeline by `DatasetBuilder.build()`.

---

## `teacher_model_synthesis.py`

Generates synthetic SFT and DPO training pairs from document chunks using a teacher
LLM. Supports OpenAI (structured outputs), Anthropic (JSON mode), and Ollama
(OpenAI-compatible local API).

### Module-level helpers

| Function | Description |
|---|---|
| `_build_usage(input_tokens, output_tokens) → UsageMetrics` | Normalises raw token counts (coercing `None` to 0, clamping negatives) into a consistent `{"input_tokens": int, "output_tokens": int}` dict. |
| `_extract_usage_input_output(response) → UsageMetrics` | Extracts usage from responses that expose `input_tokens` / `output_tokens` — covers both the OpenAI Responses API and the Anthropic messages API, which share these field names. |
| `_extract_usage_from_openai_chat(response) → UsageMetrics` | Extracts usage from OpenAI-compatible Chat Completions responses (uses `prompt_tokens` / `completion_tokens`). Used for Ollama. |
| `_split_usage_across_pairs(total_usage, pair_count) → list[UsageMetrics]` | Distributes a single API call's token usage evenly across all generated SFT pairs, with remainder tokens allocated to the first pairs. Ensures cost tracking is accurate per-pair. |

### Pydantic schemas

| Class | Fields | Purpose |
|---|---|---|
| `SFTPair` | `prompt`, `chosen` | Single SFT question-answer pair (frozen). |
| `SFTResponse` | `qa_pairs: list[SFTPair]` | Structured API response containing multiple SFT pairs. |
| `DPOResponse` | `rejected` | Structured API response containing one DPO rejected answer (frozen). |

### JSON-mode instruction constants

| Constant | Purpose |
|---|---|
| `_SFT_JSON_INSTRUCTION` | Appended to the system prompt for Anthropic/Ollama, instructing the model to respond with `{"qa_pairs": [...]}` only. |
| `_DPO_JSON_INSTRUCTION` | Appended for DPO generation, instructing the model to respond with `{"rejected": "..."}` only. |

---

### `TeacherModelSynthesizer`

#### Construction — `__init__(domain, api_key, model, provider, ollama_base_url)`

| Argument | Default | Description |
|---|---|---|
| `domain` | `"technical documentation"` | Domain injected into the system prompt (e.g. `"home appliances"`). |
| `api_key` | `None` | Explicit API key; falls back to environment variables. |
| `model` | `"gpt-4o"` | Teacher model identifier. |
| `provider` | `None` | `"openai"`, `"anthropic"`, or `"ollama"`. Auto-detected from model name if `None`. |
| `ollama_base_url` | `None` | Custom Ollama server URL. |

---

#### Provider-specific backend methods (private)

Each provider pair (SFT + DPO) is decorated with `@retry_on_rate_limit()`.

| Method | Provider | Mode | Description |
|---|---|---|---|
| `_openai_generate_sft` | OpenAI | Structured outputs | Calls `client.responses.parse` with `SFTResponse` schema. |
| `_openai_generate_dpo` | OpenAI | Structured outputs | Calls `client.responses.parse` with `DPOResponse` schema. |
| `_anthropic_raw_call` | Anthropic | JSON mode | Calls `client.messages.create`; returns raw text + usage. |
| `_ollama_raw_call` | Ollama | JSON mode | Calls `client.chat.completions.create` with `response_format={"type": "json_object"}`. |
| `_json_mode_call` | Anthropic / Ollama | JSON mode | Dispatcher: routes to `_anthropic_raw_call` or `_ollama_raw_call` based on `self.provider`. |
| `_json_mode_generate_sft` | Anthropic / Ollama | JSON mode | Appends `_SFT_JSON_INSTRUCTION` to system prompt, calls `_json_mode_call`, validates with `SFTResponse.model_validate_json`. |
| `_json_mode_generate_dpo` | Anthropic / Ollama | JSON mode | Appends `_DPO_JSON_INSTRUCTION`, validates with `DPOResponse.model_validate_json`. |

---

#### Prompt construction

##### `_build_sft_user_prompt(markdown_chunk) → str` (static)

Constructs the two-stage user prompt for SFT generation:

- **Step 1 — Focus selection**: The model identifies the single most specific,
  actionable claim in the chunk and labels it `FOCUS: <statement>`. Prevents
  question drift across unrelated sub-topics.
- **Step 2 — Question generation**: Using only the named anchor, the model generates
  exactly 3 QA pairs from three angles:
  - Angle A — Symptom-based (user describes a problem)
  - Angle B — Direct action (user asks how to perform the procedure)
  - Angle C — Edge-case or clarification (boundary condition, prerequisite, easy-to-get-wrong detail)

---

#### Public API

##### `generate_sft_pairs(markdown_chunk) → list[tuple[dict, UsageMetrics]]`

Generates diverse SFT QA pairs from one chunk. Dispatches to the OpenAI structured
output path or the JSON-mode path based on `self.provider`. Returns a list of
`(pair, usage)` tuples where `pair` has `"prompt"` and `"chosen"` keys and `usage`
is the per-pair token attribution. Returns `[]` on `JSONDecodeError` or any other
exception (logged at `WARNING`/`ERROR`).

##### `generate_dpo_rejection(prompt, chosen) → tuple[str | None, UsageMetrics]`

Generates a subtly flawed `"rejected"` answer for DPO alignment. The system prompt
enforces that the rejection must look **highly plausible** and contain a **critical
factual error** (wrong sequence, wrong component, or subtle hallucination) — not an
obviously bad response. Returns `(None, zero_usage)` on failure.

##### `process_chunk(markdown_chunk) → list[dict]`

Orchestrates the full SFT → DPO pipeline for a single chunk:

1. Calls `generate_sft_pairs` to get all QA pairs.
2. For each pair, calls `generate_dpo_rejection` to produce the rejected response.
3. Combines token counts and computes `cost_usd` via `compute_model_usage_cost`.
4. Returns a list of complete tuples:
```python
{
    "prompt":   "...",
    "chosen":   "...",
    "rejected": "...",
    "metadata": {
        "model":         "gpt-4o",
        "input_tokens":  412,
        "output_tokens": 187,
        "cost_usd":      0.0000234
    }
}
```

---

## `report_synthesis_spend.py`

A standalone CLI utility for auditing token and cost usage from synthesis JSONL cache
files produced by `PipelineCache`. Accepts a single file or a directory, and produces
both an aggregate and a per-model breakdown.

### `_SpendTotals` (dataclass)

Accumulates totals during the scan: `files_scanned`, `records_scanned`,
`tuples_scanned`, `tuples_with_metadata`, `tuples_missing_metadata`,
`input_tokens`, `output_tokens`, `cost_usd`.

### Private helpers

| Function | Description |
|---|---|
| `_resolve_inputs(input_path)` | Resolves `input_path` to a list of synthesis JSONL files. Accepts a single file or a directory (glob `synthesis_*.jsonl`). Raises `FileNotFoundError` otherwise. |
| `_safe_int(value)` | Coerces `value` to a non-negative int, returning `0` on any failure. |
| `_safe_float(value)` | Coerces `value` to a float, returning `0.0` on any failure. |
| `_iter_jsonl(path)` | Yields parsed JSON records from a JSONL file, skipping blank and malformed lines. |

### `summarize_spend(input_path) → tuple[_SpendTotals, dict[str, _SpendTotals]]`

Main computation function. Iterates over all matching JSONL files, accumulates token
and cost totals globally and per-model. If a tuple's `cost_usd` is missing or zero,
falls back to `compute_model_usage_cost` for on-the-fly estimation.

Returns `(totals, by_model)`.

### `_print_summary(totals, by_model)`

Renders the full report to stdout — aggregate metrics followed by a per-model
breakdown sorted by descending cost.

### CLI usage

```bash
python -m doctune.data.synthesis.report_synthesis_spend --input .cache/my_domain
```

| Flag | Default | Description |
|---|---|---|
| `--input` | `.cache` | Path to a synthesis JSONL file or directory containing `synthesis_*.jsonl` files |

---

## `__init__.py`

Exposes the five primary public classes at the package level:

```python
from doctune.data.synthesis import (
    TeacherModelSynthesizer,
    DatasetFilter,
    ChunkFilter,
    DiversitySelector,
    LateChunker,
)
```

---

## Dependency graph

```
     stdlib / numpy / torch
              │
              ▼
       late_chunker.py          (no doctune imports)
              │
              ▼
    diversity_selector.py
              │
              │         deduplicate_dataset.py
              │                  │
              └──────────────────┤
                                 ▼
                    teacher_model_synthesis.py
                    (doctune.utils.pricing,
                     doctune.utils.provider_utils)
                                 │
                                 ▼
                       build_dataset.py  (pipeline/)
```

---

## Data flow diagram

```
list[str] chunks
      │
      ▼
┌─────────────────┐   encode()   ┌──────────────┐
│ DiversitySelector│ ────────────►│  LateChunker │
│  (optional gate) │ ◄──indices──│  jina-v3     │
└────────┬─────────┘             └──────────────┘
         │ selected chunks
         ▼
┌─────────────────┐
│   ChunkFilter   │  is_duplicate()   →  all-MiniLM-L6-v2
│  (chunk dedup)  │
└────────┬─────────┘
         │ novel chunks
         ▼
┌──────────────────────────┐
│ TeacherModelSynthesizer  │  process_chunk()
│  ┌──────────────────┐    │     ├─ generate_sft_pairs()   → teacher model API
│  │ SFTPair ×3       │    │     └─ generate_dpo_rejection() → teacher model API
│  │ + rejected ×3    │    │
│  └──────────────────┘    │
└────────┬─────────────────┘
         │ list[dict]  (prompt / chosen / rejected / metadata)
         ▼
┌─────────────────┐
│  DatasetFilter  │  process_new_pair()  →  all-MiniLM-L6-v2
│  (prompt dedup) │
└────────┬─────────┘
         │ unique pairs
         ▼
  alignment_dataset.jsonl
```
