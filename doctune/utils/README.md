# `doctune.utils`

This folder contains shared, cross-cutting utilities utilized across the `doctune` pipeline—spanning from synthetic data generation (`data/synthesis`) down to evaluation (`eval`) and deployment (`deploy`). 

It centralizes logic around model loading, API provider wrapping, hardware management, and cost estimation to prevent duplication and drift across different pipeline stages.

---

## 📄 `model_utils.py`

Shared utilities for loading, configuring, and managing HuggingFace causal language models locally.

### Methods
* **`detect_lora_target_modules(model: nn.Module) -> list[str]`**
  Auto-detects all `nn.Linear` layers within a given model to automatically determine the optimal target modules for LoRA injection during fine-tuning.
* **`load_tokenizer(model_id: str, padding_side: str = "right") -> AutoTokenizer`**
  Loads a HuggingFace tokenizer and ensures it has a `<|pad|>` token.
* **`load_base_model(model_id: str, tokenizer: AutoTokenizer, ...) -> AutoModelForCausalLM`**
  Loads a HuggingFace causal LM, auto-detects if Flash Attention 2 is available, and resizes embeddings strictly matching the tokenizer's vocabulary.
* **`format_prompt_for_eval(tokenizer: AutoTokenizer, prompt_text: str) -> str`**
  Safely applies the model's native chat template (via `apply_chat_template`) to format simple user text for evaluation endpoints. Falls back to generic formatting.
* **`derive_run_name(model_id: str, stage: str) -> str`**
  Cleans and normalizes a HuggingFace model path into a concise slug suitable for MLflow run tracking (e.g., `meta-llama/Llama-3.1-8B` → `llama-3.1-8b-sft-v1`).
* **`clear_gpu_cache() -> None`**
  Safely clears CUDA memory buffers to prevent OOM spikes, mostly called after finishing distinct phases like DPO tuning or merging.

---

## 📄 `provider_utils.py`

High-level wrappers and client detection logic for external API providers—specifically OpenAI, Anthropic, and local endpoints via Ollama. 

### Methods
* **`detect_provider(model: str) -> str`**
  Automatically infers the appropriate backend provider (`"openai"`, `"anthropic"`, or `"ollama"`) directly from the model string.
* **`check_provider_separation(model_a: str, model_b: str) -> tuple[bool, str, str]`**
  An evaluation contamination guard. Checks if two specified models belong to different provider families, which natively prevents identical formatting/bias leakage between the training generator and the eval generator.
* **`get_alternative_models(provider: str) -> list[str]`**
  Returns suggested fallback/alternative models strictly out-of-family from the queried `provider`. 
* **`build_client(provider: str, api_key: str | None = None, base_url: str | None = None) -> Any`**
  Instantiates the natively corresponding Python SDK (`openai.OpenAI` or `anthropic.Anthropic`). Handles environmental fallbacks dynamically.
* **`@retry_on_rate_limit(max_retries: int = 3, base_delay: float = 2.0)`**
  A robust exponential backoff decorator that natively catches Rate Limits (HTTP 429) across both OpenAI and Anthropic SDKs (as well as raw HTTP requests).

---

## 📄 `pricing.py`

Centralized cost tracking tables and cost-estimation formulas for synthetic data creation pipelines. 

### Classes
* **`ModelPricing(BaseModel)`**
  A Pydantic schema representing input/output cost configuration for a given LLM tier (measured per 1M tokens).

### Globals
* **`MODEL_PRICING_PER_1M: dict[str, ModelPricing]`**
  The master lookup table of costs. Supports prefix-matching fallback resolution (e.g. `gpt-4o` effectively matches any downstream specific variants if missing).

### Methods
* **`compute_model_usage_cost(model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float`**
  Calculates the exact total USD cost of a resolved API hit based on token volumes, accounting separately for base inputs, cached inputs, and standard outputs.
* **`estimate_batch_cost(model: str, count: int, input_tokens_per_item: int, output_tokens_per_item: int) -> float`**
  Proactively estimates the USD price of a large parallel job *prior* to submission. Heavily utilized in `generate_golden_eval.py` to prevent unintentional high-cost expenditures.
