# doctune.utils — Shared utilities (model loading, provider detection, etc.).

from doctune.utils.model_utils import (
    clear_gpu_cache,
    derive_run_name,
    detect_lora_target_modules,
    format_prompt_for_eval,
    load_base_model,
    load_tokenizer,
)
from doctune.utils.provider_utils import (
    PROVIDER_ALTERNATIVES,
    build_client,
    check_provider_separation,
    detect_provider,
    get_alternative_models,
    retry_on_rate_limit,
)
from doctune.utils.pricing import (
    MODEL_PRICING_PER_1M,
    compute_model_usage_cost,
    estimate_batch_cost,
)

__all__ = [
    # model_utils
    "clear_gpu_cache",
    "derive_run_name",
    "detect_lora_target_modules",
    "format_prompt_for_eval",
    "load_base_model",
    "load_tokenizer",
    # provider_utils
    "PROVIDER_ALTERNATIVES",
    "build_client",
    "check_provider_separation",
    "detect_provider",
    "get_alternative_models",
    "retry_on_rate_limit",
    # pricing
    "MODEL_PRICING_PER_1M",
    "compute_model_usage_cost",
    "estimate_batch_cost",
]
