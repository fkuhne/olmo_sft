# Model Card: OLMo 2 1B — Domain-Adapted QA

## Model Details

| Field | Value |
|---|---|
| **Model Name** | OLMo 2 1B Domain QA |
| **Base Model** | [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) |
| **Architecture** | Transformer decoder-only (1B parameters) |
| **Fine-Tuning Method** | LoRA (r=16, α=32) → SFT + DPO |
| **Status** | Pre-training blueprint (not yet executed) |
| **License** | Apache 2.0 |

## Intended Use

This model is designed for **closed-domain question answering** over technical documentation extracted from PDF manuals. After fine-tuning:

- **Primary Use:** Answering user questions grounded in a specific PDF corpus
- **Secondary Use:** Demonstrating SFT + DPO alignment methodology on a small (1B) LLM
- **Out-of-Scope:** General knowledge QA, code generation, creative writing, or any task outside the trained domain

## Training Methodology

### Phase 1: Data Synthesis
- PDF documents are parsed via IBM Docling (layout-aware extraction)
- GPT-4o Teacher Model generates SFT question-answer pairs and DPO preference tuples
- Cosine similarity deduplication (threshold > 0.85) enforces dataset diversity

### Phase 2: SFT (Supervised Fine-Tuning)
- LoRA adapters targeting all 7 linear projections + `lm_head` + `embed_tokens`
- 3 epochs, cosine LR schedule (2e-4), BF16 precision
- Dedicated `<|pad|>` token (not reusing EOS)

### Phase 3: DPO (Direct Preference Optimization)
- Aligns the SFT model using chosen vs. rejected response pairs
- Beta sweep: [0.1, 0.25], LR sweep: [5e-6, 1e-6]
- 1 epoch per configuration

### Phase 4: Weight Merge + Deployment
- LoRA adapters merged into base model for standalone inference
- Served via vLLM with PagedAttention

## Limitations

- **Domain-locked:** The model will refuse or produce low-quality answers for out-of-domain queries
- **Synthetic data dependency:** All training data is GPT-4o-generated; no human-curated ground truth exists yet
- **Scale constraints:** 1B parameters limit complex multi-step reasoning compared to larger models
- **Language:** English only

## Evaluation

| Metric | Target |
|---|---|
| In-domain accuracy | Assessed via golden eval set (100 synthetic scenarios) |
| Out-of-domain refusal rate | > 90% |
| Baseline comparison | Planned (base model vs. fine-tuned) |

## Ethical Considerations

- The model should **not** be used as a sole source of truth for safety-critical decisions
- Synthetic training data may contain subtle biases inherited from GPT-4o
- Users should validate model outputs against original source documentation
- The DPO alignment enforces boundary behavior but cannot guarantee zero hallucination

## Citation

```bibtex
@misc{olmo2-domain-qa,
  title={OLMo 2 1B Domain-Adapted QA Pipeline},
  year={2026},
  url={https://github.com/your-username/olmo}
}
```
