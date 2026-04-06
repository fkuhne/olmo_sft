"""
train_dpo.py — Direct Preference Optimization (DPO).

Runs Phase 4 of the training pipeline: aligns the SFT-trained model using
preference tuples (chosen vs. rejected) to penalize hallucinations.

Supports hyperparameter sweeping over beta and learning rate values.

Usage:
    python train_dpo.py --model-id <huggingface-model-id>

Requirements:
    pip install -e ".[training]"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

from peft import PeftModel
from trl import DPOTrainer

from doctune.training.training_utils import (
    add_common_train_args,
    build_training_args,
    load_datasets,
)
from doctune.utils.model_utils import (
    clear_gpu_cache,
    derive_run_name,
    load_base_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class _SweepResult:
    """Holds the outcome of a single DPO sweep run."""
    run_name: str
    beta: float
    lr: float
    eval_loss: float
    rewards_chosen: float    # mean reward margin on chosen responses
    rewards_rejected: float  # mean reward margin on rejected responses
    reward_margin: float     # chosen - rejected (higher = better separation)
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.run_name} | β={self.beta} lr={self.lr:.0e} | "
            f"eval_loss={self.eval_loss:.4f} | "
            f"reward_margin={self.reward_margin:.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DPO training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Doctune DPO Training")
    add_common_train_args(parser)
    parser.add_argument(
        "--sft-adapter", type=str, default="./doctune-sft",
        help="Path to SFT LoRA adapter directory (default: ./doctune-sft, "
             "produced by train_sft.py with its default --output)",
    )
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5],
        help="DPO beta values to sweep",
    )
    parser.add_argument(
        "--lrs", type=float, nargs="+", default=[5e-6, 1e-6],
        help="Learning rates to sweep",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device train batch size for DPO (default: 2; lower than SFT due to ref model memory)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=16,
        help="Gradient accumulation steps for DPO (default: 16)",
    )
    return parser.parse_args()


def _extract_sweep_result(
    trainer: DPOTrainer,
    run_name: str,
    beta: float,
    lr: float,
) -> _SweepResult:
    """Extract final eval metrics from a completed DPOTrainer run.

    DPOTrainer exposes reward margins as ``rewards/chosen`` and
    ``rewards/rejected`` in its log history. These are the primary
    signal for sweep comparison: a higher margin means the model more
    clearly separates preferred from dis-preferred responses.

    Args:
        trainer: The completed ``DPOTrainer`` instance (before deletion).
        run_name: Human-readable run identifier for logging.
        beta: The β value used in this run.
        lr: The learning rate used in this run.

    Returns:
        Populated ``_SweepResult`` dataclass.
    """
    eval_logs = [
        entry for entry in trainer.state.log_history
        if "eval_loss" in entry
    ]
    last_eval = eval_logs[-1] if eval_logs else {}

    eval_loss        = last_eval.get("eval_loss", float("inf"))
    rewards_chosen   = last_eval.get("eval_rewards/chosen", 0.0)
    rewards_rejected = last_eval.get("eval_rewards/rejected", 0.0)
    reward_margin    = rewards_chosen - rewards_rejected

    try:
        import mlflow  # noqa: PLC0415
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params({"beta": beta, "lr": lr})
            mlflow.log_metrics({
                "eval_loss":        eval_loss,
                "rewards_chosen":   rewards_chosen,
                "rewards_rejected": rewards_rejected,
                "reward_margin":    reward_margin,
            })
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow logging failed for %s: %s", run_name, e)

    return _SweepResult(
        run_name=run_name,
        beta=beta,
        lr=lr,
        eval_loss=eval_loss,
        rewards_chosen=rewards_chosen,
        rewards_rejected=rewards_rejected,
        reward_margin=reward_margin,
        extra=last_eval,
    )


def _log_sweep_summary(results: list[_SweepResult]) -> None:
    """Print a ranked sweep summary and tag the best run in MLflow.

    Ranks runs by ``reward_margin`` descending (primary) and
    ``eval_loss`` ascending (tiebreak).

    Args:
        results: All ``_SweepResult`` instances from the sweep.
    """
    if not results:
        return

    ranked = sorted(results, key=lambda r: (-r.reward_margin, r.eval_loss))
    best = ranked[0]

    print("\n" + "=" * 60)
    print("DPO SWEEP COMPLETE — RESULTS RANKED BY REWARD MARGIN")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
        marker = "  <- BEST" if i == 1 else ""
        print(f"  {i}. {r}{marker}")
    print("=" * 60)
    print(f"  Best adapter saved to: ./{best.run_name}")
    print("=" * 60 + "\n")

    logger.info("Best DPO run: %s", best)

    try:
        import mlflow  # noqa: PLC0415
        mlflow.set_tags({
            "dpo_best_run":    best.run_name,
            "dpo_best_beta":   str(best.beta),
            "dpo_best_lr":     str(best.lr),
            "dpo_best_margin": f"{best.reward_margin:.4f}",
        })
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow summary tagging failed: %s", e)


def main() -> None:
    """Entry point for DPO training with hyperparameter sweep."""
    args = parse_args()

    # 1. Load Tokenizer & Base Model
    tokenizer = load_tokenizer(args.model_id, padding_side="right")
    base_model = load_base_model(args.model_id, tokenizer)

    # 2. Load SFT Adapters
    logger.info("Loading SFT Adapters from %s...", args.sft_adapter)
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)

    # 3. Load and Format Datasets
    dataset, eval_dataset = load_datasets(args.dataset, args.eval_dataset)

    def format_dpo_dataset(example: dict) -> dict:
        """Format a single row for DPOTrainer (prompt, chosen, rejected strings)."""
        prompt_messages = [{"role": "user", "content": example["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        return {
            "prompt": formatted_prompt,
            "chosen": f"{example['chosen']}{tokenizer.eos_token}",
            "rejected": f"{example['rejected']}{tokenizer.eos_token}",
        }

    dpo_dataset = dataset.map(format_dpo_dataset)
    dpo_eval_dataset = eval_dataset.map(format_dpo_dataset)

    # 4. Hyperparameter Sweep
    sweep_results: list[_SweepResult] = []
    base_run_name = derive_run_name(args.model_id, "dpo")

    for beta_val in args.betas:
        for lr_val in args.lrs:
            run_name = f"{base_run_name}-beta{beta_val}-lr{lr_val}"
            output_dir = f"./{run_name}"
            logger.info("Starting DPO Sweep: Beta=%s, LR=%s", beta_val, lr_val)

            training_args = build_training_args(
                output_dir=output_dir,
                run_name=run_name,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                lr=lr_val,
                remove_unused_columns=False,
            )

            # pylint: disable=unexpected-keyword-arg
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                train_dataset=dpo_dataset,
                eval_dataset=dpo_eval_dataset,
                tokenizer=tokenizer,
                beta=beta_val,
                max_length=args.max_seq_length,
                max_prompt_length=args.max_prompt_length,
            )

            logger.info("Initiating DPO Training for %s...", run_name)
            trainer.train()
            trainer.save_model(output_dir)
            logger.info("Alignment complete for %s. Saved to %s", run_name, output_dir)

            result = _extract_sweep_result(trainer, run_name, beta_val, lr_val)
            sweep_results.append(result)
            logger.info("Sweep run complete: %s", result)
            print(f"  [SWEEP] {result}")

            # Free trainer memory between sweeps
            del trainer
            clear_gpu_cache()

    _log_sweep_summary(sweep_results)

    # Final cleanup
    del model, base_model
    clear_gpu_cache()


if __name__ == "__main__":
    main()
