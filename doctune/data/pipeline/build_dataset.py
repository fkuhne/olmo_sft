"""
build_dataset.py — Phase 2 orchestrator for training data generation.

Scans a directory of PDFs, loads cached extraction chunks (produced by
``extract_dataset.py``), synthesizes SFT and DPO training data using a
teacher model, deduplicates via cosine similarity, and outputs a single
``alignment_dataset.jsonl`` file.

Intermediate results are persisted to ``.cache/<domain>/`` so that
interrupted runs can resume from the last completed chunk.

Usage::

    python build_dataset.py [--model MODEL] [--input-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from doctune.data.extraction.pdf_extractor import DoclingManualExtractor
from doctune.data.pipeline.pipeline_cache import PipelineCache
from doctune.data.pipeline.pipeline_utils import (
    add_common_cli_args,
    discover_pdfs,
    extract_chunks_cached,
    extract_device_context,
    init_extractor_and_cache,
)
from doctune.data.synthesis.deduplicate_dataset import ChunkFilter, DatasetFilter
from doctune.data.synthesis.diversity_selector import DiversitySelector
from doctune.data.synthesis.teacher_model_synthesis import TeacherModelSynthesizer

logger = logging.getLogger(__name__)

# Brief inter-chunk pause to avoid API burst rate limits.
_INTER_CHUNK_SLEEP_S = 1.0


@dataclass
class _BuildStats:  # pylint: disable=too-few-public-methods
    """Accumulates pipeline-wide statistics during the build loop."""

    total_chunks_processed: int = 0
    total_pairs_generated: int = 0
    skipped_chunks: int = 0
    pdf_count: int = 0

    def log_summary(self, unique_pairs: int) -> None:
        """Print the final pipeline summary banner."""
        print("\n============================================================")
        print("PIPELINE COMPLETE.")
        print(f"Total PDF Manuals Processed: {self.pdf_count}")
        print(f"Total Semantic Chunks Analyzed: {self.total_chunks_processed}")
        print(f"Total QA Pairs Generated: {self.total_pairs_generated}")
        print(f"Total Unique QA Pairs Retained: {unique_pairs}")
        if self.skipped_chunks:
            print(f"Chunks Skipped Due to Errors: {self.skipped_chunks}")
        print("============================================================")


class DatasetBuilder:  # pylint: disable=too-few-public-methods
    """Master orchestrator for the data curation pipeline.

    Coordinates PDF extraction (with caching), teacher-model synthesis,
    and semantic deduplication into a single end-to-end workflow.
    Supports persistent caching so that interrupted runs can resume from
    the last completed chunk.

    Args:
        input_dir: Directory containing PDF manuals.
        output_file: Path for the output JSONL dataset.
        model: Teacher model identifier.
        provider: API provider (auto-detected if ``None``).
        domain: Subject-matter domain for prompt context.
        extractor: Pre-initialised Docling extractor.
        cache: Optional pipeline cache (``None`` disables caching).
        diversity_ratio: Fraction of chunks to keep after diversity
            selection (``None`` to disable, default ``0.7``).
        chunk_sim_threshold: Cosine similarity threshold for chunk-level
            deduplication (default ``0.82``).
        pair_sim_threshold: Cosine similarity threshold for prompt-level
            deduplication (default ``0.92``).
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_dir: str,
        output_file: str,
        model: str,
        provider: str | None,
        domain: str,
        extractor: DoclingManualExtractor | None,
        cache: PipelineCache | None,
        diversity_ratio: float | None = 0.7,
        chunk_sim_threshold: float = 0.82,
        pair_sim_threshold: float = 0.92,
    ) -> None:
        self.input_dir = input_dir
        self.output_file = output_file
        self.extractor = extractor
        self.cache = cache

        print("--- INITIALIZING PHASE 2 PIPELINE ---")
        self.synthesizer = TeacherModelSynthesizer(
            domain=domain,
            model=model,
            provider=provider,
        )
        self.filter = DatasetFilter(similarity_threshold=pair_sim_threshold)
        self.chunk_filter = ChunkFilter(similarity_threshold=chunk_sim_threshold)
        self.diversity_selector: DiversitySelector | None = (
            DiversitySelector(diversity_ratio=diversity_ratio)
            if diversity_ratio is not None
            else None
        )

    # ------------------------------------------------------------------
    # Cache helper
    # ------------------------------------------------------------------
    def _cache_synthesis(
        self, pdf_hash: str | None, chunk_index: int, results: list[dict],
    ) -> None:
        """Persist synthesis results to the cache if caching is enabled."""
        if self.cache is not None and pdf_hash is not None:
            self.cache.append_synthesis_result(pdf_hash, chunk_index, results)

    # ------------------------------------------------------------------
    # Private pipeline stages
    # ------------------------------------------------------------------
    def _resume_from_cache(
        self,
        pdf_hash: str | None,
        enriched_chunks: list[str],
        stats: _BuildStats,
    ) -> set[int]:
        """Re-ingest cached synthesis results and return completed chunk indices.

        Args:
            pdf_hash: PDF file hash, or ``None`` when caching is disabled.
            enriched_chunks: Full list of chunks for this PDF.
            stats: Mutable stats accumulator.

        Returns:
            Set of chunk indices already synthesized in a previous run.
        """
        if self.cache is None or pdf_hash is None:
            return set()

        completed_indices = self.cache.get_completed_chunk_indices(pdf_hash)
        if not completed_indices:
            return set()

        cached_results = self.cache.load_all_synthesis_results(pdf_hash)
        for qa_tuple in cached_results:
            self.filter.process_new_pair(qa_tuple)
            stats.total_pairs_generated += 1

        print(
            f"  [RESUMING] {len(completed_indices)}/{len(enriched_chunks)} "
            f"chunks already processed — resuming from next."
        )
        return completed_indices

    def _select_active_chunks(
        self,
        enriched_chunks: list[str],
        completed_indices: set[int],
    ) -> list[tuple[int, str]]:
        """Resolve both filtering gates and return chunks ready for synthesis.

        Applies chunk deduplication and (optionally) diversity selection to
        all not-yet-completed chunks, returning only those that should
        proceed to the teacher-model API.

        When diversity selection is enabled the gate order is:

        1. **Chunk dedup** — eliminates near-duplicate source chunks.
        2. **Diversity selection** — keeps the most semantically varied subset.

        When diversity selection is disabled, only chunk dedup runs.

        Args:
            enriched_chunks: Full list of enriched markdown chunks for this PDF.
            completed_indices: Indices already synthesized in a previous run.

        Returns:
            Ordered ``[(index, chunk), ...]`` pairs ready for synthesis.
        """
        candidates = [
            (j, c) for j, c in enumerate(enriched_chunks)
            if j not in completed_indices
        ]
        if not candidates:
            return []

        if self.diversity_selector is not None:
            # Gate 1 — chunk dedup (pre-filters before diversity selection)
            deduped = [
                (j, c) for j, c in candidates
                if not self.chunk_filter.is_duplicate(c)
            ]
            if not deduped:
                return []

            # Gate 2 — diversity selection on deduplicated candidates
            idxs, texts = zip(*deduped)
            result = self.diversity_selector.select(list(texts))
            print(
                f"  [DiversitySelector] {result.stats['selected_chunks']}/"
                f"{result.stats['total_chunks']} chunks selected"
                + (" [sliding window]" if result.used_sliding_window else "")
            )
            return [(idxs[i], texts[i]) for i in sorted(result.selected_indices)]

        # Diversity disabled — Gate 1 only
        return [
            (j, c) for j, c in candidates
            if not self.chunk_filter.is_duplicate(c)
        ]

    def _synthesize_chunk(
        self,
        j: int,
        chunk: str,
        len_chunks: int,
        pdf_hash: str | None,
        stats: _BuildStats,
    ) -> list[dict]:
        """Call the teacher model for one chunk, caching the outcome immediately.

        Args:
            j: Zero-based chunk index (into the original ``enriched_chunks``).
            chunk: Enriched markdown chunk string.
            len_chunks: Total chunk count (used only for progress display).
            pdf_hash: PDF file hash for cache writes, or ``None`` when
                caching is disabled.
            stats: Mutable stats accumulator.

        Returns:
            List of generated QA-tuple dicts (may be empty on failure or
            when the chunk contains no actionable content).
        """
        print(
            f"  -> Synthesizing chunk {j + 1}/{len_chunks}...",
            end=" ", flush=True,
        )
        try:
            generated_tuples = self.synthesizer.process_chunk(chunk)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Chunk %d/%d failed: %s", j + 1, len_chunks, e)
            print(f"[ERROR: {type(e).__name__} — skipped]")
            stats.skipped_chunks += 1
            self._cache_synthesis(pdf_hash, j, [])
            return []

        self._cache_synthesis(pdf_hash, j, generated_tuples or [])
        return generated_tuples or []

    # ------------------------------------------------------------------
    # Main build loop
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Execute the full data curation pipeline.

        Iterates through all PDFs in ``input_dir``, extracts chunks
        (using the cache when available), synthesizes SFT/DPO pairs,
        deduplicates, and saves the result.  Individual chunk failures
        are logged and skipped without stopping the pipeline.  Caching
        ensures that completed work survives interruptions.
        """
        pdf_files = discover_pdfs(self.input_dir)

        if not pdf_files:
            print(f"CRITICAL: No PDFs found in directory '{self.input_dir}'.")
            return

        stats = _BuildStats(pdf_count=len(pdf_files))
        print(f"Found {len(pdf_files)} manuals to process. Starting pipeline...\n")

        for i, pdf_path in enumerate(pdf_files):
            device_context = extract_device_context(pdf_path)
            print("============================================================")
            print(f"Processing Manual {i + 1}/{len(pdf_files)}: {device_context}")
            print("============================================================")

            try:
                self._process_single_pdf(pdf_path, device_context, stats)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Critical error processing %s: %s", pdf_path, e)
                print(f"\nCRITICAL ERROR processing {pdf_path}: {e}")
                print("Skipping to the next manual to preserve pipeline execution...")

        stats.log_summary(unique_pairs=len(self.filter.accepted_data))
        self.chunk_filter.log_summary()
        self.filter.save_dataset(self.output_file)

    def _process_single_pdf(
        self,
        pdf_path: str,
        device_context: str,
        stats: _BuildStats,
    ) -> None:
        """Extract, synthesize, and deduplicate chunks for one PDF.

        Args:
            pdf_path: Path to the PDF file.
            device_context: Human-readable document label.
            stats: Mutable stats accumulator.
        """
        # Step 1: Extract and chunk
        enriched_chunks = extract_chunks_cached(
            pdf_path, device_context, self.extractor, self.cache,
        )
        if not enriched_chunks:
            print("  No chunks extracted — skipping this manual.")
            return

        # Step 2: Resume from cache (re-ingest results from a prior run)
        pdf_hash: str | None = (
            self.cache.get_pdf_hash(pdf_path) if self.cache is not None else None
        )
        completed_indices = self._resume_from_cache(pdf_hash, enriched_chunks, stats)
        stats.total_chunks_processed += len(completed_indices)

        # Step 3: Resolve active chunks through dedup and diversity gates
        active_chunks = self._select_active_chunks(enriched_chunks, completed_indices)

        # Chunks eliminated by the gates are cached as empty so they are
        # gracefully skipped on resume without re-running the gates.
        active_indices = {j for j, _ in active_chunks}
        for j in range(len(enriched_chunks)):
            if j not in completed_indices and j not in active_indices:
                self._cache_synthesis(pdf_hash, j, [])
                stats.total_chunks_processed += 1

        # Step 4: Synthesize and deduplicate
        for j, chunk in active_chunks:
            stats.total_chunks_processed += 1
            generated_tuples = self._synthesize_chunk(
                j, chunk, len(enriched_chunks), pdf_hash, stats,
            )

            if not generated_tuples:
                print("[Skipped: No actionable data]")
                continue

            kept_count = 0
            for qa_tuple in generated_tuples:
                stats.total_pairs_generated += 1
                if self.filter.process_new_pair(qa_tuple):
                    kept_count += 1

            print(f"[Generated {len(generated_tuples)} | Kept {kept_count}]")
            time.sleep(_INTER_CHUNK_SLEEP_S)


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Build training dataset from PDF manuals"
    )
    parser.add_argument("--model", default="gpt-4o", help="Teacher model ID")
    parser.add_argument("--provider", default=None, help="API provider (auto-detected)")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Use INFO to see dedup audit lines.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write logs to this file in addition to stderr (optional).",
    )
    parser.add_argument(
        "--output", default="alignment_dataset.jsonl", help="Output JSONL file path"
    )
    parser.add_argument(
        "--diversity-ratio", type=float, default=0.7,
        help=(
            "Fraction of chunks to keep after diversity selection (0.0–1.0). "
            "Default 0.7 keeps the 70%% most diverse chunks per document. "
            "Pass --no-diversity to disable."
        ),
    )
    parser.add_argument(
        "--no-diversity", action="store_true",
        help="Disable the diversity selector (send all chunks to synthesis)."
    )
    add_common_cli_args(parser)
    args = parser.parse_args()

    # Configure logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    Path(args.input_dir).mkdir(parents=True, exist_ok=True)

    cli_extractor, cli_cache = init_extractor_and_cache(args, init_extractor=False)

    builder = DatasetBuilder(
        input_dir=args.input_dir,
        output_file=args.output,
        model=args.model,
        provider=args.provider,
        domain=args.domain,
        extractor=cli_extractor,
        cache=cli_cache,
        diversity_ratio=None if args.no_diversity else args.diversity_ratio,
        chunk_sim_threshold=args.chunk_sim_threshold,
        pair_sim_threshold=args.pair_sim_threshold,
    )

    builder.build()
