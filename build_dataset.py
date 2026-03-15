"""
build_dataset.py — Phase 2 orchestrator for training data generation.

Scans a directory of PDFs, extracts content via Docling, synthesizes SFT
and DPO training data using a teacher model, deduplicates via cosine
similarity, and outputs a single ``alignment_dataset.jsonl`` file.

Usage:
    python build_dataset.py [--model MODEL] [--input-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time

from pdf_extractor import DoclingManualExtractor
from teacher_model_synthesis import TeacherModelSynthesizer
from deduplicate_dataset import DatasetFilter

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Master orchestrator for the data curation pipeline.

    Coordinates PDF extraction, teacher-model synthesis, and semantic
    deduplication into a single end-to-end workflow.

    Args:
        input_dir: Directory containing PDF manuals.
        output_file: Path for the output JSONL dataset.
        model: Teacher model identifier.
        provider: API provider (auto-detected if ``None``).
        domain: Subject-matter domain for prompt context.
    """

    def __init__(
        self,
        input_dir: str = "./manuals",
        output_file: str = "alignment_dataset.jsonl",
        model: str = "gpt-4o",
        provider: str | None = None,
        domain: str = "technical documentation",
    ) -> None:
        self.input_dir = input_dir
        self.output_file = output_file

        print("--- INITIALIZING PHASE 2 PIPELINE ---")
        self.extractor = DoclingManualExtractor()
        self.synthesizer = TeacherModelSynthesizer(
            domain=domain,
            model=model,
            provider=provider,
        )
        self.filter = DatasetFilter(similarity_threshold=0.85)

    def extract_device_context(self, filename: str) -> str:
        """Convert a PDF filename into a clean source context string.

        Args:
            filename: Path to the PDF file (e.g. ``"product_user_guide.pdf"``).

        Returns:
            A title-cased context label (e.g. ``"Product User Guide"``).
        """
        base_name = os.path.basename(filename).replace(".pdf", "")
        clean_name = base_name.replace("_", " ").replace("-", " ").title()
        clean_name = clean_name.replace(" Manual", "").replace(" User Guide", "")
        return clean_name

    def build(self) -> None:
        """Execute the full data curation pipeline.

        Iterates through all PDFs in ``input_dir``, extracts chunks,
        synthesizes SFT/DPO pairs, deduplicates, and saves the result.
        Individual chunk failures are logged and skipped without stopping
        the pipeline.
        """
        pdf_files = glob.glob(os.path.join(self.input_dir, "*.pdf"))

        if not pdf_files:
            print(f"CRITICAL: No PDFs found in directory '{self.input_dir}'.")
            return

        print(f"Found {len(pdf_files)} manuals to process. Starting pipeline...\n")

        total_chunks_processed = 0
        total_pairs_generated = 0
        skipped_chunks = 0

        for i, pdf_path in enumerate(pdf_files):
            device_context = self.extract_device_context(pdf_path)
            print(f"============================================================")
            print(f"Processing Manual {i + 1}/{len(pdf_files)}: {device_context}")
            print(f"============================================================")

            try:
                # Step 1: Extract and Chunk (Docling)
                enriched_chunks = self.extractor.process_manual(pdf_path, device_context)

                # Step 2: Iterate through every chunk
                for j, chunk in enumerate(enriched_chunks):
                    total_chunks_processed += 1
                    print(f"  -> Synthesizing chunk {j + 1}/{len(enriched_chunks)}...", end=" ", flush=True)

                    try:
                        generated_tuples = self.synthesizer.process_chunk(chunk)
                    except Exception as e:
                        logger.warning(
                            "Chunk %d/%d failed in %s: %s",
                            j + 1, len(enriched_chunks), device_context, e,
                        )
                        print(f"[ERROR: {type(e).__name__} — skipped]")
                        skipped_chunks += 1
                        continue

                    if not generated_tuples:
                        print("[Skipped: No actionable data]")
                        continue

                    # Step 3: Filter and Deduplicate
                    kept_count = 0
                    for qa_tuple in generated_tuples:
                        total_pairs_generated += 1
                        if self.filter.process_new_pair(qa_tuple):
                            kept_count += 1

                    print(f"[Generated {len(generated_tuples)} | Kept {kept_count}]")

                    # Brief pause to avoid API rate limits
                    time.sleep(1)

            except Exception as e:
                logger.error("Critical error processing %s: %s", pdf_path, e)
                print(f"\nCRITICAL ERROR processing {pdf_path}: {e}")
                print("Skipping to the next manual to preserve pipeline execution...")

        # Final Summary
        print("\n============================================================")
        print("PIPELINE COMPLETE.")
        print(f"Total PDF Manuals Processed: {len(pdf_files)}")
        print(f"Total Semantic Chunks Analyzed: {total_chunks_processed}")
        print(f"Total QA Pairs Generated: {total_pairs_generated}")
        print(f"Total Unique QA Pairs Retained: {len(self.filter.accepted_data)}")
        if skipped_chunks:
            print(f"Chunks Skipped Due to Errors: {skipped_chunks}")
        print("============================================================")

        self.filter.save_dataset(self.output_file)


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Build training dataset from PDF manuals")
    parser.add_argument("--model", default="gpt-4o", help="Teacher model ID")
    parser.add_argument("--provider", default=None, help="API provider (auto-detected)")
    parser.add_argument("--domain", default="technical documentation", help="Subject-matter domain")
    parser.add_argument("--input-dir", default="./manuals", help="Directory containing PDF manuals")
    parser.add_argument("--output", default="alignment_dataset.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    os.makedirs(args.input_dir, exist_ok=True)

    builder = DatasetBuilder(
        input_dir=args.input_dir,
        output_file=args.output,
        model=args.model,
        provider=args.provider,
        domain=args.domain,
    )

    builder.build()
