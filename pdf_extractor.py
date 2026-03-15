"""
pdf_extractor.py — Layout-aware PDF extraction via IBM Docling.

Parses PDF documents using Docling's vision models (DocLayNet) to
faithfully preserve tables, headings, and reading order. Produces
enriched markdown chunks with source context metadata.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class DoclingManualExtractor: # pylint: disable=too-few-public-methods
    """Extracts and chunks PDF documents using IBM Docling.

    Uses DocLayNet-based ML models to understand the visual structure
    of the PDF before extracting text. Chunks are produced based on the
    document's internal heading hierarchy.
    """

    def __init__(self) -> None:
        """Initialize the Docling converter and hierarchical chunker."""
        # pylint: disable=import-outside-toplevel
        from docling.document_converter import DocumentConverter
        from docling.chunking import HierarchicalChunker

        print("Initializing Docling Document Converter...")
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def process_manual(self, pdf_path: str, device_context: str) -> list[str]:
        """Parse a PDF and return enriched markdown chunks.

        Uses Docling's vision models to identify tables, headers, and reading
        order, then chunks by heading hierarchy and injects source metadata.

        Args:
            pdf_path: Absolute or relative path to the PDF file.
            device_context: Human-readable name of the document source
                (e.g. ``"Product User Guide"``).

        Returns:
            List of markdown-formatted text chunks, each prefixed with a
            source context header. Chunks shorter than 100 characters are
            filtered out. Returns an empty list on failure.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist (logged, not raised).
        """
        # Validate file existence before calling Docling
        if not os.path.exists(pdf_path):
            logger.error("PDF not found: %s", pdf_path)
            print(f"ERROR: PDF not found: {pdf_path}")
            return []

        if not pdf_path.lower().endswith(".pdf"):
            logger.warning("File does not have .pdf extension: %s", pdf_path)

        print(f"\nAnalyzing Structural Layout from: {pdf_path}...")

        try:
            # 1. High-Fidelity Conversion
            conv_result = self.converter.convert(pdf_path)
        except PermissionError:
            logger.error("Permission denied reading PDF: %s", pdf_path)
            print(f"ERROR: Permission denied reading PDF: {pdf_path}")
            return []
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error(
                "PDF parsing failed for %s: %s: %s",
                pdf_path, type(e).__name__, e,
            )
            print(f"ERROR: PDF parsing failed for {pdf_path}: {type(e).__name__}: {e}")
            return []

        # 2. Semantic Chunking
        try:
            chunks = self.chunker.chunk(conv_result.document)
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Chunking failed for %s: %s", pdf_path, e)
            print(f"ERROR: Chunking failed for {pdf_path}: {e}")
            return []

        # 3. Metadata Injection & Filtering
        final_dataset_chunks: list[str] = []
        for chunk in chunks:
            raw_text = chunk.text

            # Filter out tiny, useless chunks (isolated page numbers, logos, etc.)
            if len(raw_text.strip()) < 100:
                continue

            enriched_chunk = (
                f"### [Source Context: {device_context}]\n\n"
                f"{raw_text}\n"
            )
            final_dataset_chunks.append(enriched_chunk)

        print(f"Extraction complete. Yielded {len(final_dataset_chunks)} high-fidelity chunks.")
        return final_dataset_chunks


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    extractor = DoclingManualExtractor()

    TARGET_PDF = "example_manual.pdf"
    SOURCE_NAME = "Product User Guide"

    if os.path.exists(TARGET_PDF):
        enriched_chunks = extractor.process_manual(TARGET_PDF, SOURCE_NAME)

        if enriched_chunks:
            print("\n--- PREVIEW OF DOCLING CHUNK 1 ---")
            print(enriched_chunks[0])
            print("----------------------------------")
    else:
        print(f"Awaiting PDF file: {TARGET_PDF}")
