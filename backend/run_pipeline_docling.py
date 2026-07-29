"""
run_pipeline_docling.py
Docling-based ingestion pipeline.

Replaces stages 1-5 (render / layout / OCR / figures / document-build) of
run_pipeline.py with a single Docling pass, then reuses the existing
stages 7-9 unchanged:

    Docling  ->  Semantic Chunking  ->  Embedding  ->  Vector Store

Usage
-----
    # normal run (models are prefetched into ~/.cache/docling/models):
    python run_pipeline_docling.py

    # fully offline run (no network touched):
    DOCLING_OFFLINE=1 python run_pipeline_docling.py

    # single PDF (by file stem):
    python run_pipeline_docling.py "EAI Unit 1 Notes"

NOTE: this overwrites data/documents/<pdf>/document.json and repopulates the
vector store, exactly like run_pipeline.py.  Delete vector_db/ first for a
fully clean rebuild.
"""

from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Point Docling at the LOCAL model dir (backend/model/docling) first, else the
# prefetched ~/.cache/docling/models. Local-first matters: the home cache can be
# a STALE copy with mismatched RapidOCR model versions (a fresh `rapidocr`
# package expects PP-OCRv6 models that an old cache won't have).
_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "docling")
_cache = os.path.join(os.path.expanduser("~"), ".cache", "docling", "models")
if os.path.isdir(_local):
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", _local)
elif os.path.isdir(_cache):
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", _cache)

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def run(pdf_name_filter: str = None) -> None:
    _section("STAGE 1-5 — Docling ingestion (render+layout+OCR+tables+figures)")
    from ingestion.docling_ingestor import DoclingIngestor

    ingestor = DoclingIngestor()
    documents = ingestor.process_all_pdfs(pdf_name_filter)
    if not documents:
        logger.error("No documents produced. Check uploads/pdfs/")
        return
    total_elements = sum(len(v) for v in documents.values())
    print(f"\nPDFs: {len(documents)}   Elements: {total_elements}")

    _section("STAGE 7 — Semantic Chunking")
    from ingestion.semantic_chunking import SemanticDocumentChunker

    chunker = SemanticDocumentChunker()
    all_chunks = chunker.process_all_pdfs(documents)
    total_chunks = sum(len(c) for c in all_chunks.values())
    print(f"\nTotal chunks: {total_chunks}")

    _section("STAGE 7.5 — Context enrichment (deterministic heading path, no LLM)")
    from ingestion.contextualizer import Contextualizer

    contextualizer = Contextualizer()
    all_chunks = contextualizer.process_all_pdfs(all_chunks)

    _section("STAGE 8 — Embedding Generation")
    from ingestion.embedding_generator import EmbeddingGenerator

    embedder = EmbeddingGenerator()
    all_embeddings = embedder.process_all_pdfs(all_chunks)

    _section("STAGE 9 — Vector Store (ChromaDB)")
    from ingestion.vector_store import VectorStore

    store = VectorStore()
    store.process_all_pdfs(all_embeddings)

    _section("PIPELINE COMPLETE")
    print(f"  PDFs processed : {len(documents)}")
    print(f"  Total chunks   : {total_chunks}")
    print(f"  Total in DB    : {store.total_chunks()}")
    print("\nNext: start the API with  python api.py")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
