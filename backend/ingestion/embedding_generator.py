"""
embedding_generator.py
Stage 8 — Embedding Generation

Model  : BAAI/bge-base-en-v1.5  (GPU if available)
Input  : chunks from SemanticDocumentChunker
Output : chunks with embeddings + full metadata
"""

from __future__ import annotations

import os
# Models are pre-cached locally; force offline so HuggingFace does not try to
# reach the network (which fails and crashes when running offline).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from pathlib import Path
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

_LOCAL_EMBED = Path(__file__).resolve().parent.parent / "model" / "embeddings" / "bge-base-en-v1.5"
EMBED_MODEL = str(_LOCAL_EMBED) if _LOCAL_EMBED.is_dir() else "BAAI/bge-base-en-v1.5"


class EmbeddingGenerator:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Embedding device: %s", self.device)

        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)

        # Batch size — reduce if GPU OOM
        self.batch_size = 32 if self.device == "cuda" else 16

    # ---------------------------------------------------------------- #

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    # ---------------------------------------------------------------- #

    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        # Contextual Retrieval: embed the CONTEXT-enriched text (`embed_text`)
        # when the contextualizer has run; fall back to the raw child text. The
        # stored `text` stays the clean child (what the reranker/LLM see).
        texts = [c.get("embed_text") or c["text"] for c in chunks]
        vectors = self.embed_texts(texts)

        embedded: List[dict] = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append({
                "chunk_id": chunk["chunk_id"],
                "pdf_name": chunk.get("pdf_name", ""),
                "text": chunk["text"],
                "embedding": vector,
                "metadata": {
                    "pdf_name": chunk.get("pdf_name", ""),
                    "pages": chunk.get("pages", []),
                    "heading": chunk.get("heading", ""),
                    "section": chunk.get("section", ""),
                    "figure_ids": chunk.get("figure_ids", []),
                    "image_paths": chunk.get("image_paths", []),
                    "modality": chunk.get("modality", "text"),
                    "parent_id": chunk.get("parent_id", ""),
                    "parent_text": chunk.get("parent_text", ""),
                    # kept for CONTEXTUAL BM25 (retriever indexes this when present)
                    "embed_text": chunk.get("embed_text", ""),
                },
            })
        return embedded

    # ---------------------------------------------------------------- #
    # Pipeline entry points
    # ---------------------------------------------------------------- #

    def process_pdf(
        self, pdf_name: str, chunks: List[dict]
    ) -> List[dict]:
        print(f"\nGenerating embeddings: {pdf_name}  ({len(chunks)} chunks)")
        embedded = self.embed_chunks(chunks)
        print(f"  Done: {len(embedded)} embeddings  (dim={len(embedded[0]['embedding']) if embedded else 0})")
        return embedded

    def process_all_pdfs(
        self, all_chunks: Dict[str, List[dict]]
    ) -> Dict[str, List[dict]]:
        all_embeddings: Dict[str, List[dict]] = {}
        for pdf_name, chunks in all_chunks.items():
            if not chunks:
                logger.warning("No chunks for %s — skipping", pdf_name)
                continue
            all_embeddings[pdf_name] = self.process_pdf(pdf_name, chunks)
        return all_embeddings
