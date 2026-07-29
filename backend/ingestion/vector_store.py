"""
vector_store.py
Stage 9 — Vector Database (ChromaDB)

Stores embeddings + rich metadata:
  pdf_name, pages, heading, section, figure_ids, image_paths

GPU is NOT used here — ChromaDB is CPU-only.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)

COLLECTION_NAME = "multimodal_rag"


class VectorStore:

    def __init__(self):
        base = Path(__file__).resolve().parent.parent
        self.db_path = base / "vector_db"
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Offline MultiModal RAG"},
        )
        logger.info(
            "ChromaDB ready: %s  (%d chunks stored)",
            self.db_path,
            self.collection.count(),
        )

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #

    @staticmethod
    def _chunk_uid(pdf_name: str, chunk: dict) -> str:
        # Include image paths so multiple figures that share the same anchor
        # text (e.g. captionless figures in one section) stay distinct, while
        # remaining stable across re-ingestion.
        meta = chunk.get("metadata", {})
        image_paths = meta.get("image_paths", chunk.get("image_paths", []))
        key = f"{pdf_name}{chunk['text']}{','.join(image_paths)}"
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def _make_metadata(pdf_name: str, chunk: dict) -> dict:
        meta = chunk.get("metadata", {})
        pages = meta.get("pages", chunk.get("pages", []))
        figure_ids = meta.get("figure_ids", chunk.get("figure_ids", []))
        image_paths = meta.get("image_paths", chunk.get("image_paths", []))
        return {
            "pdf_name": pdf_name,
            "chunk_id": str(chunk.get("chunk_id", "")),
            "pages": ",".join(str(p) for p in pages),
            "heading": str(meta.get("heading", chunk.get("heading", ""))),
            "section": str(meta.get("section", chunk.get("section", ""))),
            "figure_ids": ",".join(str(f) for f in figure_ids),
            "image_paths": ",".join(str(p) for p in image_paths),
            "modality": str(meta.get("modality", chunk.get("modality", "text"))),
            # small-to-big: parent grouping + the larger context to return
            "parent_id": str(meta.get("parent_id", chunk.get("parent_id", ""))),
            "parent_text": str(meta.get("parent_text", chunk.get("parent_text", ""))),
            # contextual retrieval: context-enriched text for contextual BM25
            "embed_text": str(meta.get("embed_text", chunk.get("embed_text", ""))),
        }

    # ---------------------------------------------------------------- #
    # Insert
    # ---------------------------------------------------------------- #

    def store_chunks(self, pdf_name: str, embedded_chunks: List[dict]) -> None:
        ids, docs, embeddings, metadatas = [], [], [], []
        skipped = 0

        seen: set = set()
        for chunk in embedded_chunks:
            uid = self._chunk_uid(pdf_name, chunk)
            # Skip if already in the DB or already queued in this batch
            # (e.g. two captionless figures sharing the same anchor text).
            if uid in seen or self.collection.get(ids=[uid])["ids"]:
                skipped += 1
                continue
            seen.add(uid)
            ids.append(uid)
            docs.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            metadatas.append(self._make_metadata(pdf_name, chunk))

        if not ids:
            logger.info("No new chunks for %s (skipped %d duplicates)", pdf_name, skipped)
            return

        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "Inserted %d chunks for %s (skipped %d duplicates)",
            len(ids), pdf_name, skipped,
        )

    # ---------------------------------------------------------------- #
    # Query
    # ---------------------------------------------------------------- #

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        pdf_filter: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> List[dict]:
        conditions = []
        if pdf_filter:
            conditions.append({"pdf_name": pdf_filter})
        if modality:
            conditions.append({"modality": modality})
        if len(conditions) > 1:
            where = {"$and": conditions}
        elif conditions:
            where = conditions[0]
        else:
            where = None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output: List[dict] = []
        for uid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "id": uid,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })
        return output

    # ---------------------------------------------------------------- #
    # Pipeline entry points
    # ---------------------------------------------------------------- #

    def process_pdf(self, pdf_name: str, embedded_chunks: List[dict]) -> None:
        print(f"\nStoring: {pdf_name}")
        self.store_chunks(pdf_name, embedded_chunks)

    def process_all_pdfs(
        self, all_embeddings: Dict[str, List[dict]]
    ) -> None:
        for pdf_name, embedded_chunks in all_embeddings.items():
            self.process_pdf(pdf_name, embedded_chunks)
        print(f"\nVector DB updated. Total chunks: {self.collection.count()}")

    def total_chunks(self) -> int:
        return self.collection.count()

    def get_all_chunks(self, pdf_filter: Optional[str] = None) -> List[dict]:
        """Return all stored chunks (id, text, metadata) — used to build the BM25 index."""
        if self.collection.count() == 0:
            return []
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if pdf_filter:
            kwargs["where"] = {"pdf_name": pdf_filter}
        result = self.collection.get(**kwargs)
        output: List[dict] = []
        for uid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            output.append({"id": uid, "text": doc, "metadata": meta})
        return output

    def list_pdf_names(self) -> List[str]:
        """Distinct pdf_name values currently in the collection (sorted).

        Source of truth for the document list — reflects what is actually
        queryable, and stays in sync with delete_pdf automatically (unlike
        scanning the data/chunks/ folder, which lingers after a delete).
        """
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        names = {
            m.get("pdf_name", "")
            for m in result["metadatas"]
            if m and m.get("pdf_name")
        }
        return sorted(names)

    def delete_pdf(self, pdf_name: str) -> None:
        """Remove all chunks for a given PDF."""
        results = self.collection.get(where={"pdf_name": pdf_name})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info("Deleted %d chunks for %s", len(results["ids"]), pdf_name)
