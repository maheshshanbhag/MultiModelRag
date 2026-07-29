"""
retriever.py
Stage 10 — Hybrid Retrieval (Vector + BM25)

Two-stage retrieval for best recall:
  1. Dense vector search — BAAI/bge-base-en-v1.5 with query instruction prefix
  2. Sparse BM25 keyword search — catches technical terms/acronyms that
     embeddings can miss
  3. Reciprocal Rank Fusion merges both ranked lists before reranking.

Retrieves top-20 candidates.  Stage 11 (reranker) narrows to top-5.
"""

from __future__ import annotations

import os
# Force offline so query-time encoding never reaches the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)

_LOCAL_EMBED = Path(__file__).resolve().parent.parent / "model" / "embeddings" / "bge-base-en-v1.5"
EMBED_MODEL = str(_LOCAL_EMBED) if _LOCAL_EMBED.is_dir() else "BAAI/bge-base-en-v1.5"
DEFAULT_TOP_K = 20

# BGE-v1.5 recommendation: prepend instruction to retrieval queries only.
# Documents (at index time) do NOT need this prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Retriever:

    def __init__(self):
        # Default CPU so the 4GB GPU stays free for the LLM (see reranker.py).
        # Encoding one query on CPU is ~50-100ms. Override with
        # RAG_ENCODER_DEVICE=cuda|auto on a larger GPU.
        _dev = os.environ.get("RAG_ENCODER_DEVICE", "cpu").lower()
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
        )
        logger.info("Retriever device: %s", self.device)

        self.model = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.store = VectorStore()

        # BM25 index — built lazily on first query
        self._bm25 = None
        self._bm25_chunks: List[dict] = []

        # Query spell-normalizer (corpus-vocabulary; built lazily). Corrects
        # typos so a misspelled query still matches the docs. RAG_SPELLFIX=0 off.
        self._normalizer = None
        self._spellfix = os.environ.get("RAG_SPELLFIX", "1") != "0"

    # ---------------------------------------------------------------- #

    def normalize_query(self, query: str) -> str:
        """Public: corpus spell-normalized query (used by the reranker path too)."""
        return self._normalize(query)

    def _normalize(self, query: str) -> str:
        """Spell-correct the query against the corpus vocabulary (no LLM)."""
        if not self._spellfix:
            return query
        if self._normalizer is None:
            self._ensure_bm25()                       # loads _bm25_chunks once
            chunks = self._bm25_chunks or self.store.get_all_chunks()
            if chunks:
                from ingestion.query_normalizer import QueryNormalizer
                self._normalizer = QueryNormalizer.from_texts(
                    (c["metadata"].get("embed_text") if c.get("metadata") else None)
                    or c["text"] for c in chunks
                )
        if self._normalizer is None:
            return query
        fixed = self._normalizer.normalize(query)
        if fixed != query:
            logger.info("Query normalized: %r -> %r", query, fixed)
        return fixed

    # ---------------------------------------------------------------- #
    # Query encoding
    # ---------------------------------------------------------------- #

    def encode_query(self, query: str) -> List[float]:
        """Encode query with BGE instruction prefix for retrieval."""
        vec = self.model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    # ---------------------------------------------------------------- #
    # BM25 index
    # ---------------------------------------------------------------- #

    def _ensure_bm25(self) -> None:
        """Lazily build BM25 index from all stored chunks."""
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 not installed — BM25 disabled.  "
                "Run: pip install rank-bm25"
            )
            return

        self._bm25_chunks = self.store.get_all_chunks()
        if not self._bm25_chunks:
            logger.info("No chunks in DB yet; BM25 index will be empty.")
            return

        # Contextual BM25: index the context-enriched text when present (matches
        # the contextual embeddings), else the raw chunk text.
        tokenized = [
            (c["metadata"].get("embed_text") or c["text"]).lower().split()
            for c in self._bm25_chunks
        ]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d documents", len(self._bm25_chunks))

    def invalidate_bm25(self) -> None:
        """Call after inserting new chunks so the index is rebuilt on next query."""
        self._bm25 = None
        self._bm25_chunks = []
        self._normalizer = None

    def _retrieve_bm25(
        self,
        query: str,
        top_k: int,
        pdf_filter: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> List[dict]:
        self._ensure_bm25()
        if self._bm25 is None or not self._bm25_chunks:
            return []

        scores: np.ndarray = self._bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[::-1][:top_k * 2]  # over-fetch to allow filter

        results: List[dict] = []
        for idx in top_indices:
            score = float(scores[int(idx)])
            if score <= 0.0:
                continue
            chunk = self._bm25_chunks[int(idx)]
            if pdf_filter and chunk["metadata"].get("pdf_name") != pdf_filter:
                continue
            if modality and chunk["metadata"].get("modality", "text") != modality:
                continue
            results.append({
                "id":         chunk["id"],
                "text":       chunk["text"],
                "metadata":   chunk["metadata"],
                "bm25_score": score,
                "distance":   0.0,
            })
            if len(results) >= top_k:
                break

        return results

    # ---------------------------------------------------------------- #
    # Reciprocal Rank Fusion
    # ---------------------------------------------------------------- #

    @staticmethod
    def _reciprocal_rank_fusion(
        results_a: List[dict],
        results_b: List[dict],
        k: int = 60,
    ) -> List[dict]:
        """
        Merge two independently ranked result lists.
        Score(d) = Σ 1/(k + rank_i + 1) for each list where d appears.
        """
        rrf_scores: Dict[str, float] = {}
        all_chunks: Dict[str, dict] = {}

        for rank, r in enumerate(results_a):
            doc_id = r["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            all_chunks[doc_id] = r

        for rank, r in enumerate(results_b):
            doc_id = r["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in all_chunks:
                all_chunks[doc_id] = r

        ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [all_chunks[doc_id] for doc_id in ranked_ids]

    # ---------------------------------------------------------------- #
    # Main retrieval
    # ---------------------------------------------------------------- #

    # Words that signal the user explicitly wants a figure/diagram back.
    _IMAGE_INTENT = {
        "image", "images", "figure", "figures", "diagram", "diagrams",
        "picture", "pictures", "illustration", "illustrations", "chart",
        "charts", "graph", "graphs", "photo", "screenshot", "visual", "show",
    }

    @classmethod
    def wants_image(cls, query: str) -> bool:
        """True if the query is explicitly asking for a visual."""
        tokens = {t.strip(".,?!").lower() for t in query.split()}
        return bool(tokens & cls._IMAGE_INTENT)

    @staticmethod
    def _expand_to_parents(results: List[dict]) -> List[dict]:
        """Small-to-big: collapse matched CHILD chunks to their PARENT section.

        Keeps the best-ranked child per parent and swaps in the larger
        parent_text so the LLM/reranker gets full context. Self-parented chunks
        (figures, tables, ungrouped) pass through unchanged.
        """
        seen: set = set()
        expanded: List[dict] = []
        for r in results:
            meta = r.get("metadata", {})
            pid = meta.get("parent_id") or r.get("id")
            if pid in seen:
                continue
            seen.add(pid)
            ptext = meta.get("parent_text") or r["text"]
            nr = dict(r)
            nr["child_text"] = r["text"]   # the precise unit that matched
            nr["text"] = ptext             # the full parent context returned
            expanded.append(nr)
        return expanded

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        pdf_filter: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> List[dict]:
        """
        Hybrid retrieval: vector search + BM25, fused via RRF.
        Returns up to top_k chunks each with: id, text, metadata, distance.

        modality : "image" or "text" to restrict the dense search to one
                   modality; None (default) searches everything.
        """
        # 0 — Spell-normalize the query against the corpus (fixes typos so both
        #     the dense encoder and BM25 see the correctly-spelled domain terms).
        query = self._normalize(query)

        # 1 — Dense vector search
        embedding = self.encode_query(query)
        vector_results = self.store.retrieve(
            query_embedding=embedding,
            top_k=top_k,
            pdf_filter=pdf_filter,
            modality=modality,
        )

        # 2 — BM25 keyword search
        bm25_results = self._retrieve_bm25(
            query, top_k=top_k, pdf_filter=pdf_filter, modality=modality,
        )

        if not bm25_results:
            # BM25 unavailable or empty DB — fall back to vector only
            final = self._expand_to_parents(vector_results)[:top_k]
            logger.info(
                "Retrieved %d candidates (vector only) for: %r",
                len(final), query[:60],
            )
            return final

        # 3 — Fuse via Reciprocal Rank Fusion, then expand children -> parents
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results)
        final = self._expand_to_parents(merged)[:top_k]

        logger.info(
            "Retrieved %d hybrid candidates (vector=%d, bm25=%d) for: %r",
            len(final), len(vector_results), len(bm25_results), query[:60],
        )
        return final

    def retrieve_smart(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        pdf_filter: Optional[str] = None,
        n_images: int = 3,
    ) -> List[dict]:
        """Answer-first retrieval that ADDS images instead of replacing text.

        Always retrieves across all modalities (so the textual answer is
        present); if the query also asks for a visual ("show image / diagram"),
        the top image chunks are appended so a figure is guaranteed — WITHOUT
        dropping the explanation. Fixes "tell about X and show image" returning
        images only and no answer.
        """
        results = self.retrieve(query, top_k=top_k, pdf_filter=pdf_filter, modality=None)
        if not self.wants_image(query):
            return results
        images = self.retrieve(
            query, top_k=max(n_images * 2, 6), pdf_filter=pdf_filter, modality="image",
        )
        seen = {r["id"] for r in results}
        added = 0
        for im in images:
            if im["id"] in seen:
                continue
            results.append(im)
            seen.add(im["id"])
            added += 1
            if added >= n_images:
                break
        return results

    def total_chunks(self) -> int:
        return self.store.total_chunks()

    # ---------------------------------------------------------------- #
    # Debug helper
    # ---------------------------------------------------------------- #

    def print_results(self, results: List[dict]) -> None:
        if not results:
            print("No results found.")
            return
        print(f"\n{'=' * 80}")
        print(f"Retrieved {len(results)} chunks")
        print("=" * 80)
        for i, r in enumerate(results, 1):
            bm25 = f"  BM25={r['bm25_score']:.2f}" if "bm25_score" in r else ""
            print(f"\nRank {i}  |  Distance: {r['distance']:.4f}{bm25}")
            print(f"  PDF    : {r['metadata'].get('pdf_name', '')}")
            print(f"  Pages  : {r['metadata'].get('pages', '')}")
            print(f"  Heading: {r['metadata'].get('heading', '')[:80]}")
            print(f"  Text   : {r['text'][:200]}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    retriever = Retriever()
    print(f"Total chunks: {retriever.total_chunks()}")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ")
    results = retriever.retrieve(query)
    retriever.print_results(results)
