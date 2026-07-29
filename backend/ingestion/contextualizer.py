"""
contextualizer.py
Stage 7.5 — deterministic context enrichment (NO LLM).

For each chunk we build an ``embed_text`` = "<doc title> — <section> > <heading>"
prepended to the chunk's own text. ONLY ``embed_text`` is embedded (and indexed
for BM25); the clean ``text`` (shown to the reranker) and ``parent_text``
(returned to the LLM) are left untouched — so the added context improves MATCHING
only, never what the user or the LLM sees.

This lets a terse row like "950-1,000 F" match a conceptual query ("reactor
temperature range") by carrying its heading path. Pure string work — **no LLM is
called during ingestion.**

Pipeline position:  Semantic Chunking -> [Contextualizer] -> Embedding.
"""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class Contextualizer:

    @staticmethod
    def _deterministic(pdf_name: str, chunk: dict) -> str:
        """Context prefix: document title + section/heading path."""
        title = pdf_name.replace("_", " ").strip()
        parts = [title]
        path = " > ".join(dict.fromkeys(
            p for p in (chunk.get("section", ""), chunk.get("heading", ""))
            if p and p.strip() and p != title
        ))
        if path:
            parts.append(path)
        return " — ".join(parts)

    # ---------------------------------------------------------------- #

    def contextualize(self, pdf_name: str, chunks: List[dict]) -> List[dict]:
        for c in chunks:
            ctx = self._deterministic(pdf_name, c)
            c["context"] = ctx
            body = c.get("text", "")
            c["embed_text"] = f"{ctx}\n{body}".strip() if ctx else body
        logger.info("Contextualized %d chunks for %s (deterministic, no LLM)",
                    len(chunks), pdf_name)
        return chunks

    def process_all_pdfs(
        self, all_chunks: Dict[str, List[dict]]
    ) -> Dict[str, List[dict]]:
        return {pdf: self.contextualize(pdf, ch) for pdf, ch in all_chunks.items()}
