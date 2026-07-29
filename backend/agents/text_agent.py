"""
agents/text_agent.py
Text Retrieval Agent

Single responsibility: retrieve and rerank text chunks.

Pipeline:
  Query
    → Encode (BGE)
    → ChromaDB top-20
    → Cross-encoder rerank
    → top-5 chunks
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from ingestion.retriever import Retriever
from ingestion.reranker import Reranker
from agents.evaluator_agent import EvaluatorAgent
from agents.query_rewrite_agent import QueryRewriteAgent
# Grounding primitives live in agents/grounding.py and are re-exported here so
# existing importers (e.g. evaluation scripts) keep working unchanged.
from agents.grounding import GROUNDING_MIN_SCORE, GROUNDING_MIN_DENSE, _dense_sim

logger = logging.getLogger(__name__)

# Grounding gate (multi-signal): a result is trusted only when the BEST reranked
# chunk is confident on EITHER the cross-encoder rerank score (precise but its
# scale drifts per corpus) OR the dense cosine similarity (corpus-stable). The
# gate itself now lives in the EvaluatorAgent; TextAgent.run() consults it and,
# on a WEAK verdict, invokes the QueryRewriteAgent to retrieve again before giving
# up (Corrective-RAG). Thresholds + _dense_sim are defined in agents/grounding.py.


# Command/meta words that describe *how* to answer ("show me the image") rather
# than *what* about. The cross-encoder reranker scores the raw query, and these
# words tank the relevance score of on-topic chunks (e.g. "show map reduce image"
# scored ~0.1 vs "map reduce" ~0.98), tripping the grounding gate and dropping
# good results. We strip them for the RERANK query only — retrieval still sees
# the original (so wants_image() appends the figure) and so does the LLM prompt.
# Deliberately excludes ambiguous CONTENT words (map/flow/graph/chart/design/
# architecture/layout/plot) — "map" is content in "map reduce".
_RERANK_STOPWORDS = {
    "show", "me", "give", "gimme", "get", "display", "see", "view", "please",
    "image", "images", "picture", "pic", "pics", "photo", "diagram", "diagrams",
    "figure", "figures", "fig", "illustration", "screenshot", "sketch",
    "schematic", "visual", "visuals",
    "the", "a", "an", "of", "for", "my", "this", "that", "its", "on", "about",
}


def _rerank_query(query: str) -> str:
    """Strip command/meta words so the cross-encoder scores the actual topic.
    Falls back to the original query if stripping leaves nothing (e.g. a pure
    "show me the image" with no subject)."""
    kept = [w for w in query.split() if w.strip(".,?!:;\"'").lower() not in _RERANK_STOPWORDS]
    cleaned = " ".join(kept).strip()
    return cleaned or query


def adaptive_context(
    chunks: List[dict],
    budget_tokens: int = 1500,
) -> List[dict]:
    """Adaptive Context Expansion (runs AFTER reranking) — completeness-first.

    Keeps each result's FULL parent section (so no sub-points are ever lost) and
    only shrinks a result to its short child snippet when the full parent would
    blow the token budget. Because results arrive best-first, trimming falls on the
    lowest-ranked TAIL, never on the most-relevant top results. This gives the
    ~2048-token window overflow protection WITHOUT dropping content from the
    answers that matter — the earlier score-based proactive trim could omit list
    sub-points, which is why it was removed. Each reranked chunk already carries
    `child_text` (matched snippet) + `text` (full parent), so no re-ingestion.
    """
    out: List[dict] = []
    used = 0
    for r in chunks:                        # already ranked best-first
        child = r.get("child_text") or r.get("text", "")
        parent = r.get("text", "") or child
        p_tok = max(1, len(parent) // 4)    # ~4 chars/token estimate
        if used + p_tok <= budget_tokens:
            piece, lvl, tok = parent, "parent", p_tok
        else:                               # parent won't fit -> shrink to snippet
            c_tok = max(1, len(child) // 4)
            if used + c_tok > budget_tokens:
                break                       # even the snippet won't fit -> stop
            piece, lvl, tok = child, "child", c_tok
        out.append({**r, "text": piece, "_ctx_level": lvl})
        used += tok
    return out


class TextAgent:

    def __init__(self, retriever: Retriever = None, reranker: Reranker = None,
                 evaluator: EvaluatorAgent = None, rewriter: QueryRewriteAgent = None):
        self.retriever = retriever or Retriever()
        self.reranker = reranker or Reranker()
        self.evaluator = evaluator or EvaluatorAgent()
        self.rewriter = rewriter or QueryRewriteAgent()
        # Corrective-RAG: how many rewrite+retry attempts to make when the
        # Evaluator judges retrieval WEAK. 0 disables the loop (original
        # single-shot behaviour); default 1 (one corrective retry).
        self.max_retries = int(os.environ.get("RAG_MAX_RETRIES", "1"))

    # ---------------------------------------------------------------- #

    def _retrieve_rerank(
        self, query: str, top_k: int, top_n: int, pdf_filter: Optional[str],
    ) -> List[dict]:
        """One retrieval attempt: hybrid retrieve (text, plus images when the
        query asks for a visual) followed by cross-encoder reranking."""
        candidates = self.retriever.retrieve_smart(
            query=query, top_k=top_k, pdf_filter=pdf_filter,
        )
        if not candidates:
            return []
        return self.reranker.rerank(
            # spell-normalize (same as retrieval) so the cross-encoder scores the
            # corrected query too, then strip command/meta words.
            query=_rerank_query(self.retriever.normalize_query(query)),
            chunks=candidates,
            top_n=top_n,
        )

    def run(
        self,
        query: str,
        top_k_retrieve: int = 10,
        top_n_rerank: int = 5,
        pdf_filter: Optional[str] = None,
    ) -> List[dict]:
        """Corrective-RAG retrieval:

            retrieve → Evaluator → GOOD: return
                                 → WEAK: Query Rewrite → retrieve again → re-evaluate

        Returns up to ``top_n_rerank`` grounded chunks (best-first), each with
        id, text, metadata, distance, rerank_score — or ``[]`` when no attempt is
        confident, in which case the Generator Agent refuses.
        """
        cur_query = query
        best = self._retrieve_rerank(cur_query, top_k_retrieve, top_n_rerank, pdf_filter)
        verdict = self.evaluator.evaluate(best)

        attempt = 0
        while not verdict.good and attempt < self.max_retries:
            new_query = self.rewriter.rewrite(cur_query)
            if not new_query:                 # no different phrasing to try → stop
                break
            attempt += 1
            logger.info("Corrective-RAG retry %d/%d (Evaluator WEAK)",
                        attempt, self.max_retries)
            retry = self._retrieve_rerank(new_query, top_k_retrieve, top_n_rerank, pdf_filter)
            v2 = self.evaluator.evaluate(retry)
            if v2.key > verdict.key:           # keep the stronger of the two attempts
                best, verdict = retry, v2
            cur_query = new_query

        if not verdict.good:
            logger.info("Grounding gate: no confident context after %d attempt(s) — refusing",
                        attempt + 1)
            return []

        reranked = best
        # Adaptive Context Expansion (default on; set RAG_ADAPTIVE=0 to disable and
        # fall back to always-full-parent).
        if os.environ.get("RAG_ADAPTIVE", "1") != "0":
            reranked = adaptive_context(reranked)
        logger.info("TextAgent: returning %d grounded chunks", len(reranked))
        return reranked
