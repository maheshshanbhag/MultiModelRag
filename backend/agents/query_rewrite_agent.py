"""
agents/query_rewrite_agent.py
Query Rewrite Agent — reformulates a query when retrieval was judged weak.

Part of the agentic (Corrective-RAG) loop: when the Evaluator Agent reports a
weak retrieval, this agent produces an alternative phrasing and the Text Agent
retrieves again. It is RULE-BASED by default (instant, no LLM), so it never slows
down the common case where the first retrieval already succeeds. An optional
single-LLM rephrase can be enabled with RAG_REWRITE_LLM=1 for harder queries.

The rewrite returns "" when it cannot produce anything different from the input,
which signals the Text Agent to stop retrying instead of re-running an identical
search.
"""
from __future__ import annotations

import logging
import os
import string

logger = logging.getLogger(__name__)

# Interrogatives + generic stopwords. Stripping these turns a conversational
# question into a keyword query: the retriever then computes a DIFFERENT dense
# embedding (and cleaner BM25 terms) for the second attempt, which often surfaces
# passages the verbose phrasing missed.
_STOP = {
    "what", "whats", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "shall", "may", "might", "the", "a", "an", "of",
    "for", "to", "in", "on", "at", "by", "with", "about", "into", "from", "as",
    "and", "or", "please", "tell", "me", "give", "explain", "describe", "define",
    "list", "show", "you", "your", "i", "my", "we", "our", "there", "any",
    "some", "this", "that", "these", "those", "it", "its", "be", "been", "get",
}


class QueryRewriteAgent:
    def __init__(self, llm=None, use_llm: bool = None):
        self.llm = llm
        self.use_llm = (
            os.environ.get("RAG_REWRITE_LLM", "0") == "1" if use_llm is None else use_llm
        )

    def rewrite(self, query: str) -> str:
        """Return a reformulated query, or "" if no useful reformulation exists
        (i.e. it would be identical to the input)."""
        if self.use_llm and self.llm is not None:
            r = self._llm_rewrite(query)
            if r and r.strip().lower() != query.strip().lower():
                logger.info("QueryRewriteAgent (LLM): %r -> %r", query[:60], r[:60])
                return r.strip()
        rewritten = self._keyword_rewrite(query)
        if rewritten:
            logger.info("QueryRewriteAgent (rule): %r -> %r", query[:60], rewritten[:60])
        return rewritten

    # ---------------------------------------------------------------- #

    def _keyword_rewrite(self, query: str) -> str:
        toks = [w.strip(string.punctuation) for w in query.split()]
        kept = [w for w in toks if w and w.lower() not in _STOP]
        rewritten = " ".join(kept).strip()
        if not rewritten or rewritten.lower() == query.strip().lower():
            return ""
        return rewritten

    def _llm_rewrite(self, query: str) -> str:
        try:
            prompt = (
                "Rewrite the user's question as a concise search query of the key "
                "terms a document retriever should match. Return ONLY the query, "
                "no explanation.\n\n"
                f"Question: {query}\nSearch query:"
            )
            out = self.llm.generate(prompt).strip().splitlines()
            return (out[0].strip() if out else "")[:200]
        except Exception as e:  # never let a rewrite failure break the query
            logger.warning("LLM rewrite failed (%s); falling back to rule-based.", e)
            return ""
