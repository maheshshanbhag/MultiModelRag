"""
agents/evaluator_agent.py
Evaluator Agent — judges whether a retrieval is strong enough to answer.

This is the "Good retrieval?" decision node in the agentic (Corrective-RAG) loop.
It applies the multi-signal grounding gate to the reranked candidates and returns
a Verdict. The Text Agent uses that verdict to decide what happens next:

    good  -> proceed to the Generator Agent
    weak  -> hand off to the Query Rewrite Agent and retrieve again

Keeping this as its own agent (rather than an inline `if`) makes the decision
explicit, testable, and reusable across the loop's attempts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from agents.grounding import GROUNDING_MIN_SCORE, GROUNDING_MIN_DENSE, _dense_sim

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    """Outcome of evaluating one retrieval attempt."""
    good: bool
    top_score: float = 0.0
    max_dense: float = 0.0
    reason: str = ""

    @property
    def key(self):
        """Comparable strength key used to pick the better of two attempts:
        a passing verdict beats a failing one; among equals, higher dense
        similarity (corpus-stable) then higher rerank score wins."""
        return (self.good, round(self.max_dense, 4), round(self.top_score, 4))


class EvaluatorAgent:
    def __init__(self, min_score: float = GROUNDING_MIN_SCORE,
                 min_dense: float = GROUNDING_MIN_DENSE):
        self.min_score = min_score
        self.min_dense = min_dense

    def evaluate(self, reranked: List[dict]) -> Verdict:
        """Return a Verdict for a reranked candidate list (best-first)."""
        if not reranked:
            return Verdict(False, 0.0, 0.0, "no candidates")
        top_score = float(reranked[0].get("rerank_score", 0.0))
        # dense_sim is the MAX across the set: the top-1 can be a BM25-only hit
        # with no dense signal, while a lower-ranked chunk carries the strong one.
        max_dense = max((_dense_sim(c) for c in reranked), default=0.0)
        good = top_score >= self.min_score or max_dense >= self.min_dense
        reason = (f"rerank={top_score:.3f}(>={self.min_score:.2f}) OR "
                  f"dense={max_dense:.3f}(>={self.min_dense:.2f})")
        logger.info("EvaluatorAgent: %s -> %s", reason, "GOOD" if good else "WEAK")
        return Verdict(good, top_score, max_dense, reason)
