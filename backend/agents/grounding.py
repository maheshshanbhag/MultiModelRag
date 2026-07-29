"""
agents/grounding.py
Shared grounding-signal primitives used by the Text Agent and the Evaluator Agent.

Kept in one place so both agents (and any evaluation script) agree on the exact
thresholds and the dense-similarity formula. Two INDEPENDENT signals decide whether
a retrieval is trustworthy enough to answer:
  * rerank score  — cross-encoder relevance (precise, but its scale drifts per corpus)
  * dense_sim     — query<->chunk cosine from ChromaDB (corpus-stable)
A result is "grounded" when EITHER signal clears its floor. See EvaluatorAgent.
"""
from __future__ import annotations

import os

# rerank floor / dense-sim floor. Set both to 0 to disable the gate entirely.
GROUNDING_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.30"))
GROUNDING_MIN_DENSE = float(os.environ.get("RAG_MIN_DENSE", "0.60"))


def _dense_sim(chunk: dict) -> float:
    """Query<->chunk cosine similarity derived from the ChromaDB L2 distance.

    Chroma stores L2 distance on L2-normalized embeddings, where
    ``||a-b||^2 = 2 - 2*cos``, so ``cos = 1 - distance/2``. BM25-only hits carry
    no dense distance (0.0 / absent) and return 0.0 (no dense signal).
    """
    d = chunk.get("distance")
    if not d:
        return 0.0
    return 1.0 - d / 2.0
