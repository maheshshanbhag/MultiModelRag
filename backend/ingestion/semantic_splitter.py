"""
semantic_splitter.py
Embedding-based (semantic) text splitting: cut where the MEANING shifts, not at a
fixed size.

Each sentence is embedded (BGE); a break is placed where the cosine distance
between consecutive sentences exceeds a percentile threshold (a semantic boundary),
or where a hard max-length is reached. Used by SemanticDocumentChunker to split a
long section into idea-coherent children instead of arbitrary char slices.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

_SENT = re.compile(r"(?<=[.!?])\s+")


class SemanticSplitter:

    def __init__(self, model, pct: float = 90.0, max_chars: int = 450, min_chars: int = 80):
        self.model = model          # a SentenceTransformer (BGE)
        self.pct = pct              # break when distance >= this percentile of distances
        self.max_chars = max_chars  # hard cap so a piece never runs away
        self.min_chars = min_chars  # don't break into pieces smaller than this

    def split(self, text: str) -> List[str]:
        text = (text or "").strip()
        if len(text) <= self.max_chars:
            return [text] if text else []
        sents = [s.strip() for s in _SENT.split(text) if s.strip()]
        if len(sents) <= 2:
            return [text]

        embs = self.model.encode(
            sents, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        # cosine distance between consecutive sentences (unit-norm vectors)
        dists = 1.0 - np.sum(embs[:-1] * embs[1:], axis=1)
        thr = float(np.percentile(dists, self.pct)) if len(dists) else 1.0

        chunks: List[str] = []
        cur = sents[0]
        for i, s in enumerate(sents[1:]):
            boundary = dists[i] >= thr and len(cur) >= self.min_chars
            if boundary or len(cur) + len(s) + 1 > self.max_chars:
                chunks.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}"
        if cur.strip():
            chunks.append(cur)
        return chunks
