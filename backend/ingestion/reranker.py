"""
reranker.py
Stage 11 — Cross-Encoder Reranker

Model : BAAI/bge-reranker-base  (GPU if available)

Input : query  +  top-20 retrieved chunks
Output: top-5 reranked chunks (highest relevance first)

Cross-encoders score (query, chunk) jointly — dramatically more
accurate than cosine similarity for final ranking.
"""

from __future__ import annotations

import os
# Models are pre-cached locally; force offline so HuggingFace does not try the
# network (which fails when running offline).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Spurious transformers warning: bge-reranker-v2-m3 uses an XLM-RoBERTa /
# SentencePiece-Unigram tokenizer with a Metaspace pre-tokenizer (no regex),
# so the "fix_mistral_regex" hint does not apply — tokenization is correct.
# transformers logs this via its OWN handler (propagate=False), so a Python
# warnings filter or root-logger filter won't catch it; we filter the message
# on the transformers handler itself.
import logging as _logging
from transformers.utils import logging as _hf_logging


class _DropMistralRegexFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "fix_mistral_regex" not in record.getMessage()


_hf_logging.get_logger("transformers")  # force the default handler to exist
for _h in _logging.getLogger("transformers").handlers:
    _h.addFilter(_DropMistralRegexFilter())

import logging
import re
from pathlib import Path
from typing import List

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_LOCAL_RERANK = Path(__file__).resolve().parent.parent / "model" / "reranker" / "bge-reranker-v2-m3"
RERANK_MODEL = str(_LOCAL_RERANK) if _LOCAL_RERANK.is_dir() else "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_N = 5
HEADING_BOOST = 0.1     # score bonus per query content-word found in a heading

# Numeric boost: on a QUANTITATIVE query ("what range / how much / temperature /
# pressure / ratio ...") nudge chunks that actually contain a number+unit up the
# ranking, so a figure-bearing chunk isn't buried below prose. No effect on
# conceptual queries. RAG_NUMERIC_BOOST=0 disables.
NUMERIC_BOOST = float(os.environ.get("RAG_NUMERIC_BOOST", "0.15"))
_QUANT_CUE = re.compile(
    r"\b(range|much|many|temperature|temp|pressure|ratio|rate|target|density|"
    r"limit|psi|percent|value|values|how|number|degrees?|scf|bpd|tons?|lb|kg|"
    r"bar|windows?|parameters?|targets?|wt)\b", re.I)
_NUM_UNIT = re.compile(
    r"\d[\d.,]*\s?(°|deg|psi|bar|wt\s?%|%|lb|kg|bpd|ton|scf|btu|:\s?1|f\b|c\b)", re.I)

# Stop / question / image-intent words that shouldn't count as heading matches.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "be", "what", "how", "why", "which", "tell", "about", "also", "me",
    "my", "explain", "describe", "give", "show", "display", "please", "do",
    "does", "can", "into", "image", "images", "figure", "figures", "diagram",
    "diagrams", "picture", "pictures", "chart", "charts", "graph", "graphs",
    "photo", "photos", "visual", "illustration", "illustrations",
}


class Reranker:

    def __init__(self):
        # Default CPU: on a 4GB GPU the encoders must stay off the card so the
        # LLM (Ollama) gets the whole GPU and runs 100% GPU-resident. Reranking
        # ~20 short pairs on CPU costs ~1-2s. Set RAG_ENCODER_DEVICE=cuda (or
        # =auto) on a bigger GPU to put the reranker back on CUDA.
        _dev = os.environ.get("RAG_ENCODER_DEVICE", "cpu").lower()
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
        )
        logger.info("Loading reranker on %s ...", self.device)
        self.model = CrossEncoder(
            RERANK_MODEL,
            device=self.device,
            # 256 not 512: we rerank the short child unit (~140 tokens), so 256
            # never truncates and roughly halves CPU cost vs 512.
            max_length=256,
        )
        logger.info("Reranker ready: %s", RERANK_MODEL)

    # ---------------------------------------------------------------- #

    @classmethod
    def _content_terms(cls, query: str) -> set:
        terms = set()
        for tok in query.split():
            w = tok.strip(".,?!:;\"'()").lower()
            if len(w) > 2 and w not in _STOPWORDS:
                terms.add(w)
        return terms

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_n: int = DEFAULT_TOP_N,
    ) -> List[dict]:
        """
        Score all (query, chunk_text) pairs and return top_n.

        A small HEADING_BOOST is added when a chunk's heading contains query
        content words — this disambiguates items whose body text is nearly
        identical (e.g. two figures sharing one OCR caption: the figure headed
        "HBase Data Model" should beat the one headed "HBase Implementation"
        for the query "hbase data model").

        Each returned chunk has an extra key  'rerank_score'.
        """
        if not chunks:
            return []

        # Rerank the short matched CHILD unit, not the big parent section: same
        # relevance signal at ~half the tokens -> much faster on CPU. The full
        # parent text still flows to the LLM unchanged (chunk["text"]).
        pairs = [(query, c.get("child_text") or c["text"]) for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        q_terms = self._content_terms(query)
        quant = bool(_QUANT_CUE.search(query))          # is this a "number" query?
        scored: List[tuple] = []
        for score, chunk in zip(scores, chunks):
            meta = chunk.get("metadata", {})
            # Section-aware: query content-words matching the heading OR the section
            # title disambiguate near-identical bodies (e.g. "reactor" vs
            # "regenerator" temperature chunks).
            head_sec = ((meta.get("heading", "") or "") + " " +
                        (meta.get("section", "") or "")).lower()
            overlap = min(sum(1 for t in q_terms if t in head_sec), 3)
            s = float(score) + HEADING_BOOST * overlap
            # Numeric boost is TOPIC-GATED: only promote a number-bearing chunk that
            # is also on-topic (overlap > 0), so an off-topic figure isn't pulled up
            # (this fixes the earlier Q21-style churn from an untargeted boost).
            if quant and overlap > 0 and _NUM_UNIT.search(
                    chunk.get("child_text") or chunk.get("text", "") or ""):
                s += NUMERIC_BOOST
            scored.append((s, chunk))

        ranked = sorted(scored, key=lambda x: x[0], reverse=True)

        result: List[dict] = []
        for score, chunk in ranked[:top_n]:
            entry = dict(chunk)
            entry["rerank_score"] = float(score)
            result.append(entry)

        logger.info(
            "Reranked %d -> %d  |  top score: %.4f",
            len(chunks), len(result),
            result[0]["rerank_score"] if result else 0.0,
        )
        return result
