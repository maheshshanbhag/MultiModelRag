"""
agents/router.py

NOTE: Reconstructed from its call sites after the original was permanently
deleted. In the shipped flow the router is constructed with llm=None, which
means rule-based (keyword-only) routing — no extra LLM round-trip just to
classify, so the answer starts streaming sooner. ask.py/api.py do the actual
image-vs-text routing inline (Retriever.wants_image + explain-intent), so this
agent only needs to be importable and offer a lightweight rule-based route().
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Words that signal the user wants to SEE a visual.
_IMAGE_WORDS = {
    "image", "images", "figure", "figures", "diagram", "diagrams", "picture",
    "pictures", "illustration", "illustrations", "chart", "charts", "graph",
    "graphs", "photo", "screenshot", "visual", "show", "display",
}


class RouterAgent:

    def __init__(self, llm=None):
        # llm=None -> rule-based routing (no classification round-trip).
        self.llm = llm

    def route(self, query: str) -> str:
        """Return 'image' if the query explicitly asks for a visual, else 'text'.

        Rule-based by default. If an llm is provided it could be used for a
        finer classification, but the shipped pipeline routes inline in
        ask.py/api.py, so the keyword rule is the default and is enough.
        """
        toks = {t.strip(".,?!:;\"'").lower() for t in query.split()}
        return "image" if (toks & _IMAGE_WORDS) else "text"
