"""
agents/generator_agent.py
Response Generation Agent

Single responsibility: build prompt and call the local LLM.

Input  : query + reranked chunks + image paths
Output : answer string
"""

from __future__ import annotations

import logging
from typing import Iterator, List

from ingestion.prompt_builder import PromptBuilder
from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class GeneratorAgent:

    def __init__(self, llm: OllamaClient = None):
        self.llm = llm or OllamaClient()
        self.prompt_builder = PromptBuilder()

    # ---------------------------------------------------------------- #

    def run(
        self,
        query: str,
        chunks: List[dict],
        image_paths: List[str] = None,
    ) -> str:
        """Generate an answer from the provided context."""
        image_paths = image_paths or []

        if not chunks:
            return "The documents don't contain enough information to answer this."

        messages = self.prompt_builder.build_messages(
            query=query,
            chunks=chunks,
            image_paths=image_paths,
        )

        logger.info("Generating answer with %d context chunks", len(chunks))

        try:
            answer = self.llm.chat(messages)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = f"[LLM error: {exc}]"

        return answer

    def run_stream(
        self,
        query: str,
        chunks: List[dict],
        image_paths: List[str] = None,
    ) -> Iterator[str]:
        """Streaming variant — yields tokens as they arrive.

        Mirrors run(): same build_messages() chat prompt and same empty-chunk /
        error handling, so the streamed and non-streamed answers are produced
        from an identical prompt via the same /api/chat endpoint.
        """
        image_paths = image_paths or []

        if not chunks:
            yield "The documents don't contain enough information to answer this."
            return

        messages = self.prompt_builder.build_messages(
            query=query,
            chunks=chunks,
            image_paths=image_paths,
        )

        logger.info("Streaming answer with %d context chunks", len(chunks))

        try:
            yield from self.llm.chat_stream(messages)
        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc)
            yield f"[LLM error: {exc}]"
