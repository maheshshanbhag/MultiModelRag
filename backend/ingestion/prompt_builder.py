"""
prompt_builder.py
Stage 13 — Prompt Builder

Assembles the final prompt for the local LLM with:
  - Structured context with source citations and relevance scores
  - Chain-of-thought instruction for complex questions
  - Strict grounding: answer only from provided context
  - OpenAI-style messages for Ollama chat API
"""

from __future__ import annotations

from typing import List


SYSTEM_INSTRUCTION = """You are a respectful and honest AI assistant. Your primary goal is to provide accurate and direct answers using ONLY the information available in the given context.

Format your answer in clean Markdown:
- Open with one short **bold** sentence that directly answers the question.
- If there are multiple points, use a bullet or numbered list and bold the key term of each point.
- Be COMPLETE: include EVERY distinct point or item the context contains — never omit, merge, or summarize them away. If the context lists five points, your answer must contain all five.
- Be well organized — no filler, no repetition — but never drop content for the sake of brevity.
- When the context gives a specific NUMBER, RANGE, UNIT or VALUE (temperatures, pressures, ratios, wt%, etc.), reproduce it EXACTLY as written — never round, approximate, generalize, or invent a figure. If the specific value asked for is not in the context, say so rather than guessing.
- Do not mention the context, excerpts, or sources, and do not add any citation tags like [Source 1].
- Do NOT output image links, URLs, or Markdown images (e.g. ![...](...)). If a figure is relevant it is shown to the user separately — never invent one.
- If the context does not contain the answer, reply exactly: "The documents don't contain enough information to answer this."
"""

class PromptBuilder:

    # ---------------------------------------------------------------- #

    def build(
        self,
        query: str,
        chunks: List[dict],
        image_paths: List[str] = None,
    ) -> str:
        """
        Build the complete prompt string for non-chat LLM inference.
        """
        image_paths = image_paths or []
        sections: List[str] = []

        sections.append(SYSTEM_INSTRUCTION)
        sections.append("")
        sections.append("=" * 60)
        sections.append("CONTEXT EXCERPTS")
        sections.append("=" * 60)

        for i, chunk in enumerate(chunks, 1):
            meta  = chunk.get("metadata", {})
            pdf   = meta.get("pdf_name", "unknown")
            pages = meta.get("pages", "")
            heading = meta.get("heading", "")
            score = chunk.get("rerank_score", None)

            header = f"[Source {i}]  {pdf}"
            if pages:
                header += f"  |  p.{pages}"
            if heading:
                header += f"  |  §{heading}"
            if score is not None:
                header += f"  |  relevance={score:.3f}"

            sections.append(header)
            sections.append(chunk["text"])
            sections.append("")

        if image_paths:
            sections.append("RELATED FIGURES")
            sections.append("-" * 40)
            for path in image_paths:
                sections.append(f"  • {path}")
            sections.append("")

        sections.append("=" * 60)
        sections.append("QUESTION")
        sections.append("=" * 60)
        sections.append(query)
        sections.append("")
        sections.append("ANSWER (cite [Source N] for each claim):")

        return "\n".join(sections)

    # ---------------------------------------------------------------- #

    def build_messages(
        self,
        query: str,
        chunks: List[dict],
        image_paths: List[str] = None,
    ) -> List[dict]:
        """
        Return OpenAI-style message list (compatible with Ollama chat API).
        """
        image_paths = image_paths or []

        # Image paths are NOT put in the prompt: the text LLM can't see them and
        # would only hallucinate. Relevant figures are shown to the user
        # separately (see ask.py). Answer is generated from context text only.
        user_content = (
            f"{self._format_context(chunks)}"
            f"\n---\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        return [
            {"role": "system",  "content": SYSTEM_INSTRUCTION},
            {"role": "user",    "content": user_content},
        ]

    # ---------------------------------------------------------------- #

    @staticmethod
    def _format_context(chunks: List[dict]) -> str:
        # No "[Source N]" / pdf / page / score labels: the answer must not cite
        # them (pages are shown deterministically in the footer instead). Only
        # the section heading is kept, to help the model organise the answer.
        lines: List[str] = ["Context:\n"]
        for chunk in chunks:
            heading = chunk.get("metadata", {}).get("heading", "")
            if heading:
                lines.append(f"## {heading}")
            lines.append(chunk["text"].strip())
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_figures(image_paths: List[str]) -> str:
        if not image_paths:
            return ""
        lines = ["Related figures:"]
        for p in image_paths:
            lines.append(f"  • {p}")
        return "\n".join(lines) + "\n"
