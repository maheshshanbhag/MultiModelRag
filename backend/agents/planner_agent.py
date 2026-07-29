"""
agents/planner_agent.py
Planner Agent — the entry point of the query pipeline.

Given a user query, it produces a Plan describing what the answer needs: a text
response, a figure (visual), or both, and whether the user wants a figure
explained. The plan tells the orchestrator which downstream agents to involve:
the Text Agent always runs; the Image Agent and vision model are engaged only for
visual queries. Planning is RULE-BASED (no LLM), so it adds no latency before the
answer begins to stream.

(Supersedes the earlier RouterAgent, which only produced a flat text/image label.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Words that signal the user wants something explained/summarised (as opposed to a
# bare "show me the figure"). Kept identical to the API's previous explain-intent
# set so routing behaviour is unchanged.
_EXPLAIN_WORDS = {
    "explain", "describe", "what", "whats", "how", "why", "detail", "details",
    "elaborate", "tell", "define", "definition", "summarize", "summary",
    "overview", "compare", "difference", "meaning", "when", "where", "which",
}


@dataclass
class Plan:
    wants_image: bool
    wants_explain: bool

    @property
    def intent(self) -> str:
        if self.wants_image and self.wants_explain:
            return "both"
        if self.wants_image:
            return "image"
        return "text"


class PlannerAgent:
    def __init__(self, llm=None):
        # llm reserved for future LLM-based planning; today the plan is rule-based.
        self.llm = llm

    def plan(self, query: str) -> Plan:
        from ingestion.retriever import Retriever

        toks = {t.strip(".,?!:;\"'").lower() for t in query.split()}
        wants_image = Retriever.wants_image(query)
        wants_explain = bool(toks & _EXPLAIN_WORDS)
        p = Plan(wants_image=wants_image, wants_explain=wants_explain)
        logger.info("PlannerAgent: intent=%s (image=%s explain=%s)",
                    p.intent, wants_image, wants_explain)
        return p
