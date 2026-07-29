"""
ask.py
Terminal RAG chat — full pipeline with STREAMING output to the console.

Pipeline (same agents the API uses):
    retrieve (top-20) -> rerank (top-5) -> Ollama llama3, streamed token-by-token.

Usage:
    python ask.py "what is a dataframe in spark"
    python ask.py                 # interactive: type questions, blank line to quit

Requires Ollama running locally (ollama serve) with the model pulled
(ollama pull llama3) and a PDF already ingested into the vector DB.
"""
from __future__ import annotations

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import subprocess, sys
from pathlib import Path

# Re-launch under the project venv if started with a different interpreter.
_venv = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
if _venv.exists() and Path(sys.executable).resolve() != _venv.resolve():
    sys.exit(subprocess.run([str(_venv)] + sys.argv, env=os.environ.copy()).returncode)

# Windows consoles default to cp1252, which crashes on chars like -> that
# appear in the notes; force UTF-8 so streamed text prints cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import logging
# Send agent/model INFO logs to stderr at WARNING+ only, so they never
# interleave with the streamed answer on stdout.
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


def build_agents():
    """Load models once and reuse across queries."""
    from agents.router import RouterAgent
    from agents.text_agent import TextAgent
    from agents.image_agent import ImageAgent
    from agents.generator_agent import GeneratorAgent
    from llm.ollama_client import OllamaClient

    llm = OllamaClient()                          # llama3.2:3b — text answers
    vlm = OllamaClient(model="qwen2.5vl:3b")      # vision model — interprets figures
    # llm=None on the router -> keyword-only (rule-based) routing, no extra
    # round-trip just to classify, so the answer starts streaming sooner.
    router = RouterAgent(llm=None)
    text_agent = TextAgent()          # builds its own Retriever + Reranker
    image_agent = ImageAgent()
    generator = GeneratorAgent(llm=llm)
    return router, text_agent, image_agent, generator, vlm


def sources_footer(chunks) -> str:
    """One entry per PDF with ALL its page numbers merged: 'doc.pdf (p. 5, 7, 9)'.

    Deterministic (straight from chunk metadata) — more reliable than trusting
    the LLM to cite pages itself.
    """
    from collections import OrderedDict

    pages_by_pdf: "OrderedDict[str, list]" = OrderedDict()
    for c in chunks:
        m = c.get("metadata", {})
        pdf = m.get("pdf_name", "") or "unknown"
        pages_by_pdf.setdefault(pdf, [])
        raw = str(m.get("pages", "")).strip()
        for p in raw.replace(";", ",").split(","):
            p = p.strip()
            if p and p not in pages_by_pdf[pdf]:
                pages_by_pdf[pdf].append(p)

    def _key(p: str):
        # Sort pages numerically when possible, else lexically.
        try:
            return (0, int(p))
        except ValueError:
            return (1, p)

    parts = []
    for pdf, pages in pages_by_pdf.items():
        if pages:
            parts.append(f"{pdf} (p. {', '.join(sorted(pages, key=_key))})")
        else:
            parts.append(pdf)
    return "; ".join(parts)


# Words signalling the user wants a textual EXPLANATION (not just to SEE a figure).
_EXPLAIN_WORDS = {
    "explain", "describe", "what", "whats", "how", "why", "detail", "details",
    "elaborate", "tell", "define", "definition", "summarize", "summary",
    "overview", "compare", "difference", "meaning", "when", "where", "which",
}


def _has_explain_intent(query: str) -> bool:
    toks = {t.strip(".,?!:;\"'").lower() for t in query.split()}
    return bool(toks & _EXPLAIN_WORDS)


def _caption_for(image_path: str, chunks) -> str:
    """Best caption/heading for the figure, to focus the VLM (not the LLM answer)."""
    for c in chunks:
        if image_path in str(c.get("metadata", {}).get("image_paths", "")):
            m = c.get("metadata", {})
            return m.get("heading", "") or (c.get("child_text") or c.get("text", ""))[:120]
    return ""


import re

# Small llama3.2:3b sometimes emits a fabricated markdown image link despite the
# system-prompt rule against it. Strip it from the stream as a hard guarantee.
_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _filter_image_md(tokens):
    """Stream text with any markdown image links ![alt](url) removed live."""
    buf = ""
    for tok in tokens:
        buf = _IMG_MD.sub("", buf + tok)
        # If a '![' is open but not yet closed with ')', hold from it (it may
        # still complete into an image link); emit everything before it.
        i = buf.rfind("![")
        if i != -1 and ")" not in buf[i:]:
            if i:
                yield buf[:i]
            buf = buf[i:]
        else:
            if buf:
                yield buf
            buf = ""
    buf = _IMG_MD.sub("", buf)
    if buf:
        yield buf


def _ensure_only_loaded(keep_model: str, base_url: str = "http://localhost:11434") -> None:
    """Evict every Ollama model except `keep_model` so it gets the FULL 4 GB GPU.

    Critical on 4 GB: without this, keep_alive leaves llama3.2 resident when
    qwen2.5vl loads, both cram into VRAM, both spill to CPU, and the VLM's vision
    prefill crawls (measured: 16 min). We only unload models actually loaded
    (from /api/ps), so this never accidentally loads one.
    """
    import requests
    try:
        loaded = requests.get(f"{base_url}/api/ps", timeout=5).json().get("models", [])
    except Exception:
        return
    for m in loaded:
        name = m.get("name") or m.get("model", "")
        if name and name != keep_model:
            try:
                requests.post(
                    f"{base_url}/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
            except Exception:
                pass


def _vlm_prompt(query: str, caption: str) -> str:
    ctx = f' It is titled/about: "{caption}".' if caption else ""
    return (
        f'The user asked: "{query}".{ctx}\n'
        "In about 100 words, describe what this figure shows — its main components, "
        "labels, and how they relate. Write ONE short plain paragraph (no headings, "
        "no bullet lists). Focus only on the visual content of the image."
    )


def answer(query: str, agents, top_k: int = 10, top_n: int = 5) -> None:
    import time
    from ingestion.retriever import Retriever

    _t0 = time.time()
    _router, text_agent, image_agent, generator, vlm = agents

    image_intent = Retriever.wants_image(query)      # rule-based: show/diagram/figure...
    explain_intent = _has_explain_intent(query)      # rule-based: explain/what/how...

    chunks = text_agent.run(query=query, top_k_retrieve=top_k, top_n_rerank=top_n)

    # Surface a figure only from the MOST relevant chunk(s): on-topic, and none
    # when the best matches have no figure. Image-intent queries look deeper.
    image_paths = []
    if chunks:
        n_top = 3 if image_intent else 2
        _, image_paths = image_agent.run(chunks[:n_top], max_images=2)
    has_image = bool(image_paths)

    # Rule-based routing (one model on the GPU at a time — no parallel):
    #   pure "show me X image" (image intent, no explain word) + image found
    #        -> VLM only (skip llama3.2)
    #   image intent + explain word   -> llama3.2 text, THEN VLM on the image
    #   otherwise                     -> llama3.2 text only
    pure_show = image_intent and not explain_intent and has_image
    run_text = not pure_show
    run_vlm = image_intent and has_image

    print(f"\n{'=' * 70}\nQ: {query}\n{'=' * 70}\n")

    # 1) Detailed TEXT answer from llama3.2 (its strength), streamed live.
    if run_text:
        _ensure_only_loaded(generator.llm.model)   # evict qwen so llama3.2 gets the full GPU
        stream = generator.run_stream(query=query, chunks=chunks, image_paths=image_paths)
        for token in _filter_image_md(stream):     # drop any fabricated ![](url) links
            print(token, end="", flush=True)
        print("\n")

    # 2) Show the relevant figure to the user.
    if has_image:
        print("Image: " + "; ".join(image_paths))

    # 3) VLM reads the image (only when the user asked for a figure). Ollama swaps
    #    llama3.2 out and qwen2.5vl in automatically (sequential, full GPU each).
    if run_vlm:
        _ensure_only_loaded(vlm.model)   # evict llama3.2 so qwen2.5vl gets the full GPU
        caption = _caption_for(image_paths[0], chunks)
        print("\n[Image analysis — qwen2.5vl]:\n")
        for token in vlm.vlm_stream(_vlm_prompt(query, caption), [image_paths[0]]):
            print(token, end="", flush=True)
        print()

    # 4) Deterministic source footer: which PDF(s)/page(s) the answer came from.
    if chunks:
        print("\n" + "-" * 70)
        print(f"Source: {sources_footer(chunks)}")
    print(f"[{time.time() - _t0:.1f}s]")
    print()


def main() -> None:
    agents = build_agents()

    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]), agents)
        return

    print("Terminal RAG (streaming). Type a question, or blank line / Ctrl-C to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(q, agents)


if __name__ == "__main__":
    main()
