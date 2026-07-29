"""
api.py
FastAPI backend for the MultiModal RAG frontend.

NOTE: RECONSTRUCTED from the fragments read this session + the shared ask.py
helpers after the original was permanently deleted. It implements the endpoints
the frontend uses (query, streaming query, figure interpretation, document
list/ingest/delete) with the same routing (one Ollama model on the GPU at a
time). Behaviour should match; verify against the recovered original if found.

Run:  python api.py   (serves on http://0.0.0.0:8000)
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
    sys.exit(subprocess.run([str(_venv), __file__] + sys.argv[1:], env=os.environ.copy()).returncode)

import json
import logging
import re
import shutil
import threading
from typing import Iterator, List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
PDF_DIR = BASE / "uploads" / "pdfs"
FIGURE_DIR = BASE / "data" / "figures"
PDF_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MultiModal RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve extracted figures so the frontend can display them.
app.mount("/figures", StaticFiles(directory=str(FIGURE_DIR)), name="figures")


# ------------------------------------------------------------------ #
# Lazy singletons (models load once, on first use)
# ------------------------------------------------------------------ #

_retriever = None
_reranker = None
_generator_agent = None
_text_agent = None
_image_agent = None
_router_agent = None
_llm = None
_vlm = None
_store = None

_ingestion_status: dict = {}


def _get_retriever():
    global _retriever
    if _retriever is None:
        from ingestion.retriever import Retriever
        _retriever = Retriever()
    return _retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        from ingestion.reranker import Reranker
        _reranker = Reranker()
    return _reranker


def _get_store():
    # Lightweight: ChromaDB only, no embedding model — so /api/status and
    # /api/documents don't trigger a heavy model load. Shares the same DB path
    # as the retriever's store.
    global _store
    if _store is None:
        from ingestion.vector_store import VectorStore
        _store = VectorStore()
    return _store


def _get_llm():
    global _llm
    if _llm is None:
        from llm.ollama_client import OllamaClient
        _llm = OllamaClient()
    return _llm


def _get_vlm():
    global _vlm
    if _vlm is None:
        from llm.ollama_client import OllamaClient
        _vlm = OllamaClient(model="qwen2.5vl:3b")
    return _vlm


def _get_agents():
    global _text_agent, _image_agent, _generator_agent, _router_agent
    if _generator_agent is None:
        from agents.planner_agent import PlannerAgent
        from agents.text_agent import TextAgent
        from agents.image_agent import ImageAgent
        from agents.generator_agent import GeneratorAgent
        llm = _get_llm()
        # Planner Agent: entry point — classifies intent and decides which
        # downstream agents run (supersedes the old flat RouterAgent).
        _router_agent = PlannerAgent(llm=None)
        _text_agent = TextAgent(retriever=_get_retriever(), reranker=_get_reranker())
        _image_agent = ImageAgent()
        _generator_agent = GeneratorAgent(llm=llm)
    return _router_agent, _text_agent, _image_agent, _generator_agent


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20
    top_n: int = 5
    pdf_filter: Optional[str] = None


class SourceRef(BaseModel):
    pdf_name: str = ""
    pages: str = ""
    heading: str = ""
    section: str = ""
    text: str = ""
    rerank_score: float = 0.0


class FigureRef(BaseModel):
    figure_id: str = ""
    path: str = ""
    url: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    query_type: str = "text"
    sources: List[SourceRef] = []
    figures: List[FigureRef] = []
    image_paths: List[str] = []
    figure_ids: List[str] = []


class InterpretRequest(BaseModel):
    query: str = ""
    figure_path: str
    caption: str = ""


# ------------------------------------------------------------------ #
# Helpers (shared with ask.py)
# ------------------------------------------------------------------ #

_EXPLAIN_WORDS = {
    "explain", "describe", "what", "whats", "how", "why", "detail", "details",
    "elaborate", "tell", "define", "definition", "summarize", "summary",
    "overview", "compare", "difference", "meaning", "when", "where", "which",
}
_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _has_explain_intent(query: str) -> bool:
    toks = {t.strip(".,?!:;\"'").lower() for t in query.split()}
    return bool(toks & _EXPLAIN_WORDS)


def _caption_for(image_path: str, chunks) -> str:
    for c in chunks:
        if image_path in str(c.get("metadata", {}).get("image_paths", "")):
            m = c.get("metadata", {})
            return m.get("heading", "") or (c.get("child_text") or c.get("text", ""))[:120]
    return ""


def _vlm_prompt(query: str, caption: str) -> str:
    ctx = f' It is titled/about: "{caption}".' if caption else ""
    return (
        f'The user asked: "{query}".{ctx}\n'
        "In about 100 words, describe what this figure shows — its main components, "
        "labels, and how they relate. Write ONE short plain paragraph (no headings, "
        "no bullet lists). Focus only on the visual content of the image."
    )


def _filter_image_md(tokens):
    buf = ""
    for tok in tokens:
        buf = _IMG_MD.sub("", buf + tok)
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
    """Evict every Ollama model except keep_model so it gets the full 4 GB GPU."""
    try:
        loaded = requests.get(f"{base_url}/api/ps", timeout=5).json().get("models", [])
    except Exception:
        return
    for m in loaded:
        name = m.get("name") or m.get("model", "")
        if name and name != keep_model:
            try:
                requests.post(f"{base_url}/api/generate",
                              json={"model": name, "keep_alive": 0}, timeout=10)
            except Exception:
                pass


def _figure_url(path: str) -> str:
    try:
        rel = Path(path).resolve().relative_to(FIGURE_DIR.resolve())
        return f"/figures/{rel.as_posix()}"
    except Exception:
        return ""


def _figures_from(figure_ids: List[str], image_paths: List[str]) -> List[FigureRef]:
    figs: List[FigureRef] = []
    for i, p in enumerate(image_paths):
        fid = figure_ids[i] if i < len(figure_ids) else ""
        figs.append(FigureRef(figure_id=fid, path=p, url=_figure_url(p)))
    return figs


def _resolve_figure(path: str) -> Path:
    """Map a stored figure path (absolute or /figures-relative) to a real file."""
    p = Path(path)
    candidates = [p]
    rel = path.replace("/figures/", "").lstrip("/")
    candidates.append(FIGURE_DIR / rel)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise HTTPException(status_code=404, detail=f"figure not found: {path}")


def _build_sources(chunks) -> List[SourceRef]:
    out: List[SourceRef] = []
    for c in chunks:
        m = c.get("metadata", {})
        out.append(SourceRef(
            pdf_name=m.get("pdf_name", ""),
            pages=str(m.get("pages", "")),
            heading=m.get("heading", ""),
            section=m.get("section", ""),
            text=(c.get("text") or c.get("child_text") or ""),
            rerank_score=float(c.get("rerank_score", 0.0)),
        ))
    return out


def _run_retrieval(query: str, top_k: int, top_n: int, pdf_filter: Optional[str]):
    """Router + retrieve/rerank + figure collection — everything except the LLM."""
    from ingestion.retriever import Retriever

    planner, text_agent, image_agent, _ = _get_agents()
    plan = planner.plan(query)
    image_intent = plan.wants_image
    explain_intent = plan.wants_explain

    chunks = text_agent.run(
        query=query, top_k_retrieve=top_k, top_n_rerank=top_n, pdf_filter=pdf_filter,
    )

    figures: List[FigureRef] = []
    if chunks:
        n_top = 3 if image_intent else 2
        figure_ids, image_paths = image_agent.run(chunks[:n_top], max_images=2)
        figures = _figures_from(figure_ids, image_paths)

    has_image = bool(figures)
    pure_show = image_intent and not explain_intent and has_image
    run_text = not pure_show
    run_vlm = image_intent and has_image
    display_type = "image" if pure_show else ("both" if (run_text and run_vlm) else "text")
    return display_type, chunks, figures


# ------------------------------------------------------------------ #
# Query endpoints
# ------------------------------------------------------------------ #

@app.get("/")
def health():
    return {"status": "ok", "chunks": _get_store().total_chunks()}


@app.get("/api/status")
def status():
    store = _get_store()
    return {"status": "ok",
            "total_chunks": store.total_chunks(),
            "documents": store.list_pdf_names()}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    _, _, _, generator_agent = _get_agents()

    query_type, chunks, figures = _run_retrieval(
        req.query, req.top_k, req.top_n, req.pdf_filter
    )

    _ensure_only_loaded(_get_llm().model)
    answer = generator_agent.run(
        query=req.query,
        chunks=chunks,
        image_paths=[f.path for f in figures],
    )

    return QueryResponse(
        query=req.query,
        answer=answer,
        query_type=query_type,
        sources=_build_sources(chunks),
        figures=figures,
        image_paths=[f.path for f in figures],
        figure_ids=[f.figure_id for f in figures],
    )


@app.get("/api/query/stream")
def query_stream(
    query: str,
    top_k: int = 20,
    top_n: int = 5,
    pdf_filter: Optional[str] = None,
):
    """SSE stream mirroring ask.py's 3-way routing (one model on the GPU at a time)."""
    from ingestion.retriever import Retriever

    planner, text_agent, image_agent, generator_agent = _get_agents()

    plan = planner.plan(query)
    image_intent = plan.wants_image
    explain_intent = plan.wants_explain

    chunks = text_agent.run(
        query=query, top_k_retrieve=top_k, top_n_rerank=top_n, pdf_filter=pdf_filter,
    )

    figures: List[FigureRef] = []
    if chunks:
        n_top = 3 if image_intent else 2
        figure_ids, image_paths = image_agent.run(chunks[:n_top], max_images=2)
        figures = _figures_from(figure_ids, image_paths)
    has_image = bool(figures)

    pure_show = image_intent and not explain_intent and has_image
    run_text = not pure_show
    run_vlm = image_intent and has_image
    display_type = "image" if pure_show else ("both" if (run_text and run_vlm) else "text")

    def _sse() -> Iterator[str]:
        meta = {
            "query_type": display_type,
            "sources": jsonable_encoder(_build_sources(chunks)),
            "figures": jsonable_encoder(figures),
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # 1) TEXT answer (llama3.2) — skipped for a pure "show me the figure" query.
        if run_text:
            _ensure_only_loaded(_get_llm().model)
            stream = generator_agent.run_stream(
                query=query, chunks=chunks, image_paths=[f.path for f in figures],
            )
            for token in _filter_image_md(stream):
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"

        # 2) VLM figure description (qwen2.5vl) — for any image-intent query with a figure.
        if run_vlm:
            fig = figures[0]
            try:
                abs_path = str(_resolve_figure(fig.path))
            except HTTPException:
                abs_path = None
            if abs_path:
                caption = _caption_for(fig.path, chunks)
                _ensure_only_loaded(_get_vlm().model)
                yield f"event: vlm_start\ndata: {json.dumps({'figure_id': fig.figure_id, 'url': fig.url})}\n\n"
                for token in _get_vlm().vlm_stream(_vlm_prompt(query, caption), [abs_path]):
                    yield f"event: vlm_token\ndata: {json.dumps({'text': token})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/interpret_figure")
def interpret_figure(req: InterpretRequest):
    """Stream a vision-model (qwen2.5vl:3b) description of one figure."""
    full_path = _resolve_figure(req.figure_path)
    vlm = _get_vlm()

    def _stream() -> Iterator[str]:
        _ensure_only_loaded(vlm.model)
        yield from vlm.vlm_stream(_vlm_prompt(req.query, req.caption), [str(full_path)])

    return StreamingResponse(
        _stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ #
# Document management
# ------------------------------------------------------------------ #

@app.get("/api/documents")
def list_documents():
    store = _get_store()
    return {"documents": store.list_pdf_names(), "total_chunks": store.total_chunks()}


@app.delete("/api/documents/{pdf_name}")
def delete_document(pdf_name: str):
    store = _get_store()
    store.delete_pdf(pdf_name)
    if _retriever is not None:
        _retriever.invalidate_bm25()
    return {"deleted": pdf_name, "total_chunks": store.total_chunks()}


def _ingest_pdf(pdf_path: Path) -> None:
    """Full ingest for one PDF: Docling -> chunk -> contextualize -> embed -> store."""
    name = pdf_path.stem

    def _mark(msg: str):
        # "running" (not "processing") — the frontend's ingest guard checks for
        # queued/running, and shows `message`.
        _ingestion_status[name] = {"status": "running", "message": msg}

    try:
        _mark("Converting (Docling)…")
        from ingestion.docling_ingestor import DoclingIngestor
        from ingestion.semantic_chunking import SemanticDocumentChunker
        from ingestion.contextualizer import Contextualizer
        from ingestion.embedding_generator import EmbeddingGenerator

        _, elements = DoclingIngestor().process_pdf(pdf_path)
        _mark("Chunking…")
        chunks = SemanticDocumentChunker().chunk_document(name, elements)
        _mark("Contextualizing…")
        chunks = Contextualizer().contextualize(name, chunks)
        _mark("Embedding…")
        embedded = EmbeddingGenerator().embed_chunks(chunks)
        _mark("Storing…")
        store = _get_store()
        store.delete_pdf(name)                 # replace if re-ingesting
        store.store_chunks(name, embedded)
        if _retriever is not None:
            _retriever.invalidate_bm25()
        _ingestion_status[name] = {"status": "done", "message": f"{len(embedded)} chunks",
                                   "chunks": len(embedded)}
        logger.info("Ingested %s (%d chunks)", name, len(embedded))
    except Exception as exc:                    # pragma: no cover
        logger.exception("Ingest failed for %s", name)
        _ingestion_status[name] = {"status": "error", "message": str(exc), "error": str(exc)}


@app.post("/api/ingest")
async def ingest(background: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")
    dest = PDF_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    name = dest.stem
    _ingestion_status[name] = {"status": "queued", "message": "Queued…"}
    background.add_task(_ingest_pdf, dest)
    return {"pdf_name": name, "status": "queued"}


@app.get("/api/ingest/{pdf_name}")
def ingest_status(pdf_name: str):
    if pdf_name not in _ingestion_status:
        raise HTTPException(status_code=404, detail=f"No ingestion record for '{pdf_name}'")
    return _ingestion_status[pdf_name]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
