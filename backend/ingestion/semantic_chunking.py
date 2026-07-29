"""
semantic_chunking.py
Stage 7 — Heading-Aware Semantic Chunking

Rules:
  - A new heading always starts a new chunk.
  - Figures encountered inside a section are attached to that chunk.
  - Long sections are split at MAX_CHUNK_LENGTH character boundary.
  - Very short orphan paragraphs are merged with the preceding chunk.

Chunk metadata stored:
  chunk_id, pdf_name, text, pages, heading, section,
  figure_ids, image_paths
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# A "Unit N" / "Unit N.N" heading marks a hard topic boundary — force a new parent
# there so a new unit never shares a chunk-group with the previous topic.
_UNIT_RE = re.compile(r"(?i)^\s*unit\b")

MIN_CHUNK_LEN = 60
# Group consecutive (sub)sections until ~this size so a side-heading and its
# sub-topics stay together "in flow" (Docling gives flat headings, so we merge
# by size rather than by hierarchy level). Tunable via TARGET_CHUNK_LEN env.
TARGET_CHUNK_LEN = int(os.environ.get("TARGET_CHUNK_LEN", "1000"))
MAX_CHUNK_LEN = 1800        # hard cap: split a single long section beyond this
CHILD_MIN_LEN = 25          # smallest child worth embedding (small-to-big)
CHILD_SPLIT_LEN = 450       # split a child longer than this into idea-sized pieces
_MAX_TABLE_ROWS = 80        # cap row-children per table to avoid explosion
# A terse "spec line" carrying a number+unit (e.g. "ROT 950-1,000 F", "dP 5-8 psi")
# is high-value for number questions — keep it as its own child even if it's below
# CHILD_MIN_LEN, so the figure stays independently retrievable.
_NUM_UNIT = re.compile(
    r"\d[\d.,]*\s?(°|deg|psi|bar|wt\s?%|%|lb|kg|bpd|ton|scf|btu|:\s?1|f\b|c\b)", re.I)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_chunk(
    pdf_name: str,
    texts: List[str],
    pages: List[int],
    heading: str,
    section: str,
    figure_ids: List[str],
    image_paths: List[str],
    modality: str = "text",
    parent_id: Optional[str] = None,
    parent_text: Optional[str] = None,
) -> dict:
    text = " ".join(texts).strip()
    return {
        "chunk_id": str(uuid.uuid4()),
        "pdf_name": pdf_name,
        "text": text,
        "pages": sorted(set(pages)),
        "heading": heading,
        "section": section,
        "figure_ids": list(figure_ids),
        "image_paths": list(image_paths),
        # "text"  = normal prose chunk
        # "image" = a figure, retrievable via its caption/section context
        "modality": modality,
        # --- small-to-big (parent-document) ---
        # `text` (the CHILD) is what gets embedded/matched — small + precise.
        # `parent_text` is the larger enclosing section returned to the LLM.
        # Sibling children share a parent_id; standalone chunks are self-parented.
        "parent_id": parent_id or str(uuid.uuid4()),
        "parent_text": parent_text if parent_text is not None else text,
    }


def _figure_text(caption: str, heading: str, section: str,
                 pdf_name: str, page: int, context: str = "") -> str:
    """Text representation of a figure for text-anchored image retrieval.

    Captions and the enclosing heading describe a scanned figure far more
    reliably than CLIP pixel features, so we embed that text in the same
    BGE space as prose chunks.  When the caption is thin (common with scanned
    notes), the surrounding section prose is appended so the figure is still
    findable by topic.
    """
    parts = [p.strip() for p in (caption, heading, section) if p and p.strip()]
    text = ". ".join(dict.fromkeys(parts))  # dedupe while preserving order
    # Add nearby section prose for context (capped to keep the anchor focused).
    ctx = " ".join(context.split())[:400].strip()
    if ctx:
        text = f"{text}. {ctx}" if text else ctx
    if len(text) < 3:
        text = f"Figure from {pdf_name}, page {page}"
    return f"Figure: {text}"


def _parse_md_table(md: str):
    """Parse a Docling markdown pipe-table into (header, data_rows).

    Returns (None, None) when the content isn't a parseable pipe table, so the
    caller can fall back to a single whole-table chunk.
    """
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return None, None

    def cells(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    sep = lines[1]
    # A markdown separator row is only |, -, :, space -> nothing else.
    is_sep = set(sep.replace("|", "").replace(":", "").replace("-", "").strip()) == set()
    data_lines = lines[2:] if is_sep else lines[1:]
    data = [cells(l) for l in data_lines]
    data = [r for r in data if any(c for c in r)]          # drop blank rows
    if not header or not data:
        return None, None
    return header, data


# ------------------------------------------------------------------ #
# Chunker
# ------------------------------------------------------------------ #

class SemanticDocumentChunker:

    def __init__(self):
        base = Path(__file__).resolve().parent.parent
        self.document_dir = base / "data" / "documents"
        self.output_dir = base / "data" / "chunks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Semantic (embedding-based) splitting of long children. RAG_SEMANTIC_CHUNK=0
        # falls back to paragraph/char splitting. Embedder loaded lazily on 1st use.
        self._splitter = None
        self._semantic = os.environ.get("RAG_SEMANTIC_CHUNK", "1") != "0"

    def _get_splitter(self):
        if self._splitter is None and self._semantic:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                from ingestion.embedding_generator import EMBED_MODEL
                from ingestion.semantic_splitter import SemanticSplitter
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._splitter = SemanticSplitter(
                    SentenceTransformer(EMBED_MODEL, device=dev), max_chars=CHILD_SPLIT_LEN)
                logger.info("Semantic chunking ON (embedder on %s)", dev)
            except Exception as e:                     # pragma: no cover
                logger.warning("Semantic chunking unavailable (%s) — paragraph fallback", e)
                self._semantic = False
        return self._splitter

    # ---------------------------------------------------------------- #

    def chunk_document(
        self,
        pdf_name: str,
        elements: list,     # List[DocumentElement]
    ) -> List[dict]:
        chunks: List[dict] = []

        current_section = ""
        current_heading = ""              # tracked for figure/table anchoring
        # A "group" is a PARENT section; each entry in `children` is a CHILD
        # (a heading + its following paragraphs). Children are embedded for a
        # precise match; the whole group is the parent returned to the LLM.
        children: List[dict] = []         # [{"texts": [], "pages": [], "heading": ""}]
        last_caption = ""

        def group_len() -> int:
            return sum(len(t) for c in children for t in c["texts"])

        def start_child(heading_text: str) -> None:
            children.append({
                "texts": [heading_text] if heading_text else [],
                "pages": [],
                "heading": heading_text,
            })

        def cur_child() -> dict:
            if not children:
                start_child("")
            return children[-1]

        def flush_group() -> None:
            nonlocal children
            if not children:
                return
            segments = [t for c in children for t in c["texts"]]
            parent_text = " ".join(segments).strip()
            all_pages = sorted({p for c in children for p in c["pages"]})
            if len(parent_text) >= MIN_CHUNK_LEN:
                parent_id = str(uuid.uuid4())
                primary = next((c["heading"] for c in children if c["heading"]), "")
                for c in children:
                    head = c["heading"] or primary
                    ctext = " ".join(c["texts"]).strip()
                    if len(ctext) < CHILD_MIN_LEN and not _NUM_UNIT.search(ctext):
                        continue            # too small to embed; still in parent_text
                    # De-dilute: a long child is split into idea-sized pieces
                    # (<= CHILD_SPLIT_LEN) at paragraph boundaries, so a query
                    # matches a precise point instead of a whole section. Short
                    # children stay whole. The full section is still parent_text.
                    if len(ctext) <= CHILD_SPLIT_LEN:
                        pieces = [ctext]
                    else:
                        sp = self._get_splitter()
                        if sp is not None:
                            pieces = sp.split(ctext)        # split at MEANING shifts
                        else:
                            paras = [t.strip() for t in c["texts"]
                                     if t and t.strip() and t.strip() != c["heading"]]
                            pieces, buf = [], ""
                            for p in paras:
                                if buf and len(buf) + len(p) + 1 > CHILD_SPLIT_LEN:
                                    pieces.append(buf); buf = p
                                else:
                                    buf = f"{buf} {p}".strip()
                            if buf:
                                pieces.append(buf)
                        if not pieces:
                            pieces = [ctext]
                        elif head:                      # keep the heading on each piece
                            pieces = [pc if pc.lower().startswith(head.lower())
                                      else f"{head}. {pc}" for pc in pieces]
                    for pc in pieces:
                        chunks.append(_make_chunk(
                            pdf_name=pdf_name,
                            texts=[pc],
                            pages=c["pages"] or all_pages,
                            heading=head,
                            section=current_section,
                            figure_ids=[],
                            image_paths=[],
                            modality="text",
                            parent_id=parent_id,
                            parent_text=parent_text,
                        ))
            children = []

        for elem in elements:
            etype = elem.type if hasattr(elem, "type") else elem["type"]
            econtent = elem.content if hasattr(elem, "content") else elem["content"]
            epage = elem.page if hasattr(elem, "page") else elem["page"]
            esection = elem.section if hasattr(elem, "section") else elem["section"]
            efig_id = elem.figure_id if hasattr(elem, "figure_id") else elem.get("figure_id", "")
            eimg_path = elem.image_path if hasattr(elem, "image_path") else elem.get("image_path", "")

            if etype == "heading":
                # Start a NEW parent once the group reaches TARGET, OR at a hard
                # topic boundary ("Unit N ..."); otherwise this heading becomes
                # another CHILD of the same parent ("in flow").
                unit_boundary = bool(_UNIT_RE.match(econtent or ""))
                if children and (group_len() >= TARGET_CHUNK_LEN or unit_boundary):
                    flush_group()
                current_section = esection or current_section
                current_heading = econtent
                start_child(econtent)
                cur_child()["pages"].append(epage)

            elif etype in ("paragraph", "caption"):
                if etype == "caption":
                    last_caption = econtent      # anchor for the next figure
                c = cur_child()
                c["texts"].append(econtent)
                c["pages"].append(epage)
                if group_len() >= MAX_CHUNK_LEN:  # safety: bound parent size
                    flush_group()

            elif etype == "figure":
                # Standalone, self-parented image-chunk anchored on caption +
                # heading + nearby section prose (independently retrievable).
                if efig_id or eimg_path:
                    # Prefer the figure's OWN caption (anchored per-page by the
                    # ingestor) so the image is retrieved by its own text, not an
                    # adjacent figure's caption. Fall back to the last caption seen.
                    fig_caption = (econtent or "").strip() or last_caption
                    img_text = _figure_text(
                        fig_caption, current_heading, current_section,
                        pdf_name, epage,
                        # Anchor on THIS heading's prose only — not the whole
                        # accumulated group, which can span a previous topic and
                        # pollute the figure (e.g. Cassandra text glued onto an
                        # HBase figure). Keeps the figure on its own topic.
                        context=" ".join(cur_child()["texts"]),
                    )
                    chunks.append(_make_chunk(
                        pdf_name=pdf_name,
                        texts=[img_text],
                        pages=[epage],
                        heading=current_heading,
                        section=current_section,
                        figure_ids=[efig_id] if efig_id else [],
                        image_paths=[eimg_path] if eimg_path else [],
                        modality="image",
                    ))
                    last_caption = ""

            elif etype == "table":
                # Row-level chunking (small-to-big): each ROW becomes a precise,
                # header-labelled CHILD that embeds distinctly (so an exact value
                # lookup matches it), while the WHOLE table stays as parent_text
                # returned to the LLM. A "columns" summary child answers
                # "what does table X show". Falls back to one whole-table chunk if
                # the markdown isn't a parseable pipe table.
                flush_group()
                table_md = econtent.strip()
                if table_md:
                    anchor = ". ".join(dict.fromkeys(
                        p for p in (current_heading, current_section) if p))
                    parent_text = (f"Table: {anchor}\n{table_md}" if anchor
                                   else f"Table:\n{table_md}")
                    header, rows = _parse_md_table(table_md)
                    if header and rows:
                        parent_id = str(uuid.uuid4())
                        head_line = " | ".join(h for h in header if h)
                        chunks.append(_make_chunk(
                            pdf_name=pdf_name,
                            texts=[f"Table: {anchor}. Columns: {head_line}" if anchor
                                   else f"Table columns: {head_line}"],
                            pages=[epage], heading=current_heading,
                            section=current_section, figure_ids=[], image_paths=[],
                            modality="text", parent_id=parent_id, parent_text=parent_text,
                        ))
                        for row in rows[:_MAX_TABLE_ROWS]:
                            pairs = ", ".join(
                                f"{h}: {v}" for h, v in zip(header, row) if v.strip()
                            ) or " | ".join(v for v in row if v.strip())
                            if not pairs:
                                continue
                            rtext = (f"Table: {anchor}. {pairs}" if anchor
                                     else f"Table row. {pairs}")
                            chunks.append(_make_chunk(
                                pdf_name=pdf_name, texts=[rtext], pages=[epage],
                                heading=current_heading, section=current_section,
                                figure_ids=[], image_paths=[], modality="text",
                                parent_id=parent_id, parent_text=parent_text,
                            ))
                    else:
                        chunks.append(_make_chunk(
                            pdf_name=pdf_name, texts=[parent_text], pages=[epage],
                            heading=current_heading, section=current_section,
                            figure_ids=[], image_paths=[], modality="text",
                        ))

        flush_group()
        return chunks

    # ---------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------- #

    def save_chunks(self, pdf_name: str, chunks: List[dict]) -> Path:
        folder = self.output_dir / pdf_name
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / "chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        return out_path

    def load_chunks(self, pdf_name: str) -> List[dict]:
        path = self.output_dir / pdf_name / "chunks.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_document(self, pdf_name: str) -> list:
        from ingestion.document_builder import DocumentElement
        path = self.document_dir / pdf_name / "document.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [DocumentElement(**d) for d in data]

    # ---------------------------------------------------------------- #
    # Pipeline entry points
    # ---------------------------------------------------------------- #

    def process_pdf(
        self,
        pdf_name: str,
        elements: Optional[list] = None,
    ) -> List[dict]:
        print(f"Semantic chunking: {pdf_name}")
        if elements is None:
            elements = self.load_document(pdf_name)
        chunks = self.chunk_document(pdf_name, elements)
        self.save_chunks(pdf_name, chunks)
        print(f"  {len(chunks)} chunks created")
        return chunks

    def process_all_pdfs(
        self,
        documents: Optional[Dict[str, list]] = None,
    ) -> Dict[str, List[dict]]:
        if documents:
            return {
                pdf: self.process_pdf(pdf, elems)
                for pdf, elems in documents.items()
            }
        # Load from disk
        all_chunks: Dict[str, List[dict]] = {}
        if not self.document_dir.exists():
            return all_chunks
        for folder in self.document_dir.iterdir():
            if folder.is_dir():
                all_chunks[folder.name] = self.process_pdf(folder.name)
        return all_chunks
