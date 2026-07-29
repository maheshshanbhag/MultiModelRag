"""
docling_ingestor.py
Stages 1-5 replacement — single-pass ingestion via Docling.

One `DocumentConverter.convert()` call performs, internally:
    PDF render  ->  layout detection  ->  OCR (selective)
                ->  reading order     ->  table structure
                ->  figure extraction

We then map the resulting ``DoclingDocument`` onto the project's existing
``DocumentElement`` contract (heading / paragraph / caption / table / figure)
and save it to ``data/documents/<pdf>/document.json`` so the unchanged
downstream stages (semantic chunking -> embedding -> vector store) keep working.

Design notes
------------
* MIXED digital + scanned handling: ``force_full_page_ocr=False`` means the
  PDF's native text layer is read where present and OCR runs only on the
  scanned/bitmap regions of a page.  No more "convert the whole doc to scanned".
* OCR engine: Tesseract via CLI (``DOCLING_OCR_ENGINE=tesseract``, the default;
  ``easyocr`` also available). CPU-only, fully offline — needs the system
  ``tesseract`` binary + ``eng.traineddata``. RapidOCR/PaddleOCR (Chinese-origin
  PP-OCR models) were removed. Which pages get OCR at all is decided per-document/
  page by ``_wants_ocr`` and the per-page split — the engine is a drop-in swap.
* Page headers / footers carry Docling ``page_header`` / ``page_footer`` labels
  and are dropped here — this replaces the old cross-page header/footer denoise.
* Tables (new capability) are exported as Markdown and flow through as their own
  elements; figure images are exported to ``data/figures/<pdf>/page_<n>/``,
  matching the path layout the rest of the pipeline already expects.

Offline
-------
Set ``DOCLING_OFFLINE=1`` once the models are prefetched
(``docling-tools models download`` + one online OCR run to cache RapidOCR
models).  That pins HuggingFace into offline mode so no network is touched.
"""

from __future__ import annotations

import os
import re

# Offline switch must be set BEFORE importing docling / huggingface_hub.
if os.environ.get("DOCLING_OFFLINE", "0") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from ingestion.document_builder import DocumentBuilder, DocumentElement

logger = logging.getLogger(__name__)

_CUDA_DLLS_ADDED = False


def _add_cuda_dll_dirs() -> None:
    """Make ``onnxruntime-gpu`` find the CUDA 12 / cuDNN 9 DLLs that torch already
    bundles (``cublasLt64_12.dll``, ``cudnn64_9.dll`` …), plus any ``nvidia-*-cu12``
    wheels, so RapidOCR's CUDA execution provider can load.  Idempotent and
    Windows-only (a no-op elsewhere, where the loader finds them via RPATH)."""
    global _CUDA_DLLS_ADDED
    if _CUDA_DLLS_ADDED or os.name != "nt":
        return
    dirs: List[Path] = []
    try:
        import torch
        dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:                                # pragma: no cover
        pass
    try:
        import nvidia
        for base in list(getattr(nvidia, "__path__", [])):
            dirs.extend(Path(base).glob("*/bin"))
    except Exception:                                # pragma: no cover
        pass
    for d in dirs:
        try:
            if d.is_dir():
                os.add_dll_directory(str(d))
        except Exception:                            # pragma: no cover
            pass
    _CUDA_DLLS_ADDED = True


# ------------------------------------------------------------------ #
# Docling label -> DocumentElement.type mapping
# ------------------------------------------------------------------ #
_HEADING_LABELS = {"title", "section_header"}
_CAPTION_LABELS = {"caption"}
_TABLE_LABELS = {"table"}
_FIGURE_LABELS = {"picture", "chart"}
# Page furniture we deliberately discard (replaces header/footer denoise).
_SKIP_LABELS = {"page_header", "page_footer", "marker", "empty_value",
                "checkbox_selected", "checkbox_unselected", "form",
                "key_value_region"}
# Everything else textual (text, paragraph, list_item, footnote, code,
# formula, reference, document_index, ...) becomes a paragraph.

# A "Figure 3", "Fig. 1-4", "Table 2" … caption label. Docling often emits these
# as plain paragraphs (not caption elements) split above/below the image, so we
# anchor figures to them by page-proximity rather than "last caption seen".
_FIG_LABEL_RE = re.compile(
    r"(?i)\b(?:figure|fig|photo|exhibit|chart|diagram|scheme|plate|table)\.?\s*\d"
)


class DoclingIngestor:
    """Convert PDFs to ``DocumentElement`` lists using Docling."""

    def __init__(self, use_gpu: bool = None, images_scale: float = 2.0):
        # NOTE: the torch models (layout + TableFormer) move to CUDA with the GPU
        # converter. RapidOCR is DECOUPLED: it stays on CPU unless DOCLING_OCR_GPU=1
        # (requires onnxruntime-gpu; it then loads the CUDA 12 / cuDNN 9 DLLs torch
        # already bundles — see _add_cuda_dll_dirs). Gating it separately keeps the
        # 4GB VRAM budget under control on the shared card.
        self.base = Path(__file__).resolve().parent.parent
        self.pdf_dir = self.base / "uploads" / "pdfs"
        self.figure_dir = self.base / "data" / "figures"
        self.images_scale = images_scale
        # ONE page on the GPU at a time -> peak VRAM is ~constant regardless of
        # the document's total page count (33-page scanned doc peaked ~0.83 GB).
        try:
            from docling.datamodel.settings import settings
            settings.perf.page_batch_size = int(os.environ.get("DOCLING_PAGE_BATCH", "1"))
        except Exception as e:                       # pragma: no cover
            logger.warning("Could not set page_batch_size: %s", e)
        self._builder = DocumentBuilder()      # reused only for save_document()
        # Device PREFERENCE (explicit arg > DOCLING_GPU env > auto CUDA). The
        # actual per-document device is decided in _convert_with_fallback, which
        # drops to CPU when free VRAM is low or a GPU pass OOMs / drops pages.
        self._gpu_pref = use_gpu if use_gpu is not None else self._resolve_gpu()
        self._converters: dict = {}            # (device, do_ocr) -> cached converter
        # Warm the preferred (device, OCR-on) converter; the OCR-off variant is
        # built lazily the first time a born-digital PDF is ingested.
        self.converter = self._get_converter(self._gpu_pref, True)

    # ---------------------------------------------------------------- #
    # Converter configuration
    # ---------------------------------------------------------------- #

    @staticmethod
    def _resolve_gpu() -> bool:
        # GPU-first when CUDA is present (measured safe on this 4GB RTX 3050:
        # layout+TableFormer peak ~0.83 GB, no dropped pages, ~2.4x faster on a
        # digital PDF). No LLM is loaded during ingestion, so the card is free,
        # and _convert_with_fallback drops to CPU if VRAM is ever tight.
        #   DOCLING_GPU=1 -> force GPU,  DOCLING_GPU=0 -> force CPU,  unset -> auto.
        val = os.environ.get("DOCLING_GPU", "").strip().lower()
        if val in ("1", "true", "yes"):
            return True
        if val in ("0", "false", "no"):
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _ocr_gpu_pref() -> bool:
        """OCR-on-GPU is OPT-IN via DOCLING_OCR_GPU=1 (default off). It only takes
        effect on the GPU converter; the CPU fallback always keeps OCR on CPU."""
        return os.environ.get("DOCLING_OCR_GPU", "").strip().lower() in (
            "1", "true", "yes",
        )

    @staticmethod
    def _ocr_engine() -> str:
        """OCR engine: 'tesseract' (default) | 'easyocr'. RapidOCR/PaddleOCR (the
        Chinese-origin PP-OCR models) were dropped. Override via DOCLING_OCR_ENGINE."""
        val = os.environ.get("DOCLING_OCR_ENGINE", "tesseract").strip().lower()
        return "easyocr" if val in ("easyocr", "easy") else "tesseract"

    def _ocr_options(self, use_gpu: bool):
        """Build Docling OCR options for the selected engine, keeping OCR selective
        (``force_full_page_ocr=False``: native text where present, OCR only on
        bitmap regions). Tesseract is CPU-only; EasyOCR can use CUDA when
        DOCLING_OCR_GPU=1 and this is the GPU converter."""
        if self._ocr_engine() == "easyocr":
            from docling.datamodel.pipeline_options import EasyOcrOptions
            ocr_on_gpu = bool(use_gpu) and self._ocr_gpu_pref()
            if ocr_on_gpu:
                _add_cuda_dll_dirs()
            return EasyOcrOptions(
                force_full_page_ocr=False, lang=["en"], use_gpu=ocr_on_gpu,
            )
        # default: Tesseract via CLI (needs the tesseract binary + eng.traineddata).
        # lang uses 3-letter codes ("eng"). CPU-only, fully offline.
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions
        return TesseractCliOcrOptions(
            force_full_page_ocr=False,
            lang=os.environ.get("DOCLING_TESSERACT_LANG", "eng").split("+"),
            tesseract_cmd=os.environ.get("DOCLING_TESSERACT_CMD", "tesseract"),
        )

    def _get_converter(self, use_gpu: bool, do_ocr: bool):
        """Lazily build + cache one DocumentConverter per (device, do_ocr) combo so
        models load at most once each. Born-digital PDFs use the OCR-off converter
        (skips the ~1.2s/page OCR stage), scanned PDFs the OCR-on one."""
        key = (use_gpu, do_ocr)
        if key not in self._converters:
            self._converters[key] = self._build_converter(use_gpu, do_ocr)
        return self._converters[key]

    def _build_converter(self, use_gpu: bool, do_ocr: bool = True) -> DocumentConverter:
        opts = PdfPipelineOptions()

        # --- OCR engine: Tesseract (default) — see _ocr_options. force_full_page_
        # ocr=False keeps OCR selective (native text layer where present, OCR only
        # on bitmap regions). Which page-subset gets OCR at all is decided upstream
        # per-document/page by _wants_ocr / the split; do_ocr toggles this converter.
        opts.do_ocr = do_ocr
        opts.ocr_options = self._ocr_options(bool(use_gpu))
        logger.info("OCR engine: %s (do_ocr=%s)", self._ocr_engine(), do_ocr)

        # --- Tables: FAST mode by default (much quicker on CPU). Set
        #     DOCLING_TABLE_MODE=accurate for higher fidelity on complex tables. ---
        opts.do_table_structure = True
        try:
            opts.table_structure_options.do_cell_matching = True
            from docling.datamodel.pipeline_options import TableFormerMode
            _tmode = os.environ.get("DOCLING_TABLE_MODE", "fast").strip().lower()
            opts.table_structure_options.mode = (
                TableFormerMode.ACCURATE if _tmode == "accurate" else TableFormerMode.FAST
            )
        except Exception as e:                       # pragma: no cover
            logger.warning("Could not set table mode: %s", e)

        # --- Figures: keep the cropped images for the image-chunk anchors ---
        opts.generate_picture_images = True
        opts.images_scale = self.images_scale

        # --- Device: this sets the accelerator for layout + TableFormer. Opt into
        #     GPU for them via DOCLING_GPU=1. OCR's device is handled separately
        #     above (DOCLING_OCR_GPU), so it does NOT follow this flag by default. ---
        # num_threads flows through to the onnxruntime OCR engine's
        # intra_op_num_threads, so on scanned docs (where OCR runs on the CPU and
        # dominates the wall clock) this is the single biggest ingestion-speed
        # lever. Docling defaults to 4; we default to the machine's core count.
        # Override with DOCLING_NUM_THREADS.
        try:
            from docling.datamodel.pipeline_options import (
                AcceleratorOptions, AcceleratorDevice,
            )
            device = AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
            n_threads = int(os.environ.get("DOCLING_NUM_THREADS", os.cpu_count() or 4))
            opts.accelerator_options = AcceleratorOptions(
                device=device, num_threads=n_threads,
            )
        except Exception as e:                       # pragma: no cover
            logger.warning("Could not set accelerator options: %s", e)

        # Offline model resolution: explicit DOCLING_ARTIFACTS_PATH > local
        # backend/model/docling > the prefetched ~/.cache/docling/models.
        artifacts = os.environ.get("DOCLING_ARTIFACTS_PATH")
        if not artifacts:
            local = self.base / "model" / "docling"
            default_cache = Path.home() / ".cache" / "docling" / "models"
            if local.is_dir():
                artifacts = str(local)
            elif default_cache.is_dir():
                artifacts = str(default_cache)
        if artifacts:
            opts.artifacts_path = artifacts

        # PDF backend: pypdfium2 instead of the default docling-parse. The C++
        # docling-parse backend throws std::bad_alloc partway through longer docs
        # (BDA failed at page 28, dropping pages 28-33 on BOTH CPU and GPU);
        # pypdfium2 processes all pages cleanly with equal text/table/figure quality.
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=opts,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    # ---------------------------------------------------------------- #
    # DoclingDocument -> List[DocumentElement]
    # ---------------------------------------------------------------- #

    def _bbox_ints(self, prov) -> List[int]:
        if not prov:
            return [0, 0, 0, 0]
        b = prov[0].bbox
        try:
            return [int(b.l), int(b.t), int(b.r), int(b.b)]
        except Exception:
            return [0, 0, 0, 0]

    def _page_no(self, prov, fallback: int) -> int:
        if prov and getattr(prov[0], "page_no", None) is not None:
            return int(prov[0].page_no)
        return fallback

    def _save_figure(self, doc, item, pdf_name: str, page: int, idx: int) -> str:
        """Export a picture/chart image and return its saved path (or "")."""
        try:
            img = item.get_image(doc)
        except Exception:
            img = None
        if img is None:
            return ""
        page_dir = self.figure_dir / pdf_name / f"page_{page}"
        page_dir.mkdir(parents=True, exist_ok=True)
        out = page_dir / f"figure_{idx}.png"
        try:
            img.save(out)
        except Exception as e:                       # pragma: no cover
            logger.warning("Failed to save figure %s: %s", out, e)
            return ""
        return str(out)

    def build_elements(self, doc, pdf_name: str, page_map: Dict[int, int] = None
                       ) -> List[DocumentElement]:
        """Map a DoclingDocument onto DocumentElements. ``page_map`` (sub-PDF page
        -> original page, both 1-based) is supplied by the per-page OCR split so
        emitted page numbers and figure directories use the ORIGINAL page index."""
        elements: List[DocumentElement] = []
        order = 0
        page_cur = 1
        current_title = ""
        current_heading = ""
        fig_counter: Dict[int, int] = {}

        for item, _level in doc.iterate_items():
            label = getattr(item, "label", None)
            label = getattr(label, "value", label)   # enum -> str
            if label is None or label in _SKIP_LABELS:
                continue

            prov = getattr(item, "prov", None)
            raw_page = self._page_no(prov, page_cur)
            page_cur = raw_page                      # fallback tracks sub-PDF pages
            page = page_map.get(raw_page, raw_page) if page_map else raw_page
            bbox = self._bbox_ints(prov)

            def add(etype, content, image_path="", figure_id="", heading_v="",
                    section_v=""):
                nonlocal order
                elements.append(DocumentElement(
                    id=f"{etype}_{page}_{order}",
                    page=page, order=order, type=etype, bbox=bbox,
                    content=content or "", image_path=image_path,
                    section=section_v, heading=heading_v, figure_id=figure_id,
                ))
                order += 1

            if label in _HEADING_LABELS:
                text = (getattr(item, "text", "") or "").strip()
                if not text:
                    continue
                if label == "title":
                    current_title = text
                current_heading = text
                add("heading", text, heading_v=text, section_v=current_title)

            elif label in _CAPTION_LABELS:
                text = (getattr(item, "text", "") or "").strip()
                if text:
                    add("caption", text, heading_v=current_heading,
                        section_v=current_title)

            elif label in _TABLE_LABELS:
                try:
                    md = item.export_to_markdown(doc).strip()
                except Exception as e:               # pragma: no cover
                    logger.warning("Table export failed p%s: %s", page, e)
                    md = ""
                if md:
                    add("table", md, heading_v=current_heading,
                        section_v=current_title)

            elif label in _FIGURE_LABELS:
                idx = fig_counter.get(page, 0) + 1
                fig_counter[page] = idx
                img_path = self._save_figure(doc, item, pdf_name, page, idx)
                # Emit the picture's caption (if any) right BEFORE the figure so
                # the chunker's last_caption pairing anchors the image-chunk.
                try:
                    cap = (item.caption_text(doc) or "").strip()
                except Exception:
                    cap = ""
                if cap:
                    add("caption", cap, heading_v=current_heading,
                        section_v=current_title)
                fig_id = f"figure_{page}_{idx}"
                add("figure", "", image_path=img_path, figure_id=fig_id,
                    heading_v=current_heading, section_v=current_title)

            else:  # all other textual labels -> paragraph
                text = (getattr(item, "text", "") or "").strip()
                if text:
                    add("paragraph", text, heading_v=current_heading,
                        section_v=current_title)

        self._anchor_figure_captions(elements)
        return elements

    @staticmethod
    def _anchor_figure_captions(elements: List[DocumentElement]) -> None:
        """Set each figure's ``content`` to the caption text nearest to it ON THE
        SAME PAGE, so the chunker anchors the image to ITS OWN caption.

        Docling frequently emits figure captions as plain paragraphs (not caption
        elements) and splits the "Figure N" label from the description above/below
        the image; the chunker's old "last caption seen" rule then paired a figure
        with an adjacent figure's caption (retrieval returned the *next* image). We
        instead pick, within a small same-page window: the closest "Figure N" label
        + the nearest substantial sentence (fragments filtered). In place."""
        WIN = 6
        for i, e in enumerate(elements):
            if e.type != "figure" or e.content:
                continue
            page = e.page
            label, label_dist = "", 99
            descs = []
            lo, hi = max(0, i - WIN), min(len(elements), i + WIN + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                f = elements[j]
                if f.page != page or f.type not in ("paragraph", "caption"):
                    continue
                txt = (f.content or "").strip()
                if _FIG_LABEL_RE.search(txt):
                    if abs(j - i) < label_dist:
                        label, label_dist = txt, abs(j - i)
                elif len(txt) >= 12 and len(txt.split()) >= 2:  # a real sentence
                    descs.append((abs(j - i), -len(txt), txt))
            descs.sort()
            desc = descs[0][2] if descs else ""
            cap = " ".join(p for p in (label, desc) if p).strip()
            if cap:
                e.content = cap

    # ---------------------------------------------------------------- #
    # Public entry points
    # ---------------------------------------------------------------- #

    # ---------------------------------------------------------------- #
    # GPU -> CPU fallback
    # ---------------------------------------------------------------- #

    @staticmethod
    def _min_free_vram_gb() -> float:
        return float(os.environ.get("DOCLING_GPU_MIN_FREE_GB", "1.2"))

    @classmethod
    def _enough_vram(cls) -> bool:
        """True if the GPU has enough free VRAM to run a Docling pass. Guards
        against starting on GPU while an Ollama model (llama/qwen) is resident."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9 >= cls._min_free_vram_gb()
        except Exception:
            return False

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        if exc.__class__.__name__ == "OutOfMemoryError":   # torch.cuda.OutOfMemoryError
            return True
        m = str(exc).lower()
        return (
            "out of memory" in m or "bad_alloc" in m
            or ("cuda" in m and "memory" in m) or "dropped pages" in m
        )

    @staticmethod
    def _free_cuda() -> None:
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _pdf_page_count(self, pdf_path: Path) -> int:
        """PDF's true page count (pypdfium2, already a backend dep) so we can
        detect a GPU pass that silently dropped pages. 0 if it can't be read."""
        try:
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(str(pdf_path))
            try:
                return len(doc)
            finally:
                doc.close()
        except Exception:
            return 0

    def _scanned_page_set(self, pdf_path: Path) -> Tuple[List[int], int]:
        """Classify every page via pypdfium2 (metadata only — no rendering, fast):
        a page with < DOCLING_OCR_MIN_CHARS of text AND >=1 image object is
        'scanned' (needs OCR); with enough text it's 'digital'; with neither it's
        'blank' (ignored). Returns (sorted 0-based scanned page indices, total).
        On failure returns ([], 0) so callers fall back to the safe OCR-on path."""
        min_chars = int(os.environ.get("DOCLING_OCR_MIN_CHARS", "50"))
        scanned: List[int] = []
        total = 0
        try:
            import pypdfium2 as pdfium
            import pypdfium2.raw as pr
            doc = pdfium.PdfDocument(str(pdf_path))
            try:
                total = len(doc)
                for i in range(total):
                    page = doc[i]
                    tp = page.get_textpage()
                    try:
                        chars = len(tp.get_text_range().strip())
                    finally:
                        tp.close()
                    if chars >= min_chars:
                        continue                          # digital page
                    if any(o.type == pr.FPDF_PAGEOBJ_IMAGE
                           for o in page.get_objects()):
                        scanned.append(i)                 # image + ~no text => scanned
                    # else: blank page — ignored (won't force OCR on)
            finally:
                doc.close()
        except Exception as e:                           # pragma: no cover
            logger.warning("Page classification failed (%s).", e)
            return [], 0
        return scanned, total

    def _wants_ocr(self, pdf_path: Path) -> bool:
        """Document-level OCR decision (used when NOT splitting). DOCLING_DO_OCR
        overrides: '1'/'on' force, '0'/'off' disable, 'auto' (default) -> OCR on
        iff any page is scanned."""
        force = os.environ.get("DOCLING_DO_OCR", "auto").strip().lower()
        if force in ("1", "on", "true", "yes"):
            return True
        if force in ("0", "off", "false", "no"):
            return False
        scanned, total = self._scanned_page_set(pdf_path)
        if total == 0:
            return True                                  # unreadable -> safe: OCR on
        logger.info("%s: %d/%d pages scanned -> OCR %s", pdf_path.stem,
                    len(scanned), total, "on" if scanned else "off")
        return len(scanned) > 0

    def _convert_with_fallback(self, pdf_path: Path, do_ocr: bool = None):
        """Convert on GPU when preferred and VRAM allows; fall back to CPU on
        low VRAM, an OOM, or a page-dropping GPU pass. ``do_ocr`` is decided per
        document by _wants_ocr when not given (the split path passes it explicitly
        per page-subset)."""
        expected = self._pdf_page_count(pdf_path)
        if do_ocr is None:
            do_ocr = self._wants_ocr(pdf_path)
        want_gpu = self._gpu_pref

        if want_gpu and not self._enough_vram():
            logger.warning(
                "Free VRAM < %.1f GB — ingesting %s on CPU instead of GPU.",
                self._min_free_vram_gb(), pdf_path.stem,
            )
            want_gpu = False

        if want_gpu:
            try:
                result = self._get_converter(True, do_ocr).convert(str(pdf_path))
                got = len(result.document.pages)
                if expected and got < expected:
                    raise RuntimeError(f"GPU dropped pages ({got}/{expected})")
                logger.info("Ingested %s on GPU (%d pages).", pdf_path.stem, got)
                return result
            except Exception as exc:
                if not self._is_oom(exc):
                    raise                         # a real error — don't mask it
                logger.warning("GPU pass failed (%s) — retrying %s on CPU.",
                               exc, pdf_path.stem)
                self._free_cuda()

        return self._get_converter(False, do_ocr).convert(str(pdf_path))

    # ---------------------------------------------------------------- #
    # Per-page OCR split (mixed docs)
    # ---------------------------------------------------------------- #

    @staticmethod
    def _split_enabled() -> bool:
        return os.environ.get("DOCLING_OCR_SPLIT", "1").strip().lower() in (
            "1", "on", "true", "yes",
        )

    def _make_subset_pdf(self, pdf_path: Path, orig_indices: List[int]
                         ) -> Tuple[Path, Dict[int, int]]:
        """Write a temp PDF containing ``orig_indices`` (0-based) in order. Returns
        (temp path, page_map: sub-PDF page -> original page, both 1-based)."""
        import tempfile
        import pypdfium2 as pdfium
        src = pdfium.PdfDocument(str(pdf_path))
        try:
            dst = pdfium.PdfDocument.new()
            dst.import_pages(src, pages=list(orig_indices))
            fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix=f"{pdf_path.stem}_sub_")
            os.close(fd)
            dst.save(tmp)
        finally:
            src.close()
        page_map = {j + 1: orig_indices[j] + 1 for j in range(len(orig_indices))}
        return Path(tmp), page_map

    def _convert_split(self, pdf_path: Path, pdf_name: str,
                       scanned_idx: List[int], total: int) -> List[DocumentElement]:
        """Mixed doc: OCR only the scanned pages, skip OCR on the digital pages
        (two conversions), then merge elements back in original page order. Avoids
        paying the ~1.2s/page OCR-stage overhead on the digital majority."""
        scanned = set(scanned_idx)
        digital_idx = [i for i in range(total) if i not in scanned]
        logger.info("%s: MIXED -> per-page OCR split (%d scanned + %d digital pages)",
                    pdf_name, len(scanned_idx), len(digital_idx))
        merged: List[DocumentElement] = []
        for indices, do_ocr in ((scanned_idx, True), (digital_idx, False)):
            if not indices:
                continue
            sub_path, page_map = self._make_subset_pdf(pdf_path, indices)
            try:
                doc = self._convert_with_fallback(sub_path, do_ocr).document
                merged.extend(self.build_elements(doc, pdf_name, page_map))
            finally:
                try:
                    sub_path.unlink()
                except Exception:                        # pragma: no cover
                    pass
        # Restore reading order: original page, then in-subset order (a page belongs
        # to exactly one subset, so (page, order) is unique). Re-index order + id and
        # forward-fill heading/section continuity across the split boundary.
        merged.sort(key=lambda e: (e.page, e.order))
        cur_title = cur_head = ""
        for new_order, e in enumerate(merged):
            if e.type == "heading":
                cur_head = e.content or cur_head
                if e.section:
                    cur_title = e.section
            else:
                if not e.heading:
                    e.heading = cur_head
                if not e.section:
                    e.section = cur_title
            e.order = new_order
            e.id = f"{e.type}_{e.page}_{new_order}"
        return merged

    def _convert_and_build(self, pdf_path: Path, pdf_name: str) -> List[DocumentElement]:
        """Pick the cheapest correct path: forced OCR modes and pure digital/scanned
        docs use a single conversion; a MIXED doc uses the per-page split."""
        force = os.environ.get("DOCLING_DO_OCR", "auto").strip().lower()
        if force in ("1", "on", "true", "yes"):
            return self.build_elements(
                self._convert_with_fallback(pdf_path, True).document, pdf_name)
        if force in ("0", "off", "false", "no"):
            return self.build_elements(
                self._convert_with_fallback(pdf_path, False).document, pdf_name)
        scanned, total = self._scanned_page_set(pdf_path)
        if total == 0:
            return self.build_elements(                  # unreadable -> safe: OCR on
                self._convert_with_fallback(pdf_path, True).document, pdf_name)
        if not scanned:                                  # all digital -> OCR off
            return self.build_elements(
                self._convert_with_fallback(pdf_path, False).document, pdf_name)
        # MIXED. The split runs two conversions (fixed ~pipeline-init overhead), so
        # it only pays off once enough digital pages skip OCR (~1.2s each). Below
        # DOCLING_OCR_SPLIT_MIN digital pages, the plain whole-doc OCR-on pass is
        # simpler and no slower.
        n_digital = total - len(scanned)
        min_digital = int(os.environ.get("DOCLING_OCR_SPLIT_MIN", "8"))
        if (len(scanned) == total or not self._split_enabled()
                or n_digital < min_digital):
            return self.build_elements(                  # all scanned / split off / small
                self._convert_with_fallback(pdf_path, True).document, pdf_name)
        return self._convert_split(pdf_path, pdf_name, scanned, total)

    # ---------------------------------------------------------------- #

    def process_pdf(self, pdf_path: Path) -> Tuple[str, List[DocumentElement]]:
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        logger.info("Docling converting: %s", pdf_name)
        elements = self._convert_and_build(pdf_path, pdf_name)
        # Save in the existing document.json format for resume + chunker reuse.
        self._builder.save_document(pdf_name, elements)
        logger.info("  %s -> %d elements", pdf_name, len(elements))
        return pdf_name, elements

    def process_all_pdfs(
        self, pdf_name_filter: str = None
    ) -> Dict[str, List[DocumentElement]]:
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        if pdf_name_filter:
            pdfs = [p for p in pdfs if p.stem == pdf_name_filter]
        if not pdfs:
            logger.error("No PDFs found in %s", self.pdf_dir)
            return {}
        documents: Dict[str, List[DocumentElement]] = {}
        for pdf in pdfs:
            name, elements = self.process_pdf(pdf)
            documents[name] = elements
        return documents


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    flt = sys.argv[1] if len(sys.argv) > 1 else None
    DoclingIngestor().process_all_pdfs(flt)
