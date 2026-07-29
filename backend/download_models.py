"""
download_models.py
One-shot downloader for every LOCAL model the pipeline needs. Run it once after
`pip install -r requirements.txt`, from the project venv:

    backend/.venv/Scripts/python download_models.py     # Windows
    backend/.venv/bin/python download_models.py         # macOS / Linux

Downloads (all from HuggingFace — reliable, no ModelScope):
  * BAAI/bge-base-en-v1.5    -> backend/model/embeddings/bge-base-en-v1.5  (embeddings)
  * BAAI/bge-reranker-v2-m3  -> backend/model/reranker/bge-reranker-v2-m3  (reranker)
  * Docling layout + TableFormer -> backend/model/docling                  (ingestion)

OCR uses the SYSTEM Tesseract binary (install per SETUP.md) — there is nothing to
download here for OCR. RapidOCR is intentionally skipped (it is no longer used, and
its models come from ModelScope, which is the usual cause of a failed download).

Re-runnable: finished files are skipped, so a re-run resumes an interrupted download.

The two Ollama models are separate (they live in Ollama, not this folder):
    ollama pull llama3.2:3b
    ollama pull qwen2.5vl:3b
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# This script DOWNLOADS, so make sure no stray offline flag from the shell/profile
# forces HuggingFace into offline mode (the usual cause of a "couldn't reach host"
# error here). The runtime pipeline sets these itself; the downloader must not.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

BASE = Path(__file__).resolve().parent
MODEL = BASE / "model"
# Skip variants the PyTorch pipeline never uses (keeps the download smaller).
SKIP = ["*.onnx", "onnx/*", "openvino/*", "*.msgpack", "*.h5",
        "tf_model.h5", "flax_model.msgpack"]


def _hf(repo: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repo}\n           -> {dest}", flush=True)
    snapshot_download(repo, local_dir=str(dest), ignore_patterns=SKIP)


def _docling(dest: Path) -> None:
    """Download ONLY the Docling models the pipeline loads: layout + TableFormer
    (+ the small HF enrichment models), all from HuggingFace. RapidOCR is skipped
    on purpose — OCR is done by the system Tesseract binary, and RapidOCR's models
    come from ModelScope, which is the most common failure point on this step."""
    dest.mkdir(parents=True, exist_ok=True)
    from docling.utils.model_downloader import download_models
    download_models(
        output_dir=dest,
        progress=True,
        with_layout=True,
        with_tableformer=True,
        with_code_formula=True,
        with_picture_classifier=True,
        with_rapidocr=False,     # Tesseract instead — avoids the ModelScope fetch
        with_easyocr=False,      # optional alt engine downloads its own models on use
    )


def _check_tesseract() -> None:
    exe = shutil.which("tesseract") or shutil.which("tesseract.exe")
    if exe:
        print(f"  OK — Tesseract found on PATH: {exe}")
    else:
        print("  WARNING: 'tesseract' is NOT on PATH. Scanned-PDF OCR will fail until\n"
              "           you install it. Easiest (Windows 10/11):\n"
              "               winget install -e --id UB-Mannheim.TesseractOCR\n"
              "           (macOS: brew install tesseract | Linux: sudo apt install tesseract-ocr)\n"
              "           Then open a NEW terminal so PATH updates, or set\n"
              "           DOCLING_TESSERACT_CMD to the tesseract.exe path. (Digital PDFs\n"
              "           still work without it.)")


def main() -> None:
    try:
        print("[1/3] BGE embedding model")
        _hf("BAAI/bge-base-en-v1.5", MODEL / "embeddings" / "bge-base-en-v1.5")
        print("[2/3] BGE reranker model")
        _hf("BAAI/bge-reranker-v2-m3", MODEL / "reranker" / "bge-reranker-v2-m3")
        print("[3/3] Docling models (layout + TableFormer)")
        _docling(MODEL / "docling")
    except Exception as e:
        print(
            f"\nDOWNLOAD FAILED: {type(e).__name__}: {e}\n"
            "  - check your internet connection and try again (it resumes where it\n"
            "    stopped — finished files are skipped)\n"
            "  - behind a proxy/firewall? set HTTP_PROXY / HTTPS_PROXY first\n"
            "  - make sure HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are not forced on",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDONE — all local models are in {MODEL}")
    print("\nOCR engine check (uses the system Tesseract binary):")
    _check_tesseract()
    print("\nNext, pull the Ollama models:")
    print("  ollama pull llama3.2:3b")
    print("  ollama pull qwen2.5vl:3b")


if __name__ == "__main__":
    main()
