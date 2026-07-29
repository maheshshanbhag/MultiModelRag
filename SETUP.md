# Setup — Offline Multimodal RAG

A fully-offline, local RAG over PDFs: **Docling** ingestion → **hybrid retrieval
+ rerank** → **llama3.2** answer, with a **vision model** reading figures.
React frontend + FastAPI backend. Runs entirely on your machine.



## Prerequisites (install these first)
| Need | Where |
|------|-------|
| **Python 3.11** | https://www.python.org |
| **Node.js 18+** | https://nodejs.org (for the frontend) |
| **Ollama** | https://ollama.com/download |
| **Tesseract OCR 5.x** | OCR engine for scanned PDFs. **Windows (easiest):** `winget install -e --id UB-Mannheim.TesseractOCR` (accept the UAC prompt). Or the GUI installer: https://github.com/UB-Mannheim/tesseract/wiki (keep the **English** language pack, tick **Add to PATH**). macOS: `brew install tesseract`. Linux: `sudo apt install tesseract-ocr`. |
| **NVIDIA GPU + driver 555+** *(optional)* | CUDA 12.6 — CPU also works, just slower. See the CUDA note in step 2. |

---

## 1. Backend — virtual environment
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
```

## 2. Backend — install Python libraries
```bash
pip install -r requirements.txt
pip uninstall -y rapidocr     # remove the unused Chinese OCR package (we use Tesseract)
```
> `rapidocr` is pulled in by Docling's default bundle but is **never used** (OCR is
> Tesseract). Removing it guarantees nothing can ever try to fetch its models from
> ModelScope. The pipeline is verified to run fully without it. (`pip check` may
> note Docling "wants" rapidocr — harmless, ignore it.)
**CUDA note:** the file pins `torch==2.12.1+cu126` (CUDA 12.6). If your machine differs, edit the top of `requirements.txt`:
- **Different CUDA** → change `--extra-index-url .../cu126` to `cu121` / `cu124` / `cu128` and match the torch/torchvision versions.

**OCR note:** OCR uses the **system Tesseract** binary (from Prerequisites), not a Python package. Verify it's on PATH:
```bash
tesseract --version        # expect 5.x
tesseract --list-langs     # must include: eng
```
If `tesseract` isn't found, add its install folder (e.g. `C:\Program Files\Tesseract-OCR`) to PATH, or set `DOCLING_TESSERACT_CMD` to its full path.


## 3. Backend — download the ML models (~4.4 GB)
```bash
python download_models.py
```
Pulls BGE embeddings + reranker (HuggingFace) + Docling models into `backend/model/`.

## 4. Ollama — the LLM + vision model
```bash
ollama pull llama3.2:3b
ollama pull qwen2.5vl:3b
```
Keep `ollama serve` running (Ollama usually starts it automatically).

> These are the **4 GB-VRAM defaults**. On a bigger GPU, pull larger models
> instead for noticeably better answers — see **["Using a bigger model"](#using-a-bigger-model-for-best-results-gpu--12-gb)** below.

## 5. Frontend
```bash
cd ../frontend/frontend
npm install
```

## 6. Add PDFs and ingest
Put PDFs in `backend/uploads/pdfs/`, then:
```bash
cd ../../backend
.venv\Scripts\python run_pipeline_docling.py      # macOS/Linux: .venv/bin/python
```
(Or skip this and upload PDFs from the web UI in step 7 — same pipeline.)

## 7. Run it
```bash
# terminal 1 — backend  (from backend/)
.venv\Scripts\python api.py            # -> http://localhost:8000

# terminal 2 — frontend (from frontend/frontend/)
npm run dev                            # -> http://localhost:5173
```
Open the frontend, upload a PDF, and ask questions.

---

## Optional tuning (env vars)
| Var | Effect |
|-----|--------|
| `RAG_ENCODER_DEVICE=cuda` | run BGE + reranker on GPU (needs VRAM headroom) |
| `DOCLING_OCR_ENGINE=easyocr` | swap the OCR engine (default `tesseract`; `easyocr` needs `pip install easyocr`) |
| `DOCLING_NUM_THREADS=<n>` | CPU threads for OCR — the main speed lever on scanned docs (default = all cores) |
| `DOCLING_DO_OCR=1` / `0` | force OCR on / off (default `auto`: skipped on born-digital, run on scanned) |
| `DOCLING_GPU=1` | layout/tables on GPU (default auto). Leave `DOCLING_OCR_GPU` **unset** — OCR is CPU. |
| `DOCLING_TABLE_MODE=accurate` | higher-fidelity tables (slower) |
| `RAG_SEMANTIC_CHUNK=0` | disable meaning-boundary chunking (size-based fallback) |

Full ingestion knob list (per-page OCR split, thresholds, etc.) is in **`backend/exp.md`**.

### Using a bigger model for best results (GPU ≥ 12 GB)

The single biggest accuracy lever is the **text model size** — `llama3.2:3b` is a
4 GB-VRAM compromise. Do these in order of impact:

**1. Pull bigger Ollama models** (replaces step 4):
```bash
ollama pull qwen2.5:7b        # text LLM — needs ~12 GB   (qwen2.5:14b / qwen3:14b on ~24 GB)
ollama pull qwen2.5vl:7b      # vision model — sharper figure reading
```

**2. Point the code at them** — three small edits:

| File | Change this line → to | Why |
|------|-----------------------|-----|
| `backend/llm/ollama_client.py` | `DEFAULT_MODEL = "llama3.2:3b"` → `"qwen2.5:7b"` | the answer model — **biggest gain** |
| `backend/llm/ollama_client.py` | `num_ctx: int = 2048` → `8192` | fit more retrieved context in the prompt |
| `backend/api.py` | `OllamaClient(model="qwen2.5vl:3b")` → `"qwen2.5vl:7b"` | matches the bigger VLM you pulled |

**3. Feed the bigger model more context** — raise these to use its larger window:

| File | Change this → to | Effect |
|------|------------------|--------|
| `backend/api.py` | `top_k: int = 20` → `40` | retrieve more candidates |
| `backend/api.py` | `top_n: int = 5` → `10` | rerank + send more chunks |
| `backend/agents/text_agent.py` | `budget_tokens: int = 1500` → `4000` | pack more context per answer |

**4. Move the encoders + ingestion onto the GPU** (env vars, no code change — Windows `set`, Linux/Mac `export`):
```bash
set RAG_ENCODER_DEVICE=cuda      # BGE embed + reranker on GPU (faster, frees CPU)
set DOCLING_GPU=1                # layout/table models on GPU during ingestion
set DOCLING_TABLE_MODE=accurate  # higher-fidelity tables
```

**On ≥ 16 GB** you can also drop the model-eviction workaround: the `_ensure_only_loaded(...)`
call in `api.py`/`ask.py` only exists because on 4 GB the LLM and VLM can't coexist.
With headroom both stay resident and figure-queries answer faster.

**Expected result:** exact-number/table questions ~70% → ~85%, with sharper
multi-step reasoning. The 7B/14B text model is what moves the needle — everything
else is secondary tuning.

---

## Notes
- **Fully offline** after setup — all models are local; Ollama runs locally too.
  (Tesseract is a local binary; no network at OCR time.)
- **OCR engine: Tesseract** (system binary, CPU, offline). RapidOCR / PaddleOCR
  (Chinese PP-OCR models) were removed; `DOCLING_OCR_ENGINE=easyocr` is an optional
  alternative.
- **Per-document OCR:** born-digital PDFs skip OCR entirely; scanned PDFs get OCR'd;
  a **mixed** PDF OCRs only its scanned pages (the rest read their text layer).
- **Ingestion speed** (4 GB RTX 3050, layout/tables on GPU): digital ~**0.4 s/page**
  (OCR skipped), scanned ~**3 s/page** (Tesseract on CPU). Scanned time is CPU-bound,
  so it varies with CPU load.
- OCR runs on **CPU** by design — on a small card it's ~2× faster than GPU OCR and
  leaves VRAM for layout/tables. Full knob list in `backend/exp.md`.
