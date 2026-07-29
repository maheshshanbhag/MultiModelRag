# MultiModal RAG Backend — Explained in Full Detail

This document explains the **entire backend**, every concept, and every important
number (chunk sizes, `top_k`, thresholds, model settings…) in **simple language**.

If you have never seen this project before, read it top-to-bottom. If you just
want a number, jump to the **[Big Table of Numbers](#12-the-big-table-of-every-number)** at the end.

---

## 1. What is this backend, in one paragraph?

It is an **offline "chat with your PDFs" system**. You upload PDF documents
(digital or scanned). The backend reads them, cuts them into small pieces,
turns each piece into a list of numbers (an *embedding*), and stores them in a
local database. Later, when you ask a question, it finds the most relevant
pieces, hands them to a **local AI model (Ollama)**, and the model writes an
answer using **only** those pieces. It can also show you **figures/diagrams**
from the PDF and even have a **vision model describe a figure**. Everything runs
on your own machine — no internet, no cloud, no API keys.

The two things it can do are called the two **flows**:

1. **Ingestion flow** — "read and remember the PDFs" (done once per document).
2. **Query flow** — "answer a question" (done every time you ask).

---

## 2. The core idea: what is RAG?

**RAG = Retrieval-Augmented Generation.**

A plain LLM (like llama) only knows what it learned during training. It does not
know what is inside *your* PDF, and if you ask, it will often **make things up**
("hallucinate").

RAG fixes this in 3 steps:

1. **Retrieve** — search your documents for the pieces most related to the question.
2. **Augment** — paste those pieces into the prompt as "context".
3. **Generate** — ask the LLM to answer using **only** that context.

So the LLM becomes a "reader + writer" on top of *your* documents, instead of
guessing from memory. If the documents don't contain the answer, the system is
designed to **refuse** rather than invent one.

---

## 3. Project map — which file does what

```
backend/
├── run_pipeline_docling.py   ← runs the whole INGESTION flow (offline batch)
├── api.py                    ← the web server (FastAPI) the frontend talks to
├── ask.py                    ← terminal chat (same pipeline, streamed to console)
├── eval.py                   ← measures quality (recall, hit-rate, coverage)
├── eval_faq.py               ← per-document regression check
├── download_models.py        ← one-time model downloader
├── requirements.txt          ← the Python libraries needed
│
├── ingestion/                ← everything that turns PDFs into searchable data
│   ├── docling_ingestor.py   ← PDF → text/tables/figures (Docling)   [stages 1-5]
│   ├── document_builder.py   ← the "DocumentElement" data shape + save/load
│   ├── semantic_chunking.py  ← splits a document into chunks          [stage 7]
│   ├── semantic_splitter.py  ← splits a long chunk where meaning shifts
│   ├── contextualizer.py     ← adds heading path to each chunk        [stage 7.5]
│   ├── embedding_generator.py← chunk text → embedding vector          [stage 8]
│   ├── vector_store.py       ← saves vectors in ChromaDB              [stage 9]
│   ├── query_normalizer.py   ← fixes typos in a query using your docs' words
│   ├── retriever.py          ← hybrid search (vector + BM25 + RRF)    [stage 10]
│   ├── reranker.py           ← re-scores results for precision        [stage 11]
│   └── prompt_builder.py     ← builds the final prompt for the LLM    [stage 13]
│
├── agents/                   ← the "decision makers" of the query flow
│   ├── planner_agent.py      ← reads intent: text? image? explain?
│   ├── router.py             ← older/simpler version of the planner
│   ├── text_agent.py         ← the main retrieval brain (Corrective-RAG)
│   ├── evaluator_agent.py    ← "is this retrieval good enough to answer?"
│   ├── grounding.py          ← the shared thresholds & similarity math
│   ├── query_rewrite_agent.py← rephrases the query when retrieval was weak
│   ├── image_agent.py        ← picks which figure(s) to show
│   └── generator_agent.py    ← builds prompt + calls the LLM
│
├── llm/
│   └── ollama_client.py      ← talks to Ollama (the local LLM/VLM server)
│
├── model/                    ← the downloaded AI models live here (offline)
│   ├── embeddings/bge-base-en-v1.5
│   ├── reranker/bge-reranker-v2-m3
│   └── docling/              ← layout + table + OCR models
│
├── uploads/pdfs/             ← the raw PDFs you add
├── data/documents/<pdf>/document.json  ← parsed elements per PDF
├── data/chunks/<pdf>/chunks.json       ← the chunks per PDF
├── data/figures/<pdf>/page_N/*.png     ← extracted figure images
└── vector_db/                ← the ChromaDB database (the searchable memory)
```

**Two numbering systems appear in the code:**
The comments call ingestion steps "Stage 1-5, 7, 8, 9" and query steps "Stage
10, 11, 13". These are just historical labels from an older pipeline; some
numbers (6, 12) no longer exist. Don't worry about the exact numbers — the
**order** is what matters, and this doc follows the real order.

---

## 4. The AI models used (and their key settings)

The system uses **four** models. Two are "encoder" models (they turn text into
numbers or scores) and two are "generator" models (they write text), served by
Ollama.

| Model | Job | Where it runs | Key numbers |
|---|---|---|---|
| **BAAI/bge-base-en-v1.5** | Embeddings: turn text into a **768-number vector** | CPU at query time, GPU at ingestion | `normalize_embeddings=True`; query gets an instruction prefix |
| **BAAI/bge-reranker-v2-m3** | Cross-encoder: **re-score** the top results for precision | CPU (default) | `max_length=256` tokens |
| **llama3.2:3b** (via Ollama) | Writes the **text answer** | GPU (100% resident) | `temperature=0.2`, `num_ctx=2048`, `keep_alive=10m` |
| **qwen2.5vl:3b** (via Ollama) | **Vision** model: describes a figure image | GPU | `num_predict=350` (hard length cap) |

**Why these specific settings? (the 4GB GPU story)**

This whole project is tuned to run on a **small 4GB GPU** (an RTX 3050). That
single constraint explains most of the "weird" numbers:

- **`llama3.2:3b` instead of a bigger 8B model** — the 3B model fits *entirely*
  in 4GB and runs 100% on the GPU. The 8B model spilled ~56% onto the CPU and
  was 5-7× slower.
- **`num_ctx=2048`** (the LLM's context window) — 2048 tokens keeps llama3.2
  fully on the GPU. Setting it to 4096 pushed ~20% onto the CPU and slowed it
  down. Real prompts here are ~1275 tokens in + ~300 out, so 2048 is enough.
- **Encoders on CPU** — the embedding model and reranker are forced onto the
  **CPU** at query time so the LLM gets the *whole* GPU. Encoding one query on
  CPU costs only ~50-100ms; reranking ~20 short pairs costs ~1-2s. A fair trade.
- **One model on the GPU at a time** — llama3.2 and qwen2.5vl cannot both fit in
  4GB. So before running one, the code **evicts** the other (`_ensure_only_loaded`,
  explained in §9.4). Without this, both spill to CPU and the vision model
  crawled (measured: 16 minutes for one image!).
- **`keep_alive=10m`** — Ollama keeps the model loaded in VRAM for 10 minutes
  between questions, so you don't pay the ~1.3s reload cost every time.

---

## 5. THE INGESTION FLOW — turning a PDF into searchable memory

This runs when you add a document. The orchestrator is
`run_pipeline_docling.py` (batch) or the `/api/ingest` endpoint (one file at a
time from the frontend). The steps:

```
PDF ─► Docling ─► Semantic Chunking ─► Contextualizer ─► Embedding ─► Vector Store
      (parse)     (cut into pieces)     (add headings)    (→ vectors)   (save to DB)
```

### 5.1 Docling — reading the PDF (`docling_ingestor.py`)

**Docling** is a document-understanding library. **One** `convert()` call does a
lot internally:

```
PDF render → layout detection → OCR (only where needed) → reading order
          → table structure → figure extraction
```

Key design decisions and numbers:

- **Mixed digital + scanned handling** (`force_full_page_ocr=False`). If a page
  already has real text (a "text layer"), Docling reads it directly. OCR
  (reading text from an image) only runs on the scanned/bitmap parts. This is
  faster and more accurate than OCR-ing everything.
- **OCR engine: Tesseract** (`TesseractCliOcrOptions`, `lang=["eng"]`), selected
  via `DOCLING_OCR_ENGINE` (default `tesseract`; `easyocr` also available). CPU-
  only, fully offline (needs the system `tesseract` binary + `eng.traineddata`).
  RapidOCR/PaddleOCR (the Chinese-origin PP-OCR models) were **removed**. The
  engine is a drop-in swap — only `opts.ocr_options` changes; per-page detection,
  the split, layout/GPU routing, chunking and embedding are engine-agnostic.
- **Tables** are exported as **Markdown** (pipe tables) and flow through as their
  own elements. `DOCLING_TABLE_MODE` is `fast` by default (quicker on CPU);
  `accurate` is available for complex tables.
- **Figures** are cropped and saved as PNGs at **`images_scale=2.0`** (2× zoom
  for a sharper crop) into `data/figures/<pdf>/page_<n>/figure_<i>.png`.
  Each figure is **anchored to its own caption** (`_anchor_figure_captions`):
  Docling often emits captions as plain paragraphs split above/below the image, so
  we pair a figure with the closest "Figure N" label + nearest sentence *on its
  page* rather than the last caption seen — fixing image queries that used to
  return the *adjacent* figure.
- **Headers/footers are dropped.** Page-number/running-title junk carries a
  `page_header`/`page_footer` label and is discarded, so it never pollutes a chunk.
- **`page_batch_size=1`** — process **one page on the GPU at a time**. This keeps
  peak GPU memory ~constant no matter how many pages the document has (a 33-page
  scanned doc peaked at only ~0.83GB).
- **PDF backend: `pypdfium2`** (not the default docling-parse). The default C++
  backend threw a memory error partway through long docs and **dropped pages
  28-33**; pypdfium2 reads all pages cleanly.
- **GPU→CPU fallback.** Ingestion prefers the GPU (~2.4× faster) but automatically
  drops to CPU if free VRAM is below **`1.2GB`** (`DOCLING_GPU_MIN_FREE_GB`), or
  if a GPU pass runs out of memory, or if it silently dropped pages. This is safe
  because no LLM is loaded during ingestion.

**Output:** a list of `DocumentElement`s, saved to
`data/documents/<pdf>/document.json`.

### 5.2 What is a `DocumentElement`? (`document_builder.py`)

Every piece Docling finds becomes one `DocumentElement` — a simple record:

```python
DocumentElement(
    id, page, order, type, bbox, content,
    image_path, section, heading, figure_id
)
```

- **`type`** is one of: `heading`, `paragraph`, `caption`, `table`, `figure`.
- **`bbox`** = the box coordinates `[left, top, right, bottom]` on the page.
- **`heading` / `section`** = the nearest heading and the document title, carried
  along so later steps know "what topic is this piece under".

These are saved as JSON so you don't have to re-parse the PDF to re-chunk it.

### 5.3 Semantic Chunking — cutting the document into pieces (`semantic_chunking.py`)

A "chunk" is a small, searchable piece of the document. **Why cut at all?**
Because you can't search or embed a whole 30-page PDF as one blob — you need
small, focused pieces so a search can match the *exact* relevant paragraph.

This is the most detail-heavy file. The important **rules and numbers**:

| Constant | Value | Meaning |
|---|---|---|
| `MIN_CHUNK_LEN` | 60 | A parent section shorter than 60 chars is thrown away (too small to be useful). |
| `TARGET_CHUNK_LEN` | 1000 | Group consecutive headings+text together until ~1000 chars, so related sub-topics stay in one "parent". |
| `MAX_CHUNK_LEN` | 1800 | Hard cap — if a single section exceeds 1800 chars, force a split. |
| `CHILD_MIN_LEN` | 25 | The smallest "child" piece worth embedding on its own. |
| `CHILD_SPLIT_LEN` | 450 | A child longer than 450 chars is split into smaller idea-sized pieces. |
| `_MAX_TABLE_ROWS` | 80 | At most 80 rows of a table become their own searchable pieces (avoids explosion). |

**The big idea here: "small-to-big" (parent/child chunks).**

This is a clever trick. Each chunk actually stores **two** texts:

- **`text` (the CHILD)** — a small, precise piece (one idea, ~one paragraph).
  This is what gets **embedded and searched**. Small = precise matching.
- **`parent_text` (the PARENT)** — the larger surrounding section. This is what
  gets **handed to the LLM** to actually answer from. Big = full context.

So you get the **best of both**: you *find* things precisely (small child), but
you *answer* with complete context (big parent). Sibling children that came from
the same section share a `parent_id` so they can be traced back to one parent.

**How chunking actually walks the document:**

- A **heading** normally becomes a new *child* inside the current parent group.
  But it starts a **brand-new parent** when the group already reached
  `TARGET_CHUNK_LEN` (1000), **or** at a hard topic boundary — a heading starting
  with **"Unit"** (regex `^\s*unit\b`), because a new unit is a new topic.
- **Paragraphs and captions** get appended to the current child. If the group
  grows past `MAX_CHUNK_LEN` (1800) it is flushed early for safety.
- A **long child** (>450 chars) is split into smaller pieces — either by
  *meaning* (the semantic splitter, §5.4) or by paragraph/character fallback —
  and the heading is stuck onto each piece so it stays self-describing.
- A tiny child (<25 chars) is normally dropped… **except** if it contains a
  **number + unit** (like `"ROT 950-1,000 F"` or `"dP 5-8 psi"`). Those "spec
  lines" are high-value for number questions, so they're kept as their own
  searchable child. (Regex `_NUM_UNIT` catches `°, psi, bar, wt%, %, lb, kg,
  bpd, ton, scf, btu, :1, F, C`.)

**Figures become their own "image chunk."** A figure has no words, so how do you
search for it? You **anchor it on text**: its caption + heading + nearby section
prose (capped at **400 chars** to stay focused). That text is embedded in the
same space as normal text, so a query like "show the HBase data model diagram"
can match the figure through its description. This is far more reliable than
trying to match raw pixels. The chunk is marked `modality: "image"` and carries
the `image_paths` to the PNG file.

**Tables get row-level chunking.** A markdown table is parsed into header + rows.
Then:
- One "columns summary" child is made (answers "what does this table show").
- **Each row** (up to 80) becomes its own child like `"Table: <heading>. col1:
  val1, col2: val2"`, so an exact value lookup matches that one row.
- The **whole table** stays as the `parent_text` given to the LLM.
- If the markdown isn't a clean table, it falls back to one whole-table chunk.

**Output:** `data/chunks/<pdf>/chunks.json`. Each chunk has: `chunk_id`,
`pdf_name`, `text`, `pages`, `heading`, `section`, `figure_ids`, `image_paths`,
`modality`, `parent_id`, `parent_text`.

### 5.4 Semantic Splitter — split where the meaning changes (`semantic_splitter.py`)

When a child is too long, we could just cut it every 450 characters — but that
might cut a sentence in half mid-idea. The semantic splitter is smarter: it cuts
where the **topic actually shifts**.

How:
1. Split the text into sentences.
2. Embed each sentence (BGE) into a vector.
3. Measure the **cosine distance** between each pair of *consecutive* sentences.
4. Put a break wherever that distance is **large** — meaning the next sentence is
   about something different.

Key numbers: break when the distance is in the top **`pct=90`** percentile (i.e.
the biggest jumps), never make a piece smaller than **`min_chars=80`**, and never
bigger than **`max_chars=450`**. If the text is already ≤450 chars or has ≤2
sentences, it's left whole.

### 5.5 Contextualizer — give each chunk its "address" (`contextualizer.py`)

Problem: a terse chunk like `"950-1,000 F"` is impossible to find by the query
"reactor temperature range" — the words don't overlap.

Solution (no AI, pure string work): prepend the chunk's **heading path** to the
text that gets embedded. It builds:

```
embed_text = "<document title> — <section> > <heading>\n<chunk text>"
```

So `"950-1,000 F"` becomes something like
`"FCC Hand Book — Operating Conditions > Reactor Temperature\n950-1,000 F"`. Now
the conceptual query matches, because the heading words are in the embedded text.

Important subtlety — **only `embed_text` gets this prefix**:
- `embed_text` → what is **embedded** and **indexed for keyword search**. (matching)
- `text` (clean child) → what the **reranker** sees.
- `parent_text` → what the **LLM** sees.

So the added "address" improves **matching only** — it never shows up in the
answer. This technique is called **Contextual Retrieval**.

### 5.6 Embedding Generation — text into numbers (`embedding_generator.py`)

An **embedding** is a list of numbers (a *vector*) that represents the *meaning*
of a text. Texts with similar meaning get similar vectors. This is how the
computer does "search by meaning" instead of "search by exact words".

- Model: **BAAI/bge-base-en-v1.5** → produces a **768-dimensional** vector.
- It embeds **`embed_text`** (the context-enriched version from §5.5), falling
  back to the raw child text if the contextualizer didn't run.
- **`normalize_embeddings=True`** — every vector is scaled to length 1. This makes
  distance math clean (cosine similarity ↔ L2 distance become interchangeable).
- Runs on the **GPU** during ingestion (`batch_size=32` on GPU, `16` on CPU).

**Output:** each chunk now also has an `embedding` (list of 768 floats) plus all
its metadata.

### 5.7 Vector Store — the searchable database (`vector_store.py`)

The vectors + metadata are saved into **ChromaDB**, a local vector database, in
the `vector_db/` folder. Collection name: `multimodal_rag`.

- **ChromaDB is CPU-only** here (no GPU) — it just stores and searches vectors.
- Each chunk gets a **stable unique ID** = an MD5 hash of
  `pdf_name + text + image_paths`. "Stable" means re-ingesting the same PDF
  produces the same IDs, so **duplicates are skipped** instead of piling up.
- **Distance metric: L2** (straight-line distance) on the normalized vectors.
  Because vectors are unit-length, this relates directly to cosine similarity
  (see §9.2).
- Stored metadata per chunk: `pdf_name`, `chunk_id`, `pages`, `heading`,
  `section`, `figure_ids`, `image_paths`, `modality`, `parent_id`, `parent_text`,
  `embed_text`.

It also has helpers the app uses: `total_chunks()`, `list_pdf_names()` (the
source of truth for "which documents exist"), `get_all_chunks()` (used to build
the keyword index), and `delete_pdf()` (remove a document).

**At this point ingestion is done.** The PDF is now a set of searchable vectors.

---

## 6. THE QUERY FLOW — answering a question

This is what happens every time you ask something. Entry points: `ask.py`
(terminal) and `api.py` (web). The shape:

```
Question
  │
  ├─► Planner: is this asking for text? an image? an explanation?   (§7)
  │
  ├─► Text Agent:  retrieve → rerank → evaluate → (rewrite & retry) → adaptive context   (§8, §9)
  │        │
  │        ├─ Retriever: fix typos → dense search + BM25 → RRF merge → child→parent
  │        ├─ Reranker: cross-encoder re-scores the top results, keeps best 5
  │        ├─ Evaluator: is the best result confident enough? (grounding gate)
  │        └─ if weak → Query Rewrite → retrieve again (Corrective-RAG)
  │
  ├─► Image Agent: grab the figure(s) from the best chunks           (§10)
  │
  ├─► Generator: build prompt from the chunks → llama3.2 writes answer (streamed)  (§11)
  │
  ├─► VLM (optional): qwen2.5vl describes the figure image            (§9.4)
  │
  └─► Sources footer: "which PDF + pages did this come from"          (§11)
```

The important **k-values** (how many items at each stage):

| Setting | ask.py | api.py | eval.py | Meaning |
|---|---|---|---|---|
| `top_k` (retrieve) | **10** | **20** | **10** | how many candidates to pull from the DB |
| `top_n` (rerank) | **5** | **5** | **5** | how many to keep after re-scoring |

(`retriever.py`'s own default is `DEFAULT_TOP_K = 20`; the callers pass their own
value. More `top_k` = higher recall but slower reranking.)

---

## 7. The Planner — understanding what you want (`planner_agent.py`)

Before searching, the system reads your **intent** using simple keyword rules
(no LLM, so there's zero delay before the answer starts):

- **`wants_image`** — true if your query contains a visual word: `image, figure,
  diagram, picture, chart, graph, photo, screenshot, visual, show`… →
  "the user wants to *see* a figure."
- **`wants_explain`** — true if your query contains an explaining word: `explain,
  describe, what, how, why, detail, tell, define, summarize, compare`… →
  "the user wants a *written explanation*."

The combination gives the intent:

| wants_image | wants_explain | intent | What runs |
|---|---|---|---|
| no | — | **text** | llama3.2 text answer only |
| yes | yes | **both** | llama3.2 text answer **then** the vision model describes the figure |
| yes | no | **image** | "just show me the figure" → vision model only, skip the text answer |

(`router.py` is the older, simpler version that only returned a flat
`"text"`/`"image"` label. `planner_agent.py` superseded it.)

---

## 8. The Retriever — hybrid search (`retriever.py`)

This is the search engine. It uses **two different search methods and merges
them**, because each catches things the other misses.

### 8.0 First: fix typos (`query_normalizer.py`)

Before searching, the query is spell-corrected **against your documents' own
vocabulary** (not a generic dictionary). If you type `"rector temperature"`, and
your petroleum handbook contains `"reactor"` but never `"rector"`, it corrects to
`"reactor"`. It builds a word-frequency dictionary from all chunks, then for each
query word finds the closest real corpus word within **edit distance 1** (or 2
for words ≥5 letters). It leaves ACRONYMS (all-caps like `MAT`, `HCN`), numbers,
and short words (<3 letters) alone, and only corrects to a word that is the same
length or longer (real typos usually *drop* a letter). Toggle with `RAG_SPELLFIX=0`.

### 8.1 Method 1: Dense vector search (search by meaning)

- The query gets an instruction prefix (a BGE requirement):
  `"Represent this sentence for searching relevant passages: "`.
- It's embedded into a 768-vector, and ChromaDB returns the `top_k` chunks whose
  vectors are closest.
- **Strength:** finds things by *meaning*, even with different words
  ("car" matches "automobile").

### 8.2 Method 2: BM25 keyword search (search by exact words)

- **BM25** is a classic keyword-ranking algorithm (like a smart Ctrl-F). It's
  built lazily on the first query from all stored chunks (indexing `embed_text`
  so it matches the contextual embeddings).
- **Strength:** nails exact technical terms, acronyms, and codes that embeddings
  can blur (e.g. an exact part number or `"HDFS"`). It over-fetches `top_k * 2`
  then filters.
- It's rebuilt whenever documents are added or deleted (`invalidate_bm25()`).

### 8.3 Merging the two: Reciprocal Rank Fusion (RRF)

Now we have two ranked lists (one from meaning, one from keywords). RRF merges
them fairly using only the **rank position**, not the raw scores (which are on
different scales and can't be compared directly):

```
score(chunk) = Σ  1 / (k + rank + 1)      for each list the chunk appears in
```

with the constant **`k = 60`** (a standard RRF value). A chunk that ranks high in
*both* lists gets the highest merged score. Chunks are then sorted by this score.

### 8.4 Small-to-big expansion (child → parent)

After merging, `_expand_to_parents` swaps each matched **child** for its full
**parent section** (`parent_text`), keeping the best-ranked child per parent.
So the search matched a precise child, but what moves forward is the *full
context*. The precise child is remembered as `child_text` (the reranker still
uses it).

### 8.5 `retrieve_smart` — text answer AND figures together

The plain `retrieve()` searches everything. `retrieve_smart()` adds a rule for
image queries: it **always** retrieves text (so you still get an explanation),
and *additionally* appends the top image chunks (`n_images=3`) when the query
asks for a visual. This fixed a bug where "tell me about X **and show the image**"
returned images only and no explanation.

**Output of the retriever:** up to `top_k` candidate chunks, each with `id`,
`text` (the parent), `child_text` (the precise match), `metadata`, and `distance`.

---

## 9. The Reranker, Evaluator, and Corrective-RAG

### 9.1 The Reranker — precision re-scoring (`reranker.py`)

The retriever is fast but a bit rough. The **reranker** is slow but very
accurate, so we only run it on the ~10-20 survivors.

- Model: **BAAI/bge-reranker-v2-m3**, a **cross-encoder**. Unlike embeddings
  (which score the query and chunk *separately* then compare), a cross-encoder
  reads the **query and chunk together** and outputs one relevance score. This is
  much more accurate for final ranking.
- It scores the short **`child_text`** (~140 tokens), not the big parent —
  `max_length=256` never truncates it, and it's ~2× faster than scoring at 512.
- Returns the **`top_n = 5`** highest-scoring chunks, each with a new `rerank_score`.

**Two score boosts** nudge the ranking:

- **`HEADING_BOOST = 0.1`** per query content-word found in the chunk's heading
  or section title (capped at 3 words → max **+0.3**). This breaks ties between
  chunks with near-identical body text (e.g. the figure headed "HBase Data Model"
  beats the one headed "HBase Implementation" for the query "hbase data model").
- **`NUMERIC_BOOST = 0.15`** — on a *quantitative* query (contains words like
  `range, much, temperature, pressure, ratio, psi, how many`…), a chunk that
  actually contains a number+unit gets bumped up **but only if it's already
  on-topic** (heading overlap > 0), so an off-topic number isn't pulled up.

There's also a `_rerank_query()` cleanup (in `text_agent.py`) that strips
command words like "show me the image" *before* reranking, because those words
tanked the relevance score of on-topic chunks (e.g. "show map reduce image"
scored ~0.1 vs "map reduce" ~0.98). Content words like "map" are deliberately kept.

### 9.2 The Evaluator — "is this good enough to answer?" (`evaluator_agent.py` + `grounding.py`)

This is the **anti-hallucination gate**. Before letting the LLM answer, the
Evaluator checks whether the best result is actually confident. It uses **two
independent signals** and passes if **either** clears its floor:

| Signal | Floor | Why |
|---|---|---|
| **rerank score** (`GROUNDING_MIN_SCORE`) | **0.30** | Precise, but its scale drifts per document set. |
| **dense similarity** (`GROUNDING_MIN_DENSE`) | **0.60** | Corpus-stable cosine similarity. |

**Dense similarity math:** ChromaDB gives an L2 distance on unit vectors, and the
code converts it to cosine similarity with:

```
cosine = 1 − distance / 2
```

The dense signal is taken as the **maximum across all 5 chunks** (the #1 chunk
might be a keyword-only hit with no dense signal, while a lower chunk carries the
strong meaning-match).

If **neither** signal passes for **any** attempt → the system returns **no
chunks** → the Generator refuses with *"The documents don't contain enough
information to answer this."* This is how out-of-scope questions get refused
instead of hallucinated. (Set both floors to 0 to disable the gate.)

### 9.3 Corrective-RAG — try again with a better query (`text_agent.py` + `query_rewrite_agent.py`)

This is the "agentic" loop that makes the Text Agent smart:

```
retrieve → Evaluator says GOOD?  → yes → use it
                                 → no  → rewrite the query → retrieve again → re-evaluate
```

- Controlled by **`RAG_MAX_RETRIES = 1`** (one corrective retry by default; 0
  disables the loop).
- The **Query Rewrite Agent** is rule-based by default (instant, no LLM): it
  strips question-words and stopwords (`what, how, is, the, explain…`) to turn a
  chatty question into a lean keyword query. That produces a *different*
  embedding on the retry, which often surfaces passages the verbose phrasing
  missed. It returns `""` if it can't make anything different (→ stop retrying).
- An optional single-LLM rewrite exists behind `RAG_REWRITE_LLM=1`.
- Between the original and the retry, the **stronger** attempt is kept (compared
  by: passing beats failing, then higher dense sim, then higher rerank score).

### 9.4 Adaptive Context Expansion — fit the budget without losing content (`text_agent.py`)

Before sending chunks to the LLM, `adaptive_context()` makes sure they fit the
LLM's small window without cutting important content.

- Budget: **`budget_tokens = 1500`** (estimated at **~4 characters per token**).
- It walks the chunks **best-first** and gives each its **full parent section**
  if it fits. Only when the budget is nearly full does it **shrink a chunk down
  to its short child snippet** instead of dropping it. If even the snippet won't
  fit, it stops.
- Because results are best-first, any trimming falls on the **least-relevant
  tail**, never the top answers. This protects list sub-points from being lost.
- Toggle off with `RAG_ADAPTIVE=0` (falls back to always-full-parent).

---

## 10. The Image Agent — picking figures to show (`image_agent.py`)

Once the best chunks are known, the Image Agent pulls out any figures attached to
them. Because figures were anchored on text at ingest time (§5.3), a relevant
chunk already carries `image_paths` in its metadata.

- It looks at only the **most relevant** chunks: the top **3** if the query wants
  an image, else the top **2** (so a figure is surfaced only when it's genuinely
  on-topic).
- It de-duplicates and returns at most **`max_images = 2`** figure paths.

The figure is then shown to the user directly (the text LLM never "sees" the
image — see §11).

---

## 11. The Generator — writing the answer (`generator_agent.py` + `prompt_builder.py` + `ollama_client.py`)

### 11.1 Building the prompt (`prompt_builder.py`)

The prompt sent to llama is a two-message chat: a **system message** (the rules)
and a **user message** (the context + the question).

The **system rules** (`SYSTEM_INSTRUCTION`) are strict, and each rule fixes a
real failure mode:

- Answer using **ONLY** the given context (no outside knowledge). → grounding.
- Open with **one short bold sentence** that directly answers.
- Use bullet/numbered lists for multiple points, **bold the key term**.
- **Be COMPLETE** — include *every* distinct point; if the context lists five
  points, the answer must have all five. → stops the model from summarizing away detail.
- Reproduce **numbers, ranges, units, and values EXACTLY** — never round,
  approximate, or invent. → critical for the technical/petroleum docs.
- **Do not** mention "the context", add citation tags like `[Source 1]`, or
  output image links/markdown images `![](...)`. → keeps the answer clean.
- If the context lacks the answer, reply **exactly**: *"The documents don't
  contain enough information to answer this."*

The **user message** contains the chunk texts (only the section `## heading` is
kept as a label — no source/page/score clutter, because the answer must not cite
them) followed by `Question: ...` and `Answer:`.

**Figures are NOT put in the text prompt.** The text model can't see images and
would only hallucinate about them, so image paths are stripped from the prompt;
the actual figure is shown to the user separately.

### 11.2 Calling the LLM (`ollama_client.py`)

`OllamaClient` wraps the local **Ollama** HTTP server (`http://localhost:11434`).
Key points:

- Default model **`llama3.2:3b`**, `temperature=0.2` (low = focused/consistent;
  eval uses `0.0` for reproducibility), `num_ctx=2048`, `keep_alive=10m`.
- **Streaming** — `chat_stream()` yields tokens one at a time so the answer appears
  live in the terminal/UI instead of after a long pause.
- **`vlm_stream()`** handles the vision model: it base64-encodes the image file
  and streams a description from qwen2.5vl, hard-capped at **`num_predict=350`**
  tokens (VLMs ignore "be brief" and ramble otherwise).

A safety net in `ask.py`/`api.py` (`_filter_image_md`) strips any fabricated
`![](...)` markdown image link out of the stream live, in case the small model
disobeys the "no image links" rule.

### 11.3 The GPU juggling — `_ensure_only_loaded` (the 4GB trick)

Because only **one** Ollama model fits in 4GB, before running a model the code
calls `_ensure_only_loaded(model)`:
1. Ask Ollama which models are currently loaded (`/api/ps`).
2. **Evict** every model except the one we're about to use (POST with
   `keep_alive: 0`).

So for a "text + image" query: it evicts qwen → runs llama for the text answer →
evicts llama → runs qwen for the figure description. Each gets the **full GPU**,
one at a time. This turned a 16-minute worst case into seconds.

### 11.4 The sources footer

After the answer, the system prints a **deterministic** citation built straight
from chunk metadata (not from the LLM, which is unreliable at citing):

```
Source: FCC_Hand_Book.pdf (p. 5, 7, 9); Unit 1 - SIDS.pdf (p. 2)
```

One entry per PDF with all its page numbers merged and sorted.

---

## 12. How the vision (figure) path works end to end

When you ask something like *"explain and show the MapReduce diagram"*:

1. Planner sets `wants_image=true`, `wants_explain=true` → intent **"both"**.
2. Text Agent retrieves + reranks; because it's an image query,
   `retrieve_smart` also appends image chunks.
3. Image Agent grabs the figure PNG from the top 3 chunks (max 2 images).
4. `_ensure_only_loaded(llama)` → llama3.2 streams the **text answer**.
5. The figure is shown to the user (`Image: .../figure_1.png`).
6. `_ensure_only_loaded(qwen)` → qwen2.5vl streams a **~100-word description** of
   that image, focused by a prompt that includes the figure's caption/heading.

For a bare *"show me the diagram"* (no explain word) → intent **"image"** → it
**skips** the llama text answer entirely and goes straight to the vision model.

---

## 13. The web server (`api.py`) and terminal chat (`ask.py`)

Both run the exact same pipeline; they differ only in how they deliver it.

**`ask.py`** — a terminal chat. Loads the models once, then loops reading
questions and streaming answers to the console. Good for quick testing.

**`api.py`** — a **FastAPI** server on port **8000** for the frontend. Models are
**lazy singletons** (loaded once on first use). Endpoints:

| Endpoint | What it does |
|---|---|
| `GET /` and `/api/status` | health check + how many chunks/documents exist |
| `POST /api/query` | ask a question, get a full JSON answer + sources + figures |
| `GET /api/query/stream` | same, but **Server-Sent Events** streaming (meta → tokens → vlm tokens → done) |
| `POST /api/interpret_figure` | stream a vision-model description of one figure |
| `GET /api/documents` | list ingested PDFs |
| `POST /api/ingest` | upload a PDF; ingest it in the **background** |
| `GET /api/ingest/{name}` | poll ingestion progress (queued → running → done/error) |
| `DELETE /api/documents/{name}` | remove a PDF and its chunks |

Extracted figures are served as static files under `/figures/...` so the frontend
can display them. CORS is open (`allow_origins=["*"]`) for local development.

---

## 14. How quality is measured (`eval.py`, `eval_faq.py`)

You can't improve what you don't measure. `eval.py` scores the system on a set of
questions with **known correct answers** (`eval/golden_set.json`):

- **Retrieval metrics (no LLM needed):**
  - **`recall@k`** — is the correct (pdf, page) anywhere in the top-`k` retrieved? (baseline: **100%** at k=10)
  - **`hit@n`** — is it in the top-`n` after reranking? (baseline: **95%** at n=5)
  - **`MRR`** — 1 / (rank of the first correct chunk); higher = correct answer ranked nearer the top.
- **Answer metrics (`--full`, runs llama at `temperature=0`):**
  - **keypoint coverage** — what fraction of the expected key facts appear in the answer? (baseline: **~96%**)
  - **refusal accuracy** — for out-of-scope questions, did it correctly **refuse** instead of hallucinating?

`eval_faq.py` / `--faq` mode is a **per-document regression check**: for each new
document's known Q&A, it verifies in-scope questions are answerable and
out-of-scope ones get refused — using the **same grounding gate** production uses,
so it needs no LLM.

---

## 15. Offline & configuration — environment variables

Everything is designed to run **fully offline**. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` are set so HuggingFace never touches the network; models
are pre-downloaded into `model/`. The Ollama models live inside Ollama itself
(`ollama pull llama3.2:3b` and `ollama pull qwen2.5vl:3b`).

Useful toggles (all optional — the defaults are tuned already):

| Env var | Default | Effect |
|---|---|---|
| `RAG_ENCODER_DEVICE` | `cpu` | Put the embedder/reranker on `cuda` or `auto` (only worth it on a bigger GPU). |
| `RAG_MAX_RETRIES` | `1` | Corrective-RAG retries when retrieval is weak. |
| `RAG_ADAPTIVE` | `1` | Adaptive context expansion on/off. |
| `RAG_SPELLFIX` | `1` | Query typo-correction on/off. |
| `RAG_SEMANTIC_CHUNK` | `1` | Semantic (meaning-based) splitting of long chunks. |
| `RAG_MIN_SCORE` / `RAG_MIN_DENSE` | `0.30` / `0.60` | The two grounding-gate floors. |
| `RAG_NUMERIC_BOOST` | `0.15` | Boost for number-bearing chunks on quantitative queries. |
| `RAG_REWRITE_LLM` | `0` | Use an LLM (not rules) to rewrite weak queries. |
| `TARGET_CHUNK_LEN` | `1000` | Target parent-chunk size during ingestion. |
| `DOCLING_GPU` | auto | Force GPU (`1`) / CPU (`0`) for layout + TableFormer. |
| `DOCLING_OCR_ENGINE` | `tesseract` | OCR engine: `tesseract` (default, CPU, needs the tesseract binary + `eng` data) or `easyocr` (pip, torch). RapidOCR/PaddleOCR removed. |
| `DOCLING_TESSERACT_CMD` | `tesseract` | Path to the tesseract executable if not on PATH. |
| `DOCLING_DO_OCR` | `auto` | `auto` scans every page and runs OCR iff **any** page is scanned (image + ~no text) — so mixed docs (e.g. a mostly-digital handbook with a few scanned pages) keep their scanned text. Skips OCR only when the whole doc is born-digital. `1`/`on` forces OCR; `0`/`off` disables. |
| `DOCLING_OCR_MIN_CHARS` | `50` | Per-page text floor for the `auto` scan: a page below it **with an image** counts as scanned; below it with no image is a blank page (ignored). |
| `DOCLING_OCR_SPLIT` | `1` | For MIXED docs, OCR *only* the scanned pages: convert scanned + digital page-subsets separately (OCR on/off) and merge in page order. `0` disables (whole-doc OCR-on). |
| `DOCLING_OCR_SPLIT_MIN` | `8` | Min digital pages before the split is worth its two-conversion overhead; smaller mixed docs use the whole-doc OCR-on pass. |
| `DOCLING_OCR_GPU` | `0` | Only affects `easyocr` (Tesseract is CPU-only). Leave off on small cards. |
| `DOCLING_NUM_THREADS` | `os.cpu_count()` | CPU threads for layout/OCR accelerator. Docling's own default is only 4. |
| `DOCLING_TABLE_MODE` | `fast` | `accurate` for higher-fidelity tables. |
| `DOCLING_OFFLINE` | `0` | Pin Docling fully offline once models are cached. |
| `DOCLING_PAGE_BATCH` | `1` | Pages processed on the GPU at once (only ~6% gain — ingestion is CPU-bound, GPU ~1% util). |
| `DOCLING_GPU_MIN_FREE_GB` | `1.2` | Free-VRAM floor before Docling falls back to CPU. |

**Ingestion is CPU-bound, not GPU-bound.** Per-stage profile of a 33-page
*digital* PDF (Docling's `profile_pipeline_timings`): **ocr 39 s** (fired on
every page!), page_parse 8.6 s, layout **7.3 s (GPU)**, table_structure 2.5 s
(GPU). The GPU does <10 s of work; OCR on the CPU is the whole cost — and Docling
runs OCR on every page even when a text layer exists. So the wins are CPU-side:

* **Skip OCR on born-digital PDFs** (`DOCLING_DO_OCR=auto`, default): the ingestor
  scans every page (text-layer chars + image presence) and turns OCR off only
  when the *whole* doc is digital; a single scanned page anywhere keeps its text.
  Fully-digital 33-page PDF: **52 s → 12 s (0.4 s/page), ~4.2×.**
* **Per-page OCR split for MIXED docs** (`DOCLING_OCR_SPLIT=1`, default): instead
  of paying the ~1.2 s/page OCR-stage overhead on the digital majority, the
  ingestor converts the scanned pages (OCR on) and digital pages (OCR off) as two
  subsets and merges the elements back in original page order (page numbers +
  figure dirs remapped). 30-digital + 3-scanned: **54 s → 40 s (1.38×)**; the win
  grows with the digital-page count (a 377-digital + 7-scanned handbook ≈ 2.5–3×).
  Kicks in only above `DOCLING_OCR_SPLIT_MIN` digital pages (two conversions have
  fixed overhead). Tradeoff: text *inside images on digital pages* isn't OCR'd
  (the VLM figure-description path covers it); force whole-doc OCR with
  `DOCLING_OCR_SPLIT=0` or `DOCLING_DO_OCR=1`.
* **OCR on CPU with all cores** (`DOCLING_NUM_THREADS`) for scanned docs: 33-page
  scanned PDF **~190 s (5.8 s/page)** vs GPU-OCR **>13 min unfinished**.
* Layout/TableFormer on GPU (`DOCLING_GPU=1`) — small but free.

Real PaddlePaddle-GPU OCR is **not usable in-process** on Windows: it and torch
ship incompatible, identically-named `cudnn_cnn64_9.dll` (one-DLL-per-name per
process → `WinError 127`); RapidOCR runs the same PP-OCR models via ONNX.
Docling has **no PyMuPDF backend** (only pypdfium2) and the PDF backend isn't the
bottleneck anyway.

---

## 16. The big table of every number

Everything numeric in one place, for quick reference.

### Models
| Thing | Value |
|---|---|
| Embedding model | BAAI/bge-base-en-v1.5 |
| Embedding dimension | **768** |
| Reranker model | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Reranker `max_length` | **256** tokens |
| Text LLM | llama3.2:3b |
| Vision model (VLM) | qwen2.5vl:3b |
| LLM temperature | **0.2** (eval: 0.0) |
| LLM context window `num_ctx` | **2048** tokens |
| LLM `keep_alive` | **10m** |
| VLM output cap `num_predict` | **350** tokens |
| VLM target length | ~**100** words |
| Embedding batch size | **32** GPU / **16** CPU |

### Retrieval & ranking
| Thing | Value |
|---|---|
| `top_k` retrieve (ask.py / eval) | **10** |
| `top_k` retrieve (api / retriever default) | **20** |
| `top_n` rerank (everywhere) | **5** |
| RRF constant `k` | **60** |
| BM25 over-fetch | `top_k × 2` |
| Image chunks added (`retrieve_smart`) | **3** |
| Image chunks searched | `max(n_images×2, 6)` |
| Chunks the Image Agent looks at | top **3** (image) / **2** (text) |
| Max images shown | **2** |
| Grounding rerank floor `MIN_SCORE` | **0.30** |
| Grounding dense floor `MIN_DENSE` | **0.60** |
| Dense similarity formula | `cos = 1 − distance/2` |
| Heading boost | **+0.1** per word, capped at 3 (**max +0.3**) |
| Numeric boost | **+0.15** (on-topic quantitative queries only) |
| Corrective-RAG retries | **1** |
| Adaptive context budget | **1500** tokens (~**4** chars/token) |
| Query typo edit distance | 1 (or 2 for words ≥5 letters) |

### Chunking (ingestion)
| Thing | Value |
|---|---|
| `MIN_CHUNK_LEN` (drop parent below) | **60** chars |
| `TARGET_CHUNK_LEN` (group up to) | **1000** chars |
| `MAX_CHUNK_LEN` (hard split) | **1800** chars |
| `CHILD_MIN_LEN` (smallest child) | **25** chars |
| `CHILD_SPLIT_LEN` (split child above) | **450** chars |
| Max table rows chunked | **80** |
| Figure context cap | **400** chars |
| Semantic split break percentile | **90** |
| Semantic split min/max piece | **80** / **450** chars |
| Docling image scale | **2.0×** |
| Docling page batch | **1** |
| Docling VRAM floor | **1.2 GB** |

### Quality baselines (eval)
| Metric | Value |
|---|---|
| recall@10 | **100%** |
| hit@5 | **95%** |
| keypoint coverage | **~96%** |

### Infrastructure
| Thing | Value |
|---|---|
| API server port | **8000** |
| Ollama URL | http://localhost:11434 |
| Vector DB | ChromaDB (`vector_db/`), collection `multimodal_rag` |
| Distance metric | L2 on normalized vectors |
| Target GPU | 4 GB (RTX 3050) |

---

## 17. One-line summary of the whole thing

> **Ingest:** Docling reads each PDF → it's cut into small precise "child" pieces
> (that carry a big "parent" for context) → each piece gets its heading path added
> → embedded into 768-number vectors → stored in ChromaDB.
>
> **Ask:** fix typos → search by meaning + by keywords → merge (RRF) → re-score the
> top with a cross-encoder → check it's confident enough (else rewrite & retry, or
> refuse) → fit the best context into 1500 tokens → llama3.2 writes a grounded,
> complete answer → optionally show a figure and let qwen2.5vl describe it → cite
> the exact PDFs and pages.
>
> All local, all offline, all tuned to fit a 4 GB GPU by running exactly one model
> at a time.
