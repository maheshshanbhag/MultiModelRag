# Offline MultiModal RAG — Frontend

A React + Tailwind UI for the FastAPI backend in `api.py`. Layout: a document
sidebar (upload, status, filter) on the left, a chat panel with streaming
answers, source citations, and clickable figures on the right — matching the
reference screenshots.

## Setup

```bash
npm install
cp .env.example .env   # edit VITE_API_BASE if your backend isn't on :8000
npm run dev
```

Then open the printed local URL (default `http://localhost:5173`). Make sure
`api.py` is running (`python api.py`, default `http://localhost:8000`) — CORS
is already open on the backend (`allow_origins=["*"]`).

## What's wired up

- **Upload** — drag/drop or browse a PDF → `POST /api/ingest`, then polls
  `GET /api/ingest/{pdf_name}` every 1.5s until `done`/`error`.
- **Document list** — `GET /api/documents`, with delete via
  `DELETE /api/documents/{pdf_name}`. Click a document to scope the next
  query to it (`pdf_filter`).
- **Chat** — `GET /api/query/stream` (SSE): renders the `meta` frame
  (query type badge, source chips, figure thumbnails) immediately, then
  streams `token` frames into the answer as they arrive.
- **Figures** — clicking a thumbnail calls `POST /api/interpret_figure`
  (plain-text stream) and shows the vision-model description in a modal.
- **Status pill** in the header pings `GET /api/status` on load.

## Structure

```
src/
  api.js                 fetch/SSE client for every api.py endpoint
  App.jsx                state + wiring
  components/
    Sidebar.jsx          upload dropzone, document list, stats
    ChatPanel.jsx         message list, suggestions, input bar
    MessageBubble.jsx     one turn (user or RAG answer)
    RouteBadge.jsx        router query_type pill (text / figure / both)
    SourceChips.jsx       expandable retrieved-chunk citations
    FigureGrid.jsx        figure thumbnails + VLM-interpretation modal
```
