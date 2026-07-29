// Thin client for the FastAPI backend defined in api.py.
// Override the host by setting VITE_API_BASE in a .env file if the
// backend isn't running on localhost:8000.

export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function asJson(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function getStatus() {
  const res = await fetch(`${API_BASE}/api/status`);
  return asJson(res);
}

export async function listDocuments() {
  const res = await fetch(`${API_BASE}/api/documents`);
  return asJson(res);
}

export async function deleteDocument(pdfName) {
  const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(pdfName)}`, {
    method: "DELETE",
  });
  return asJson(res);
}

export async function ingestDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/ingest`, { method: "POST", body: form });
  return asJson(res);
}

export async function getIngestStatus(pdfName) {
  const res = await fetch(`${API_BASE}/api/ingest/${encodeURIComponent(pdfName)}`);
  return asJson(res);
}

export function figureUrl(url) {
  return `${API_BASE}${url}`;
}

/**
 * Runs the query as an SSE stream (GET /api/query/stream), mirroring ask.py.
 * Callbacks:
 *   onMeta({query_type, sources, figures})  once, before anything else
 *   onToken(text)                           llama answer tokens (skipped for pure "show" queries)
 *   onVlmStart({figure_id, url})            once, when the vision model starts describing a figure
 *   onVlmToken(text)                        qwen2.5vl figure-description tokens
 *   onDone()                                stream finished
 * Returns an abort() function.
 */
export function streamQuery(
  { query, topK = 20, topN = 5, pdfFilter = null },
  { onMeta, onToken, onVlmStart, onVlmToken, onDone, onError }
) {
  const params = new URLSearchParams({
    query,
    top_k: String(topK),
    top_n: String(topN),
  });
  if (pdfFilter) params.set("pdf_filter", pdfFilter);

  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/api/query/stream?${params.toString()}`, {
        signal: controller.signal,
        headers: { Accept: "text/event-stream" },
      });
      if (!res.ok || !res.body) {
        throw new Error(`Query stream failed (${res.status})`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const rawEvent = buf.slice(0, idx);
          buf = buf.slice(idx + 2);

          let eventName = "message";
          let data = "";
          for (const line of rawEvent.split("\n")) {
            if (line.startsWith("event:")) eventName = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (!data) continue;

          if (eventName === "meta") {
            onMeta?.(JSON.parse(data));
          } else if (eventName === "token") {
            onToken?.(JSON.parse(data).text);
          } else if (eventName === "vlm_start") {
            onVlmStart?.(JSON.parse(data));
          } else if (eventName === "vlm_token") {
            onVlmToken?.(JSON.parse(data).text);
          } else if (eventName === "done") {
            onDone?.();
          }
        }
      }
      onDone?.();
    } catch (err) {
      if (err.name !== "AbortError") onError?.(err);
    }
  })();

  return () => controller.abort();
}

/**
 * Streams a vision-model description of one figure (POST /api/interpret_figure).
 * Plain-text stream, not SSE. Returns an abort() function.
 */
export function streamInterpretFigure(
  { figurePath, query = "", caption = "" },
  { onToken, onDone, onError }
) {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/api/interpret_figure`, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ figure_path: figurePath, query, caption }),
      });
      if (!res.ok || !res.body) {
        throw new Error(`Interpret figure failed (${res.status})`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        onToken?.(decoder.decode(value, { stream: true }));
      }
      onDone?.();
    } catch (err) {
      if (err.name !== "AbortError") onError?.(err);
    }
  })();

  return () => controller.abort();
}
