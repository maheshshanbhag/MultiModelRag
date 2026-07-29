import React, { useCallback, useEffect, useRef, useState } from "react";
import { Circle, Sparkles } from "lucide-react";
import Sidebar from "./components/Sidebar.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import LandingPage from "./components/LandingPage";
import {
  getStatus,
  listDocuments,
  ingestDocument,
  getIngestStatus,
  deleteDocument,
  streamQuery,
} from "./api.js";

let idCounter = 0;
const nextId = () => `m${++idCounter}`;

export default function App() {
  const [entered, setEntered] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [totalChunks, setTotalChunks] = useState(-1);
  const [backendUp, setBackendUp] = useState(null); // null = unknown, true/false
  const [activeFilter, setActiveFilter] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isBusy, setIsBusy] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const pollTimers = useRef({});

  const refreshDocuments = useCallback(async () => {
    try {
      const data = await listDocuments();
      setTotalChunks(data.total_chunks ?? -1);
      setDocuments((prev) => {
        const running = prev.filter((d) => d.status !== "done");
        const runningByName = new Map(running.map((d) => [d.name, d]));
        const dbNames = new Set(data.documents);
        // Docs already in the DB (finished, or keep their live status if still marked running).
        const fromDb = data.documents.map((name) => {
          const existing = runningByName.get(name);
          return existing && existing.status !== "done"
            ? existing
            : { name, status: "done", chunks: null };
        });
        // Keep in-progress uploads that aren't in the DB yet (still converting) —
        // otherwise a fast upload finishing would drop a slow one from the list.
        const pending = running.filter((d) => !dbNames.has(d.name));
        return [...pending, ...fromDb];
      });
      setBackendUp(true);
    } catch {
      setBackendUp(false);
    }
  }, []);

  useEffect(() => {
    getStatus()
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false));
    refreshDocuments();
  }, [refreshDocuments]);

  // Clear any in-flight ingestion poll timers on unmount so they can't fire
  // (and call setState) after the component is gone.
  useEffect(() => {
    const timers = pollTimers.current;
    return () => {
      Object.values(timers).forEach(clearInterval);
    };
  }, []);

  const pollIngestion = useCallback(
    (pdfName) => {
      const tick = async () => {
        try {
          const s = await getIngestStatus(pdfName);
          setDocuments((prev) =>
            prev.map((d) => (d.name === pdfName ? { ...d, status: s.status, message: s.message } : d))
          );
          if (s.status === "done" || s.status === "error") {
            clearInterval(pollTimers.current[pdfName]);
            delete pollTimers.current[pdfName];
            if (s.status === "done") refreshDocuments();
          }
        } catch {
          clearInterval(pollTimers.current[pdfName]);
          delete pollTimers.current[pdfName];
        }
      };
      tick();
      pollTimers.current[pdfName] = setInterval(tick, 1500);
    },
    [refreshDocuments]
  );

  const handleUpload = useCallback(
    async (file) => {
      setIsUploading(true);
      const pdfName = file.name.replace(/\.pdf$/i, "");
      setDocuments((prev) => [
        { name: pdfName, status: "queued", message: "Uploading…" },
        ...prev.filter((d) => d.name !== pdfName),
      ]);
      try {
        await ingestDocument(file);
        pollIngestion(pdfName);
      } catch (err) {
        setDocuments((prev) =>
          prev.map((d) => (d.name === pdfName ? { ...d, status: "error", message: err.message } : d))
        );
      } finally {
        setIsUploading(false);
      }
    },
    [pollIngestion]
  );

  const handleDelete = useCallback(async (pdfName) => {
    // Stop polling this doc so a stale timer can't re-add it after deletion
    // (its ingestion, if any, is cancelled server-side).
    if (pollTimers.current[pdfName]) {
      clearInterval(pollTimers.current[pdfName]);
      delete pollTimers.current[pdfName];
    }
    setDocuments((prev) => prev.filter((d) => d.name !== pdfName));
    try {
      await deleteDocument(pdfName);
    } finally {
      setActiveFilter((f) => (f === pdfName ? null : f));
      refreshDocuments();
    }
  }, [refreshDocuments]);

  // A PDF is "processing" while it's uploading or its ingestion is queued/running.
  // Querying during this window hits a half-built index (chunks aren't in Chroma
  // and the BM25 index isn't refreshed until ingestion finishes), so we block it.
  const isIngesting =
    isUploading ||
    documents.some((d) => d.status === "queued" || d.status === "running");

  const handleSend = useCallback(
    (query) => {
      if (isIngesting) return; // guard the programmatic path too, not just the UI
      const userMsg = { id: nextId(), role: "user", content: query };
      const assistantId = nextId();
      const assistantMsg = {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "thinking",
        query,
        queryType: null,
        sources: [],
        figures: [],
        description: "",
        describing: false,
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsBusy(true);

      const patch = (id, fn) =>
        setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));

      streamQuery(
        { query, pdfFilter: activeFilter },
        {
          onMeta: (meta) =>
            patch(assistantId, (m) => ({
              ...m,
              status: "streaming",
              queryType: meta.query_type,
              sources: meta.sources,
              figures: meta.figures,
            })),
          onToken: (text) =>
            patch(assistantId, (m) => ({ ...m, content: m.content + text })),
          onVlmStart: () =>
            patch(assistantId, (m) => ({ ...m, describing: true })),
          onVlmToken: (text) =>
            patch(assistantId, (m) => ({ ...m, description: m.description + text })),
          onDone: () => {
            patch(assistantId, (m) => ({ ...m, status: "done", describing: false }));
            setIsBusy(false);
          },
          onError: (err) => {
            patch(assistantId, (m) => ({
              ...m,
              status: "done",
              describing: false,
              error: err.message || "Something went wrong",
            }));
            setIsBusy(false);
          },
        }
      );
    },
    [activeFilter, isIngesting]
  );
if (!entered) {
  return <LandingPage onEnter={() => setEntered(true)} />;
}
  return (
  <div className="h-screen bg-base-950 flex flex-col">

    {/* Navbar */}
    <header className="h-16 shrink-0 border-b border-base-700 bg-base-900/80 backdrop-blur flex items-center justify-between px-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-accent-blue/15 border border-accent-blue/25 flex items-center justify-center">
          <Sparkles size={16} className="text-accent-blue" />
        </div>
        <div>
          <h1 className="text-[15px] font-semibold text-ink-100 leading-tight">
            Offline Multimodal RAG
          </h1>
          <p className="text-[11px] text-ink-700 leading-tight">
            Local document Q&amp;A · figures read by a vision model
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 rounded-full border border-base-600 bg-base-800 px-3 py-1.5">
        <Circle
          size={8}
          className={
            backendUp === null
              ? "fill-ink-700 text-ink-700"
              : backendUp
              ? "fill-accent-green text-accent-green"
              : "fill-accent-red text-accent-red"
          }
        />
        <span className="text-[12px] font-medium text-ink-300">
          {backendUp === null ? "Connecting…" : backendUp ? "Connected" : "Offline"}
        </span>
      </div>
    </header>

    {/* Two-column layout: documents | chat (answers, figures & sources inline) */}
    <main className="flex flex-1 overflow-hidden">

      <Sidebar
        documents={documents}
        activeFilter={activeFilter}
        onSelectFilter={setActiveFilter}
        onUpload={handleUpload}
        onDelete={handleDelete}
        totalChunks={totalChunks}
        isUploading={isUploading}
      />

      <ChatPanel
        messages={messages}
        onSend={handleSend}
        isBusy={isBusy}
        isIngesting={isIngesting}
        activeFilter={activeFilter}
        onClearFilter={() => setActiveFilter(null)}
      />

    </main>

  </div>
  );
}
