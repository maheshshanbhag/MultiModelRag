import React, { useCallback, useRef, useState } from "react";
import { UploadCloud, FileText, X, Loader2 } from "lucide-react";

function StatusDot({ status }) {
  const color =
    status === "done"
      ? "bg-accent-green"
      : status === "error"
      ? "bg-accent-red"
      : status === "running" || status === "queued"
      ? "bg-accent-amber dot-pulse"
      : "bg-ink-700";
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${color}`} />;
}

export default function Sidebar({
  documents,
  activeFilter,
  onSelectFilter,
  onUpload,
  onDelete,
  totalChunks,
  isUploading,
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(
    (files) => {
      const pdf = Array.from(files).find((f) => f.name.toLowerCase().endsWith(".pdf"));
      if (pdf) onUpload(pdf);
    },
    [onUpload]
  );

  return (
    <aside className="w-72 shrink-0 h-full flex flex-col border-r border-base-700 bg-base-900 shadow-xl">
      <div className="px-4 pt-5 pb-3">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-7 h-7 rounded-md bg-accent-blue/15 flex items-center justify-center">
            <FileText size={14} className="text-accent-blue" />
          </div>
          <div>
            <p className="text-ink-100 text-sm font-semibold leading-tight">Offline RAG</p>
            <p className="text-ink-700 text-[11px] leading-tight">multimodal document Q&A</p>
          </div>
        </div>

        <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase mb-2">
          Upload
        </p>
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed px-4 py-6 cursor-pointer transition-colors ${
            dragOver
              ? "border-accent-blue bg-accent-blue/10"
              : "border-base-600 bg-base-800/60 hover:border-base-500 hover:bg-base-800"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(e) => e.target.files?.length && handleFiles(e.target.files)}
          />
          {isUploading ? (
            <Loader2 size={20} className="text-accent-blue animate-spin" />
          ) : (
            <UploadCloud size={20} className="text-ink-500" />
          )}
          <span className="text-[13px] text-ink-300 font-medium">
            {isUploading ? "Uploading…" : "Drop a PDF or browse"}
          </span>
          <span className="text-[11px] text-ink-700">Digital or scanned</span>
        </label>
      </div>

      <div className="px-4 pb-2 flex items-center justify-between">
        <p className="text-[11px] font-medium tracking-wide text-ink-500 uppercase">
          Documents · {documents.length}
        </p>
        {activeFilter && (
          <button
            onClick={() => onSelectFilter(null)}
            className="text-[11px] text-accent-blue hover:underline"
          >
            clear filter
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1">
        {documents.length === 0 && (
          <p className="text-ink-700 text-[12px] px-2 py-4 text-center">
            No documents yet. Upload a PDF to get started.
          </p>
        )}
        {documents.map((doc) => {
          const selected = activeFilter === doc.name;
          return (
            <div
              key={doc.name}
              onClick={() => onSelectFilter(selected ? null : doc.name)}
              className={`group relative flex items-start gap-2.5 rounded-md px-2.5 py-2.5 cursor-pointer border transition-colors ${
                selected
                  ? "bg-accent-bluedim/60 border-accent-blue/40"
                  : "border-transparent hover:bg-base-800"
              }`}
            >
              <div className="w-7 h-7 mt-0.5 rounded bg-accent-red/10 flex items-center justify-center shrink-0">
                <FileText size={13} className="text-accent-red/80" />
              </div>
              <div className="min-w-0 flex-1">
                <p
                  className={`text-[13px] font-medium truncate ${
                    selected ? "text-accent-blue" : "text-ink-100"
                  }`}
                  title={doc.name}
                >
                  {doc.name}
                </p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <StatusDot status={doc.status} />
                  <span className="text-[11px] text-ink-700 font-mono">
                    {doc.status === "running" || doc.status === "queued"
                      ? doc.message || "processing…"
                      : doc.chunks != null
                      ? `${doc.chunks} chunks`
                      : "indexed"}
                  </span>
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(doc.name);
                }}
                className="opacity-0 group-hover:opacity-100 text-ink-700 hover:text-accent-red transition-opacity p-1 -m-1"
                aria-label={`Remove ${doc.name}`}
              >
                <X size={13} />
              </button>
            </div>
          );
        })}
      </div>

      <div className="border-t border-base-700 px-4 py-3 grid grid-cols-2 gap-2">
        <div className="rounded-md bg-base-800 px-3 py-2 text-center">
          <p className="text-base font-semibold text-ink-100 font-mono">{documents.length}</p>
          <p className="text-[10px] uppercase tracking-wide text-ink-700">documents</p>
        </div>
        <div className="rounded-md bg-base-800 px-3 py-2 text-center">
          <p className="text-base font-semibold text-ink-100 font-mono">
            {totalChunks >= 0 ? totalChunks : "—"}
          </p>
          <p className="text-[10px] uppercase tracking-wide text-ink-700">chunks</p>
        </div>
      </div>
    </aside>
  );
}
