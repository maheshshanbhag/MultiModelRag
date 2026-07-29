import React, { useState } from "react";
import { FileText, BookOpen, ChevronDown } from "lucide-react";

export default function SourceChips({ sources }) {
  const [openIdx, setOpenIdx] = useState(null);
  if (!sources || sources.length === 0) return null;

  const uniquePdfs = [...new Set(sources.map((s) => s.pdf_name).filter(Boolean))];

  return (
    <div className="mt-3">
      <div className="flex flex-wrap gap-1.5">
        {sources.map((s, i) => {
          const label = s.heading || s.section || `chunk ${i + 1}`;
          const open = openIdx === i;
          return (
            <button
              key={i}
              onClick={() => setOpenIdx(open ? null : i)}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-colors ${
                open
                  ? "border-accent-blue/50 bg-accent-bluedim/50 text-ink-100"
                  : "border-base-600 bg-base-800 text-ink-300 hover:border-base-500"
              }`}
            >
              <FileText size={11} className="text-ink-500 shrink-0" />
              <span className="truncate max-w-[160px]">{label}</span>
              {s.pages && <span className="text-ink-700 font-mono">p.{s.pages}</span>}
              <ChevronDown
                size={11}
                className={`text-ink-700 transition-transform ${open ? "rotate-180" : ""}`}
              />
            </button>
          );
        })}
      </div>

      {openIdx !== null && sources[openIdx] && (
        <div className="mt-2 rounded-md border border-base-600 bg-base-800/70 px-3 py-2.5">
          <p className="text-[11px] text-ink-300 leading-relaxed line-clamp-6">
            {sources[openIdx].text}
          </p>
          {sources[openIdx].rerank_score ? (
            <p className="mt-1.5 text-[10px] text-ink-700 font-mono">
              rerank score {sources[openIdx].rerank_score.toFixed(3)}
            </p>
          ) : null}
        </div>
      )}

      {uniquePdfs.length > 0 && (
        <p className="mt-2 flex items-center gap-1.5 text-[11px] text-ink-300">
          <BookOpen size={11} />
          Source{uniquePdfs.length > 1 ? "s" : ""}: {uniquePdfs.join(", ")}
        </p>
      )}
    </div>
  );
}
