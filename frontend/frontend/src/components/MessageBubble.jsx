import React from "react";
import { Sparkles } from "lucide-react";
import RouteBadge from "./RouteBadge.jsx";
import SourceChips from "./SourceChips.jsx";
import FigureGrid from "./FigureGrid.jsx";
import Markdown from "./Markdown.jsx";

function Avatar({ role }) {
  const isUser = role === "user";
  return (
    <div
      className={`w-6 h-6 shrink-0 rounded-full flex items-center justify-center text-[11px] font-semibold ${
        isUser ? "bg-accent-blue/20 text-accent-blue" : "bg-accent-purple/20 text-accent-purple"
      }`}
    >
      {isUser ? "Y" : "R"}
    </div>
  );
}

function ThinkingDots({ label }) {
  return (
    <div className="flex items-center gap-2 text-ink-500 text-[13px]">
      <span className="flex gap-1">
        <span className="w-1.5 h-1.5 rounded-full bg-ink-500 dot-pulse" />
        <span className="w-1.5 h-1.5 rounded-full bg-ink-500 dot-pulse" style={{ animationDelay: "0.15s" }} />
        <span className="w-1.5 h-1.5 rounded-full bg-ink-500 dot-pulse" style={{ animationDelay: "0.3s" }} />
      </span>
      {label}
    </div>
  );
}

export default function MessageBubble({ message }) {
  const { role, content, queryType, sources, figures, description, describing, status, error } = message;

  if (role === "user") {
    return (
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <Avatar role="user" />
          <span className="text-[12px] font-medium text-ink-500">You</span>
        </div>
        <div className="pl-8">
          <p className="text-[14px] text-ink-100 leading-relaxed">{content}</p>
        </div>
      </div>
    );
  }

  const hasText = Boolean(content);
  const hasFigures = figures && figures.length > 0;
  const hasDescription = Boolean(description);
  const waitingForVlm = describing && !hasDescription;
  const done = status === "done";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <Avatar role="assistant" />
        <span className="text-[12px] font-medium text-ink-500">RAG</span>
        {queryType && <RouteBadge queryType={queryType} />}
      </div>

      <div className="pl-8">
        <div className="rounded-xl border border-base-600 bg-base-800/70 px-4 py-3.5 shadow-panel space-y-3">
          {error ? (
            <p className="text-[13px] text-accent-red">{error}</p>
          ) : (
            <>
              {/* Answer text — or a generating indicator until it arrives.
                  Pure "show me the figure" queries (queryType "image") have no
                  text answer, so no indicator is shown for them. */}
              {hasText ? (
                <div className="text-[14px] text-ink-100">
                  <Markdown>{content}</Markdown>
                  {status === "streaming" && !describing && <span className="caret" />}
                </div>
              ) : (
                !done &&
                queryType !== "image" && (
                  <ThinkingDots label={queryType ? "generating answer…" : "retrieving & generating…"} />
                )
              )}

              {hasFigures && <FigureGrid figures={figures} />}

              {/* Vision-model description of the figure (streams in on its own). */}
              {(describing || hasDescription) && (
                <div className="rounded-lg border border-accent-purple/25 bg-accent-purple/5 px-3 py-2.5">
                  <div className="flex items-center gap-1.5 text-[11px] font-medium text-accent-purple mb-1.5">
                    <Sparkles size={12} />
                    Figure description
                    <span className="text-ink-700 font-normal">· qwen2.5vl</span>
                  </div>
                  {waitingForVlm ? (
                    <ThinkingDots label="reading the figure with the vision model… (~60s on first run)" />
                  ) : (
                    <p className="text-[13px] text-ink-300 leading-relaxed whitespace-pre-wrap">
                      {description}
                      {describing && <span className="caret" />}
                    </p>
                  )}
                </div>
              )}

              {sources && sources.length > 0 && <SourceChips sources={sources} />}

              {done && !hasText && !hasFigures && !hasDescription && (
                <p className="text-[13px] text-ink-500">
                  No relevant content found in your notes for this question.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
