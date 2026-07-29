import React, { useEffect, useRef, useState } from "react";
import { Send, Search, Filter, X, Loader2 } from "lucide-react";
import MessageBubble from "./MessageBubble.jsx";

export default function ChatPanel({ messages, onSend, isBusy, isIngesting, activeFilter, onClearFilter }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  // Sending is blocked while a query is streaming (isBusy) or a PDF is being
  // indexed (isIngesting) — querying a half-built index gives wrong answers.
  const sendDisabled = isBusy || isIngesting;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function submit(text) {
    const q = (text ?? input).trim();
    if (!q || sendDisabled) return;
    onSend(q);
    setInput("");
  }

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-base-950">
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center px-6">
              <div className="w-12 h-12 rounded-xl bg-accent-blue/10 border border-accent-blue/20 flex items-center justify-center mb-4">
                <Search size={20} className="text-accent-blue" />
              </div>
              <p className="text-ink-100 text-[15px] font-medium">Ask anything about your documents</p>
              <p className="text-ink-500 text-[13px] mt-1.5 max-w-md leading-relaxed">
                Answers cite the retrieved pages. Ask to “show” or “explain” a diagram and the
                figure appears inline, read by the vision model.
              </p>
            </div>
          )}
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-base-700 bg-base-900/60 px-6 py-4">
        <div className="max-w-5xl mx-auto">
          {activeFilter && (
            <div className="mb-2 inline-flex items-center gap-1.5 rounded-md bg-accent-bluedim/50 border border-accent-blue/30 px-2 py-1 text-[11px] text-accent-blue">
              <Filter size={11} />
              scoped to {activeFilter}
              <button onClick={onClearFilter} className="hover:text-ink-100">
                <X size={11} />
              </button>
            </div>
          )}

          {isIngesting && (
            <div className="mb-2 inline-flex items-center gap-1.5 rounded-md bg-accent-amber/10 border border-accent-amber/30 px-2 py-1 text-[11px] text-accent-amber">
              <Loader2 size={11} className="animate-spin" />
              Indexing a document — you can ask questions once it finishes.
            </div>
          )}

          <div className=" flex items-center gap-2 rounded-2xl shadow-lg border border-base-600 bg-base-800 px-4 py-3">
            <Search size={15} className="text-ink-700 shrink-0" />
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.isComposing && submit()}
              disabled={sendDisabled}
              placeholder={
                isIngesting
                  ? "Indexing a document…"
                  : "Ask anything about your documents…"
              }
              className="flex-1 bg-transparent text-[14px] text-ink-100 placeholder:text-ink-700 outline-none disabled:cursor-not-allowed"
            />
            <button
              onClick={() => submit()}
              disabled={sendDisabled || !input.trim()}
              className="text-ink-700 hover:text-accent-blue disabled:opacity-30 disabled:hover:text-ink-700 transition-colors"
              aria-label="Send"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
