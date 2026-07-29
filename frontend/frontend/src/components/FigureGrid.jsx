import React from "react";
import { ImageIcon } from "lucide-react";
import { figureUrl } from "../api.js";

// Figures shown inline in the chat. Clicking opens the full-size image in a new
// tab — no vision-model call here (the AI description streams separately below).
export default function FigureGrid({ figures }) {
  if (!figures || figures.length === 0) return null;

  return (
    <div className="mt-3">
      <div className="flex items-center gap-1.5 text-[11px] font-medium text-ink-500 mb-1.5">
        <ImageIcon size={12} />
        {figures.length > 1 ? `${figures.length} figures` : "Figure"}
      </div>
      <div className="flex flex-wrap gap-3">
        {figures.map((fig, i) => (
          <a
            key={fig.figure_id || i}
            href={figureUrl(fig.url)}
            target="_blank"
            rel="noreferrer"
            title="Open full image"
            className="block rounded-lg overflow-hidden border border-base-600 bg-base-950 hover:border-accent-blue/60 transition-colors"
          >
            <img
              src={figureUrl(fig.url)}
              alt={fig.figure_id || `figure ${i + 1}`}
              className="max-h-72 w-auto object-contain"
            />
          </a>
        ))}
      </div>
    </div>
  );
}
