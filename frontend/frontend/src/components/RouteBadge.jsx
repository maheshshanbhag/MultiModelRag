import React from "react";
import { Grid2x2 } from "lucide-react";

const LABELS = {
  text: "text",
  image: "figure",
  both: "text + figure",
};

export default function RouteBadge({ queryType }) {
  if (!queryType) return null;
  return (
    <div className="inline-flex items-center gap-1.5 rounded-md bg-accent-amber/10 border border-accent-amber/25 px-2 py-1 text-[11px] font-medium text-accent-amber">
      <Grid2x2 size={12} />
      <span>{LABELS[queryType] || queryType}</span>
    </div>
  );
}
