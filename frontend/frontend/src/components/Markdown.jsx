import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Renders the LLM answer as markdown (bold, headings, lists, code, and GFM
// tables). Styling lives in index.css under `.md`. Wide tables get a horizontal
// scroll wrapper so they never blow out the chat bubble width.
const COMPONENTS = {
  table: ({ node, ...props }) => (
    <div className="md-tablewrap">
      <table {...props} />
    </div>
  ),
  // Guard: react-markdown won't emit raw <img>, but strip any that slip through
  // (llama sometimes fabricates image links; backend already filters these).
  img: () => null,
};

export default function Markdown({ children }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children || ""}
      </ReactMarkdown>
    </div>
  );
}
