import React from "react";
import { Database, ArrowRight } from "lucide-react";

export default function LandingPage({ onEnter }) {
  return (
    <div className="h-screen w-full flex items-center justify-center bg-base-950">

      <div className="text-center max-w-2xl px-6">

        <div className="w-20 h-20 mx-auto rounded-2xl bg-blue-600/20 flex items-center justify-center mb-6">
          <Database size={36} className="text-blue-400" />
        </div>

        <h1 className="text-5xl font-bold text-white">
          Offline Multi-Agent RAG
        </h1>

        <p className="text-gray-300 mt-4 text-lg">
          A fully offline document intelligence system with retrieval,
          figures, tables, and multi-agent reasoning.
        </p>

        <button
          onClick={onEnter}
          className="mt-8 px-6 py-3 rounded-2xl bg-blue-600 hover:bg-blue-700 text-white flex items-center gap-2 mx-auto"
        >
          Enter 
          <ArrowRight size={18} />
        </button>

      </div>
    </div>
  );
}