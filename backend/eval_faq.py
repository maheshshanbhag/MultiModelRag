"""
eval_faq.py
End-to-end FAQ evaluation WITH answer generation, measuring the full pipeline
(retrieve -> corrective loop -> generate) against reference answers.

Unlike eval.py --faq (retrieval-only), this routes every question through
TextAgent.run() — so it exercises the Corrective-RAG loop — then generates an
answer with llama3.2 and scores it against the FAQ reference answer by embedding
cosine similarity (an automated proxy for "does the answer match the reference").

It runs the whole set twice — loop OFF (max_retries=0) and loop ON (max_retries=1)
— to quantify the loop's effect.

    python eval_faq.py                       # both passes, generation on
    python eval_faq.py --faq eval/faq/fcc_qa.json --top-k 20 --top-n 5

Buckets (BGE answer<->reference cosine): correct >=0.72, partial 0.60-0.72,
weak <0.60; refused when the grounding gate returns no context.
"""
from __future__ import annotations

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import subprocess, sys
from pathlib import Path

_venv = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
if _venv.exists() and Path(sys.executable).resolve() != _venv.resolve():
    sys.exit(subprocess.run([str(_venv)] + sys.argv, env=os.environ.copy()).returncode)

import argparse, json, logging
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])

CORRECT, PARTIAL = 0.72, 0.60   # cosine thresholds


def _bucket(answered: bool, sim: float) -> str:
    if not answered:
        return "refused"
    if sim != sim:                      # NaN (no reference to compare)
        return "answered"
    if sim >= CORRECT:
        return "correct"
    if sim >= PARTIAL:
        return "partial"
    return "weak"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faq", default="eval/faq/fcc_qa.json")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).parent / "eval" / "faq" / "results.json"))
    args = ap.parse_args()

    items = json.loads(Path(args.faq).read_text(encoding="utf-8"))

    from agents.text_agent import TextAgent
    from agents.generator_agent import GeneratorAgent
    from llm.ollama_client import OllamaClient

    ta = TextAgent()
    gen = GeneratorAgent(llm=OllamaClient(temperature=0.0))
    emb = ta.retriever.model

    # count corrective retries per question via a thin wrapper on the rewriter
    fired = {"n": 0}
    _orig_rewrite = ta.rewriter.rewrite
    def _counting_rewrite(q):
        r = _orig_rewrite(q)
        if r:
            fired["n"] += 1
        return r
    ta.rewriter.rewrite = _counting_rewrite

    def _sim(a: str, b: str) -> float:
        if not a or not b:
            return float("nan")
        va = emb.encode(a, normalize_embeddings=True, convert_to_numpy=True)
        vb = emb.encode(b, normalize_embeddings=True, convert_to_numpy=True)
        return float(np.dot(va, vb))

    report = {}
    for retries in (0, 1):
        ta.max_retries = retries
        rows = []
        retries_fired = 0
        for it in items:
            q = it["question"]
            ref = (it.get("answer") or "").strip()
            before = fired["n"]
            chunks = ta.run(q, top_k_retrieve=args.top_k, top_n_rerank=args.top_n)
            this_fired = fired["n"] - before
            retries_fired += 1 if this_fired else 0
            answered = bool(chunks)
            ans = gen.run(query=q, chunks=chunks, image_paths=[])
            sim = _sim(ans, ref) if answered else float("nan")
            rows.append({
                "question": q, "answered": answered, "retry_fired": bool(this_fired),
                "sim": None if sim != sim else round(sim, 3),
                "bucket": _bucket(answered, sim),
                "answer": ans[:400],
            })
        # aggregate
        from collections import Counter
        cnt = Counter(r["bucket"] for r in rows)
        sims = [r["sim"] for r in rows if r["sim"] is not None]
        substantive = cnt.get("correct", 0) + cnt.get("partial", 0)
        report[f"retries_{retries}"] = {
            "buckets": dict(cnt),
            "answered": sum(1 for r in rows if r["answered"]),
            "refused": sum(1 for r in rows if not r["answered"]),
            "retries_fired": retries_fired,
            "substantive": substantive,
            "substantive_pct": round(100 * substantive / len(rows), 1),
            "mean_sim": round(float(np.mean(sims)), 3) if sims else None,
            "rows": rows,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ASCII-safe console summary
    print("\n" + "=" * 64)
    print(f"FAQ evaluation — {len(items)} questions   (top_k={args.top_k} top_n={args.top_n})")
    print("=" * 64)
    for retries in (0, 1):
        r = report[f"retries_{retries}"]
        tag = "LOOP OFF (baseline)" if retries == 0 else "LOOP ON  (corrective)"
        print(f"\n{tag}   max_retries={retries}")
        print(f"  buckets        : {r['buckets']}")
        print(f"  answered/refused: {r['answered']}/{r['refused']}")
        print(f"  retries fired  : {r['retries_fired']}")
        print(f"  substantive    : {r['substantive']}/{len(items)} = {r['substantive_pct']}%")
        print(f"  mean answer-sim: {r['mean_sim']}")
    print(f"\nfull per-question results -> {args.out}")


if __name__ == "__main__":
    main()
