"""
eval.py
Evaluation harness for the RAG system — turns "looks good on one query" into
measured numbers.

Layers:
  1. RETRIEVAL (objective, no LLM): for a question with a known correct
     (pdf, page), measure whether retrieval + rerank surface a matching chunk.
       - recall@k : gold (pdf,page) in the top-k RETRIEVED candidates
       - hit@n    : gold (pdf,page) in the top-n RERANKED chunks
       - rank/MRR : position of the first gold chunk after rerank
  2. COMPLETENESS (--full, runs llama3.2, temp=0):
       - in-scope Q : fraction of expected keypoints present in the answer
       - out-of-scope Q (should_refuse) : did it REFUSE instead of hallucinating?

Question sets (JSON list; each item may carry a "type" for grouping):
    eval/golden_set.json       — topic coverage (default)
    eval/typed_questions.json  — question-TYPE coverage + out-of-scope refusal
Item fields: question, [expected_pdf], [expected_pages], [expected_keypoints],
             [image_query], [should_refuse], [type].

Usage:
    python eval.py                                   # retrieval-only, golden set
    python eval.py --full                            # + answer completeness
    python eval.py --golden eval/typed_questions.json --full
    python eval.py --faq eval/faq/fcc_qa.json        # per-document FAQ relevance
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stderr)])

# Phrases that count as a correct refusal for an out-of-scope question.
_REFUSAL = (
    "don't contain enough", "do not contain enough", "doesn't contain enough",
    "not contain enough", "not enough information", "don't have enough",
    "no information", "cannot answer", "can't answer", "unable to answer",
)


# ------------------------------------------------------------------ #
# Matching (pdf + page aware — page numbers repeat across documents)
# ------------------------------------------------------------------ #

def _parse_pages(raw) -> set:
    out = set()
    for p in str(raw).replace(";", ",").split(","):
        p = p.strip()
        if p.isdigit():
            out.add(int(p))
    return out


def _matches(chunk: dict, item: dict, image_only: bool) -> bool:
    meta = chunk.get("metadata", {})
    if image_only and not (meta.get("modality") == "image" or meta.get("image_paths")):
        return False
    pdf = item.get("expected_pdf")
    if pdf and meta.get("pdf_name", "") != pdf:
        return False
    return bool(_parse_pages(meta.get("pages", "")) & set(item.get("expected_pages", [])))


def _recall(chunks: list, item: dict, image_only: bool) -> bool:
    return any(_matches(c, item, image_only) for c in chunks)


def _rank(chunks: list, item: dict, image_only: bool):
    for i, c in enumerate(chunks, 1):
        if _matches(c, item, image_only):
            return i
    return None


# ------------------------------------------------------------------ #
# FAQ-relevance mode (reusable per-document regression check)
# ------------------------------------------------------------------ #

def _faq_files(path: Path) -> list:
    return sorted(path.glob("*_qa.json")) if path.is_dir() else [path]


def run_faq(args) -> None:
    """For each new document, verify retrieval surfaces relevant context for its
    known questions AND refuses its out-of-scope questions — using the SAME
    multi-signal grounding gate production uses. No LLM needed (pure retrieval).

    FAQ file (``<doc>_qa.json``): a JSON list where each item is either
        {"question": "...", "answer": "..."}          -> must be ANSWERED
        {"question": "...", "should_refuse": true}     -> must be REFUSED
    """
    import numpy as np
    from ingestion.retriever import Retriever
    from ingestion.reranker import Reranker
    from agents.text_agent import (_rerank_query, GROUNDING_MIN_SCORE,
                                    GROUNDING_MIN_DENSE, _dense_sim)

    files = _faq_files(Path(args.faq))
    if not files:
        print(f"No *_qa.json files found at {args.faq}")
        return
    retriever, reranker = Retriever(), Reranker()
    emb = retriever.model

    grand_ok = grand_n = 0
    for fp in files:
        items = json.loads(fp.read_text(encoding="utf-8"))
        print(f"\n{'#' * 78}\n  {fp.name}   ({len(items)} questions)   "
              f"gate: rerank>={GROUNDING_MIN_SCORE} OR dense>={GROUNDING_MIN_DENSE}\n{'#' * 78}")
        print(f"{'exp':>6} {'question':50} {'rrank':>6} {'dense':>6} {'a_sim':>6} {'result':>7}")
        print("-" * 78)
        ok = n = 0
        for it in items:
            q = it["question"]
            refuse = bool(it.get("should_refuse"))
            ref = (it.get("answer") or "").strip()
            cands = retriever.retrieve_smart(q, top_k=args.top_k)
            reranked = reranker.rerank(_rerank_query(q), cands, top_n=args.top_n)
            top = float(reranked[0].get("rerank_score", 0.0)) if reranked else 0.0
            dense = max((_dense_sim(c) for c in reranked), default=0.0)
            grounded = bool(reranked) and (top >= GROUNDING_MIN_SCORE or dense >= GROUNDING_MIN_DENSE)

            asim = float("nan")
            if ref and reranked:               # answer<->retrieved similarity (relevance)
                rt = "\n".join((c.get("text") or c.get("child_text") or "") for c in reranked[:3])
                asim = float(np.dot(emb.encode(ref, normalize_embeddings=True, convert_to_numpy=True),
                                    emb.encode(rt, normalize_embeddings=True, convert_to_numpy=True)))

            passed = (not grounded) if refuse else grounded
            ok += passed; n += 1
            exp = "REFUSE" if refuse else "answer"
            res = "PASS" if passed else "FAIL"
            asim_s = f"{asim:6.2f}" if asim == asim else "     -"
            print(f"{exp:>6} {q[:50]:50} {top:6.2f} {dense:6.2f} {asim_s} {res:>7}"
                  f"{'  <<<' if not passed else ''}")

        print("-" * 78)
        print(f"  {fp.name}: {ok}/{n} correct   "
              f"(in-scope answered / out-of-scope refused as expected)")
        grand_ok += ok; grand_n += n

    print(f"\n{'=' * 78}\nTOTAL across {len(files)} file(s): {grand_ok}/{grand_n} "
          f"({grand_ok / grand_n:.0%}) behaved as expected\n{'=' * 78}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser(description="RAG evaluation harness")
    ap.add_argument("--full", action="store_true",
                    help="also run generation + measure completeness/refusal (slower)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "eval" / "golden_set.json"))
    ap.add_argument("--faq", default=None,
                    help="FAQ-relevance mode: path to a <doc>_qa.json file (or a folder "
                         "of them). Verifies retrieval answers in-scope questions and "
                         "refuses out-of-scope ones, using the production grounding gate.")
    args = ap.parse_args()

    if args.faq:                                  # per-document regression check
        run_faq(args)
        return

    items = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    print(f"Loaded {len(items)} questions from {args.golden}")

    from ingestion.retriever import Retriever
    from ingestion.reranker import Reranker
    retriever = Retriever()
    reranker = Reranker()

    llm = pb = adaptive = None
    min_score = min_dense = 0.0
    dsim = lambda c: 0.0
    if args.full:
        from ingestion.prompt_builder import PromptBuilder
        from llm.ollama_client import OllamaClient
        from agents.text_agent import (adaptive_context, GROUNDING_MIN_SCORE,
                                        GROUNDING_MIN_DENSE, _dense_sim)
        pb = PromptBuilder()
        adaptive = adaptive_context
        min_score = GROUNDING_MIN_SCORE
        min_dense = GROUNDING_MIN_DENSE
        dsim = _dense_sim
        llm = OllamaClient(temperature=0.0)

    rows = []
    for item in items:
        q = item["question"]
        img = bool(item.get("image_query"))
        refuse = bool(item.get("should_refuse"))
        has_pages = bool(item.get("expected_pages"))

        cands = retriever.retrieve_smart(q, top_k=args.top_k)
        reranked = reranker.rerank(q, cands, top_n=args.top_n)

        recall = _recall(cands, item, img) if has_pages else None
        rank = _rank(reranked, item, img) if has_pages else None

        cov = None
        missing = []
        if args.full and (refuse or (not img and item.get("expected_keypoints"))):
            top = float(reranked[0].get("rerank_score", 0.0)) if reranked else 0.0
            dense = max((dsim(c) for c in reranked), default=0.0)
            grounded = bool(reranked) and (top >= min_score or dense >= min_dense)
            if not grounded:                     # multi-signal grounding gate -> refuse
                ans = "the documents don't contain enough information to answer this."
            else:
                ans = llm.chat(pb.build_messages(q, adaptive([dict(c) for c in reranked]))).lower()
            if refuse:
                refused = any(p in ans for p in _REFUSAL)
                cov = 1.0 if refused else 0.0
                missing = [] if refused else ["DID NOT REFUSE (hallucinated)"]
            else:
                kps = item["expected_keypoints"]
                missing = [k for k in kps if k.lower() not in ans]
                cov = (len(kps) - len(missing)) / len(kps)

        rows.append({"q": q, "type": item.get("type", ""), "img": img, "refuse": refuse,
                     "recall": recall, "rank": rank, "cov": cov, "missing": missing})

    _scorecard(rows, args)


def _scorecard(rows: list, args) -> None:
    def pct(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    print("\n" + "=" * 82)
    print(f"{'TYPE':11} {'QUESTION':46} {'rec':>4} {'rnk':>4} {'cov':>5}")
    print("-" * 82)
    for r in rows:
        rec = "-" if r["recall"] is None else ("OK" if r["recall"] else "MISS")
        rnk = str(r["rank"]) if r["rank"] else "-"
        if r["cov"] is None:
            cov = "img" if r["img"] else "-"
        elif r["refuse"]:
            cov = "REF" if r["cov"] >= 1 else "HALL"
        else:
            cov = f"{r['cov']*100:.0f}%"
        bad = (r["recall"] is False) or (r["cov"] is not None and r["cov"] < 0.6)
        print(f"{r['type'][:11]:11} {r['q'][:46]:46} {rec:>4} {rnk:>4} {cov:>5}{'  <<<' if bad else ''}")

    print("=" * 82)
    print("AGGREGATE")
    rec_all = pct([1.0 if r["recall"] else 0.0 for r in rows if r["recall"] is not None])
    hit_all = pct([1.0 if r["rank"] else 0.0 for r in rows if r["recall"] is not None])
    mrr = pct([(1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows if r["recall"] is not None])
    cov_in = pct([r["cov"] for r in rows if r["cov"] is not None and not r["refuse"]])
    ref = [r for r in rows if r["refuse"] and r["cov"] is not None]
    if rec_all is not None:
        print(f"  recall@{args.top_k} : {rec_all:.2%}    hit@{args.top_n} : {hit_all:.2%}    MRR : {mrr:.3f}")
    if cov_in is not None:
        print(f"  keypoint coverage (in-scope answers) : {cov_in:.2%}")
    if ref:
        print(f"  refusal accuracy (out-of-scope)      : {pct([r['cov'] for r in ref]):.2%}  ({len(ref)} Q)")

    # per-type breakdown
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"] or "(untyped)"].append(r)
    print("\n  By question type:")
    for t, rs in by_type.items():
        rc = pct([1.0 if x["recall"] else 0.0 for x in rs if x["recall"] is not None])
        cv = pct([x["cov"] for x in rs if x["cov"] is not None])
        rc_s = f"recall {rc:.0%}" if rc is not None else "recall n/a"
        cv_s = f"cov {cv:.0%}" if cv is not None else "cov n/a"
        print(f"    {t:14} ({len(rs):2}) : {rc_s:14} {cv_s}")

    fails = [r for r in rows if (r["recall"] is False) or (r["cov"] is not None and r["cov"] < 0.6)]
    if fails:
        print("\n  FAILURES to look at:")
        for r in fails:
            why = []
            if r["recall"] is False:
                why.append("gold page not retrieved")
            if r["cov"] is not None and r["cov"] < 0.6:
                why.append(f"low cov {r['cov']*100:.0f}% missing={r['missing']}")
            print(f"    - [{r['type']}] {r['q']}  ({'; '.join(why)})")
    print()


if __name__ == "__main__":
    main()
