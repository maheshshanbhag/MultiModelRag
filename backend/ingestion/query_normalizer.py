"""
query_normalizer.py
Query-time spell normalization against the CORPUS vocabulary (no LLM, no deps).

Typos hurt retrieval — BM25 is exact-match ("standpip" != "standpipe") and dense
similarity degrades as key terms are garbled. This corrects each query word toward
the nearest word that actually appears in the indexed chunks, so a misspelled query
matches the documents again. Because the dictionary is built from YOUR documents,
it fixes domain terms a generic spellchecker would miss (e.g. "rector" -> "reactor",
since "rector" never occurs in a petroleum handbook but "reactor" does).

Only misspelled words are touched:
  * words already in the corpus vocabulary are kept as-is (incl. common words
    like "why/do/we" — the retriever down-weights those anyway),
  * ALL-CAPS tokens (acronyms: MAT, HCN) are left alone,
  * numbers, units and short tokens (<3 letters) are never rewritten.

Runs in a few ms/query; correct words skip straight through (dict lookup), so
only the actual typos pay the edit-distance cost.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_FIX_TOKEN = re.compile(r"[A-Za-z]{3,}")     # only correct alphabetic words len>=3
_VOCAB_TOKEN = re.compile(r"[a-z]{2,}")      # vocabulary tokens (incl. 2-char words)
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


class QueryNormalizer:

    def __init__(self, vocab: Counter):
        self.vocab = vocab
        self.words = set(vocab)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "QueryNormalizer":
        v: Counter = Counter()
        for t in texts:
            v.update(_VOCAB_TOKEN.findall((t or "").lower()))
        return cls(v)

    # ---------------------------------------------------------------- #

    @staticmethod
    def _edits1(word: str) -> set:
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in _LETTERS]
        inserts = [a + c + b for a, b in splits for c in _LETTERS]
        return set(deletes + transposes + replaces + inserts)

    def _best(self, w: str) -> str:
        """Most frequent corpus word within edit distance 1, then 2; else w.

        Only corrects to a candidate that is NOT shorter than the query word: a
        real typo usually DROPS a letter (so the fix ADDS one, e.g. catlyst ->
        catalyst), whereas "correcting" a valid word to a shorter one tends to
        mangle it (e.g. flow -> low, image -> ...). If no same-or-longer corpus
        word is within reach, the word is left untouched.
        """
        if w in self.words:
            return w
        e1 = [c for c in self._edits1(w) if c in self.words and len(c) >= len(w)]
        if e1:
            return max(e1, key=lambda x: self.vocab[x])
        if len(w) >= 5:                      # distance-2 only for longer words
            e2 = {c2 for c1 in self._edits1(w)
                  for c2 in self._edits1(c1) if c2 in self.words and len(c2) >= len(w)}
            if e2:
                return max(e2, key=lambda x: self.vocab[x])
        return w

    def _fix(self, m: re.Match) -> str:
        w = m.group(0)
        if w.isupper():                      # acronym (MAT, HCN) — leave it
            return w
        lw = w.lower()
        if lw in self.words:                 # already a corpus word — keep
            return w
        best = self._best(lw)
        if best == lw:
            return w
        return best.capitalize() if w[0].isupper() else best

    def normalize(self, query: str) -> str:
        if not query or not self.words:
            return query
        return _FIX_TOKEN.sub(self._fix, query)
