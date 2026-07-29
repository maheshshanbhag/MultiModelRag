"""
agents/image_agent.py

NOTE: Reconstructed from its call sites (ask.py / api.py) after the original was
permanently deleted — functionally equivalent, may differ cosmetically.

Image Retrieval Agent — surfaces the figure(s) attached to the most relevant
retrieved chunks. Figures are anchored on text at ingest time (see
semantic_chunking._figure_text), so by the time we get reranked chunks their
metadata already carries `image_paths` / `figure_ids` (comma-joined strings).

    figure_ids, image_paths = ImageAgent().run(chunks, max_images=2)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ImageAgent:

    def run(self, chunks: List[dict], max_images: int = 2) -> Tuple[List[str], List[str]]:
        """Return (figure_ids, image_paths) from the given chunks, best-first,
        de-duplicated and capped at max_images."""
        figure_ids: List[str] = []
        image_paths: List[str] = []
        seen: set = set()

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            paths = [p.strip() for p in str(meta.get("image_paths", "")).split(",") if p.strip()]
            fids = [f.strip() for f in str(meta.get("figure_ids", "")).split(",") if f.strip()]
            for i, path in enumerate(paths):
                if path in seen:
                    continue
                seen.add(path)
                image_paths.append(path)
                figure_ids.append(fids[i] if i < len(fids) else "")
                if len(image_paths) >= max_images:
                    return figure_ids, image_paths

        return figure_ids, image_paths
