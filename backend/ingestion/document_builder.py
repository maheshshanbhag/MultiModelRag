"""
document_builder.py

NOTE: This file was RECONSTRUCTED from its usage across the codebase after the
original was permanently deleted. It is functionally equivalent (fields + the
save/load JSON contract match every call site) but may differ cosmetically from
your original. Replace with the recovered original if file-recovery succeeds.

Defines the DocumentElement contract that Docling ingestion emits and the
semantic chunker consumes, plus DocumentBuilder for persisting a document's
elements to  data/documents/<pdf>/document.json.

DocumentElement fields (inferred from docling_ingestor.py construction and
semantic_chunking.py access):
    id, page, order, type, bbox, content, image_path, section, heading, figure_id
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class DocumentElement:
    id: str = ""
    page: int = 1
    order: int = 0
    type: str = "paragraph"        # heading | paragraph | caption | table | figure
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    content: str = ""
    image_path: str = ""
    section: str = ""
    heading: str = ""
    figure_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentBuilder:
    """Persists / loads the per-document element list as document.json."""

    def __init__(self):
        self.base = Path(__file__).resolve().parent.parent
        self.document_dir = self.base / "data" / "documents"

    # ---------------------------------------------------------------- #

    def save_document(self, pdf_name: str, elements: List[DocumentElement]) -> Path:
        folder = self.document_dir / pdf_name
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / "document.json"
        data = [e.to_dict() if isinstance(e, DocumentElement) else e for e in elements]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path

    def load_document(self, pdf_name: str) -> List[DocumentElement]:
        path = self.document_dir / pdf_name / "document.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [DocumentElement(**d) for d in data]
