from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from tika import detector as tika_detector
from tika import parser as tika_parser

from app.core.service_logging import service_logger
from app.services.chunking import ChunkingService, RawChunk


class DocumentProcessor:
    def __init__(self, chunking_service: ChunkingService) -> None:
        self.chunking_service = chunking_service

    @service_logger()
    def load_text(self, file_path: Path) -> str:
        parsed = self._parse_with_tika(file_path)
        if parsed.text:
            return parsed.text

        raw_bytes = self._safe_read_bytes(file_path)
        return self._decode_bytes(raw_bytes) if raw_bytes else ""

    @service_logger()
    def chunk_file(self, file_path: Path, mode: str | None = None) -> list[RawChunk]:
        text = self.load_text(file_path)
        return self.chunking_service.chunk(text, mode=mode)

    @dataclass
    class ParsedDocument:
        text: str
        metadata: Dict[str, Any]
        mime_type: Optional[str] = None
        file_name: Optional[str] = None
        file_type: str = "unknown"

    def _parse_with_tika(self, file_path: Path) -> ParsedDocument:
        file_name = file_path.name
        try:
            result = tika_parser.from_file(str(file_path), xmlContent=False)
        except Exception:
            return self.ParsedDocument(text="", metadata={}, mime_type=None, file_name=file_name)

        metadata = result.get("metadata") or {}
        raw_text = result.get("content") or ""
        mime_type = metadata.get("Content-Type") or self._detect_mime(file_path)
        cleaned = self._normalize_text(raw_text)
        file_type = self._detect_file_type(mime_type, cleaned)
        return self.ParsedDocument(
            text=cleaned,
            metadata=metadata,
            mime_type=mime_type,
            file_name=file_name,
            file_type=file_type,
        )

    def _detect_file_type(self, mime_type: Optional[str], text: str) -> str:
        if not mime_type:
            return "unknown"

        normalized = mime_type.lower()

        if normalized.startswith("image/"):
            return "image"

        if normalized in {"application/pdf", "application/x-pdf"}:
            return "pdf_image" if not text else "pdf"

        if normalized in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
            return "csv"

        if normalized in {
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }:
            return "word"

        if normalized in {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
        }:
            return "excel"

        if normalized.startswith("text/"):
            return "text"

        return "unknown"

    def _detect_mime(self, file_path: Path) -> Optional[str]:
        try:
            return tika_detector.from_file(str(file_path))
        except Exception:
            return None

    def _safe_read_bytes(self, file_path: Path) -> bytes:
        try:
            return file_path.read_bytes()
        except Exception:
            return b""

    def _decode_bytes(self, raw_bytes: bytes) -> str:
        if not raw_bytes:
            return ""

        candidates = ["utf-8", "utf-16", "utf-16le", "utf-16be", "gb18030", "gbk", "latin-1"]
        best_text = ""
        best_score = float("inf")

        for encoding in candidates:
            try:
                decoded = raw_bytes.decode(encoding)
            except Exception:
                continue
            score = self._garbled_score(decoded)
            if score < best_score:
                best_text = decoded
                best_score = score

        if not best_text:
            best_text = raw_bytes.decode("utf-8", errors="replace")

        return self._normalize_text(best_text)

    def _garbled_score(self, text: str) -> float:
        if not text:
            return float("inf")
        bad_chars = sum(1 for ch in text if ch == "\ufffd" or ch == "\x00")
        ctrl_chars = sum(1 for ch in text if ord(ch) < 9 or (13 < ord(ch) < 32))
        return (bad_chars + ctrl_chars) / max(len(text), 1)

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\x00", "")
        cleaned = cleaned.lstrip("\ufeff")
        return cleaned.strip(" \t\n")
