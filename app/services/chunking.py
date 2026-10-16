from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

from langchain_core.documents import Document

try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        HTMLHeaderTextSplitter,
        MarkdownHeaderTextSplitter,
        PythonCodeTextSplitter,
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
    )
except ImportError:  # pragma: no cover - optional dependency
    CharacterTextSplitter = None
    HTMLHeaderTextSplitter = None
    MarkdownHeaderTextSplitter = None
    PythonCodeTextSplitter = None
    RecursiveCharacterTextSplitter = None
    TokenTextSplitter = None

from app.core.config import Settings
from app.core.service_logging import service_logger


class ChunkingMode(str, Enum):
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"
    MARKDOWN_HEADER = "markdown_header"
    HTML_HEADER = "html_header"
    PYTHON_CODE = "python_code"


@dataclass
class RawChunk:
    chunk_id: str
    text: str
    index: int


LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")

class LangChainTextChunker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chunk(self, text: str, mode: ChunkingMode) -> List[RawChunk]:
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("langchain-text-splitters is not installed")

        normalized = self._normalize_text(text)
        if not normalized.strip():
            return []

        splitter = self._build_splitter(mode)
        chunks = self._split_text(splitter, normalized, mode)
        return [RawChunk(str(uuid.uuid4()), body, idx) for idx, body in enumerate(chunks) if body.strip()]

    def _build_splitter(self, mode: ChunkingMode):
        chunk_size = max(1, self.settings.chunk_size)
        overlap = max(0, min(self.settings.overlap_size, chunk_size - 1))
        separators = self.settings.chunk_separators

        if mode == ChunkingMode.CHARACTER:
            separator = separators[0] if separators else "\n\n"
            return CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                is_separator_regex=self.settings.chunk_is_separator_regex,
            )

        if mode == ChunkingMode.TOKEN:
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                encoding_name=self.settings.chunk_tokenizer_name,
            )

        if mode == ChunkingMode.MARKDOWN_HEADER:
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
                strip_headers=False,
            )

        if mode == ChunkingMode.HTML_HEADER:
            return HTMLHeaderTextSplitter(
                headers_to_split_on=[("h1", "h1"), ("h2", "h2"), ("h3", "h3")],
                return_each_element=False,
            )

        if mode == ChunkingMode.PYTHON_CODE:
            return PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators or ["\n\n", "\n", " ", ""],
        )

    def _split_text(self, splitter, text: str, mode: ChunkingMode) -> List[str]:
        parts = splitter.split_text(text)
        if not parts:
            return []
        if isinstance(parts[0], Document):
            parts = [part.page_content for part in parts]
        else:
            parts = [str(part) for part in parts]

        if mode == ChunkingMode.MARKDOWN_HEADER:
            expanded: List[str] = []
            for part in parts:
                expanded.extend(self._split_markdown_list_items(part))
            return expanded

        return parts

    def _split_markdown_list_items(self, text: str) -> List[str]:
        lines = text.splitlines()
        chunks: List[str] = []
        buffer: List[str] = []
        item: List[str] = []

        def flush_buffer() -> None:
            if buffer:
                blob = "\n".join(buffer).strip()
                if blob:
                    chunks.append(blob)
                buffer.clear()

        def flush_item() -> None:
            if item:
                blob = "\n".join(item).strip()
                if blob:
                    chunks.append(blob)
                item.clear()

        for line in lines:
            if LIST_ITEM_RE.match(line):
                flush_buffer()
                flush_item()
                item.append(line)
                continue

            if item:
                if not line.strip() or line.startswith("    ") or line.startswith("\t"):
                    item.append(line)
                else:
                    flush_item()
                    buffer.append(line)
                continue

            buffer.append(line)

        flush_item()
        flush_buffer()
        return chunks

    def _normalize_text(self, text: str) -> str:
        src = text.replace("\r\n", "\n").replace("\r", "\n")
        src = re.sub(r"(https?://[^\s]+)\n([^\s]+)", r"\1\2", src)
        src = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", src)
        return src


class ChunkingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.langchain_chunker = LangChainTextChunker(settings)

    @service_logger()
    def chunk(self, text: str, mode: str | None = None) -> List[RawChunk]:
        chunk_mode = ChunkingMode(mode or self.settings.chunk_mode)
        return self.langchain_chunker.chunk(text, chunk_mode)

    @staticmethod
    @service_logger("app.services.chunking.ChunkingService.to_documents")
    def to_documents(chunks: Sequence[RawChunk], source: str) -> List[Document]:
        docs: List[Document] = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.index,
                        "source": source,
                    },
                )
            )
        return docs
