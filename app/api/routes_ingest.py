from __future__ import annotations

import uuid
from pathlib import Path

import logging
import traceback

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.container import get_container
from app.schemas import IngestResponse

router = APIRouter(tags=["ingest"])
logger = logging.getLogger(__name__)


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str | None = Form(None),
    chunk_mode: str | None = Form(None),
    bm25_language: str | None = Form(None),
) -> IngestResponse:
    container = get_container()
    settings = container.settings
    collection_name = collection_name or settings.default_collection

    suffix = Path(file.filename or "").suffix
    target_name = f"{uuid.uuid4().hex}{suffix}"
    target_path = container.settings.uploads_dir / target_name

    data = await file.read()
    target_path.write_bytes(data)

    try:
        container.collection_config.ensure_collection(collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        chunk_count = container.ingestion_service.ingest_file(
            file_path=target_path,
            collection_name=collection_name,
            chunk_mode=chunk_mode,
            bm25_language=bm25_language,
        )
    except Exception as exc:
        logger.error(
            "Ingestion failed for %s\n%s",
            target_path,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="Ingestion failed") from exc

    return IngestResponse(collection_name=collection_name, chunk_count=chunk_count)
