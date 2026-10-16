from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_milvus import router as milvus_router
from app.core.config import get_settings
from app.core.request_logging import configure_logging, request_logging_middleware
from app.schemas import HealthResponse

settings = get_settings()
configure_logging()
app = FastAPI(title=settings.app_name)

app.middleware("http")(request_logging_middleware)

app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(ingest_router, prefix=settings.api_prefix)
app.include_router(milvus_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "Reflex UI runs separately. See README for details."}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        metadata={"milvus_uri": settings.milvus_uri, "chat_model": settings.chat_model},
    )
