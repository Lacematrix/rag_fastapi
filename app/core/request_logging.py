from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import Request, Response


MAX_BODY_LOG_BYTES = 4096


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _summarize_body(body: bytes, content_type: str) -> Dict[str, Any]:
    if not body:
        return {"body_bytes": 0}

    if "multipart/form-data" in content_type or "application/octet-stream" in content_type:
        return {"body_bytes": len(body), "body_truncated": True}

    text: Optional[str]
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return {"body_bytes": len(body), "body_truncated": True}

    if len(text) > MAX_BODY_LOG_BYTES:
        text = text[:MAX_BODY_LOG_BYTES]
        return {"body_preview": text, "body_bytes": len(body), "body_truncated": True}

    if "application/json" in content_type:
        try:
            payload = json.loads(text)
            return {"body": payload, "body_bytes": len(body)}
        except json.JSONDecodeError:
            pass

    return {"body_preview": text, "body_bytes": len(body)}


async def request_logging_middleware(request: Request, call_next) -> Response:
    logger = logging.getLogger("app.request")
    request_id = uuid.uuid4().hex
    start = time.perf_counter()

    body = await request.body()
    content_type = request.headers.get("content-type", "")

    log_payload = {
        "event": "request.start",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "client": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    if body:
        log_payload.update(_summarize_body(body, content_type))

    logger.info("request.start %s", log_payload)

    async def receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(request.scope, receive)

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "request.error %s",
            {
                "event": "request.error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["x-request-id"] = request_id

    logger.info(
        "request.end %s",
        {
            "event": "request.end",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response
