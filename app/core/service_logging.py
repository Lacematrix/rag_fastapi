from __future__ import annotations

import asyncio
import inspect
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

MAX_STRING_LEN = 200
MAX_REPR_LEN = 200
MAX_ITEMS = 10
MAX_DEPTH = 2


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _sanitize(value: Any, depth: int = MAX_DEPTH) -> Any:
    if depth <= 0:
        return {"type": type(value).__name__}

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if len(value) > MAX_STRING_LEN:
            return {"type": "str", "len": len(value), "preview": _truncate(value, MAX_STRING_LEN)}
        return value

    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        items = list(value.items())
        preview = {str(k): _sanitize(v, depth - 1) for k, v in items[:MAX_ITEMS]}
        return {"type": "dict", "size": len(value), "preview": preview}

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        preview = [_sanitize(item, depth - 1) for item in seq[:MAX_ITEMS]]
        return {"type": type(value).__name__, "size": len(seq), "preview": preview}

    return {"type": type(value).__name__, "repr": _truncate(repr(value), MAX_REPR_LEN)}


def _collect_args(func: Callable[..., Any], args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(func)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        bound.arguments.pop("self", None)
        bound.arguments.pop("cls", None)
        return {name: _sanitize(value) for name, value in bound.arguments.items()}
    except Exception:
        return {"args": _sanitize(args), "kwargs": _sanitize(kwargs)}


def service_logger(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = name or f"{func.__module__}.{func.__qualname__}"
        logger = logging.getLogger("app.service")

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                payload = _collect_args(func, args, kwargs)
                start = time.perf_counter()
                logger.info("service.start %s", {"event": "service.start", "name": func_name, "args": payload})
                try:
                    result = await func(*args, **kwargs)
                except Exception:
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    logger.exception(
                        "service.error %s",
                        {"event": "service.error", "name": func_name, "duration_ms": duration_ms},
                    )
                    raise
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(
                    "service.end %s",
                    {"event": "service.end", "name": func_name, "duration_ms": duration_ms},
                )
                return result

            async_wrapper.__signature__ = inspect.signature(func)
            return async_wrapper

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = _collect_args(func, args, kwargs)
            start = time.perf_counter()
            logger.info("service.start %s", {"event": "service.start", "name": func_name, "args": payload})
            try:
                result = func(*args, **kwargs)
            except Exception:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.exception(
                    "service.error %s",
                    {"event": "service.error", "name": func_name, "duration_ms": duration_ms},
                )
                raise
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "service.end %s",
                {"event": "service.end", "name": func_name, "duration_ms": duration_ms},
            )
            return result

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator
