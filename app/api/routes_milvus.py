from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter
from pymilvus import Collection, utility

from app.core.container import get_container
from app.schemas import CollectionConfigView, MilvusCollectionView, MilvusFieldView

router = APIRouter(tags=["milvus"])


def _is_vector_dtype(raw_dtype: object) -> bool:
    return "VECTOR" in str(raw_dtype).upper()


def _is_viewable_dtype(raw_dtype: object) -> bool:
    return not _is_vector_dtype(raw_dtype)


@router.get("/milvus/config/collections", response_model=List[CollectionConfigView])
def list_configured_collections() -> List[CollectionConfigView]:
    container = get_container()
    specs = container.collection_config.list_specs()
    return [CollectionConfigView(name=spec.name, description=spec.description) for spec in specs]


@router.get("/milvus/collections", response_model=List[MilvusCollectionView])
def list_collections() -> List[MilvusCollectionView]:
    container = get_container()
    # Ensure the Milvus connection is initialized.
    _ = container.vector_store_service

    rows: List[MilvusCollectionView] = []
    for name in utility.list_collections():
        try:
            collection = Collection(name)
            num_entities = int(collection.num_entities)
        except Exception:
            num_entities = 0
        rows.append(MilvusCollectionView(name=name, num_entities=num_entities))
    return rows


@router.get("/milvus/collections/{collection_name}/schema", response_model=List[MilvusFieldView])
def get_collection_schema(collection_name: str) -> List[MilvusFieldView]:
    container = get_container()
    _ = container.vector_store_service

    collection = Collection(collection_name)
    fields: List[MilvusFieldView] = []
    for field in collection.schema.fields:
        fields.append(
            MilvusFieldView(
                name=field.name,
                dtype=str(field.dtype),
                is_primary=bool(field.is_primary),
                auto_id=bool(field.auto_id),
                is_vector=_is_vector_dtype(field.dtype),
                viewable=_is_viewable_dtype(field.dtype),
            )
        )
    return fields


@router.get("/milvus/collections/{collection_name}/entities", response_model=List[Dict[str, Any]])
def get_collection_entities(
    collection_name: str,
    limit: int = 50,
    offset: int = 0,
    output_fields: str | None = None,
) -> List[Dict[str, Any]]:
    container = get_container()
    _ = container.vector_store_service

    collection = Collection(collection_name)
    collection.load()

    vector_fields = {
        field.name
        for field in collection.schema.fields
        if _is_vector_dtype(field.dtype)
    }
    viewable_fields = [field.name for field in collection.schema.fields if _is_viewable_dtype(field.dtype)]

    if output_fields:
        fields = [item.strip() for item in output_fields.split(",") if item.strip() and item.strip() not in vector_fields]
    else:
        fields = viewable_fields

    if not fields:
        fields = viewable_fields or ["id"]

    expr = 'id != ""'
    rows = []
    try:
        try:
            rows = collection.query(expr=expr, output_fields=fields, limit=limit, offset=offset)
        except TypeError:
            rows = collection.query(expr=expr, output_fields=fields, limit=limit)
    except Exception as e:
        # If query fails (potentially due to sparse_embedding issues), retry with only non-sparse fields
        safe_fields = [f for f in fields if f != "sparse_embedding"]
        if safe_fields and safe_fields != fields:
            try:
                try:
                    rows = collection.query(expr=expr, output_fields=safe_fields, limit=limit, offset=offset)
                except TypeError:
                    rows = collection.query(expr=expr, output_fields=safe_fields, limit=limit)
            except Exception:
                # If still failing, return empty results or just the ID field
                pass

    results: List[Dict[str, Any]] = []
    for row in rows:
        normalized: Dict[str, Any] = dict(row)
        if "metadata" in normalized and isinstance(normalized["metadata"], str):
            try:
                normalized["metadata"] = json.loads(normalized["metadata"])
            except (json.JSONDecodeError, TypeError):
                normalized["metadata"] = normalized["metadata"]
        results.append(normalized)
    return results
