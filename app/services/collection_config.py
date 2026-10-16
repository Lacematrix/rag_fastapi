from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings


@dataclass
class CollectionFieldSpec:
    name: str
    dtype: str
    is_primary: bool = False
    auto_id: bool = False
    max_length: Optional[int] = None
    dim: Optional[int] = None


@dataclass
class CollectionSpec:
    name: str
    description: str = ""
    embedding_dim: Optional[int] = None
    enable_sparse_embedding: bool = False
    index_params: Dict[str, Any] = field(default_factory=dict)
    fields: List[CollectionFieldSpec] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class CollectionConfigService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._specs: Dict[str, CollectionSpec] = {}
        self._load_from_file(settings.collection_config_path)

    def list_specs(self) -> List[CollectionSpec]:
        return list(self._specs.values())

    def list_collection_names(self) -> List[str]:
        return sorted(self._specs.keys())

    def get_spec(self, name: str) -> Optional[CollectionSpec]:
        return self._specs.get(name)

    def ensure_collection(self, name: str) -> None:
        if not self._specs:
            return
        if name not in self._specs:
            raise ValueError(f"Unknown collection name: {name}")

    def _load_from_file(self, path: Path) -> None:
        if not path or not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return

        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            fields = []
            for field_item in item.get("fields", []) or []:
                if not isinstance(field_item, dict):
                    continue
                fields.append(
                    CollectionFieldSpec(
                        name=str(field_item.get("name", "")).strip(),
                        dtype=str(field_item.get("dtype", "")).strip(),
                        is_primary=bool(field_item.get("is_primary", False)),
                        auto_id=bool(field_item.get("auto_id", False)),
                        max_length=field_item.get("max_length"),
                        dim=field_item.get("dim"),
                    )
                )
            spec = CollectionSpec(
                name=name,
                description=str(item.get("description", "")),
                embedding_dim=item.get("embedding_dim"),
                index_params=item.get("index_params") or {},
                fields=fields,
                examples=item.get("examples") or [],
            )
            self._specs[name] = spec
