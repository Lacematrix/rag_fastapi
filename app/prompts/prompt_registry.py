from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import PromptTemplate


class PromptRegistry:
    templates_dir = Path(__file__).resolve().parent / "templates"

    @classmethod
    @lru_cache(maxsize=64)
    def get_template_text(cls, name: str) -> str:
        path = cls.templates_dir / f"{name}.st"
        return path.read_text(encoding="utf-8")

    @classmethod
    def render(cls, name: str, **kwargs: str) -> str:
        template = PromptTemplate.from_template(cls.get_template_text(name))
        return template.format(**kwargs)