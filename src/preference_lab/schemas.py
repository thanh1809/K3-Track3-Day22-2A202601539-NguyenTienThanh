from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize_text(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value.strip()


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if not isinstance(chosen, str):
            return rejected

        normalized_chosen = _normalize_text(chosen)
        normalized_rejected = _normalize_text(rejected)
        similarity = SequenceMatcher(None, normalized_chosen, normalized_rejected).ratio()
        if normalized_chosen == normalized_rejected or similarity >= 0.97:
            raise ValueError("chosen and rejected must differ")
        return rejected
