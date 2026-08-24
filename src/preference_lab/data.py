from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b")


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def _contains_pii(example: PreferenceExample) -> bool:
    text = f"{example.prompt}\n{example.chosen}\n{example.rejected}"
    return bool(_EMAIL_RE.search(text) or _PHONE_RE.search(text))


def load_jsonl(path: str | Path, *, reject_pii: bool = False) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Raises ValueError with line numbers for malformed JSON, schema errors,
    duplicate prompts, and optional obvious PII patterns.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            try:
                example = PreferenceExample.model_validate(raw)
            except ValidationError as exc:
                raise ValueError(
                    f"{input_path}:{line_no}: invalid preference example: {exc}"
                ) from exc

            normalized = _normalize_prompt(example.prompt)
            if normalized in seen_prompts:
                first_line = seen_prompts[normalized]
                raise ValueError(
                    f"{input_path}:{line_no}: duplicate prompt; first seen on line {first_line}"
                )
            if reject_pii and _contains_pii(example):
                raise ValueError(f"{input_path}:{line_no}: possible PII detected")

            seen_prompts[normalized] = line_no
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    *,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    All rows with the same normalized prompt stay in the same split.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(_normalize_prompt(example.prompt), []).append(example)

    shuffled_groups = list(groups.values())
    random.Random(seed).shuffle(shuffled_groups)

    target_val_size = max(1, round(len(examples) * validation_ratio))
    validation_groups: list[list[PreferenceExample]] = []
    validation_count = 0
    for group in shuffled_groups:
        if validation_count >= target_val_size and validation_groups:
            break
        validation_groups.append(group)
        validation_count += len(group)

    validation_ids = {id(example) for group in validation_groups for example in group}
    train = [example for example in examples if id(example) not in validation_ids]
    validation = [example for group in validation_groups for example in group]
    return train, validation
