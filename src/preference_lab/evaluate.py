from __future__ import annotations

import json
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return pairwise accuracy, counting ties as half-credit."""
    if not examples:
        return 0.0
    if len(examples) != len(chosen_scores) or len(examples) != len(rejected_scores):
        raise ValueError("examples, chosen_scores, and rejected_scores must have the same length")

    wins = 0
    ties = 0
    for chosen, rejected in zip(chosen_scores, rejected_scores, strict=True):
        if chosen > rejected:
            wins += 1
        elif chosen == rejected:
            ties += 1
    return (wins + 0.5 * ties) / len(examples)

def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
