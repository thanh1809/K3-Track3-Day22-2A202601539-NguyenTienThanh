from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import pairwise_accuracy, write_metrics

app = typer.Typer(help="Preference alignment lab CLI")

@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")

@app.command()
def evaluate(config: Annotated[Path, typer.Option("--config", "-c")]) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])

    chosen_scores = [_deterministic_response_score(example.prompt, example.chosen) for example in examples]
    rejected_scores = [
        _deterministic_response_score(example.prompt, example.rejected) for example in examples
    ]
    margins = [chosen - rejected for chosen, rejected in zip(chosen_scores, rejected_scores, strict=True)]
    metrics = {
        "pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores),
        "mean_score_margin": sum(margins) / len(margins) if margins else 0.0,
    }
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


def _deterministic_response_score(prompt: str, response: str) -> float:
    """CPU-safe proxy for response quality when no model logprobs are available."""
    prompt_terms = _content_terms(prompt)
    response_terms = _content_terms(response)
    if not response_terms:
        return float("-inf")

    overlap = len(prompt_terms & response_terms) / max(1, len(prompt_terms))
    length_score = min(len(response_terms), 80) / 80.0
    specificity = math.log1p(len(set(response_terms))) / 5.0
    quality_markers = sum(
        marker in response.casefold()
        for marker in [
            "while",
            "during",
            "because",
            "for example",
            "helps",
            "used",
            "learn",
        ]
    )
    return overlap + length_score + specificity + 0.05 * quality_markers


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "how",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "while",
        "with",
    }
    cleaned = "".join(char.casefold() if char.isalnum() else " " for char in text)
    return {term for term in cleaned.split() if term not in stopwords and len(term) > 2}


if __name__ == "__main__":
    app()
