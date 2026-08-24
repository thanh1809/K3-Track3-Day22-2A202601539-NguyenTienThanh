from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected

def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_load_jsonl_reports_line_number() -> None:
    bad_dir = Path("outputs/test_artifacts")
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad_file = bad_dir / "bad.jsonl"
    bad_file.write_text('{"prompt": "missing end quote}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r":1: invalid JSON"):
        load_jsonl(bad_file)


def test_split_keeps_prompts_in_one_split() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.25, seed=7)

    train_prompts = {example.prompt for example in train}
    val_prompts = {example.prompt for example in val}
    assert train_prompts.isdisjoint(val_prompts)
