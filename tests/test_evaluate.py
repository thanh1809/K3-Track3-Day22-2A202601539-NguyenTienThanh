from preference_lab.evaluate import pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_counts_tie_as_half_credit() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="a", rejected="b"),
    ]
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0]) == 0.75
