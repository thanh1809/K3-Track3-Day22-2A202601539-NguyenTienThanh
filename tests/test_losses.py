import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_prefers_lower_loss_when_policy_margin_beats_reference() -> None:
    # Using negative logprobs (log probabilities are <= 0)
    good = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )
    bad = dpo_loss(
        np.array([-1.5]),
        np.array([-0.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )
    assert good < bad


def test_orpo_loss_adds_preference_penalty() -> None:
    # Using negative logprobs
    loss = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    assert loss > 1.0
