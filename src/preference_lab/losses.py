from __future__ import annotations

import numpy as np


def _as_float_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _check_same_shape(*arrays: np.ndarray) -> None:
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all input arrays must have the same shape")


def _negative_log_sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.asarray(np.logaddexp(0.0, -logits), dtype=np.float64)


def _log1mexp(logp: np.ndarray) -> np.ndarray:
    logp = np.minimum(logp, -np.finfo(np.float64).eps)
    return np.asarray(
        np.where(
            logp > -np.log(2.0),
            np.log(-np.expm1(logp)),
            np.log1p(-np.exp(logp)),
        ),
        dtype=np.float64,
    )


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    DPO compares the policy preference log-ratio against the reference
    preference log-ratio and minimizes -log(sigmoid(beta * margin)).
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_chosen = _as_float_array(policy_chosen_logps, "policy_chosen_logps")
    policy_rejected = _as_float_array(policy_rejected_logps, "policy_rejected_logps")
    ref_chosen = _as_float_array(ref_chosen_logps, "ref_chosen_logps")
    ref_rejected = _as_float_array(ref_rejected_logps, "ref_rejected_logps")
    _check_same_shape(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    policy_log_ratio = policy_chosen - policy_rejected
    ref_log_ratio = ref_chosen - ref_rejected
    logits = beta * (policy_log_ratio - ref_log_ratio)
    return float(np.mean(_negative_log_sigmoid(logits)))


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    The objective is mean(SFT NLL + lambda * preference penalty), where the
    preference penalty is -log(sigmoid(log-odds(chosen) - log-odds(rejected))).
    """
    if lambda_orpo < 0:
        raise ValueError("lambda_orpo must be non-negative")
    nll = _as_float_array(sft_nll, "sft_nll")
    chosen = _as_float_array(chosen_logps, "chosen_logps")
    rejected = _as_float_array(rejected_logps, "rejected_logps")
    _check_same_shape(nll, chosen, rejected)

    chosen_log_odds = chosen - _log1mexp(chosen)
    rejected_log_odds = rejected - _log1mexp(rejected)
    preference_penalty = _negative_log_sigmoid(chosen_log_odds - rejected_log_odds)
    return float(np.mean(nll + lambda_orpo * preference_penalty))
