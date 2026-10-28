"""Scoring helpers: calibration error, lift, and bootstrap intervals that resample by group."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE with equal-frequency bins (bins of equal width are nearly empty at a 1% base rate)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins) / len(y))


def lift_at(y, score, frac: float = 0.1) -> float:
    """Positive rate in the top ``frac`` by score divided by the overall positive rate."""
    y = np.asarray(y, float)
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def reliability_table(y, p, n_bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.array_split(np.argsort(p, kind="stable"), n_bins)
    return [
        {"n": len(b), "mean_pred": float(p[b].mean()), "observed": float(y[b].mean())} for b in bins
    ]


def group_bootstrap(
    fn: Callable[..., float],
    arrays: tuple,
    groups,
    n_boot: int = 300,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval of ``fn(*arrays, sample_weight=w)`` when whole
    groups (customers) are resampled with replacement. ``w`` is the number of times each row's
    group was drawn, so rows of one customer move together across snapshots."""
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    k = codes.max() + 1
    rng = np.random.default_rng(seed)
    point = fn(*arrays, sample_weight=np.ones(len(codes)))
    reps = []
    for _ in range(n_boot):
        draw = np.bincount(rng.integers(0, k, k), minlength=k)
        reps.append(fn(*arrays, sample_weight=draw[codes].astype(float)))
    lo, hi = np.nanquantile(reps, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def auc_w(y, s, sample_weight=None) -> float:
    return float(roc_auc_score(y, s, sample_weight=sample_weight))


def ap_w(y, s, sample_weight=None) -> float:
    return float(average_precision_score(y, s, sample_weight=sample_weight))


def percentile_ci(values, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
