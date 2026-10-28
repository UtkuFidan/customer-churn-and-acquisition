"""Profit-based targeting of a win-back message.

A repeat purchase is rare (about 1% in 90 days), so "save the churner" does not describe the
decision. The message is assumed to raise the repurchase probability of the customer it reaches by
a relative factor ``lift``: p -> p * (1 + lift). The incremental orders p * lift earn the platform
margin; the message costs ``send_cost`` whether or not it works.
"""

from __future__ import annotations

import numpy as np


def expected_gain(p_repeat, order_value, send_cost: float, lift: float, margin_rate: float):
    """Expected net gain of messaging each customer.

    gain_i = p_i * lift * margin_rate * order_value_i - send_cost
    """
    return (
        np.asarray(p_repeat) * lift * margin_rate * np.asarray(order_value, dtype=float) - send_cost
    )


def realised_gain(repeated, next_gmv, send_cost: float, lift: float, margin_rate: float):
    """Semi-synthetic realised gain from observed outcomes.

    If the message multiplies the repurchase probability by (1 + lift), a fraction
    lift / (1 + lift) of the repurchases seen among messaged customers is incremental. Each of
    those earns ``margin_rate`` of the GMV of the purchases the customer actually made."""
    share = lift / (1.0 + lift)
    return (
        np.asarray(repeated, dtype=float) * share * margin_rate * np.asarray(next_gmv, dtype=float)
        - send_cost
    )


def profit_curve(gain, rank_score=None):
    """Cumulative profit when targeting customers in descending ``rank_score`` order.

    Returns (fraction_targeted, cumulative_profit), both of length n+1 starting at (0, 0).
    ``rank_score`` defaults to ``gain`` (the profit-optimal ordering)."""
    gain = np.asarray(gain, dtype=float)
    order = np.argsort(-(gain if rank_score is None else np.asarray(rank_score)), kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(gain[order])])
    frac = np.arange(len(gain) + 1) / len(gain)
    return frac, cum


def optimal_targeting(gain, rank_score=None):
    """(best_fraction, best_profit, n_targeted) of the profit curve."""
    frac, cum = profit_curve(gain, rank_score)
    i = int(np.argmax(cum))
    return float(frac[i]), float(cum[i]), i


def profit_at(gain, rank_score, n_targeted: int) -> float:
    """Cumulative profit of the first ``n_targeted`` customers in ``rank_score`` order."""
    _, cum = profit_curve(gain, rank_score)
    return float(cum[int(n_targeted)])
