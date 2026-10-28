"""Customer lifetime value with BG/NBD (purchase counts) and Gamma-Gamma (order value).

Fitting uses the ``lifetimes`` package. The parts that are plain arithmetic (discounting a
forecast of purchases, risk/value segmentation) live here so they can be tested on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data

DAYS_PER_MONTH = 30.0


def rfm_split(
    orders: pd.DataFrame, calibration_end: str, observation_end: str, value_col: str = "gmv"
) -> pd.DataFrame:
    """Frequency / recency / age (days) / mean repeat value up to ``calibration_end`` and the
    purchases observed afterwards. One transaction per customer-day."""
    tx = orders.groupby(["customer_unique_id", "day"], as_index=False)[value_col].sum()
    return calibration_and_holdout_data(
        tx,
        "customer_unique_id",
        "day",
        calibration_period_end=calibration_end,
        observation_period_end=observation_end,
        freq="D",
        monetary_value_col=value_col,
    )


def fit_bgnbd(freq, recency, age, penalizer: float = 0.001) -> BetaGeoFitter:
    return BetaGeoFitter(penalizer_coef=penalizer).fit(freq, recency, age)


def fit_gamma_gamma(freq, monetary, penalizer: float = 0.001) -> GammaGammaFitter:
    """Fitted on repeat buyers only (a mean repeat value exists only if frequency > 0)."""
    m = (np.asarray(freq) > 0) & (np.asarray(monetary) > 0)
    return GammaGammaFitter(penalizer_coef=penalizer).fit(
        np.asarray(freq)[m], np.asarray(monetary)[m]
    )


def discounted_clv(
    expected_purchases_up_to,
    horizon_months: int,
    value_per_order,
    margin_rate: float,
    monthly_discount: float,
):
    """CLV = margin * order value * sum_m [N(m) - N(m-1)] / (1 + d)^m.

    ``expected_purchases_up_to(t_days)`` returns the expected cumulative purchases by day t for
    every customer; month m ends at ``m * 30`` days and its purchases are discounted m months."""
    value_per_order = np.asarray(value_per_order, dtype=float)
    total = np.zeros_like(value_per_order)
    prev = np.zeros_like(value_per_order)
    for m in range(1, horizon_months + 1):
        cum = np.asarray(expected_purchases_up_to(m * DAYS_PER_MONTH), dtype=float)
        total += (cum - prev) / (1.0 + monthly_discount) ** m
        prev = cum
    return margin_rate * value_per_order * total


def value_segments(value, propensity, q_value: float = 0.8, q_propensity: float = 0.8):
    """2x2 split: high value = top (1 - q_value) of CLV; high propensity likewise.

    'Core' is high/high. 'Value at risk' is high value with low repurchase propensity."""
    v, p = np.asarray(value), np.asarray(propensity)
    hv = v >= np.quantile(v, q_value)
    hp = p >= np.quantile(p, q_propensity)
    return np.where(
        hv & hp,
        "Core",
        np.where(hv, "Value at risk", np.where(hp, "Active, lower value", "Long tail")),
    )
