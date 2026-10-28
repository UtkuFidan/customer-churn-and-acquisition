import numpy as np
import pandas as pd

from cust_ml.churn.clv import (
    DAYS_PER_MONTH,
    discounted_clv,
    fit_bgnbd,
    rfm_split,
    value_segments,
)


def test_discounted_clv_matches_the_geometric_sum_for_a_constant_rate():
    rate = 0.02  # purchases per day
    value, margin, d, months = np.array([100.0, 50.0]), 0.1, 0.01, 12

    def cum(t):
        return np.full(2, rate * t)

    got = discounted_clv(cum, months, value, margin, d)
    per_month = rate * DAYS_PER_MONTH
    expect = margin * value * sum(per_month / (1 + d) ** m for m in range(1, months + 1))
    assert np.allclose(got, expect)


def test_no_discount_no_margin_is_expected_revenue():
    got = discounted_clv(lambda t: np.array([t / 30.0]), 6, np.array([10.0]), 1.0, 0.0)
    assert np.isclose(got[0], 6 * 10.0)  # one purchase a month for 6 months


def test_bgnbd_recovers_parameters_and_forecasts_a_holdout():
    rng = np.random.default_rng(3)
    r, alpha, a, b, n, T, T_hold = 0.6, 12.0, 0.8, 2.5, 4000, 365.0, 180.0
    lam = rng.gamma(r, 1 / alpha, n)  # purchase rate per day
    p = rng.beta(a, b, n)  # dropout probability after each purchase
    freq, rec, hold = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        t, alive = 0.0, True
        while alive:
            t += rng.exponential(1 / lam[i])
            if t > T + T_hold:
                break
            if t <= T:
                freq[i] += 1
                rec[i] = t
            else:
                hold[i] += 1
            alive = rng.random() >= p[i]
    m = fit_bgnbd(freq, rec, np.full(n, T), penalizer=0.0)
    assert abs(m.params_["r"] - r) / r < 0.35
    assert abs(m.params_["alpha"] - alpha) / alpha < 0.35
    pred = m.predict(T_hold, freq, rec, np.full(n, T))
    # holdout purchases are forecast to within a few percent in total and by frequency bucket
    assert abs(pred.sum() / hold.sum() - 1) < 0.1
    zero = freq == 0
    assert abs(pred[zero].sum() / max(hold[zero].sum(), 1) - 1) < 0.35


def test_rfm_split_counts_repeat_purchases_per_customer_day():
    tx = pd.DataFrame(
        {
            "customer_unique_id": ["a", "a", "a", "a", "b"],
            "day": pd.to_datetime(
                ["2017-01-01", "2017-01-11", "2017-01-11", "2017-03-01", "2017-01-05"]
            ),
            "gmv": [10.0, 20.0, 30.0, 40.0, 5.0],
        }
    )
    d = rfm_split(tx, "2017-02-01", "2017-04-01")
    a, b = d.loc["a"], d.loc["b"]
    assert a["frequency_cal"] == 1  # two orders on 11 Jan are one occasion
    assert a["recency_cal"] == 10 and a["T_cal"] == 31
    assert a["frequency_holdout"] == 1 and b["frequency_cal"] == 0
    assert np.isclose(a["monetary_value_cal"], 50.0)  # mean repeat value: the 11 Jan basket


def test_value_segments_cover_four_cells_and_rank_by_value():
    v = np.arange(100.0)
    p = np.tile(np.arange(10.0), 10)
    seg = value_segments(v, p, 0.8, 0.8)
    assert set(seg) == {"Core", "Value at risk", "Active, lower value", "Long tail"}
    assert (seg[v >= np.quantile(v, 0.8)] != "Long tail").all()
