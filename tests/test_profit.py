import numpy as np

from cust_ml.churn.profit import (
    expected_gain,
    optimal_targeting,
    profit_at,
    profit_curve,
    realised_gain,
)


def test_expected_gain_formula():
    # p * lift * margin * value - cost
    g = expected_gain([0.02, 0.001], [200.0, 200.0], send_cost=0.05, lift=0.5, margin_rate=0.1)
    assert np.allclose(g, [0.02 * 0.5 * 0.1 * 200 - 0.05, 0.001 * 0.5 * 0.1 * 200 - 0.05])


def test_realised_gain_credits_only_the_incremental_share_of_repurchases():
    g = realised_gain([1, 0], [300.0, 0.0], send_cost=0.1, lift=0.5, margin_rate=0.1)
    # lift / (1 + lift) = 1/3 of a repurchase is attributed to the message
    assert np.allclose(g, [300 * 0.1 / 3 - 0.1, -0.1])


def test_profit_curve_is_cumulative_and_sorted_by_gain():
    gain = np.array([-2.0, 30.0, 5.0, -10.0])
    frac, cum = profit_curve(gain)
    assert np.allclose(frac, [0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(cum, [0, 30, 35, 33, 23])


def test_optimal_targeting_stops_at_last_positive_gain():
    gain = np.array([-2.0, 30.0, 5.0, -10.0])
    assert optimal_targeting(gain) == (0.5, 35.0, 2)


def test_ranking_by_probability_only_can_be_worse():
    p = np.array([0.9, 0.5])
    value = np.array([10.0, 1000.0])
    gain = expected_gain(p, value, send_cost=20.0, lift=1.0, margin_rate=1.0)  # [-11, 480]
    assert optimal_targeting(gain)[1] == 480.0
    assert profit_at(gain, p, 1) == -11.0
