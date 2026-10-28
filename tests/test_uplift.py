import numpy as np
import pytest

from cust_ml.acq import uplift as U
from cust_ml.acq.uplift import (
    BinnedCurves,
    auuc,
    default_learner,
    qini_coefficient,
    qini_curve,
    score_all,
    uplift_curve,
)


def fixture():
    # 4 treated, 4 control, hand-checkable. Sorted by score desc: indices 0..7.
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([8, 7, 6, 5, 4, 3, 2, 1.0])
    return y, t, score


def test_qini_curve_hand_computed():
    y, t, s = fixture()
    x, q = qini_curve(y, t, s)
    # k=1:1-0*1/0->1 ; k=2:2 ; k=3:2 ; k=4:2 ; k=5: 2-1*4/1=-2 ; k=8: 2-1*4/4=1
    assert np.allclose(q[[0, 1, 2, 3, 4, 5, 8]], [0, 1, 2, 2, 2, -2, 1])
    assert x[-1] == 1.0 and len(x) == 9


def test_uplift_curve_end_is_ate_times_n():
    y, t, s = fixture()
    _, u = uplift_curve(y, t, s)
    assert np.isclose(u[-1], 8 * (2 / 4 - 1 / 4))


def synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.05 + 0.2 * x * t).astype(int)  # uplift grows with x
    return x, t, y


def test_good_ordering_beats_reversed_and_random():
    x, t, y = synthetic()
    rng = np.random.default_rng(1)
    assert qini_coefficient(y, t, x) > 0 > qini_coefficient(y, t, -x)
    assert qini_coefficient(y, t, x) > qini_coefficient(y, t, rng.random(len(x)))
    assert auuc(y, t, x) > auuc(y, t, -x)


def test_random_score_has_near_zero_qini():
    x, t, y = synthetic(40000, seed=2)
    vals = [qini_coefficient(y, t, np.random.default_rng(s).random(len(y))) for s in range(8)]
    # the head of a random ranking has few controls, so single draws can be noisy: use the median
    assert abs(np.median(vals)) < 5e-4


def test_matches_scikit_uplift_if_installed():
    sk = pytest.importorskip("sklift.metrics")
    rng = np.random.default_rng(1)
    n = 500
    s = rng.random(n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.1 + 0.2 * s * t).astype(int)
    _, q = qini_curve(y, t, s)
    _, ref_q = sk.qini_curve(y, s, t)
    _, u = uplift_curve(y, t, s)
    _, ref_u = sk.uplift_curve(y, s, t)
    assert np.allclose(q, ref_q) and np.allclose(u, ref_u)


def test_binned_curves_agree_with_the_exact_functions():
    x, t, y = synthetic(30000, seed=3)
    bc = BinnedCurves(y, t, {"s": x}, n_bins=1000)
    m = bc.metrics("s")
    assert np.isclose(m["qini"], qini_coefficient(y, t, x), atol=2e-4)
    assert np.isclose(m["auuc"], auuc(y, t, x), atol=2e-4)
    assert np.isclose(m["auuc_random"], U.auuc_random(y, t), atol=1e-12)


def test_poisson_bootstrap_interval_covers_and_orders_methods():
    x, t, y = synthetic(30000, seed=4)
    noise = np.random.default_rng(0).random(len(x))
    bc = BinnedCurves(y, t, {"good": x, "noise": noise}, n_bins=200)
    boots = bc.bootstrap(["good", "noise"], n_boot=60, seed=0)
    point = bc.metrics("good")["qini"]
    lo, hi = np.quantile(boots["good"]["qini"], [0.025, 0.975])
    assert lo < point < hi
    assert (boots["good"]["qini"] - boots["noise"]["qini"] > 0).mean() > 0.95


def test_all_scorers_find_planted_heterogeneity():
    """Uplift grows with feature a; baseline response depends on feature b instead, so a plain
    response model ranks by the wrong thing."""
    import pandas as pd

    rng = np.random.default_rng(5)
    n = 30000
    X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    t = rng.integers(0, 2, n)
    p = 0.02 + 0.25 * X["b"] * (1 - t * 0.0) + 0.15 * X["a"] * t
    y = (rng.random(n) < p).astype(int)
    tr, te = np.arange(n) < 20000, np.arange(n) >= 20000
    learner = default_learner("lgbm", 0)
    learner.set_params(n_estimators=60, min_child_samples=50)
    sc = score_all(X[tr], y[tr], t[tr], X[te], learner)
    q = {k: qini_coefficient(y[te], t[te], v) for k, v in sc.items()}
    for k in ("two_model", "x_learner", "class_transformation"):
        assert q[k] > q["response_model"], q
    assert q["x_learner"] > 0
