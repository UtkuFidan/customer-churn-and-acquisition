import numpy as np

from cust_ml.metrics import auc_w, expected_calibration_error, group_bootstrap, lift_at


def test_ece_is_zero_when_bin_means_match():
    y = np.array([0, 0, 0, 1] * 25)
    p = np.full(100, 0.25)
    assert expected_calibration_error(y, p, 5) < 1e-12
    assert expected_calibration_error(y, np.full(100, 0.75), 5) == 0.5


def test_lift_at_top_decile():
    y = np.zeros(100)
    y[:10] = 1
    s = np.linspace(1, 0, 100)
    assert lift_at(y, s, 0.1) == 10.0 and lift_at(y, -s, 0.1) == 0.0


def test_group_bootstrap_resamples_whole_groups_and_covers_the_point():
    rng = np.random.default_rng(0)
    n_groups, per = 400, 3
    groups = np.repeat(np.arange(n_groups), per)
    y = (rng.random(n_groups * per) < 0.3).astype(int)
    s = y * 0.8 + rng.random(n_groups * per)
    point, lo, hi = group_bootstrap(auc_w, (y, s), groups, n_boot=100, seed=1)
    assert lo < point < hi and hi - lo < 0.1
    # deterministic given the seed
    assert group_bootstrap(auc_w, (y, s), groups, n_boot=100, seed=1) == (point, lo, hi)
