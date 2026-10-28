import numpy as np
import pandas as pd
import pytest

from cust_ml.churn.models import KINDS, Calibrator, ChurnModel, time_splits


def test_time_splits_never_train_on_labels_that_close_after_validation_starts():
    cutoffs = np.repeat(pd.date_range("2017-01-31", periods=12, freq="ME"), 5)
    h = 90
    for tr, va in time_splits(cutoffs, h, n_splits=3, block=2):
        first_val = pd.Timestamp(cutoffs[va].min())
        assert (pd.to_datetime(cutoffs[tr]) + pd.Timedelta(days=h) <= first_val).all()
        assert set(tr).isdisjoint(va)
        assert pd.to_datetime(cutoffs[tr]).max() < first_val


def test_time_splits_validation_blocks_are_consecutive_and_later():
    cutoffs = np.repeat(pd.date_range("2017-01-31", periods=12, freq="ME"), 3)
    blocks = [np.unique(cutoffs[va]) for _, va in time_splits(cutoffs, 30, 3, 2)]
    starts = [b.min() for b in blocks]
    assert starts == sorted(starts) and len({tuple(b) for b in blocks}) == 3


def test_time_splits_refuse_when_there_is_no_room():
    cutoffs = np.repeat(pd.date_range("2017-01-31", periods=3, freq="ME"), 2)
    with pytest.raises(ValueError):
        list(time_splits(cutoffs, 90, n_splits=3, block=2))


def frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "recency_days": rng.integers(0, 300, n).astype(float),
            "monetary_sum": rng.gamma(2, 80, n),
            "delay_last": np.where(rng.random(n) < 0.2, np.nan, rng.normal(-8, 6, n)),
            "payment_type_last": rng.choice(["credit_card", "boleto", "voucher"], n),
            "customer_state": rng.choice(["SP", "RJ", None], n),
        }
    )
    logit = -3 - 0.01 * X["recency_days"] + 0.003 * X["monetary_sum"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.mark.parametrize("kind", KINDS)
def test_every_family_fits_handles_missing_and_unseen_categories(kind):
    X, y = frame()
    params = {"catboost": {"iterations": 20, "depth": 3}, "lightgbm": {"n_estimators": 20}}.get(
        kind, {"n_estimators": 20} if kind == "xgboost" else {}
    )
    m = ChurnModel(kind, params, seed=0, neg_frac=0.5).fit(X, y)
    Xn = X.head(20).copy()
    Xn.loc[Xn.index[0], "payment_type_last"] = "never_seen"
    p = m.predict_proba(Xn)
    assert p.shape == (20,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_negative_downsampling_keeps_every_positive():
    X, y = frame()
    m = ChurnModel("lightgbm", {"n_estimators": 5}, neg_frac=0.1).fit(X, y)
    assert m.n_fit_pos_ == y.sum() and m.n_fit_rows_ < 0.3 * len(y)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_repairs_a_systematically_overconfident_score(method):
    rng = np.random.default_rng(1)
    true_p = rng.random(20000) * 0.05
    y = (rng.random(20000) < true_p).astype(int)
    raw = np.clip(true_p * 4, 0, 1)  # four times too high
    cal = Calibrator(method).fit(raw, y).predict(raw)
    assert abs(cal.mean() - y.mean()) < 0.002 < abs(raw.mean() - y.mean())
