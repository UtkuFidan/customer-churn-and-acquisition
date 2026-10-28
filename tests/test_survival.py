import numpy as np

from cust_ml.churn import survival as S


def rows():
    return [
        # a: first order delivered day 8, reviewed day 10; buys again 40 days later
        dict(customer="a", purchase="2017-01-01", delivered=8, review=5, review_after=2),
        dict(customer="a", purchase="2017-02-20", delivered=5, review=4),
        # b: never buys again; late delivery, bad review
        dict(
            customer="b",
            purchase="2017-01-01",
            delivered=30,
            estimated=20,
            review=1,
            review_after=1,
        ),
        # c: buys again on day 4, before the first order is delivered: excluded
        dict(customer="c", purchase="2017-01-01", delivered=8, review=5),
        dict(customer="c", purchase="2017-01-05", delivered=5),
        # d: first order never delivered: excluded
        dict(customer="d", purchase="2017-01-01", delivered=None),
        # e: first order known only after the end of follow-up: excluded
        dict(customer="e", purchase="2017-06-01", delivered=15),
    ]


def test_cohort_origin_is_when_the_first_order_is_fully_known(make):
    c, n = S.first_purchase_cohort(make(rows()), "2017-06-10")
    assert set(c.index) == {"a", "b"}
    assert n["repurchased_before_t0"] == 1 and n["no_delivery_yet"] == 1 and n["past_end"] == 1
    # a: t0 = review answered on 2017-01-11 (delivery + 2 days); second purchase 2017-02-20
    assert c.loc["a", "t0"].normalize().strftime("%Y-%m-%d") == "2017-01-11"
    assert c.loc["a", "duration"] == 40 and c.loc["a", "event"] == 1
    # b: censored at the end of follow-up
    assert c.loc["b", "event"] == 0
    assert c.loc["b", "duration"] == (
        np.datetime64("2017-06-10") - np.datetime64("2017-02-01")
    ).astype(int)


def test_delivery_delay_band_and_flags(make):
    c, _ = S.first_purchase_cohort(make(rows()), "2017-06-10")
    assert c.loc["b", "late"] == 1 and c.loc["b", "delay_band"] == "8+ days late"
    assert c.loc["a", "late"] == 0 and c.loc["b", "low_review"] == 1


def test_concordance_ci_is_one_for_a_perfect_ranking_and_half_for_noise():
    rng = np.random.default_rng(0)
    d = rng.exponential(100, 400)
    e = np.ones(400, int)
    c = S.concordance_ci(d, e, -d, n_boot=30)
    assert c[0] == 1.0 and c[1] == 1.0
    noise = S.concordance_ci(d, e, rng.random(400), n_boot=60, seed=1)
    assert noise[1] < 0.5 < noise[2]


def test_boosted_cox_ranks_the_planted_risk_factor_first():
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({"risk": rng.normal(size=n), "noise": rng.normal(size=n)})
    T = rng.exponential(100 * np.exp(-0.8 * X["risk"]))
    cens = rng.exponential(150, n)
    dur, ev = np.minimum(T, cens).clip(0.5), (T <= cens).astype(int)
    m = S.fit_boosted_cox(X, dur, ev, {"n_rounds": 60})
    r = S.predict_boosted(m, X)
    assert S.concordance_ci(dur, ev, r, n_boot=20)[0] > 0.6
