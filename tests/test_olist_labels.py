import numpy as np
import pandas as pd

from cust_ml.churn.olist import FEATURES_CAT, FEATURES_NUM, snapshot

CUT = "2017-06-30"
H = 90


def base_rows():
    return [
        # a: bought before the cutoff, comes back 10 days after it
        dict(customer="a", purchase="2017-05-01", delivered=8),
        dict(customer="a", purchase="2017-07-10", delivered=5),
        # b: bought before the cutoff, never again
        dict(customer="b", purchase="2017-05-02", delivered=8),
        # c: next purchase exactly on the last day of the window
        dict(customer="c", purchase="2017-04-01", delivered=8),
        dict(customer="c", purchase="2017-09-28", delivered=5),  # cutoff + 90 days
        # d: next purchase one day after the window closes
        dict(customer="d", purchase="2017-04-01", delivered=8),
        dict(customer="d", purchase="2017-09-29", delivered=5),
        # e: first ever purchase after the cutoff, so not in the population
        dict(customer="e", purchase="2017-07-15", delivered=8),
        # f: two orders on the same day before the cutoff and nothing later
        dict(customer="f", purchase="2017-06-01", delivered=6, hour=9),
        dict(customer="f", purchase="2017-06-01", delivered=7, hour=15),
        # g: bought on the cutoff day itself, still in transit, never again
        dict(customer="g", purchase="2017-06-30", delivered=None),
    ]


def test_label_window_is_open_after_cutoff_and_inclusive_at_the_end(make):
    s = snapshot(make(base_rows()), CUT, H)
    assert set(s.index) == {"a", "b", "c", "d", "f", "g"}  # e starts after the cutoff
    assert s.loc["a", "churn"] == 0
    assert s.loc["b", "churn"] == 1
    assert s.loc["c", "churn"] == 0  # day cutoff + 90 counts
    assert s.loc["d", "churn"] == 1  # day cutoff + 91 does not
    assert s.loc["g", "churn"] == 1


def test_orders_on_the_same_day_are_one_purchase_occasion(make):
    s = snapshot(make(base_rows()), CUT, H)
    assert s.loc["f", "frequency"] == 1 and s.loc["f", "n_orders"] == 2
    # a same-day second order must not turn into a repeat purchase inside the label window
    assert s.loc["f", "churn"] == 1


def test_next_gmv_only_counts_the_label_window(make):
    rows = base_rows()
    rows[1]["gmv"] = 250.0  # a, inside the window
    rows[6]["gmv"] = 999.0  # d, outside the window
    s = snapshot(make(rows), CUT, H)
    assert s.loc["a", "next_gmv"] == 250.0
    assert s.loc["d", "next_gmv"] == 0.0


def test_features_do_not_depend_on_anything_after_the_cutoff(make):
    rows = base_rows()
    s1 = snapshot(make(rows), CUT, H)
    future = [r for r in rows if pd.Timestamp(r["purchase"]) > pd.Timestamp(CUT)]
    for r in future:
        r["gmv"] = 12345.0
        r["delivered"] = 1
        r["review"] = 1
    rows2 = (
        [r for r in rows if r not in future]
        + future
        + [dict(customer="b", purchase="2017-08-01", delivered=3, review=1, gmv=777.0)]
    )
    s2 = snapshot(make(rows2), CUT, H)
    cols = FEATURES_NUM + FEATURES_CAT
    pd.testing.assert_frame_equal(
        s1[cols].sort_index(), s2.loc[s1.index, cols].sort_index(), check_dtype=False
    )


def test_delivery_and_review_known_only_once_they_have_happened(make):
    rows = [
        # delivered 3 days after the cutoff: at the cutoff the parcel is still in transit
        dict(customer="x", purchase="2017-06-25", delivered=8, estimated=5, review=1),
        # delivered and reviewed before the cutoff
        dict(
            customer="y", purchase="2017-06-01", delivered=5, estimated=10, review=2, review_after=3
        ),
        # delivered before, but the review arrives after the cutoff
        dict(
            customer="z",
            purchase="2017-06-20",
            delivered=5,
            estimated=10,
            review=1,
            review_after=20,
        ),
    ]
    s = snapshot(make(rows), CUT, H)
    assert np.isnan(s.loc["x", "delay_last"]) and s.loc["x", "n_undelivered"] == 1
    assert np.isnan(s.loc["x", "review_score_last"]) and s.loc["x", "n_reviews"] == 0
    assert s.loc["y", "delay_last"] == -5 and s.loc["y", "review_score_last"] == 2
    assert s.loc["z", "delay_last"] == -5 and np.isnan(s.loc["z", "review_score_last"])


def test_recency_and_tenure_are_measured_from_the_cutoff(make):
    s = snapshot(make(base_rows()), CUT, H)
    assert s.loc["b", "recency_days"] == 59 and s.loc["b", "tenure_days"] == 59
    assert s.loc["g", "recency_days"] == 0
