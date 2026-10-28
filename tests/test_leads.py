import numpy as np
import pandas as pd

from cust_ml.acq import leads as L


def tables():
    mql = pd.DataFrame(
        {
            "mql_id": ["m1", "m2", "m3", "m4"],
            "first_contact_date": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-20", "2018-02-15"]
            ),
            "landing_page_id": ["p1", "p1", "p2", "p1"],
            "origin": ["social", "social", None, "email"],
        }
    )
    deals = pd.DataFrame(
        {
            "mql_id": ["m1", "m4"],
            "seller_id": ["s1", "s4"],
            "won_date": pd.to_datetime(["2018-01-10", "2018-03-20"]),
            "business_segment": ["pet", "toys"],
            "lead_type": ["online", "online"],
            "business_type": ["reseller", "reseller"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "seller_id": ["s1", "s1", "s1"],
            "price": [100.0, 40.0, 7.0],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "order_status": ["delivered", "delivered", "canceled"],
            "order_purchase_timestamp": pd.to_datetime(["2018-01-05", "2018-02-01", "2018-02-02"]),
        }
    )
    return {"mql": mql, "deals": deals}, items, orders


def test_value_counts_only_sales_after_the_deal_and_not_cancelled():
    t, items, orders = tables()
    df = L.build_leads(t, items, orders).set_index("mql_id")
    assert df.loc["m1", "gmv_after_win"] == 40.0  # o1 predates the deal, o3 is cancelled
    assert df.loc["m4", "gmv_after_win"] == 0.0
    assert np.isnan(df.loc["m2", "gmv_after_win"])  # never won: no value


def test_prior_counts_exclude_the_lead_itself_and_older_leads_age_out():
    t, items, orders = tables()
    df = L.build_leads(t, items, orders).set_index("mql_id")
    assert df.loc["m1", "page_leads_30d"] == 0 and df.loc["m2", "page_leads_30d"] == 1
    assert df.loc["m4", "page_leads_30d"] == 0  # m1/m2 are more than 30 days earlier
    assert df.loc["m4", "origin_leads_30d"] == 0


def test_training_label_is_the_status_known_at_the_split_date():
    t, items, orders = tables()
    df = L.build_leads(t, items, orders)
    asof = L._as_of(df, pd.Timestamp("2018-03-01"))
    assert dict(zip(df["mql_id"], asof, strict=True)) == {"m1": 1, "m2": 0, "m3": 0, "m4": 0}
    # m4 closes on 2018-03-20: open on 1 March, won if you look later
    assert L._as_of(df, pd.Timestamp("2018-04-01"))[3] == 1


def test_features_contain_only_what_is_known_when_the_lead_arrives():
    t, items, orders = tables()
    df = L.build_leads(t, items, orders)
    X = L.feature_frame(df, pages=["p1"])
    assert set(X.columns) == {
        "origin",
        "landing_page",
        "weekday",
        "page_leads_30d",
        "origin_leads_30d",
    }
    assert X.loc[2, "landing_page"] == "other" and X.loc[2, "origin"] == "missing"


def test_cv_folds_use_only_earlier_leads_and_as_of_labels():
    t, items, orders = tables()
    df = pd.concat([L.build_leads(t, items, orders)] * 1, ignore_index=True)
    for tr, va, _ in L.cv_folds(df):
        if len(tr) and len(va):
            assert df["first_contact_date"].iloc[tr].max() < df["first_contact_date"].iloc[va].min()
