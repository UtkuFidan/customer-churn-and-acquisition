"""Olist e-commerce tables -> one clean order table -> customer snapshots at a cutoff date.

Unit of analysis is ``customer_unique_id`` (``customer_id`` is minted per order). A purchase
occasion is a customer-day: several orders placed on the same calendar day by one customer
(mostly one basket split across sellers) count once. Without that, 31% of "repeat" gaps are zero
days and the churn label would mostly measure basket splitting.

``snapshot`` is the only place where features and labels are built. Features see the world as
it was at the cutoff: a delivery date or a review timestamp that falls after the cutoff is
treated as unknown. The label looks only at purchase days in (cutoff, cutoff + horizon].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
EXCLUDED_STATUS = ("canceled", "unavailable")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_orders(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per order that was not cancelled or unavailable, with item, payment, review,
    delivery and distance attributes. Time-varying columns keep their own timestamps."""
    cust = t["customers"][
        ["customer_id", "customer_unique_id", "customer_state", "customer_zip_code_prefix"]
    ]
    o = t["orders"].merge(cust, on="customer_id", how="inner")
    o = o[~o["order_status"].isin(EXCLUDED_STATUS)].copy()
    o["day"] = o["order_purchase_timestamp"].dt.normalize()
    o["delay_days"] = (
        o["order_delivered_customer_date"] - o["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    o["delivery_days"] = (
        o["order_delivered_customer_date"] - o["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # items: totals, plus the most expensive item decides category and seller
    it = t["items"].merge(
        t["products"][["product_id", "product_category_name"]], on="product_id", how="left"
    )
    it = it.merge(t["categories"], on="product_category_name", how="left")
    it["category"] = it["product_category_name_english"].fillna(it["product_category_name"])
    it = it.merge(t["sellers"][["seller_id", "seller_state", "seller_zip_code_prefix"]], how="left")
    geo = (
        t["geo"].groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
    ).mean()
    it = it.join(geo.add_prefix("s_"), on="seller_zip_code_prefix")
    it = it.merge(o[["order_id", "customer_zip_code_prefix", "customer_state"]], on="order_id")
    it = it.join(geo.add_prefix("c_"), on="customer_zip_code_prefix")
    it["distance_km"] = haversine_km(
        it.s_geolocation_lat, it.s_geolocation_lng, it.c_geolocation_lat, it.c_geolocation_lng
    )
    it["same_state"] = (it["seller_state"] == it["customer_state"]).astype(float)
    top = it.sort_values("price", ascending=False).drop_duplicates("order_id")
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        price_sum=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        n_sellers=("seller_id", "nunique"),
        distance_km=("distance_km", "mean"),
        same_state=("same_state", "mean"),
    )
    agg = agg.join(top.set_index("order_id")[["category", "seller_state"]])
    o = o.merge(agg, left_on="order_id", right_index=True, how="left")
    o["gmv"] = o["price_sum"] + o["freight_sum"]
    o["freight_share"] = o["freight_sum"] / o["gmv"].where(o["gmv"] > 0)

    pay = t["payments"].sort_values("payment_value", ascending=False)
    main = pay.drop_duplicates("order_id").set_index("order_id")["payment_type"]
    pagg = (
        t["payments"]
        .groupby("order_id")
        .agg(installments=("payment_installments", "max"), paid=("payment_value", "sum"))
    )
    pagg["payment_type"] = main
    o = o.merge(pagg, left_on="order_id", right_index=True, how="left")

    rv = t["reviews"].copy()
    rv["review_ts"] = rv["review_answer_timestamp"].fillna(rv["review_creation_date"])
    rv["review_has_text"] = rv["review_comment_message"].fillna("").str.strip().ne("")
    rv = rv.sort_values("review_ts").drop_duplicates("order_id", keep="last")
    o = o.merge(
        rv[["order_id", "review_score", "review_has_text", "review_ts"]], on="order_id", how="left"
    )
    cols = [
        "order_id", "customer_unique_id", "customer_state", "day", "order_purchase_timestamp",
        "order_delivered_customer_date", "delay_days", "delivery_days", "n_items", "price_sum",
        "freight_sum", "gmv", "freight_share", "n_sellers", "distance_km", "same_state",
        "category", "installments", "payment_type", "review_score", "review_has_text", "review_ts",
    ]  # fmt: skip
    return (
        o[cols]
        .sort_values(["customer_unique_id", "order_purchase_timestamp"])
        .reset_index(drop=True)
    )


def top_categories(orders: pd.DataFrame, upto: pd.Timestamp, k: int = 20) -> list[str]:
    """The k most frequent categories among orders placed on or before ``upto``."""
    return orders.loc[orders["day"] <= upto, "category"].value_counts().head(k).index.tolist()


FEATURES_NUM = [
    "recency_days", "tenure_days", "frequency", "n_orders", "monetary_sum", "monetary_mean",
    "avg_gap_days", "n_items_mean", "price_mean", "freight_share_mean", "delay_last",
    "delay_mean", "delay_max", "late_share", "n_late", "delivery_days_mean",
    "days_since_delivery", "n_undelivered", "review_score_last", "review_score_mean",
    "low_review_share", "n_reviews", "review_text_share", "installments_max",
    "installments_last", "voucher_share", "distance_km_mean", "distance_km_last",
    "same_state_share", "n_sellers_mean",
]  # fmt: skip
FEATURES_CAT = ["payment_type_last", "category_last", "customer_state"]


def snapshot(
    orders: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Customer-level features at ``cutoff`` and the churn label for the next ``horizon_days``.

    Population: customers with at least one purchase on or before the cutoff.
    ``churn`` = 1 if the customer has no purchase day in (cutoff, cutoff + horizon_days].
    Also returned (for evaluation only, never as features): ``next_gmv`` (GMV of purchases in
    the label window) and ``cutoff``.
    """
    cutoff = pd.Timestamp(cutoff).normalize()
    edge = cutoff + DAY  # an event is known at the cutoff if it happened before this instant
    obs = orders.loc[orders["day"] <= cutoff].copy()

    # information that did not exist yet at the cutoff is hidden
    seen_delivery = obs["order_delivered_customer_date"] < edge
    for c in ("delay_days", "delivery_days"):
        obs[c] = obs[c].where(seen_delivery)
    seen_review = obs["review_ts"] < edge
    obs["review_score"] = obs["review_score"].where(seen_review)
    obs["review_has_text"] = obs["review_has_text"].where(seen_review).astype(float)
    obs["delivered_ts"] = obs["order_delivered_customer_date"].where(seen_delivery)

    obs["new_occasion"] = ~obs.duplicated(["customer_unique_id", "day"])
    obs["late"] = (obs["delay_days"] > 0).astype(float).where(obs["delay_days"].notna())
    obs["low_review"] = (obs["review_score"] <= 2).astype(float).where(obs["review_score"].notna())
    obs["voucher"] = (obs["payment_type"] == "voucher").astype(float)
    obs["undelivered"] = (~seen_delivery).astype(float)
    obs["category_g"] = obs["category"]
    if categories is not None:
        obs["category_g"] = obs["category"].where(obs["category"].isin(categories), "other")

    g = obs.groupby("customer_unique_id", sort=False)
    f = g.agg(
        n_orders=("order_id", "size"),
        frequency=("new_occasion", "sum"),
        first_day=("day", "min"),
        last_day=("day", "max"),
        monetary_sum=("gmv", "sum"),
        n_items_mean=("n_items", "mean"),
        price_mean=("price_sum", "mean"),
        freight_share_mean=("freight_share", "mean"),
        delay_last=("delay_days", "last"),
        delay_mean=("delay_days", "mean"),
        delay_max=("delay_days", "max"),
        late_share=("late", "mean"),
        n_late=("late", "sum"),
        delivery_days_mean=("delivery_days", "mean"),
        last_delivered=("delivered_ts", "max"),
        n_undelivered=("undelivered", "sum"),
        review_score_last=("review_score", "last"),
        review_score_mean=("review_score", "mean"),
        low_review_share=("low_review", "mean"),
        n_reviews=("review_score", "count"),
        review_text_share=("review_has_text", "mean"),
        installments_max=("installments", "max"),
        installments_last=("installments", "last"),
        voucher_share=("voucher", "mean"),
        distance_km_mean=("distance_km", "mean"),
        distance_km_last=("distance_km", "last"),
        same_state_share=("same_state", "mean"),
        n_sellers_mean=("n_sellers", "mean"),
        payment_type_last=("payment_type", "last"),
        category_last=("category_g", "last"),
        customer_state=("customer_state", "last"),
    )  # fmt: skip
    f["recency_days"] = (cutoff - f["last_day"]).dt.days
    f["tenure_days"] = (cutoff - f["first_day"]).dt.days
    f["monetary_mean"] = f["monetary_sum"] / f["frequency"]
    f["avg_gap_days"] = (f["last_day"] - f["first_day"]).dt.days / (f["frequency"] - 1).where(
        f["frequency"] > 1
    )
    f["days_since_delivery"] = (edge - f["last_delivered"]).dt.total_seconds() / 86400
    f = f.drop(columns=["first_day", "last_day", "last_delivered"])

    fut = orders.loc[(orders["day"] > cutoff) & (orders["day"] <= cutoff + horizon_days * DAY)]
    nxt = fut.groupby("customer_unique_id")["gmv"].sum()
    f["next_gmv"] = nxt.reindex(f.index).fillna(0.0)
    f["churn"] = (~f.index.isin(nxt.index)).astype(int)
    f["cutoff"] = cutoff
    f["avg_order_value"] = f["monetary_mean"]
    return f[FEATURES_NUM + FEATURES_CAT + ["avg_order_value", "next_gmv", "churn", "cutoff"]]


def make_snapshots(
    orders: pd.DataFrame, cutoffs: list, horizon_days: int, categories: list[str]
) -> pd.DataFrame:
    """Stack snapshots; index is customer_unique_id (repeated across cutoffs)."""
    return pd.concat([snapshot(orders, c, horizon_days, categories) for c in cutoffs])
