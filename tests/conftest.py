import pandas as pd
import pytest

ORDER_COLS = [
    "order_id", "customer_unique_id", "customer_state", "day", "order_purchase_timestamp",
    "order_delivered_customer_date", "delay_days", "delivery_days", "n_items", "price_sum",
    "freight_sum", "gmv", "freight_share", "n_sellers", "distance_km", "same_state", "category",
    "installments", "payment_type", "review_score", "review_has_text", "review_ts",
]  # fmt: skip


def make_orders(rows: list[dict]) -> pd.DataFrame:
    """Order table in the shape ``build_orders`` returns, from a few hand-written rows.

    Each row needs customer, purchase (date string) and may set delivered (days after purchase),
    estimated (days after purchase), gmv, review (score) and review_after (days after delivery)."""
    out = []
    for i, r in enumerate(rows):
        ts = pd.Timestamp(r["purchase"]) + pd.Timedelta(hours=r.get("hour", 10))
        deliv = ts + pd.Timedelta(days=r["delivered"]) if r.get("delivered") is not None else pd.NaT
        est = ts + pd.Timedelta(days=r.get("estimated", 20))
        gmv = r.get("gmv", 100.0)
        has_review = r.get("review") is not None and deliv is not pd.NaT
        out.append(
            {
                "order_id": f"o{i}",
                "customer_unique_id": r["customer"],
                "customer_state": "SP",
                "day": ts.normalize(),
                "order_purchase_timestamp": ts,
                "order_delivered_customer_date": deliv,
                "delay_days": (deliv - est).total_seconds() / 86400
                if deliv is not pd.NaT
                else None,
                "delivery_days": (deliv - ts).total_seconds() / 86400
                if deliv is not pd.NaT
                else None,
                "n_items": 1,
                "price_sum": gmv * 0.9,
                "freight_sum": gmv * 0.1,
                "gmv": gmv,
                "freight_share": 0.1,
                "n_sellers": 1,
                "distance_km": 500.0,
                "same_state": 0.0,
                "category": "toys",
                "installments": 1,
                "payment_type": "credit_card",
                "review_score": r["review"] if has_review else None,
                "review_has_text": bool(r.get("text", False)) if has_review else None,
                "review_ts": deliv + pd.Timedelta(days=r.get("review_after", 2))
                if has_review
                else pd.NaT,
            }
        )
    df = pd.DataFrame(out)[ORDER_COLS]
    for c in ("delay_days", "delivery_days", "review_score"):
        df[c] = df[c].astype(float)
    return df.sort_values(["customer_unique_id", "order_purchase_timestamp"]).reset_index(drop=True)


@pytest.fixture
def make():
    return make_orders
