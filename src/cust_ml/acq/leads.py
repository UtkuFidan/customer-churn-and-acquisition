"""Lead scoring on the Olist Marketing Funnel: P(a marketing-qualified lead closes).

Only fields that exist when the lead arrives are features: origin channel, landing page and the
contact date. Business segment, lead type, behaviour profile and the other fields of the
closed-deals table are filled in by the sales team after a deal is won, so using them would leak
the label. They are used afterwards, to describe who the won sellers are.

Time is respected twice. The split is by first-contact date. And a training label is the status
known at the split date: a lead that closes later counts as open (0) in the training data, as it
would have on that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cust_ml.churn import models

TEST_START = pd.Timestamp("2018-04-01")
CV_STARTS = [pd.Timestamp(d) for d in ("2018-02-01", "2018-03-01", "2018-04-01")]
VALUE_DAYS = 90
MIN_PAGE_LEADS = 40


def build_leads(tables: dict, items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """One row per MQL: arrival-time features, eventual outcome and, for won deals, the seller's
    GMV in the ``VALUE_DAYS`` days after the deal (NaN when that window is not yet complete)."""
    m, d = tables["mql"].copy(), tables["deals"]
    m = m.merge(d[["mql_id", "seller_id", "won_date"]], on="mql_id", how="left")
    m["won"] = m["won_date"].notna().astype(int)
    m["origin"] = m["origin"].fillna("missing")
    m["weekday"] = m["first_contact_date"].dt.weekday
    m = m.sort_values("first_contact_date").reset_index(drop=True)
    # arrival intensity: leads from the same page / channel in the 30 days before (not counting
    # the lead itself), a causal proxy for campaign activity
    for col, name in (("landing_page_id", "page_leads_30d"), ("origin", "origin_leads_30d")):
        m[name] = _prior_counts(m, col, 30)
    # seller GMV after winning
    ok = orders.loc[~orders["order_status"].isin(("canceled", "unavailable")),
                    ["order_id", "order_purchase_timestamp"]]  # fmt: skip
    s = items[["order_id", "seller_id", "price"]].merge(ok, on="order_id")
    w = m.loc[m["won"] == 1, ["mql_id", "seller_id", "won_date"]].merge(
        s, on="seller_id", how="left"
    )
    inside = (w["order_purchase_timestamp"] > w["won_date"]) & (
        w["order_purchase_timestamp"] <= w["won_date"] + pd.Timedelta(days=VALUE_DAYS)
    )
    gmv = w.assign(v=w["price"].where(inside, 0.0)).groupby("mql_id")["v"].sum()
    m["gmv_after_win"] = m["mql_id"].map(gmv)
    return m


def _prior_counts(m: pd.DataFrame, col: str, days: int) -> np.ndarray:
    out = np.zeros(len(m))
    t = m["first_contact_date"].to_numpy()
    for _, idx in m.groupby(col).indices.items():
        times = t[idx]  # sorted because m is sorted
        lo = np.searchsorted(times, times - np.timedelta64(days, "D"), side="left")
        out[idx] = np.arange(len(idx)) - lo
    return out


def describe_won(tables: dict) -> dict:
    """Post-win attributes of closed deals (not features)."""
    d = tables["deals"]
    return {
        "segment_top": d["business_segment"].value_counts().head(5).to_dict(),
        "lead_type_top": d["lead_type"].value_counts().head(4).to_dict(),
        "business_type": d["business_type"].value_counts().to_dict(),
    }


def _as_of(df: pd.DataFrame, when: pd.Timestamp) -> np.ndarray:
    """Label known on ``when``: won and the deal date is on or before it."""
    return ((df["won"] == 1) & (df["won_date"] <= when)).astype(int).to_numpy()


def feature_frame(df: pd.DataFrame, pages: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["origin"] = df["origin"]
    X["landing_page"] = df["landing_page_id"].where(df["landing_page_id"].isin(pages), "other")
    X["weekday"] = df["weekday"]
    X["page_leads_30d"] = df["page_leads_30d"]
    X["origin_leads_30d"] = df["origin_leads_30d"]
    return X


def frequent_pages(df: pd.DataFrame) -> list[str]:
    vc = df["landing_page_id"].value_counts()
    return vc[vc >= MIN_PAGE_LEADS].index.tolist()


def cv_folds(df: pd.DataFrame):
    """Expanding-window folds on first-contact date. Training labels are 'as of' the fold start."""
    for a, b in zip(CV_STARTS[:-1], CV_STARTS[1:], strict=True):
        tr = np.flatnonzero((df["first_contact_date"] < a).to_numpy())
        va = np.flatnonzero(
            ((df["first_contact_date"] >= a) & (df["first_contact_date"] < b)).to_numpy()
        )
        yield tr, va, _as_of(df, a)


def cv_auc(kind, params, df, X, seed=0):
    aucs = []
    for tr, va, y_asof in cv_folds(df):
        m = models.ChurnModel(kind, params, seed).fit(X.iloc[tr], y_asof[tr])
        aucs.append(roc_auc_score(df["won"].to_numpy()[va], m.predict_proba(X.iloc[va])))
    return aucs


def tune(kind, df, X, n_trials, seed=0, timeout=600):
    def objective(trial):
        return float(np.mean(cv_auc(kind, models.suggest(trial, kind), df, X, seed)))

    study = models.optuna.create_study(
        direction="maximize", sampler=models.optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, float(study.best_value)


# --------------------------------------------------------------------------- prioritisation


def channel_close_rate(train: pd.DataFrame, y_asof: np.ndarray, smoothing: float = 20.0) -> dict:
    """Smoothed close rate per origin from the training period (the heuristic to beat)."""
    base = y_asof.mean()
    g = (
        pd.DataFrame({"o": train["origin"].to_numpy(), "y": y_asof})
        .groupby("o")["y"]
        .agg(["sum", "count"])
    )
    return ((g["sum"] + smoothing * base) / (g["count"] + smoothing)).to_dict()


def channel_value(train: pd.DataFrame, k: float = 5.0) -> tuple[dict, float]:
    """Mean 90-day GMV of won leads per origin (shrunk to the global mean), training period."""
    w = train[(train["won"] == 1) & train["gmv_after_win"].notna()]
    glob = float(w["gmv_after_win"].mean())
    g = w.groupby("origin")["gmv_after_win"].agg(["mean", "count"])
    shrunk = (g["mean"] * g["count"] + glob * k) / (g["count"] + k)
    return shrunk.to_dict(), glob


def prioritise(df: pd.DataFrame, scores: dict[str, np.ndarray], fractions=(0.1, 0.2, 0.3, 0.5)):
    """Work the top share of leads under each ranking; returns closed deals and the realised
    90-day GMV of the leads worked. Random targeting is its expectation (share x total)."""
    won = df["won"].to_numpy()
    gmv = df["gmv_after_win"].fillna(0.0).to_numpy()
    n = len(df)
    rows = []
    for f in fractions:
        k = int(round(n * f))
        rows.append(
            {
                "policy": "random (expected)",
                "share": f,
                "leads": k,
                "closed": float(won.sum() * f),
                "gmv_90d": float(gmv.sum() * f),
            }
        )
        for name, s in scores.items():
            top = np.argsort(-s, kind="stable")[:k]
            rows.append(
                {
                    "policy": name,
                    "share": f,
                    "leads": k,
                    "closed": float(won[top].sum()),
                    "gmv_90d": float(gmv[top].sum()),
                }
            )
    return rows
