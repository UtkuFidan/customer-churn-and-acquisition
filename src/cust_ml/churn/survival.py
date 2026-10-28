"""Time to the second purchase: cohort construction, Kaplan-Meier, Cox and boosted Cox.

Origin (t0) is the moment the first order is fully known to the customer and to us: its delivery
date, or the review date if later. Everything used as a covariate (delay, review) exists by t0.
Customers who bought again before t0 are left out (and counted), otherwise the covariates would
be defined partly by the future. Customers still waiting for their first delivery are left out
as well. Follow-up is censored at ``end``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index

DELAY_BANDS = [-np.inf, -14, -7, 0, 7, np.inf]
DELAY_LABELS = [
    "14+ days early",
    "7-14 days early",
    "0-7 days early",
    "1-7 days late",
    "8+ days late",
]
# right=True: (-inf,-14], (-14,-7], (-7,0], (0,7], (7,inf)


def first_purchase_cohort(
    orders: pd.DataFrame, end: str | pd.Timestamp, categories: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """One row per eligible customer: duration (days from t0), event, and covariates known at t0.

    Returns (cohort, counts) where counts documents who was dropped and why."""
    end = pd.Timestamp(end)
    o = orders.sort_values(["customer_unique_id", "order_purchase_timestamp"])
    first_day = o.groupby("customer_unique_id")["day"].transform("min")
    f = o[o["day"] == first_day].copy()
    n_customers = f["customer_unique_id"].nunique()
    f["known_ts"] = f[["order_delivered_customer_date", "review_ts"]].max(axis=1)
    f.loc[f["order_delivered_customer_date"].isna(), "known_ts"] = pd.NaT
    f["pay_voucher"] = (f["payment_type"] == "voucher").astype(float)
    f["low"] = (f["review_score"] <= 2).astype(float).where(f["review_score"].notna())
    big = f.sort_values("gmv", ascending=False).drop_duplicates("customer_unique_id")
    g = f.groupby("customer_unique_id")
    c = g.agg(
        first_day=("day", "min"),
        t0=("known_ts", "max"),
        delay_days=("delay_days", "max"),
        delivery_days=("delivery_days", "max"),
        review_score=("review_score", "mean"),
        review_has_text=("review_has_text", "max"),
        gmv=("gmv", "sum"),
        price=("price_sum", "sum"),
        freight_share=("freight_share", "mean"),
        n_items=("n_items", "sum"),
        installments=("installments", "max"),
        voucher=("pay_voucher", "max"),
        distance_km=("distance_km", "mean"),
        same_state=("same_state", "mean"),
    )
    c = c.join(big.set_index("customer_unique_id")[["payment_type", "category", "customer_state"]])
    days = o.drop_duplicates(["customer_unique_id", "day"])
    second = (
        days.assign(rk=days.groupby("customer_unique_id").cumcount())
        .query("rk == 1")
        .set_index("customer_unique_id")["day"]
    )
    c["second_day"] = second.reindex(c.index)
    counts = {"customers": int(n_customers)}
    ok = c["t0"].notna()
    counts["no_delivery_yet"] = int((~ok).sum())
    c = c[ok]
    t0_day = c["t0"].dt.normalize()
    early = c["second_day"].notna() & (c["second_day"] <= t0_day)
    counts["repurchased_before_t0"] = int(early.sum())
    c = c[~early]
    counts["past_end"] = int((c["t0"] > end).sum())
    c = c[c["t0"] <= end].copy()
    t0_day = c["t0"].dt.normalize()
    event = c["second_day"].notna() & (c["second_day"] <= end)
    stop = c["second_day"].where(event, end)
    c["duration"] = ((stop - t0_day).dt.days).clip(lower=0.5)
    c["event"] = event.astype(int)
    c["late"] = (c["delay_days"] > 0).astype(int)
    c["delay_band"] = pd.cut(c["delay_days"], DELAY_BANDS, labels=DELAY_LABELS)
    c["review_band"] = pd.cut(
        c["review_score"], [0, 2.5, 3.5, 5], labels=["1-2 stars", "3 stars", "4-5 stars"]
    )
    c["low_review"] = (c["review_score"] <= 2).astype(int)
    if categories is not None:
        c["category"] = c["category"].where(c["category"].isin(categories), "other")
    counts["eligible"] = int(len(c))
    counts["repurchased"] = int(c["event"].sum())
    return c, counts


def km_summary(cohort: pd.DataFrame, by: str, times=(90, 180)) -> tuple[dict, dict]:
    """Kaplan-Meier by group: share bought again by each horizon (1 - S(t)) with 95% CI, plus
    a log-rank test across groups. Also returns the fitted KM objects for plotting."""
    d = cohort.dropna(subset=[by])
    out = {"groups": {}, "n": int(len(d))}
    fits = {}
    for g, sub in d.groupby(by, observed=True):
        km = KaplanMeierFitter().fit(sub["duration"], sub["event"], label=str(g))
        fits[str(g)] = km
        row = {"n": int(len(sub)), "events": int(sub["event"].sum())}
        for t in times:
            s = float(km.survival_function_at_times(t).iloc[0])
            ci = km.confidence_interval_
            lo = float(ci.iloc[:, 0].asof(t))
            hi = float(ci.iloc[:, 1].asof(t))
            row[f"repurchase_{t}d"] = 1 - s
            row[f"repurchase_{t}d_ci"] = [1 - hi, 1 - lo]
        out["groups"][str(g)] = row
    lr = multivariate_logrank_test(d["duration"], d[by].astype(str), d["event"])
    out["logrank_p"] = float(lr.p_value)
    return out, fits


COX_NUM = [
    "delay_days",
    "review_score",
    "log_price",
    "freight_share",
    "log_distance",
    "installments",
]


def cox_design(c: pd.DataFrame, states: list[str] | None = None, cats: list[str] | None = None):
    """Covariates at t0 for Cox: numeric controls, indicators, dummies for payment type, top
    states and categories. Column set is fixed by ``states``/``cats`` so train and test align."""
    X = pd.DataFrame(index=c.index)
    X["late"] = c["late"].astype(float)
    X["low_review"] = c["low_review"].astype(float)
    X["review_score"] = c["review_score"]
    X["review_missing"] = c["review_score"].isna().astype(float)
    X["review_has_text"] = c["review_has_text"].astype(float).fillna(0.0)
    X["log_price"] = np.log1p(c["price"])
    X["freight_share"] = c["freight_share"]
    X["log_distance"] = np.log1p(c["distance_km"])
    X["installments"] = c["installments"].clip(upper=12)
    X["voucher"] = c["voucher"]
    X["n_items"] = c["n_items"].clip(upper=5)
    X["same_state"] = c["same_state"]
    states = states or c["customer_state"].value_counts().head(6).index.tolist()
    cats = cats or c["category"].value_counts().head(8).index.tolist()
    for s in states:
        X[f"state_{s}"] = (c["customer_state"] == s).astype(float)
    for k in cats:
        X[f"cat_{k}"] = (c["category"] == k).astype(float)
    X["pay_boleto"] = (c["payment_type"] == "boleto").astype(float)
    return X, states, cats


def fill_missing(X: pd.DataFrame, ref: pd.Series | None = None):
    ref = X.median() if ref is None else ref
    return X.fillna(ref), ref


def fit_cox(X: pd.DataFrame, duration, event, penalizer: float = 0.01):
    d = X.copy()
    d["duration"], d["event"] = np.asarray(duration), np.asarray(event)
    keep = [c for c in X.columns if X[c].std() > 0]
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d[keep + ["duration", "event"]], "duration", "event")
    return cph, keep


def hazard_ratios(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.columns = ["hr", "lo", "hi", "p"]
    return s


def concordance_ci(duration, event, risk, n_boot=300, seed=0, alpha=0.05):
    """Harrell's C (higher risk -> earlier event) with a percentile bootstrap over customers."""
    d, e, r = np.asarray(duration), np.asarray(event), np.asarray(risk)
    point = concordance_index(d, -r, e)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        reps.append(concordance_index(d[i], -r[i], e[i]))
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def fit_boosted_cox(X: pd.DataFrame, duration, event, params: dict | None = None, seed=0):
    """Gradient-boosted Cox model (XGBoost ``survival:cox``): negative labels are censored."""
    p = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "tree_method": "hist",
        "max_depth": 2,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "lambda": 10.0,
        "seed": seed,
        "nthread": -1,
        "verbosity": 0,
    }
    p.update(params or {})
    n_rounds = p.pop("n_rounds", 200)
    y = np.where(np.asarray(event) == 1, np.asarray(duration), -np.asarray(duration))
    return xgb.train(p, xgb.DMatrix(X, label=y), num_boost_round=n_rounds)


def predict_boosted(model, X: pd.DataFrame):
    return model.predict(xgb.DMatrix(X))
