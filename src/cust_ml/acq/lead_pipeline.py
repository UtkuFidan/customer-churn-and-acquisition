"""Olist Marketing Funnel: tune, evaluate and use a lead-close model under a capacity budget."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cust_ml import plots
from cust_ml.acq import leads as L
from cust_ml.churn import models
from cust_ml.data import load_funnel_tables, load_olist_tables
from cust_ml.metrics import ap_w, auc_w, group_bootstrap

TRIALS = {"logreg": 12, "lightgbm": 40, "xgboost": 30, "catboost": 15}
VALUE_COMPLETE_BY = pd.Timestamp("2018-08-29") - pd.Timedelta(days=L.VALUE_DAYS)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _paired_closed_ci(won, a, b, share, seed, n_boot=1000):
    """Closed deals when working the top ``share`` by score ``a`` minus by score ``b``, with the
    worked sets held fixed and leads resampled with replacement."""
    k = int(round(len(won) * share))
    in_a = np.zeros(len(won), bool)
    in_a[np.argsort(-a, kind="stable")[:k]] = True
    in_b = np.zeros(len(won), bool)
    in_b[np.argsort(-b, kind="stable")[:k]] = True
    rng = np.random.default_rng(seed)
    diffs = [
        won[i][in_a[i]].sum() - won[i][in_b[i]].sum()
        for i in (rng.integers(0, len(won), len(won)) for _ in range(n_boot))
    ]
    return {
        "difference": float(won[in_a].sum() - won[in_b].sum()),
        "ci95": [float(v) for v in np.quantile(diffs, [0.025, 0.975])],
    }


def run(data_dir="data", results_dir="results", docs_dir="docs", trials=None, seed=42):
    trials = {**TRIALS, **(trials or {})}
    tables = load_funnel_tables(data_dir)
    ol = load_olist_tables(data_dir, only=("items", "orders"))
    df = L.build_leads(tables, ol["items"], ol["orders"])
    train = df[df["first_contact_date"] < L.TEST_START].reset_index(drop=True)
    test = df[df["first_contact_date"] >= L.TEST_START].reset_index(drop=True)
    y_train = L._as_of(train, L.TEST_START)
    yt = test["won"].to_numpy()
    pages = L.frequent_pages(train)
    X, Xt = L.feature_frame(train, pages), L.feature_frame(test, pages)
    out = {
        "seed": seed,
        "leads": int(len(df)),
        "closed_deals": int(df["won"].sum()),
        "close_rate": float(df["won"].mean()),
        "train": {
            "leads": len(train),
            "contact_dates": [
                str(train.first_contact_date.min().date()),
                str(train.first_contact_date.max().date()),
            ],
            "closed_as_of_split": int(y_train.sum()),
            "close_rate_as_of_split": float(y_train.mean()),
        },
        "test": {
            "leads": len(test),
            "contact_dates": [
                str(test.first_contact_date.min().date()),
                str(test.first_contact_date.max().date()),
            ],
            "closed": int(yt.sum()),
            "close_rate": float(yt.mean()),
        },
        "close_rate_by_quarter": {
            str(k): float(v)
            for k, v in df.groupby(df["first_contact_date"].dt.to_period("Q"))["won"].mean().items()
        },
        "days_contact_to_win_quantiles": {
            str(q): float((df["won_date"] - df["first_contact_date"]).dt.days.quantile(q))
            for q in (0.5, 0.75, 0.9)
        },
        "won_sellers_with_orders_in_ecommerce_data": float(
            (df.loc[df["won"] == 1, "gmv_after_win"].fillna(0) > 0).mean()
        ),
        "won_profile": L.describe_won(tables),
    }
    res, preds = {}, {}
    for kind in models.KINDS:
        _log(f"tuning {kind}")
        best, cv = L.tune(kind, train, X, trials[kind], seed)
        m = models.ChurnModel(kind, best, seed).fit(X, y_train)
        p = m.predict_proba(Xt)
        g = np.arange(len(test))
        auc = group_bootstrap(auc_w, (yt, p), g, 500, seed)
        ap = group_bootstrap(ap_w, (yt, p), g, 500, seed)
        top = np.argsort(-p, kind="stable")[: int(round(0.1 * len(p)))]
        res[kind] = {
            "params": best,
            "cv_auc_mean": cv,
            "test_auc": auc[0],
            "test_auc_ci": [auc[1], auc[2]],
            "test_pr_auc": ap[0],
            "test_pr_auc_ci": [ap[1], ap[2]],
            "test_top_decile_lift": float(yt[top].mean() / yt.mean()),
        }
        preds[kind] = p
    rate = L.channel_close_rate(train, y_train)
    heur = test["origin"].map(rate).to_numpy()
    auc = group_bootstrap(auc_w, (yt, heur), np.arange(len(test)), 500, seed)
    ap = group_bootstrap(ap_w, (yt, heur), np.arange(len(test)), 500, seed)
    res["channel_rate_heuristic"] = {
        "test_auc": auc[0],
        "test_auc_ci": [auc[1], auc[2]],
        "test_pr_auc": ap[0],
        "test_pr_auc_ci": [ap[1], ap[2]],
    }
    out["models"] = res
    best_kind = max(models.KINDS, key=lambda k: res[k]["cv_auc_mean"])
    out["selected"] = best_kind
    p = preds[best_kind]

    # lift by channel
    ch = (
        pd.DataFrame({"origin": test["origin"], "won": yt, "p": p})
        .groupby("origin")
        .agg(leads=("won", "size"), closed=("won", "sum"), mean_predicted=("p", "mean"))
    )
    ch["observed_close_rate"] = ch["closed"] / ch["leads"]
    ch["lift_vs_all"] = ch["observed_close_rate"] / yt.mean()
    cut = np.quantile(p, 0.7)
    ch["share_in_top_30pct"] = pd.Series(p >= cut).groupby(test["origin"].to_numpy()).mean()
    ch = ch.sort_values("leads", ascending=False).reset_index()
    out["by_channel"] = ch.to_dict("records")
    plots.lift_by_channel_figure(
        ch[ch["leads"] >= 40].to_dict("records"), Path(docs_dir) / "leads_by_channel.png"
    )

    # budgeted prioritisation
    vmap, vglob = L.channel_value(train)
    value = test["origin"].map(vmap).fillna(vglob).to_numpy()
    scores = {
        "channel close rate": heur,
        "model P(close)": p,
        "model P(close) x channel GMV": p * value,
    }
    out["prioritisation_closed_deals"] = L.prioritise(test, scores)
    out["model_vs_channel_rule_closed_ci95"] = _paired_closed_ci(
        yt, scores["model P(close)"], heur, 0.3, seed
    )
    mask = ((test["won"] == 0) | (test["won_date"] <= VALUE_COMPLETE_BY)).to_numpy()
    comp = test[mask].reset_index(drop=True)
    out["prioritisation_gmv"] = L.prioritise(comp, {k: v[mask] for k, v in scores.items()})
    out["prioritisation_gmv_population"] = {
        "leads": int(len(comp)),
        "closed": int(comp["won"].sum()),
        "dropped_won_with_open_window": int(len(test) - len(comp)),
        "gmv_window_days": L.VALUE_DAYS,
        "channel_gmv_assumed": {k: float(v) for k, v in vmap.items()},
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "olist_leads.json").write_text(json.dumps(out, indent=1, default=float))
    return out
