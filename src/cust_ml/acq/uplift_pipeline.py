"""Uplift experiments: Criteo Uplift v2.1 (stratified sample) and Hillstrom (comparison)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from cust_ml import plots
from cust_ml.acq import uplift as U
from cust_ml.data import CRITEO_FEATURES, criteo_full_rows, load_hillstrom, sample_criteo
from cust_ml.metrics import percentile_ci

SAMPLE_FRAC = 0.3  # of the 13,979,592-row file, stratified by (treatment, visit, conversion)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate(y, t, scores: dict, n_boot: int, seed: int, n_bins: int = 1000) -> dict:
    """Qini coefficient and AUUC with bootstrap CIs for each scorer, and the paired difference
    in Qini coefficient to the response model. Also the overall effect and incremental
    outcomes per 100 customers when the top 30% by score are treated."""
    y, t = np.asarray(y), np.asarray(t)
    bc = U.BinnedCurves(y, t, scores, n_bins)
    boots = bc.bootstrap(list(scores), n_boot, seed)
    rows = {}
    for name in scores:
        point = bc.metrics(name)
        b = boots[name]
        row = {
            "qini": point["qini"],
            "qini_ci": list(percentile_ci(b["qini"])),
            "auuc": point["auuc"],
            "auuc_ci": list(percentile_ci(b["auuc"])),
            "auuc_random": point["auuc_random"],
            "incremental_per_100_at_top_30pct": 100 * point["uplift_at_30"],
            "incremental_per_100_at_top_30pct_ci": [
                100 * v for v in percentile_ci(b["uplift_at_30"])
            ],
        }
        if name != "response_model":
            d = b["qini"] - boots["response_model"]["qini"]
            row["qini_minus_response"] = point["qini"] - bc.metrics("response_model")["qini"]
            row["qini_minus_response_ci"] = list(percentile_ci(d))
            row["share_of_bootstraps_above_response"] = float((d > 0).mean())
        rows[name] = row
    ate = float(y[t == 1].mean() - y[t == 0].mean())
    return {
        "methods": rows,
        "treated_rate": float(y[t == 1].mean()),
        "control_rate": float(y[t == 0].mean()),
        "ate": ate,
        "n": int(len(y)),
        "n_outcomes": int(y.sum()),
    }


def run_criteo(data_dir="data", n_boot=200, seed=0, docs_dir="docs"):
    _log("sampling Criteo")
    df = sample_criteo(data_dir, SAMPLE_FRAC, seed)
    full = criteo_full_rows(data_dir, SAMPLE_FRAC, seed)
    X = df[CRITEO_FEATURES]
    t = df["treatment"].to_numpy()
    cell = t * 4 + df["visit"].to_numpy() * 2 + df["conversion"].to_numpy()
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.4, stratify=cell, random_state=seed
    )
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    out = {
        "source_rows": full,
        "sample_rows": int(len(df)),
        "sample_fraction": SAMPLE_FRAC,
        "train_rows": int(len(idx_tr)),
        "test_rows": int(len(idx_te)),
        "treated_share": float(t.mean()),
        "outcomes": {},
    }
    learner = U.default_learner("lgbm", seed, big=True)
    panels = {}
    for outcome in ("visit", "conversion"):
        _log(f"criteo {outcome}")
        y = df[outcome].to_numpy()
        sc = U.score_all(Xtr, y[idx_tr], t[idx_tr], Xte, learner)
        out["outcomes"][outcome] = evaluate(y[idx_te], t[idx_te], sc, n_boot, seed)
        bc = U.BinnedCurves(y[idx_te], t[idx_te], sc, 200)
        panels[outcome] = {k: bc.curve(k) for k in sc}
    plots.qini_figure(
        panels, Path(docs_dir) / "qini_criteo.png",
        "Qini curves, Criteo Uplift v2.1 sample (hold-out 40%, LightGBM)",
    )  # fmt: skip
    return out


HILL_CATS = ["history_segment", "zip_code", "channel"]


def run_hillstrom(data_dir="data", n_boot=200, seed=0, docs_dir="docs"):
    df = load_hillstrom(data_dir)
    t = (df["segment"] != "No E-Mail").astype(int).to_numpy()
    X = df[
        ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
    ]
    out = {"n": int(len(df)), "treated": int(t.sum()), "outcomes": {}}
    panels = {}
    for outcome in ("visit", "conversion"):
        y = df[outcome].to_numpy()
        out["outcomes"][outcome] = {}
        for kind in ("lgbm", "logreg"):
            _log(f"hillstrom {outcome} {kind}")
            sc = U.cross_fit(X, y, t, kind, 5, seed)
            out["outcomes"][outcome][kind] = evaluate(y, t, sc, n_boot, seed, 500)
            if kind == "lgbm":
                bc = U.BinnedCurves(y, t, sc, 200)
                panels[outcome] = {k: bc.curve(k) for k in sc}
    plots.qini_figure(
        panels, Path(docs_dir) / "qini_hillstrom.png",
        "Qini curves, Hillstrom e-mail test (5-fold cross-fitted, LightGBM)",
    )  # fmt: skip
    return out


def run(data_dir="data", results_dir="results", docs_dir="docs", n_boot=200, seed=0):
    out = {"criteo": run_criteo(data_dir, n_boot, seed, docs_dir)}
    out["hillstrom"] = run_hillstrom(data_dir, n_boot, seed, docs_dir)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "uplift.json").write_text(json.dumps(out, indent=1, default=float))
    return out
