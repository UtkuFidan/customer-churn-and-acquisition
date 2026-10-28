"""Olist churn pipeline: snapshots -> tuned models -> calibration -> profit targeting ->
survival -> CLV -> SHAP. One call to ``run`` writes ``results/olist_churn.json`` and figures.

Design constants live at the top. Horizon, cutoffs and economic assumptions are explained in the
README; changing them here changes every downstream number.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from cust_ml import plots
from cust_ml.churn import clv as clvmod
from cust_ml.churn import explain, models, olist, profit, survival
from cust_ml.data import load_olist_tables
from cust_ml.metrics import (
    ap_w,
    auc_w,
    expected_calibration_error,
    group_bootstrap,
    lift_at,
    reliability_table,
)

HORIZON = 90
DATA_END = pd.Timestamp("2018-08-29")  # last day with a complete order feed
TRAIN_CUTOFFS = list(pd.date_range("2017-03-31", "2017-12-31", freq="ME"))
TEST_CUTOFFS = [pd.Timestamp(d) for d in ("2018-03-31", "2018-04-30", "2018-05-31")]
FEATURES = olist.FEATURES_NUM + olist.FEATURES_CAT
SURV_TRAIN_T0_END = pd.Timestamp("2017-09-30")
SURV_TRAIN_CENSOR = pd.Timestamp("2018-03-31")
CLV_CALIBRATION_END = "2018-03-31"
TRIALS = {"logreg": 12, "lightgbm": 30, "xgboost": 25, "catboost": 10}
NEG_FRAC = 0.3  # tree models train on all positives and 30% of negatives

# Economic assumptions, all in BRL, all made up. See README.
ASSUMPTIONS = {
    "send_cost": 0.02,  # one win-back message (push / e-mail blend)
    "lift": 0.30,  # relative rise in repurchase probability caused by the message
    "margin_rate": 0.10,  # platform margin as share of GMV
    "clv_horizon_months": 12,
    "clv_monthly_discount": 0.01,
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(data_dir, art_dir):
    cache = Path(art_dir) / "olist_snapshots.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    orders = olist.build_orders(load_olist_tables(data_dir))
    cats = olist.top_categories(orders, TRAIN_CUTOFFS[-1])
    train = olist.make_snapshots(orders, TRAIN_CUTOFFS, HORIZON, cats)
    test = olist.make_snapshots(orders, TEST_CUTOFFS, HORIZON, cats)
    out = {"orders": orders, "train": train, "test": test, "categories": cats}
    Path(art_dir).mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, cache)
    return out


def data_facts(orders: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Numbers behind the churn definition: repeat rate, gap distribution, horizon coverage."""
    occ = orders.drop_duplicates(["customer_unique_id", "day"])
    rk = occ.groupby("customer_unique_id").cumcount()
    first = occ[rk == 0].set_index("customer_unique_id")["day"]
    second = occ[rk == 1].set_index("customer_unique_id")["day"]
    gap = (second - first.loc[second.index]).dt.days
    raw_gap = (
        orders.assign(rk=orders.groupby("customer_unique_id").cumcount())
        .query("rk <= 1")
        .pivot(index="customer_unique_id", columns="rk", values="order_purchase_timestamp")
        .dropna()
    )
    same_day = ((raw_gap[1] - raw_gap[0]).dt.days == 0).mean()
    snap = lambda d: {  # noqa: E731
        "customers": int(len(d)),
        "repeat_rows": int((1 - d.churn).sum()),
        "repeat_rate": float((1 - d.churn).mean()),
    }
    return {
        "orders_used": int(len(orders)),
        "purchase_days": int(len(occ)),
        "customers": int(orders["customer_unique_id"].nunique()),
        "customers_with_repeat": int(len(second)),
        "repeat_share": float(len(second) / len(first)),
        "orders_share_of_gaps_zero_days_before_collapsing": float(same_day),
        "gap_days_quantiles": {str(q): float(gap.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "share_of_second_purchases_within": {
            str(n): float((gap <= n).mean()) for n in (30, 60, 90, 120, 180)
        },
        "horizon_days": HORIZON,
        "train": snap(train) | {"cutoffs": [str(c.date()) for c in TRAIN_CUTOFFS]},
        "test": snap(test) | {"cutoffs": [str(c.date()) for c in TEST_CUTOFFS]},
        "test_per_cutoff": {str(c.date()): snap(test[test["cutoff"] == c]) for c in TEST_CUTOFFS},
    }


# ---------------------------------------------------------------------------- models


def train_models(snaps, art_dir, trials, seed):
    """Tune each family, calibrate, evaluate on the later snapshots. Returns state + results."""
    tr, te = snaps["train"], snaps["test"]
    X, y = tr[FEATURES], (1 - tr["churn"]).to_numpy()
    Xt, yt = te[FEATURES], (1 - te["churn"]).to_numpy()
    cut = tr["cutoff"].to_numpy()
    params_file = Path(art_dir) / "olist_best_params.json"
    saved = json.loads(params_file.read_text()) if params_file.exists() else {}
    state, rows = {}, {}
    for kind in models.KINDS:
        nf = 1.0 if kind == "logreg" else NEG_FRAC
        if kind in saved and saved[kind].get("trials") == trials[kind]:
            best, cv_auc = saved[kind]["params"], saved[kind]["cv_auc"]
        else:
            _log(f"tuning {kind} ({trials[kind]} trials)")
            best, cv_auc, _ = models.tune(kind, X, y, cut, HORIZON, trials[kind], seed, nf)
            saved[kind] = {"params": best, "cv_auc": cv_auc, "trials": trials[kind]}
            params_file.write_text(json.dumps(saved, indent=1))
        pos, oof, fold_aucs = models.cv_oof(kind, best, X, y, cut, HORIZON, seed, nf)
        # choose calibration on the earlier half of the OOF blocks, check on the later half
        oof_cut = cut[pos]
        half = np.sort(np.unique(oof_cut))[len(np.unique(oof_cut)) // 2]
        a, b = oof_cut < half, oof_cut >= half
        brier = {}
        for method in ("none", "sigmoid", "isotonic"):
            cal = models.Calibrator(method).fit(oof[a], y[pos][a])
            brier[method] = float(brier_score_loss(y[pos][b], cal.predict(oof[b])))
        method = min(brier, key=brier.get)
        calibrator = models.Calibrator(method).fit(oof, y[pos])
        _log(f"fitting {kind} on all training rows; calibration={method}")
        model = models.ChurnModel(kind, best, seed, nf).fit(X, y)
        raw = model.predict_proba(Xt)
        p = calibrator.predict(raw)
        state[kind] = {"model": model, "calibrator": calibrator, "raw": raw, "p": p}
        groups = te.index.to_numpy()
        auc = group_bootstrap(auc_w, (yt, p), groups, 200, seed)
        ap = group_bootstrap(ap_w, (yt, p), groups, 200, seed)
        rows[kind] = {
            "params": best,
            "cv_auc_mean": cv_auc,
            "cv_auc_folds": [float(v) for v in fold_aucs],
            "calibration": method,
            "calibration_brier_check": brier,
            "test_auc": auc[0],
            "test_auc_ci": [auc[1], auc[2]],
            "test_pr_auc_repeat": ap[0],
            "test_pr_auc_repeat_ci": [ap[1], ap[2]],
            "test_top_decile_lift": lift_at(yt, p, 0.1),
            "test_brier": float(brier_score_loss(yt, p)),
            "test_brier_uncalibrated": float(brier_score_loss(yt, np.clip(raw, 0, 1))),
            "test_logloss": float(log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6))),
            "test_ece": expected_calibration_error(yt, p),
            "test_ece_uncalibrated": expected_calibration_error(yt, raw),
        }
    base = float(y.mean())
    rec = -Xt["recency_days"].to_numpy()
    r_auc = group_bootstrap(auc_w, (yt, rec), te.index.to_numpy(), 200, seed)
    r_ap = group_bootstrap(ap_w, (yt, rec), te.index.to_numpy(), 200, seed)
    results = {
        "models": rows,
        "base_rate_train": base,
        "base_rate_test": float(yt.mean()),
        "brier_constant_train_rate": float(brier_score_loss(yt, np.full(len(yt), base))),
        "recency_only_baseline": {
            "test_auc": r_auc[0],
            "test_auc_ci": [r_auc[1], r_auc[2]],
            "test_pr_auc_repeat": r_ap[0],
            "test_pr_auc_repeat_ci": [r_ap[1], r_ap[2]],
        },
    }
    best_kind = max(rows, key=lambda k: rows[k]["cv_auc_mean"])
    results["selected"] = best_kind
    results["selection_rule"] = "highest mean ROC AUC over the time-aware validation blocks"
    return state, results, best_kind


# ---------------------------------------------------------------------------- profit


def profit_analysis(te: pd.DataFrame, p_rep, assumptions: dict, seed: int) -> tuple[dict, dict]:
    a = assumptions
    y = (1 - te["churn"]).to_numpy()
    aov = te["avg_order_value"].fillna(te["avg_order_value"].median()).to_numpy()
    gain = profit.expected_gain(p_rep, aov, a["send_cost"], a["lift"], a["margin_rate"])
    real = profit.realised_gain(
        y, te["next_gmv"].to_numpy(), a["send_cost"], a["lift"], a["margin_rate"]
    )
    n = len(te)
    n_rule = int((gain > 0).sum())  # contact whoever has a positive expected gain
    rank_p = np.asarray(p_rep)
    rank_rec = -te["recency_days"].to_numpy()
    rng = np.random.default_rng(seed)
    rank_rand = rng.random(n)

    def at(rank, k):
        return profit.profit_at(real, rank, k)

    f_h, best_hind, k_h = profit.optimal_targeting(real, gain)  # hindsight, ranked by gain
    out = {
        "assumptions": a,
        "customers_scored": n,
        "campaigns": te["cutoff"].nunique(),
        "rule": "contact if expected gain > 0",
        "contacted": n_rule,
        "contacted_share": n_rule / n,
        "min_p_repeat_contacted": float(rank_p[gain > 0].min()) if n_rule else None,
        "expected_profit_rule": float(gain[gain > 0].sum()),
        "realised_profit": {
            "model_expected_gain_rule": at(gain, n_rule),
            "same_count_ranked_by_p_repeat": at(rank_p, n_rule),
            "same_count_ranked_by_recency": at(rank_rec, n_rule),
            "same_count_random_expected": float(real.mean() * n_rule),
            "contact_everyone": float(real.sum()),
        },
        "hindsight_best_share_ranked_by_gain": f_h,
        "hindsight_best_profit": best_hind,
    }
    # paired customer bootstrap of model-minus-recency at the rule's contact count
    codes = np.unique(te.index.to_numpy(), return_inverse=True)[1]
    k = codes.max() + 1
    top_m = np.zeros(n, bool)
    top_m[np.argsort(-gain, kind="stable")[:n_rule]] = True
    top_r = np.zeros(n, bool)
    top_r[np.argsort(-rank_rec, kind="stable")[:n_rule]] = True
    diffs = []
    for _ in range(300):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[codes]
        diffs.append(float((w * real * top_m).sum() - (w * real * top_r).sum()))
    out["model_minus_recency_ci95"] = [float(v) for v in np.quantile(diffs, [0.025, 0.975])]

    # sensitivity: contact share and realised profit under other assumptions
    grid = []
    for sc in (0.02, 0.05, 0.10, 0.20):
        for lf in (0.15, 0.30, 0.60):
            g = profit.expected_gain(p_rep, aov, sc, lf, a["margin_rate"])
            r = profit.realised_gain(y, te["next_gmv"].to_numpy(), sc, lf, a["margin_rate"])
            kk = int((g > 0).sum())
            order = np.argsort(-g, kind="stable")[:kk]
            grid.append(
                {
                    "send_cost": sc,
                    "lift": lf,
                    "contacted_share": kk / n,
                    "realised_profit": float(r[order].sum()),
                    "contact_everyone": float(r.sum()),
                }
            )
    out["sensitivity"] = grid
    curves = {
        "Ranked by expected gain": profit.profit_curve(real, gain),
        "Ranked by P(repeat)": profit.profit_curve(real, rank_p),
        "Ranked by recency": profit.profit_curve(real, rank_rec),
        "Random order": profit.profit_curve(real, rank_rand),
    }
    return out, {"curves": curves, "n_rule": n_rule, "n": n, "gain": gain}


# ---------------------------------------------------------------------------- survival


def survival_analysis(orders, cats, seed):
    cohort_all, counts = survival.first_purchase_cohort(orders, DATA_END, cats)
    res = {"cohort": counts}
    figs = {}

    def km_block(by):
        s, fits = survival.km_summary(cohort_all, by)
        return s, fits

    res["km_delay_band"], figs["delay"] = km_block("delay_band")
    res["km_review_band"], figs["review"] = km_block("review_band")
    res["km_late"], _ = km_block("late")

    # headline: late delivery and bad reviews, with the numbers behind it
    g = cohort_all.dropna(subset=["review_score"])
    combo = np.select(
        [
            (g.late == 1) & (g.low_review == 1),
            g.late == 1,
            g.low_review == 1,
        ],
        ["late and 1-2 stars", "late, 3-5 stars", "on time, 1-2 stars"],
        default="on time, 3-5 stars",
    )
    both = g.assign(both=combo)
    res["km_late_x_review"], _ = survival.km_summary(both, "both")
    on, late = res["km_late"]["groups"]["0"], res["km_late"]["groups"]["1"]
    low = res["km_review_band"]["groups"]["1-2 stars"]
    high = res["km_review_band"]["groups"]["4-5 stars"]
    res["excess_lost_repurchases_180d"] = {
        "late_delivery": (on["repurchase_180d"] - late["repurchase_180d"]) * late["n"],
        "bad_review": (high["repurchase_180d"] - low["repurchase_180d"]) * low["n"],
        "total_repurchases_observed": counts["repurchased"],
    }

    X_all, states, cat_list = survival.cox_design(cohort_all)
    X_all = X_all.drop(columns=["review_score"])
    X_all, _ = survival.fill_missing(X_all)
    ratios = {}
    for name, cols in (
        ("late_only", ["late"]),
        ("late_and_bad_review", ["late", "low_review"]),
        ("full", list(X_all.columns)),
    ):
        cph, keep = survival.fit_cox(X_all[cols], cohort_all["duration"], cohort_all["event"])
        hr = survival.hazard_ratios(cph)
        ratios[name] = {
            k: {c: float(v) for c, v in hr.loc[k].items()}
            for k in ("late", "low_review")
            if k in hr.index
        }
    res["cox_hazard_ratios"] = ratios

    # predictive comparison on a later cohort
    train = cohort_all[cohort_all["t0"] <= SURV_TRAIN_T0_END].copy()
    after = train["second_day"].notna() & (train["second_day"] > SURV_TRAIN_CENSOR)
    train["event"] = np.where(after, 0, train["event"])
    cens = np.where(after | train["second_day"].isna(), SURV_TRAIN_CENSOR, train["second_day"])
    train["duration"] = (pd.to_datetime(cens) - train["t0"].dt.normalize()).dt.days.clip(lower=0.5)
    train = train[train["t0"] <= SURV_TRAIN_CENSOR]
    test = cohort_all[
        (cohort_all["t0"] > SURV_TRAIN_T0_END) & (cohort_all["t0"] <= SURV_TRAIN_CENSOR)
    ]
    Xtr, _, _ = survival.cox_design(train, states, cat_list)
    Xte, _, _ = survival.cox_design(test, states, cat_list)
    Xtr, ref = survival.fill_missing(Xtr)
    Xte, _ = survival.fill_missing(Xte, ref)
    cph, keep = survival.fit_cox(Xtr, train["duration"], train["event"])
    risk_cox = cph.predict_partial_hazard(Xte[keep]).to_numpy()
    booster = survival.fit_boosted_cox(Xtr, train["duration"], train["event"], seed=seed)
    risk_gb = survival.predict_boosted(booster, Xte)
    res["prediction"] = {
        "train_customers": int(len(train)),
        "train_events": int(train["event"].sum()),
        "test_customers": int(len(test)),
        "test_events": int(test["event"].sum()),
        "train_t0_up_to": str(SURV_TRAIN_T0_END.date()),
        "train_follow_up_censored_at": str(SURV_TRAIN_CENSOR.date()),
        "c_index": {},
    }
    for name, r in (("cox", risk_cox), ("boosted_cox_xgboost", risk_gb)):
        c = survival.concordance_ci(test["duration"], test["event"], r, 300, seed)
        res["prediction"]["c_index"][name] = {"estimate": c[0], "ci95": [c[1], c[2]]}
    # delivery-only benchmark: how far do delay and review alone get
    only = ["late", "low_review", "review_missing"]
    cph2, k2 = survival.fit_cox(Xtr[only], train["duration"], train["event"])
    r2 = cph2.predict_partial_hazard(Xte[k2]).to_numpy()
    c = survival.concordance_ci(test["duration"], test["event"], r2, 300, seed)
    res["prediction"]["c_index"]["cox_delivery_and_review_only"] = {
        "estimate": c[0],
        "ci95": [c[1], c[2]],
    }
    return res, figs


# ---------------------------------------------------------------------------- CLV


def clv_analysis(orders, snaps_test, p_rep, a, seed):
    d = clvmod.rfm_split(orders, CLV_CALIBRATION_END, str(DATA_END.date()))
    bgf = clvmod.fit_bgnbd(d["frequency_cal"], d["recency_cal"], d["T_cal"])
    hold_days = float(d["duration_holdout"].iloc[0])
    pred = bgf.predict(hold_days, d["frequency_cal"], d["recency_cal"], d["T_cal"])
    actual = d["frequency_holdout"]
    rep = d[d["frequency_cal"] > 0]
    ggf = clvmod.fit_gamma_gamma(rep["frequency_cal"], rep["monetary_value_cal"])
    corr_fm = float(np.corrcoef(rep["frequency_cal"], rep["monetary_value_cal"])[0, 1])
    by_freq = (
        pd.DataFrame(
            {
                "freq": d["frequency_cal"].clip(upper=3),
                "pred": pred,
                "actual": actual,
            }
        )
        .groupby("freq")
        .agg(customers=("pred", "size"), predicted=("pred", "mean"), actual=("actual", "mean"))
    )
    dec = pd.qcut(pred.rank(method="first"), 10, labels=False)
    by_dec = pd.DataFrame({"pred": pred, "actual": actual}).groupby(dec).mean()
    # Gamma-Gamma needs a repeat purchase to say anything about a customer's order value, so
    # one-time buyers get the mean order value of the calibration period instead
    pop_value = float(orders.loc[orders["day"] <= pd.Timestamp(CLV_CALIBRATION_END), "gmv"].mean())
    avg_value = ggf.conditional_expected_average_profit(
        d["frequency_cal"], d["monetary_value_cal"].where(d["frequency_cal"] > 0, 0.0)
    ).where(d["frequency_cal"] > 0, pop_value)
    clv = clvmod.discounted_clv(
        lambda t: bgf.conditional_expected_number_of_purchases_up_to_time(
            t, d["frequency_cal"], d["recency_cal"], d["T_cal"]
        ),
        a["clv_horizon_months"],
        avg_value,
        a["margin_rate"],
        a["clv_monthly_discount"],
    )
    clv = pd.Series(np.asarray(clv), index=d.index, name="clv")
    res = {
        "calibration_end": CLV_CALIBRATION_END,
        "holdout_days": hold_days,
        "customers": int(len(d)),
        "bgnbd_params": {k: float(v) for k, v in bgf.params_.items()},
        "gamma_gamma_params": {k: float(v) for k, v in ggf.params_.items()},
        "corr_frequency_monetary_repeaters": corr_fm,
        "holdout_purchases": {
            "predicted_total": float(pred.sum()),
            "actual_total": float(actual.sum()),
            "corr_customer_level": float(np.corrcoef(pred, actual)[0, 1]),
            "by_calibration_frequency": {
                str(int(k)): {c: float(v) for c, v in row.items()} for k, row in by_freq.iterrows()
            },
            "by_predicted_decile_top": {c: float(v) for c, v in by_dec.iloc[-1].items()},
            "by_predicted_decile_bottom": {c: float(v) for c, v in by_dec.iloc[0].items()},
        },
        "mean_order_value_population": pop_value,
        "mean_order_value_repeaters_gg": float(avg_value[d["frequency_cal"] > 0].mean()),
        "mean_order_value_repeaters_observed": float(
            d.loc[d["frequency_cal"] > 0, "monetary_value_cal"].mean()
        ),
        "clv_12m_quantiles": {str(q): float(clv.quantile(q)) for q in (0.5, 0.9, 0.99)},
        "clv_12m_total": float(clv.sum()),
    }
    # value-at-risk segments on the first test snapshot (same date as the CLV calibration end)
    first = snaps_test["cutoff"] == pd.Timestamp(CLV_CALIBRATION_END)
    snap = snaps_test[first].copy()
    snap["p_repeat"] = np.asarray(p_rep)[first.to_numpy()]
    snap["clv"] = clv.reindex(snap.index).to_numpy()
    snap = snap.dropna(subset=["clv"])
    snap["segment"] = clvmod.value_segments(snap["clv"], snap["p_repeat"])
    seg = snap.groupby("segment").agg(
        customers=("clv", "size"),
        clv_total=("clv", "sum"),
        mean_clv=("clv", "mean"),
        mean_p_repeat=("p_repeat", "mean"),
        repeat_rate_90d=("churn", lambda s: 1 - s.mean()),
    )
    seg["share_of_clv"] = seg["clv_total"] / seg["clv_total"].sum()
    res["segments"] = {k: {c: float(v) for c, v in row.items()} for k, row in seg.iterrows()}
    res["segment_snapshot"] = CLV_CALIBRATION_END
    res["segment_rule"] = "high = top 20% (CLV, and calibrated 90-day repeat probability)"
    return res, {"pred": pred, "actual": actual, "by_dec": by_dec}


# ---------------------------------------------------------------------------- SHAP


def shap_analysis(model, te, p_rep, gain, art_dir, results_dir, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(te), size=min(15000, len(te)), replace=False)
    Xs = te[FEATURES].iloc[idx]
    sv, Z = explain.shap_values(model, Xs)
    imp = explain.global_importance(sv, list(Z.columns), top=12)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=Z.columns)
    rank = mean_abs.rank(ascending=False).astype(int)
    focus = [
        "delay_last", "delay_mean", "late_share", "review_score_last", "review_score_mean",
        "low_review_share", "review_text_share", "freight_share_mean",
    ]  # fmt: skip
    last = np.flatnonzero((te["cutoff"] == te["cutoff"].max()).to_numpy())
    top_idx = last[np.argsort(-gain[last], kind="stable")[:10]]
    Xt = te[FEATURES].iloc[top_idx]
    svt, Zt = explain.shap_values(model, Xt)
    codes = explain.reason_codes(svt, Zt, 3)
    sample = pd.DataFrame(
        {
            "rank": np.arange(1, len(top_idx) + 1),
            "p_repeat_90d": np.asarray(p_rep)[top_idx].round(4),
            "reason_codes": codes,
        }
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    sample.to_csv(Path(results_dir) / "olist_reason_codes_sample.csv", index=False)
    res = {
        "rows_explained": int(len(Xs)),
        "top_features": imp.to_dict("records"),
        "delivery_and_review_features": {
            f: {"mean_abs_shap": float(mean_abs[f]), "rank": int(rank[f]), "of": int(len(rank))}
            for f in focus
        },
        "example_reason_code": codes[0],
    }
    return res, (sv, Z)


# ---------------------------------------------------------------------------- entry point


def run(
    data_dir="data",
    results_dir="results",
    docs_dir="docs",
    art_dir="artifacts",
    trials: dict | None = None,
    seed: int = 42,
):
    trials = {**TRIALS, **(trials or {})}
    snaps = prepare(data_dir, art_dir)
    orders, tr, te = snaps["orders"], snaps["train"], snaps["test"]
    out = {"seed": seed, "assumptions": ASSUMPTIONS}
    out["data"] = data_facts(orders, tr, te)
    _log(f"snapshots: train {len(tr):,} rows, test {len(te):,} rows")

    state, res, best_kind = train_models(snaps, art_dir, trials, seed)
    out["models"] = res
    p_rep = state[best_kind]["p"]
    yt = (1 - te["churn"]).to_numpy()
    out["calibration_table"] = reliability_table(yt, p_rep, 10)

    # shap needs a tree model: use the best tree family if logistic regression won
    tree_kind = max(
        (k for k in models.KINDS if k != "logreg"), key=lambda k: res["models"][k]["cv_auc_mean"]
    )
    _log("profit targeting")
    out["profit"], pf = profit_analysis(te, p_rep, ASSUMPTIONS, seed)
    _log("survival")
    out["survival"], sfigs = survival_analysis(orders, snaps["categories"], seed)
    _log("clv")
    out["clv"], cl = clv_analysis(orders, te, p_rep, ASSUMPTIONS, seed)
    _log(f"shap ({tree_kind})")
    out["shap"], (sv, Z) = shap_analysis(
        state[tree_kind]["model"], te, state[tree_kind]["p"], pf["gain"], art_dir, results_dir, seed
    )
    out["shap"]["model"] = tree_kind

    plots.profit_figure(pf["curves"], pf["n_rule"] / pf["n"], Path(docs_dir) / "profit_curve.png")
    plots.km_bands_figure(sfigs["delay"], Path(docs_dir) / "km_delay_band.png")
    plots.calibration_figure(
        yt, {k: state[k]["p"] for k in models.KINDS}, Path(docs_dir) / "calibration.png"
    )
    plots.clv_holdout_figure(cl["by_dec"], Path(docs_dir) / "clv_holdout.png")
    plots.shap_summary_figure(sv, Z, Path(docs_dir) / "shap_summary.png")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "olist_churn.json").write_text(json.dumps(out, indent=1, default=float))
    joblib.dump(state[best_kind], Path(art_dir) / "olist_selected_model.joblib")
    return out
