"""IBM Telco churn as a classic benchmark: the same four model families on a textbook table.

Included only to show how different an easy, snapshot-labelled churn problem looks from the
time-based Olist one. Random 75/25 split, light Optuna tuning on 3-fold CV AUC.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cust_ml.churn import models
from cust_ml.data import load_telco
from cust_ml.metrics import ap_w, auc_w, group_bootstrap


def clean_telco(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    d = raw.drop(columns=["customerID"]).copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"].replace(" ", np.nan))
    y = (d.pop("Churn") == "Yes").astype(int).to_numpy()
    for c in d.select_dtypes(exclude="number").columns:
        d[c] = d[c].astype(str)
    return d, y


def _cv_auc(kind, params, X, y, seed) -> float:
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in cv.split(X, y):
        m = models.ChurnModel(kind, params, seed).fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])))
    return float(np.mean(aucs))


def run(data_dir="data", trials=15, seed=42) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = clean_telco(load_telco(data_dir))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    out = {"n": int(len(y)), "churn_rate": float(y.mean()), "test_n": int(len(yte)), "models": {}}
    for kind in models.KINDS:

        def objective(trial, kind=kind):
            return _cv_auc(kind, models.suggest(trial, kind), Xtr, ytr, seed)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=trials)
        m = models.ChurnModel(kind, study.best_params, seed).fit(Xtr, ytr)
        p = m.predict_proba(Xte)
        g = np.arange(len(yte))
        auc = group_bootstrap(auc_w, (yte, p), g, 300, seed)
        ap = group_bootstrap(ap_w, (yte, p), g, 300, seed)
        out["models"][kind] = {
            "cv_auc": float(study.best_value),
            "test_auc": auc[0],
            "test_auc_ci": [auc[1], auc[2]],
            "test_pr_auc": ap[0],
            "test_pr_auc_ci": [ap[1], ap[2]],
            "test_brier": float(brier_score_loss(yte, p)),
        }
    return out
