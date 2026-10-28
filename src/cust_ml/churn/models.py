"""Churn models on stacked snapshots: time-aware CV, Optuna tuning and calibration.

All models predict the rare event (a purchase in the next N days, ``y = 1 - churn``); churn
probability is one minus that. Splits never mix time: a validation block is only scored by a model
trained on snapshots whose label window closed before the block's first cutoff.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

KINDS = ("logreg", "lightgbm", "xgboost", "catboost")
N_JOBS = -1


def time_splits(cutoffs, horizon_days: int, n_splits: int = 3, block: int = 2):
    """Expanding-window, purged splits over snapshot cutoffs.

    The last ``n_splits * block`` distinct cutoffs are cut into validation blocks. The training
    rows of a fold are the snapshots whose label window (cutoff + horizon) ends before the
    block's first cutoff, so no training label is observed after validation features exist.
    Yields (train_positions, valid_positions) as integer index arrays."""
    cut = pd.to_datetime(pd.Series(np.asarray(cutoffs)))
    uniq = np.sort(cut.unique())
    if len(uniq) < n_splits * block + 1:
        raise ValueError("not enough distinct cutoffs for the requested splits")
    first = len(uniq) - n_splits * block
    for k in range(n_splits):
        v = uniq[first + k * block : first + (k + 1) * block]
        limit = v[0] - np.timedelta64(horizon_days, "D")
        tr = np.flatnonzero((cut <= limit).to_numpy())
        va = np.flatnonzero(cut.isin(v).to_numpy())
        if len(tr) == 0:
            raise ValueError("empty training fold; use fewer splits or earlier cutoffs")
        yield tr, va


class ChurnModel:
    """Uniform wrapper: ``fit(X, y)``, ``predict_proba(X)`` -> P(y = 1) as a 1-D array.

    Preprocessing is fitted inside ``fit`` (training rows only). Logistic regression gets median
    imputation with missing-value flags, log-compressed money and scaling; the tree models keep
    NaN as a signal (an undelivered order or a missing review is information) and see ordinal
    codes for the three categorical columns; CatBoost takes the strings directly."""

    def __init__(self, kind: str, params: dict | None = None, seed: int = 0, neg_frac: float = 1.0):
        self.kind, self.params, self.seed, self.neg_frac = kind, dict(params or {}), seed, neg_frac

    def _split(self, X):
        cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
        return [c for c in X.columns if c not in cat], cat

    def fit(self, X: pd.DataFrame, y):
        self.num_, self.cat_ = self._split(X)
        y = np.asarray(y)
        if self.kind != "logreg" and self.neg_frac < 1.0:
            # keep every positive and a random share of negatives: faster boosting at a ~1%
            # base rate; probabilities are re-calibrated afterwards, so the shift is harmless
            rng = np.random.default_rng(self.seed)
            keep = (y == 1) | (rng.random(len(y)) < self.neg_frac)
            X, y = X.iloc[np.flatnonzero(keep)], y[keep]
        self.n_fit_rows_, self.n_fit_pos_ = len(y), int(y.sum())
        if self.kind == "logreg":
            C = self.params.get("C", 0.1)
            self.model_ = Pipeline(
                [
                    ("prep", self._lr_prep()),
                    ("lr", LogisticRegression(C=C, max_iter=3000)),
                ]
            )
            self.model_.fit(X, y)
        elif self.kind == "catboost":
            self.model_ = CatBoostClassifier(
                **self.params,
                random_seed=self.seed,
                verbose=0,
                thread_count=N_JOBS,
                allow_writing_files=False,
            )
            self.model_.fit(self._cb(X), y, cat_features=self.cat_)
        else:
            self.enc_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            ).fit(X[self.cat_])
            if self.kind == "lightgbm":
                self.model_ = LGBMClassifier(
                    **self.params, random_state=self.seed, n_jobs=N_JOBS, verbose=-1
                )
            else:
                self.model_ = XGBClassifier(
                    **self.params,
                    random_state=self.seed,
                    n_jobs=N_JOBS,
                    verbosity=0,
                    tree_method="hist",
                )
            self.model_.fit(self._ord(X), y)
        return self

    def _lr_prep(self):
        log_cols = [c for c in self.num_ if c in _LOG_COLS]
        plain = [c for c in self.num_ if c not in _LOG_COLS]
        num = lambda: [  # noqa: E731
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]
        return ColumnTransformer(
            [
                ("log", Pipeline([("log", FunctionTransformer(_log1p_clip))] + num()), log_cols),
                ("num", Pipeline(num()), plain),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    self.cat_,
                ),
            ]
        )

    def _ord(self, X):
        Z = X.copy()
        Z[self.cat_] = self.enc_.transform(X[self.cat_])
        return Z

    def _cb(self, X):
        Z = X.copy()
        Z[self.cat_] = Z[self.cat_].astype(object).where(Z[self.cat_].notna(), "missing")
        return Z

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "logreg":
            return self.model_.predict_proba(X)[:, 1]
        if self.kind == "catboost":
            return self.model_.predict_proba(self._cb(X))[:, 1]
        return self.model_.predict_proba(self._ord(X))[:, 1]

    def transform_for_trees(self, X: pd.DataFrame) -> pd.DataFrame:
        """Feature frame in the form the booster was trained on (for SHAP)."""
        return self._cb(X) if self.kind == "catboost" else self._ord(X)


_LOG_COLS = {
    "monetary_sum", "monetary_mean", "price_mean", "distance_km_mean", "distance_km_last",
    "recency_days", "tenure_days", "avg_gap_days", "days_since_delivery", "delivery_days_mean",
}  # fmt: skip


def _log1p_clip(X):
    """log1p on non-negative columns (negative inputs, none expected, are clipped to 0)."""
    return np.log1p(np.clip(np.asarray(X, float), 0, None))


def suggest(trial: optuna.Trial, kind: str) -> dict:
    if kind == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if kind == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 4, 48),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        }
    if kind == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        }
    return {
        "iterations": trial.suggest_int("iterations", 100, 400, step=50),
        "depth": trial.suggest_int("depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.4, 1.0),
    }


def cv_oof(kind, params, X, y, cutoffs, horizon_days, seed=0, neg_frac=1.0, n_splits=3, block=2):
    """Out-of-fold predictions on the validation blocks. Returns (positions, preds, fold_aucs)."""
    pos, preds, aucs = [], [], []
    for tr, va in time_splits(cutoffs, horizon_days, n_splits, block):
        m = ChurnModel(kind, params, seed, neg_frac).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])
        pos.append(va)
        preds.append(p)
        aucs.append(roc_auc_score(y[va], p))
    return np.concatenate(pos), np.concatenate(preds), aucs


def tune(kind, X, y, cutoffs, horizon_days, n_trials, seed=0, neg_frac=1.0, timeout=900):
    """Optuna TPE; objective is the mean ROC AUC over the time-aware validation blocks."""

    def objective(trial):
        params = suggest(trial, kind)
        return float(np.mean(cv_oof(kind, params, X, y, cutoffs, horizon_days, seed, neg_frac)[2]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)  # whichever comes first
    return study.best_params, float(study.best_value), study


class Calibrator:
    """Maps raw scores to probabilities. ``method`` is 'sigmoid' (Platt on the logit),
    'isotonic' or 'none'. Fitted on out-of-fold scores, never on the training fit."""

    def __init__(self, method: str):
        self.method = method

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        elif self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "sigmoid":
            return self.m_.predict_proba(self._logit(p))[:, 1]
        if self.method == "isotonic":
            return self.m_.predict(p)
        return p
