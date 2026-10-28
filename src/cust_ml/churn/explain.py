"""SHAP attributions for the tree models and per-customer reason codes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from catboost import Pool

from cust_ml.churn.models import ChurnModel


def shap_values(model: ChurnModel, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values on the log-odds of a repeat purchase, and the feature frame they refer to."""
    if model.kind == "logreg":
        raise ValueError("SHAP here is for the tree models")
    Z = model.transform_for_trees(X)
    if model.kind == "catboost":
        sv = model.model_.get_feature_importance(
            Pool(Z, cat_features=model.cat_), type="ShapValues"
        )
        return sv[:, :-1], Z
    sv = shap.TreeExplainer(model.model_).shap_values(Z)
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv), Z


def global_importance(sv: np.ndarray, names: list[str], top: int = 10) -> pd.DataFrame:
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    return imp.head(top).rename("mean_abs_shap").rename_axis("feature").reset_index()


def _fmt(v) -> str:
    return v if isinstance(v, str) else f"{v:.5g}"


def reason_codes(sv: np.ndarray, X: pd.DataFrame, k: int = 3) -> list[str]:
    """Top-k features by |SHAP| per row as 'feature=value (+shap)'; positive pushes the
    repeat-purchase probability up."""
    names = X.columns.to_numpy()
    vals = X.to_numpy()
    out = []
    for i in range(len(X)):
        top = np.argsort(-np.abs(sv[i]))[:k]
        out.append("; ".join(f"{names[j]}={_fmt(vals[i, j])} ({sv[i, j]:+.2f})" for j in top))
    return out
