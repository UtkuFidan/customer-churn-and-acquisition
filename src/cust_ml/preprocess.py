"""Leakage-safe preprocessing: everything data-dependent lives inside an sklearn Pipeline,
so statistics (medians, scaling, category sets) are learned from the training fold only."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def build_preprocessor(
    num_cols: list[str], cat_cols: list[str], scale: bool = True
) -> ColumnTransformer:
    """Median-impute + (optionally) scale numerics; most-frequent-impute + one-hot categoricals.

    Unseen categories at predict time are ignored (all-zero row) instead of raising."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(X: pd.DataFrame, estimator, scale: bool = True) -> Pipeline:
    num, cat = split_columns(X)
    return Pipeline([("prep", build_preprocessor(num, cat, scale)), ("model", estimator)])
