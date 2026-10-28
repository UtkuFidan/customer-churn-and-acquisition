"""Uplift modelling: Qini / uplift curves with bootstrap intervals, and four scorers.

Scorers: a plain response model (ignores treatment), two-model (T-learner), X-learner
(Kunzel et al. 2019) and class transformation (Jaskowski and Jaroszewicz 2012).
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cust_ml.preprocess import make_pipeline

METHODS = ("response_model", "two_model", "x_learner", "class_transformation")


def _sorted_counts(y, t, score, w=None):
    order = np.argsort(-np.asarray(score), kind="stable")
    y, t = np.asarray(y)[order], np.asarray(t)[order]
    w = np.ones(len(y)) if w is None else np.asarray(w, float)[order]
    return (
        np.cumsum(w * t),
        np.cumsum(w * (1 - t)),
        np.cumsum(w * y * t),
        np.cumsum(w * y * (1 - t)),
    )


def qini_curve(y, t, score):
    """Qini curve: Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k) after targeting the top-k by score.

    Returns (x, q) with x = fraction of the population targeted (0..1) and q the cumulative
    incremental conversions (outcome units); starts at (0, 0)."""
    n_t, n_c, y_t, y_c = _sorted_counts(y, t, score)
    q = y_t - y_c * n_t / np.maximum(n_c, 1)
    n = len(q)
    return np.concatenate([[0.0], np.arange(1, n + 1) / n]), np.concatenate([[0.0], q])


def uplift_curve(y, t, score):
    """Uplift curve: U(k) = k * (Y_T/N_T - Y_C/N_C): incremental conversions if the top-k were
    treated, using the average uplift of the top-k extrapolated to k customers."""
    n_t, n_c, y_t, y_c = _sorted_counts(y, t, score)
    k = n_t + n_c
    u = k * (y_t / np.maximum(n_t, 1) - y_c / np.maximum(n_c, 1))
    n = len(u)
    return np.concatenate([[0.0], np.arange(1, n + 1) / n]), np.concatenate([[0.0], u])


def _area(x, y):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def qini_coefficient(y, t, score) -> float:
    """Area between the Qini curve and the random-targeting diagonal, as a share of the
    population (x in 0..1, q divided by N), so it is comparable across sample sizes."""
    x, q = qini_curve(y, t, score)
    return (_area(x, q) - 0.5 * q[-1]) / len(score)


def auuc(y, t, score) -> float:
    """Area under the uplift curve (x in 0..1, uplift divided by N). Compare against
    ``auuc_random`` (the area under the straight line to the overall ATE)."""
    x, u = uplift_curve(y, t, score)
    return _area(x, u) / len(score)


def auuc_random(y, t) -> float:
    y, t = np.asarray(y), np.asarray(t)
    ate = y[t == 1].mean() - y[t == 0].mean()
    return 0.5 * ate


def incremental_at(y, t, score, n_targeted: int) -> float:
    """Estimated incremental conversions if the top ``n_targeted`` customers by score are
    treated (uplift-curve estimate from the randomised treatment/control split)."""
    _, u = uplift_curve(y, t, score)
    return float(u[int(np.clip(n_targeted, 0, len(u) - 1))])


class BinnedCurves:
    """Qini coefficient and AUUC from curves evaluated at ``n_bins`` equal-size cuts of the
    ranking, with optional row weights. Used for the bootstrap on millions of rows: each
    replicate is one weighted pass instead of a sort. Differences to the exact functions above
    are tested to be small."""

    def __init__(self, y, t, scores: dict[str, np.ndarray], n_bins: int = 1000):
        self.y, self.t = np.asarray(y, float), np.asarray(t, float)
        n = len(self.y)
        self.edges = np.linspace(0, n, n_bins + 1).astype(int)
        self.order = {k: np.argsort(-np.asarray(s), kind="stable") for k, s in scores.items()}
        self.n = n

    def _stats(self, name, w):
        o = self.order[name]
        w = w[o]
        t, y = self.t[o], self.y[o]
        cut = self.edges[1:] - 1
        cs = lambda a: np.concatenate([[0.0], np.cumsum(a)[cut]])  # noqa: E731
        return cs(w * t), cs(w * (1 - t)), cs(w * y * t), cs(w * y * (1 - t)), cs(w)

    def curve(self, name):
        """(x, Qini value) at the bin edges, unweighted; for plotting."""
        n_t, n_c, y_t, y_c, k = self._stats(name, np.ones(self.n))
        return k / k[-1], y_t - y_c * n_t / np.maximum(n_c, 1e-12)

    def metrics(self, name, w=None) -> dict[str, float]:
        w = np.ones(self.n) if w is None else w
        n_t, n_c, y_t, y_c, k = self._stats(name, w)
        total = k[-1]
        x = k / total
        q = y_t - y_c * n_t / np.maximum(n_c, 1e-12)
        u = k * (y_t / np.maximum(n_t, 1e-12) - y_c / np.maximum(n_c, 1e-12))
        u[0] = 0.0
        return {
            "qini": (_area(x, q) - 0.5 * q[-1]) / total,
            "auuc": _area(x, u) / total,
            "auuc_random": 0.5 * u[-1] / total,
            "uplift_at_30": float(np.interp(0.3, x, u)) / total,
        }

    def bootstrap(self, names, n_boot=200, seed=0) -> dict[str, dict[str, np.ndarray]]:
        """Poisson(1) row weights, shared by all methods so differences are paired."""
        rng = np.random.default_rng(seed)
        out = {n: {} for n in names}
        for _ in range(n_boot):
            w = rng.poisson(1.0, self.n).astype(float)
            for n in names:
                for m, v in self.metrics(n, w).items():
                    out[n].setdefault(m, []).append(v)
        return {n: {m: np.array(v) for m, v in d.items()} for n, d in out.items()}


def default_learner(kind: str = "lgbm", seed: int = 0, big: bool = False):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=0.1)
    return LGBMClassifier(
        n_estimators=300 if big else 150,
        learning_rate=0.05 if big else 0.03,
        num_leaves=31 if big else 6,
        min_child_samples=1000 if big else 300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _regressor_like(learner):
    if isinstance(learner, LGBMClassifier):
        return LGBMRegressor(**learner.get_params())
    from sklearn.linear_model import Ridge

    return Ridge(alpha=10.0)


def _fit(X, y, learner, scale, sample_weight=None):
    pipe = make_pipeline(X, clone(learner), scale=scale)
    pipe.fit(X, y, **({"model__sample_weight": sample_weight} if sample_weight is not None else {}))
    return pipe


def _predict(pipe, X):
    m = pipe[-1]
    return pipe.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else pipe.predict(X)


def response_scores(X_tr, y_tr, X_te, learner, scale=False):
    """Plain response ('propensity') model P(Y=1|X), ignoring treatment: the baseline that
    targets likely converters rather than persuadable ones."""
    return _predict(_fit(X_tr, y_tr, learner, scale), X_te)


def two_model_scores(X_tr, y_tr, t_tr, X_te, learner, scale=False):
    """Two-model (T-learner): uplift = P(Y=1|X, treated) - P(Y=1|X, control)."""
    t_tr = np.asarray(t_tr)
    m1 = _fit(X_tr[t_tr == 1], np.asarray(y_tr)[t_tr == 1], learner, scale)
    m0 = _fit(X_tr[t_tr == 0], np.asarray(y_tr)[t_tr == 0], learner, scale)
    return _predict(m1, X_te) - _predict(m0, X_te)


def x_learner_scores(X_tr, y_tr, t_tr, X_te, learner, scale=False):
    """X-learner. Stage 1 fits outcome models per arm. Stage 2 regresses the imputed individual
    effects (observed minus the other arm's prediction) on X, once per arm. The two effect
    estimates are blended with the treatment share (the propensity in a randomised test)."""
    y_tr, t_tr = np.asarray(y_tr), np.asarray(t_tr)
    X1, X0 = X_tr[t_tr == 1], X_tr[t_tr == 0]
    y1, y0 = y_tr[t_tr == 1], y_tr[t_tr == 0]
    mu1, mu0 = _fit(X1, y1, learner, scale), _fit(X0, y0, learner, scale)
    d1 = y1 - _predict(mu0, X1)  # effect imputed for treated
    d0 = _predict(mu1, X0) - y0  # effect imputed for control
    reg = _regressor_like(learner)
    tau1, tau0 = _fit(X1, d1, reg, scale), _fit(X0, d0, reg, scale)
    g = t_tr.mean()  # P(treated), constant by design
    return g * _predict(tau0, X_te) + (1 - g) * _predict(tau1, X_te)


def class_transform_scores(X_tr, y_tr, t_tr, X_te, learner, scale=False):
    """Class transformation (Jaskowski & Jaroszewicz): Z = 1 if (T=1,Y=1) or (T=0,Y=0).
    With treatment/control balanced by sample weights, uplift = 2 * P(Z=1|X) - 1."""
    y_tr, t_tr = np.asarray(y_tr), np.asarray(t_tr)
    z = (y_tr == t_tr).astype(int)
    w = np.where(t_tr == 1, 0.5 / t_tr.mean(), 0.5 / (1 - t_tr.mean()))
    return 2 * _predict(_fit(X_tr, z, learner, scale, w), X_te) - 1


def score_all(X_tr, y_tr, t_tr, X_te, learner, scale=False) -> dict[str, np.ndarray]:
    return {
        "response_model": response_scores(X_tr, y_tr, X_te, learner, scale),
        "two_model": two_model_scores(X_tr, y_tr, t_tr, X_te, learner, scale),
        "x_learner": x_learner_scores(X_tr, y_tr, t_tr, X_te, learner, scale),
        "class_transformation": class_transform_scores(X_tr, y_tr, t_tr, X_te, learner, scale),
    }


def cross_fit(X, y, t, learner_kind="lgbm", n_splits=5, seed=0) -> dict[str, np.ndarray]:
    """Out-of-fold scores for every row from each method, so one Qini curve covers the whole
    randomised sample (much lower variance than a single hold-out)."""
    scale = learner_kind == "logreg"
    learner = default_learner(learner_kind, seed)
    y, t = np.asarray(y), np.asarray(t)
    out = {k: np.zeros(len(y)) for k in METHODS}
    strat = t * 2 + y
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strat):
        part = score_all(X.iloc[tr], y[tr], t[tr], X.iloc[te], learner, scale)
        for k, v in part.items():
            out[k][te] = v
    return out
