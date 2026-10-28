"""Small, consistent matplotlib figures (saved at ~100 dpi to keep the repo light)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"a": "#2a6fdb", "b": "#e4572e", "c": "#2e9c6a", "g": "#8a8f98"}
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "axes.titleweight": "bold",
    }
)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def profit_figure(curves: dict, share: float, path, currency="R$"):
    """curves: label -> (fraction targeted, cumulative realised profit)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [C["a"], C["c"], C["b"], C["g"]]
    for (lab, (f, p)), col in zip(curves.items(), colors, strict=False):
        ax.plot(f * 100, p, label=lab, color=col, lw=2 if col == C["a"] else 1.4)
    ax.axvline(share * 100, color=C["a"], ls="--", lw=1)
    ax.text(share * 100 + 0.8, ax.get_ylim()[0], "expected gain > 0", fontsize=8, color=C["a"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set(
        xlabel="% of customers messaged (test snapshots)",
        ylabel=f"Cumulative realised profit ({currency})",
        title="Win-back message: profit by targeting order",
    )
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def km_bands_figure(fits: dict, path, horizon=270):
    """Share of first-time customers who have bought again, by first-order delivery band."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    cols = [C["c"], C["a"], "#5aa0d8", C["b"], "#a3331a"]
    for (lab, km), col in zip(fits.items(), cols, strict=False):
        cd = km.cumulative_density_.iloc[:, 0]
        ax.step(cd.index, cd.to_numpy() * 100, where="post", color=col, label=lab, lw=1.6)
    ax.set_xlim(0, horizon)
    ax.set(
        xlabel="Days since the first order was delivered",
        ylabel="Bought again, % of first-time customers",
        title="Time to second purchase by delivery delay",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, path)


def calibration_figure(y, preds: dict, path, n_bins=10):
    from cust_ml.metrics import reliability_table

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    top = 0.0
    for (lab, p), col in zip(preds.items(), [C["g"], C["a"], C["b"], C["c"]], strict=False):
        t = reliability_table(y, p, n_bins)
        x = [r["mean_pred"] * 100 for r in t]
        o = [r["observed"] * 100 for r in t]
        top = max(top, max(x), max(o))
        ax.plot(x, o, marker="o", ms=3, label=lab, color=col, lw=1.3)
    ax.plot([0, top], [0, top], color="k", ls=":", lw=1)
    ax.set(
        xlabel="Mean predicted repeat probability, % (decile bins)",
        ylabel="Observed repeat rate, %",
        title="Calibration on the test snapshots",
    )
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def clv_holdout_figure(by_dec, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(1, len(by_dec) + 1)
    ax.bar(x - 0.2, by_dec["pred"], 0.4, color=C["a"], label="BG/NBD prediction")
    ax.bar(x + 0.2, by_dec["actual"], 0.4, color=C["b"], label="Observed")
    ax.set(
        xlabel="Decile of predicted purchases (holdout window)",
        ylabel="Mean purchases per customer",
        title="BG/NBD against the holdout window",
    )
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def qini_figure(panels: dict, path, title):
    """panels: outcome -> {method: (x_fraction, qini_values)}; one subplot per outcome."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    names = {
        "response_model": "Response model",
        "two_model": "Two-model",
        "x_learner": "X-learner",
        "class_transformation": "Class transformation",
    }
    colors = {
        "response_model": C["g"],
        "two_model": C["a"],
        "x_learner": C["c"],
        "class_transformation": C["b"],
    }
    for ax, (outcome, curves) in zip(axes, panels.items(), strict=False):
        first = next(iter(curves.values()))
        ax.plot([0, 1], [0, first[1][-1]], color="k", ls=":", lw=1, label="Random targeting")
        for lab, (x, q) in curves.items():
            ax.plot(x, q, label=names.get(lab, lab), color=colors.get(lab), lw=1.6)
        ax.set(xlabel="Fraction targeted (ranked by score)", title=outcome)
    axes[0].set_ylabel("Cumulative incremental outcomes (Qini)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    _save(fig, path)


def lift_by_channel_figure(rows: list[dict], path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    names = [r["origin"] for r in rows]
    y = np.arange(len(rows))
    ax.barh(
        y + 0.2, [r["observed_close_rate"] * 100 for r in rows], 0.4, color=C["g"], label="Observed"
    )
    ax.barh(
        y - 0.2, [r["mean_predicted"] * 100 for r in rows], 0.4, color=C["a"], label="Model mean"
    )
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel="Close rate on test leads, %", title="Close rate by origin channel")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def shap_summary_figure(sv, X_t, path, max_display=12):
    import shap

    fig = plt.figure(figsize=(7, 4.6))
    shap.summary_plot(sv, X_t, max_display=max_display, show=False, plot_size=None)
    plt.title("SHAP summary (log-odds of a repeat purchase)", fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close("all")


__all__ = [
    "profit_figure",
    "km_bands_figure",
    "calibration_figure",
    "clv_holdout_figure",
    "qini_figure",
    "lift_by_channel_figure",
    "shap_summary_figure",
]
