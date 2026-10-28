"""``cust-ml`` command line: churn | leads | uplift | benchmark."""

from __future__ import annotations

import argparse
import json


def _common(p):
    p.add_argument("--data-dir", default="data", help="cache directory for raw datasets")
    p.add_argument("--results-dir", default="results", help="where metric JSON files are written")
    p.add_argument("--docs-dir", default="docs", help="where figures are written")
    p.add_argument("--seed", type=int, default=42)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cust-ml", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("churn", help="Olist: snapshots, models, profit, survival, CLV, SHAP")
    _common(p)
    p.add_argument(
        "--art-dir", default="artifacts", help="cache for snapshots and tuned parameters"
    )
    p.add_argument(
        "--trials", type=int, default=None, help="Optuna trials per family (default: per-family)"
    )

    p = sub.add_parser("leads", help="Olist Marketing Funnel lead scoring and prioritisation")
    _common(p)
    p.add_argument("--trials", type=int, default=None)

    p = sub.add_parser("uplift", help="Criteo (sampled) and Hillstrom uplift comparison")
    _common(p)
    p.add_argument("--bootstrap", type=int, default=200)

    p = sub.add_parser("benchmark", help="IBM Telco, the classic churn table")
    _common(p)
    p.add_argument("--trials", type=int, default=15)

    a = ap.parse_args(argv)
    if a.cmd == "churn":
        from cust_ml.churn import pipeline

        trials = None if a.trials is None else dict.fromkeys(pipeline.TRIALS, a.trials)
        pipeline.run(a.data_dir, a.results_dir, a.docs_dir, a.art_dir, trials, a.seed)
    elif a.cmd == "leads":
        from cust_ml.acq import lead_pipeline

        trials = None if a.trials is None else dict.fromkeys(lead_pipeline.TRIALS, a.trials)
        lead_pipeline.run(a.data_dir, a.results_dir, a.docs_dir, trials, a.seed)
    elif a.cmd == "uplift":
        from cust_ml.acq import uplift_pipeline

        uplift_pipeline.run(a.data_dir, a.results_dir, a.docs_dir, a.bootstrap, 0)
    else:
        from pathlib import Path

        from cust_ml.churn import telco

        out = telco.run(a.data_dir, a.trials, a.seed)
        Path(a.results_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.results_dir) / "telco_benchmark.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
