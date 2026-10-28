"""Dataset access. Raw files live in ``data/`` (gitignored) and are fetched on first use.

Olist (two datasets) and the Criteo fallback come from Kaggle through the official ``kaggle``
CLI, which needs your own credentials. Hillstrom and Telco are plain HTTP downloads.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)
HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"

OLIST_SLUG = "olistbr/brazilian-ecommerce"
FUNNEL_SLUG = "olistbr/marketing-funnel-olist"
CRITEO_KAGGLE_SLUG = "arashnic/uplift-modeling"  # copy of the official v2.1 file

CRITEO_FEATURES = [f"f{i}" for i in range(12)]


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cust-ml/0.2"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    return dest


def kaggle_download(slug: str, dest: Path) -> None:
    """``kaggle datasets download`` into ``dest``, unzipped. Requires the kaggle CLI + login."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"The kaggle CLI is needed to fetch {slug}. Install it (pip install kaggle), "
            f"log in, or place the unzipped files in {dest}."
        ) from e


def _kaggle_dir(data_dir: str | Path, name: str, slug: str, probe: str) -> Path:
    d = Path(data_dir) / name
    if not (d / probe).exists():
        kaggle_download(slug, d)
    return d


def load_olist_tables(
    data_dir: str | Path = "data", only: tuple[str, ...] | None = None
) -> dict[str, pd.DataFrame]:
    """The Olist e-commerce tables used here, keyed by short name (``only`` limits the load)."""
    d = _kaggle_dir(data_dir, "olist", OLIST_SLUG, "olist_orders_dataset.csv")
    ts = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    readers = {
        "orders": lambda: pd.read_csv(d / "olist_orders_dataset.csv", parse_dates=ts),
        "customers": lambda: pd.read_csv(d / "olist_customers_dataset.csv"),
        "items": lambda: pd.read_csv(d / "olist_order_items_dataset.csv"),
        "payments": lambda: pd.read_csv(d / "olist_order_payments_dataset.csv"),
        "reviews": lambda: pd.read_csv(
            d / "olist_order_reviews_dataset.csv",
            parse_dates=["review_creation_date", "review_answer_timestamp"],
        ),
        "products": lambda: pd.read_csv(d / "olist_products_dataset.csv"),
        "sellers": lambda: pd.read_csv(d / "olist_sellers_dataset.csv"),
        "geo": lambda: pd.read_csv(
            d / "olist_geolocation_dataset.csv",
            usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
        ),
        "categories": lambda: pd.read_csv(d / "product_category_name_translation.csv"),
    }
    return {k: f() for k, f in readers.items() if only is None or k in only}


def load_funnel_tables(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    d = _kaggle_dir(data_dir, "funnel", FUNNEL_SLUG, "olist_closed_deals_dataset.csv")
    return {
        "mql": pd.read_csv(
            d / "olist_marketing_qualified_leads_dataset.csv", parse_dates=["first_contact_date"]
        ),
        "deals": pd.read_csv(d / "olist_closed_deals_dataset.csv", parse_dates=["won_date"]),
    }


def load_telco(data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_csv(_fetch(TELCO_URL, Path(data_dir) / "telco.csv"))


def load_hillstrom(data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_csv(_fetch(HILLSTROM_URL, Path(data_dir) / "hillstrom.csv"))


def _criteo_raw_path(data_dir: Path) -> Path:
    """Path to the full v2.1 CSV (plain or gzipped). Tries Criteo's own host, then Kaggle."""
    for name in ("criteo-uplift-v2.1.csv.gz", "criteo-uplift-v2.1.csv"):
        if (data_dir / name).exists():
            return data_dir / name
    try:
        return _fetch(CRITEO_URL, data_dir / "criteo-uplift-v2.1.csv.gz")
    except Exception as e:  # certificate or network failure on the official host
        print(f"Criteo host unreachable ({type(e).__name__}); falling back to Kaggle copy.")
    kdir = data_dir / "criteo_kaggle"
    if not any(kdir.glob("*.csv*")):
        kaggle_download(CRITEO_KAGGLE_SLUG, kdir)
    found = sorted(kdir.glob("*.csv*"))
    if not found:
        raise FileNotFoundError(f"no CSV found in {kdir}")
    return found[0]


def sample_criteo(
    data_dir: str | Path = "data", frac: float = 0.3, seed: int = 0, chunksize: int = 2_000_000
) -> pd.DataFrame:
    """Stratified sample of Criteo Uplift v2.1 (13,979,592 rows).

    Rows are drawn at the same rate ``frac`` within each (treatment, visit, conversion) cell,
    chunk by chunk, so the full file never sits in memory and the treatment share, visit rate
    and conversion rate are preserved. The sample is cached as a pickle in ``data_dir``, with the
    size of the full file in a JSON sidecar (``criteo_full_rows``).
    """
    data_dir = Path(data_dir)
    cache = data_dir / f"criteo_sample_{frac}_{seed}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    raw = _criteo_raw_path(data_dir)
    rng = np.random.default_rng(seed)
    parts, n_total = [], 0
    opener = gzip.open if raw.suffix == ".gz" else open
    with opener(raw, "rt") as fh:
        for chunk in pd.read_csv(fh, chunksize=chunksize):
            n_total += len(chunk)
            cell = (
                chunk["treatment"].to_numpy() * 4
                + chunk["visit"].to_numpy() * 2
                + chunk["conversion"].to_numpy()
            )
            keep = np.zeros(len(chunk), dtype=bool)
            for c in np.unique(cell):
                idx = np.flatnonzero(cell == c)
                k = int(round(len(idx) * frac))
                keep[rng.choice(idx, size=k, replace=False)] = True
            part = chunk.loc[keep].drop(columns=["exposure"], errors="ignore")
            parts.append(part.astype({c: "float32" for c in CRITEO_FEATURES}))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    cache.with_suffix(".json").write_text(json.dumps({"criteo_full_rows": n_total}))
    return df


def criteo_full_rows(data_dir: str | Path, frac: float = 0.3, seed: int = 0) -> int:
    meta = Path(data_dir) / f"criteo_sample_{frac}_{seed}.json"
    return int(json.loads(meta.read_text())["criteo_full_rows"])
