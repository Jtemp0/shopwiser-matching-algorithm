"""Central path configuration for the ShopWiser pipeline (cwd-independent)."""

import sys
from pathlib import Path

# src/shopwiser/paths.py -> parents[2] = repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_on_syspath() -> None:
    """Put the repo root on ``sys.path`` so ``import tests.…`` works under ``python -m``."""
    r = str(PROJECT_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUTS = PROJECT_ROOT / "data" / "outputs"
DATA_OUTPUTS_CLUSTERS = DATA_OUTPUTS / "clusters"
DATA_OUTPUTS_CLUSTERS_SAMPLE = DATA_OUTPUTS / "clusters_sample"
DATA_OUTPUTS_ML_CLUSTERS = DATA_OUTPUTS / "ml_clusters"
DATA_OUTPUTS_ML_CLUSTERS_SAMPLE = DATA_OUTPUTS / "ml_clusters_sample"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"

RAW_CSV = "raw.csv"
RAW_1000_CSV = "raw_1000.csv"
ALL_DEC_DATA_CSV = "all_dec_data.csv"
NORMALIZED_PRODUCTS_CSV = "normalized_products.csv"
NORMALIZED_SAMPLE_CSV = "normalized_products_sample.csv"


def raw_csv_path(*, sample: bool = False) -> Path:
    """Full scrape vs ~1000-row development slice in ``data/raw/``."""
    return DATA_RAW / (RAW_1000_CSV if sample else RAW_CSV)


def normalized_products_path(*, sample: bool = False) -> Path:
    """Processed file written by the normalisation step (full vs sample run)."""
    name = NORMALIZED_SAMPLE_CSV if sample else NORMALIZED_PRODUCTS_CSV
    return DATA_PROCESSED / name


def cluster_outputs_path(*, sample: bool = False) -> Path:
    """Cluster CSVs and audit samples (separate folder so sample runs do not overwrite)."""
    return DATA_OUTPUTS_CLUSTERS_SAMPLE if sample else DATA_OUTPUTS_CLUSTERS


def ml_matching_outputs_path(*, sample: bool = False) -> Path:
    """FAISS + LightGBM matching outputs (separate from heuristic clusters)."""
    return DATA_OUTPUTS_ML_CLUSTERS_SAMPLE if sample else DATA_OUTPUTS_ML_CLUSTERS


# Backwards-compatible defaults (full dataset)
INPUT_RAW = raw_csv_path(sample=False)
NORMALIZED_PRODUCTS = normalized_products_path(sample=False)
