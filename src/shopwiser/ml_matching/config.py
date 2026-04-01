"""Configuration for the ML-based Matching Pipeline."""

from pathlib import Path

from shopwiser.paths import ml_matching_outputs_path, normalized_products_path

# Set by configure_paths() before each run (mirrors heuristic clustering pattern).
INPUT_CSV: Path = normalized_products_path(sample=False)
OUTPUT_DIR: Path = ml_matching_outputs_path(sample=False)


def configure_paths(*, sample: bool = False) -> None:
    """Point input/output at full vs sample artefacts under ``data/``."""
    global INPUT_CSV, OUTPUT_DIR
    INPUT_CSV = normalized_products_path(sample=sample)
    OUTPUT_DIR = ml_matching_outputs_path(sample=sample)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Retrieval (Level A) Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Fast, highly accurate semantic search model
TOP_K_CANDIDATES = 25                 # How many candidates to pull per retailer

# Gating (Level B) Configuration
# Reject pairs before ML scoring if the base unit size differs by more than 20%
SIZE_GATE_TOLERANCE = 0.20

# Ranker (Level C) Configuration
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'is_unbalance': True,
    'verbose': -1,
    'random_state': 42,
    'num_threads': 1,           
    'force_col_wise': True      

}

LGBM_NUM_BOOST_ROUNDS = 100

# Final Decision Configuration
ACCEPT_THRESHOLD = 0.50   # Minimum ML probability to accept a match
MARGIN_THRESHOLD = 0.10   # The best match must beat the second-best match by this much
