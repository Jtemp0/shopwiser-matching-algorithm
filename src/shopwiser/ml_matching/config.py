"""Configuration for the ML-based Matching Pipeline."""

from pathlib import Path

from shopwiser.paths import ml_matching_outputs_path, normalized_products_path

INPUT_CSV: Path = normalized_products_path(sample=False)
OUTPUT_DIR: Path = ml_matching_outputs_path(sample=False)


def configure_paths(*, sample: bool = False) -> None:
    global INPUT_CSV, OUTPUT_DIR
    INPUT_CSV = normalized_products_path(sample=sample)
    OUTPUT_DIR = ml_matching_outputs_path(sample=sample)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Level A: Retrieval
EMBEDDING_MODEL = 'all-mpnet-base-v2'
TOP_K_CANDIDATES = 150

# Level B: Gating — 30% relative size difference allowed before LightGBM
SIZE_GATE_TOLERANCE = 0.30

# Level C: Ranker
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'is_unbalance': True,
    'verbose': -1,
    'random_state': 42,
    'num_threads': 1,
    'force_col_wise': True,
}

LGBM_NUM_BOOST_ROUNDS = 150

# Final Decision Logic
# ACCEPT_THRESHOLD: minimum match_prob for the forward argmax direction
# REVERSE_THRESHOLD: minimum match_prob for the reverse confirmation (softer —
#   the reverse path scores slightly lower when brand extraction is asymmetric)
ACCEPT_THRESHOLD = 0.13
REVERSE_THRESHOLD = 0.09
MARGIN_THRESHOLD = 0.04

# Cluster completion: looser than ACCEPT_THRESHOLD but tight enough to prevent
# weak-transitive chains from bridging incomplete clusters.  Precision-critical
# because completion edges are only validated by already-present cluster members,
# which themselves may have been admitted with marginal evidence.
COMPLETION_THRESHOLD = 0.15
COMPLETION_MAX_DELTA = 0.12   # size agreement required for completion bridges

# Cluster-guided retrieval: after initial clustering, for each incomplete cluster
# query the missing SM's FAISS index with the cluster's mean embedding.
CLUSTER_GUIDED_TOP_K = 75
CLUSTER_GUIDED_MIN_SIM = 0.58
CLUSTER_GUIDED_MAX_DELTA = 0.12   # tightened from 0.20 — centroid drift compounds
CLUSTER_GUIDED_MIN_FUZZ = 65

# Blob splitting: clusters larger than MAX_CLUSTER_SIZE are almost certainly blobs
# formed by transitive closure of weak edges.  They are rebuilt from scratch using
# only scored pairs above BLOB_SPLIT_THRESHOLD — a stricter gate that retains
# genuine same-product links while breaking false-positive chains.
MAX_CLUSTER_SIZE = 12
BLOB_SPLIT_THRESHOLD = 0.50

# Hard size gate at edge acceptance: pairs where both unit values are known and
# delta_size exceeds this value are rejected even if match_prob ≥ ACCEPT_THRESHOLD.
# Consistent with the silver-negative definition (same_brand + delta_size > 0.20 → neg)
# and the blob re-admission gate (delta_size ≤ 0.15).  Prevents same-brand,
# different-size transitive chains from forming blob clusters.
ACCEPT_SIZE_GATE = 0.20
