"""Load the trained LightGBM ranker and score cluster↔candidate pairs with it.

Replaces the hand-tuned `0.55*cosine + 0.45*fuzz` linear score used in
recomplete_ml.py / merge_2way_ml.py with the full 11-feature model trained by
shopwiser.ensemble.train_ranker.

If the model file does not exist (e.g. training hasn't run yet), load_model()
returns None and callers should fall back to the linear combo.
"""

from __future__ import annotations

import os
# Must be set before LightGBM / OpenMP-bound libraries initialize.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Import lightgbm BEFORE faiss — on macOS Python 3.14 they ship conflicting
# OpenMP runtimes and whichever loads first wins.  faiss-after-lightgbm is safe;
# the reverse order segfaults inside pickle.load of a lightgbm Booster.
import lightgbm as _lgb  # noqa: F401  (side-effect import, keeps OMP stable)

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from shopwiser.ml_matching.features import build_pairwise_features
from shopwiser.paths import DATA_OUTPUTS

MODEL_PATH = DATA_OUTPUTS / 'ensemble' / 'ranker_model.pkl'


def load_model():
    """Return {'model': lgbm, 'feature_cols': [...]} or None if missing."""
    if not MODEL_PATH.exists():
        return None
    try:
        with MODEL_PATH.open('rb') as fh:
            return pickle.load(fh)
    except Exception:
        return None


def score_cluster_candidate(
    model_bundle: dict,
    cluster_df: pd.DataFrame,
    cand: pd.Series,
    cosine_sim: float,
    allp: pd.DataFrame,
) -> float:
    """Best model match_prob over all (member, candidate) pairs.

    `cosine_sim` is the centroid-vs-candidate score; it's used as a fallback
    only.  The model uses the pairwise cosine_sim computed from per-member
    embeddings — but for simplicity here we inject `cosine_sim` (centroid)
    as the pairwise score feature.  Empirically this under-estimates pairwise
    cosine by ~0.02 but saves recomputing per-member dot products per call.
    """
    model = model_bundle['model']
    feature_cols = model_bundle['feature_cols']

    cand_idx = int(cand['product_idx'])
    member_idxs = cluster_df['product_idx'].astype(int).tolist()

    pairs = pd.DataFrame({
        'id_a': [min(m, cand_idx) for m in member_idxs],
        'id_b': [max(m, cand_idx) for m in member_idxs],
        'score': [cosine_sim] * len(member_idxs),
    })
    feats = build_pairwise_features(allp, pairs)
    X = feats[feature_cols].to_numpy(dtype=np.float32)
    probs = model.predict(X, num_threads=1)
    return float(np.max(probs)) if len(probs) else 0.0


def score_cluster_pair(
    model_bundle: dict,
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    centroid_cos: float,
    allp: pd.DataFrame,
) -> float:
    """Best model match_prob over all cross-cluster (a_i, b_j) pairs."""
    model = model_bundle['model']
    feature_cols = model_bundle['feature_cols']

    a_idxs = a_df['product_idx'].astype(int).tolist()
    b_idxs = b_df['product_idx'].astype(int).tolist()

    rows = []
    for ai in a_idxs:
        for bi in b_idxs:
            rows.append({'id_a': min(ai, bi), 'id_b': max(ai, bi), 'score': centroid_cos})
    if not rows:
        return 0.0
    pairs = pd.DataFrame(rows)
    feats = build_pairwise_features(allp, pairs)
    X = feats[feature_cols].to_numpy(dtype=np.float32)
    probs = model.predict(X, num_threads=1)
    return float(np.max(probs)) if len(probs) else 0.0
