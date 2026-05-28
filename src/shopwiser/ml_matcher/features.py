"""Feature engineering for product pairs."""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Conflict vocabulary lives in shopwiser.conflict_tokens (single source of truth).
from shopwiser.conflict_tokens import check_hard_conflict as _check_hard_conflict_bool


def check_hard_conflict(name_a: str, name_b: str) -> int:
    """1 when any hard-conflict gate fires between the two product names, else 0.

    Thin int-returning wrapper around the canonical implementation in
    ``shopwiser.conflict_tokens.check_hard_conflict``. Kept as int because
    the value feeds directly into the LightGBM ranker as a numeric feature.
    """
    return int(_check_hard_conflict_bool(name_a, name_b))


def _own_brand_int(series: pd.Series) -> pd.Series:
    """CSV/bool-safe 0/1 flags for own-brand columns."""
    return series.astype(str).str.lower().isin(('true', '1', 'yes')).astype(int)


def _best_delta_size(uv_a: np.ndarray, pq_a: np.ndarray,
                     uv_b: np.ndarray, pq_b: np.ndarray) -> np.ndarray:
    """Compute relative size difference tolerant of one-sided multipack reporting.

    Supermarkets inconsistently report multipacks: one SM may store the TOTAL
    (1650 ml, pq=NaN) while another stores PER-UNIT (275 ml, pq=6).  A naïve
    base_size = uv/pq gives 1650 vs 45.8 — a 97% delta that gates out a valid pair.

    Fix: compute delta under three interpretations and take the minimum:
      1. per-unit vs per-unit : (uv_a/pq_a) vs (uv_b/pq_b)
      2. A-total vs B-total-from-pack : uv_a vs (uv_b × pq_b)
      3. A-total-from-pack vs B-total : (uv_a × pq_a) vs uv_b
    """
    def _rd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        hi = np.maximum(np.abs(x), np.abs(y))
        return np.where(hi > 1e-5, np.abs(x - y) / hi, 0.0)

    d1 = _rd(uv_a / pq_a, uv_b / pq_b)          # per-unit vs per-unit
    d2 = _rd(uv_a, uv_b * pq_b)                  # A total vs B total-from-pack
    d3 = _rd(uv_a * pq_a, uv_b)                  # A total-from-pack vs B total

    return np.minimum(np.minimum(d1, d2), d3)


def build_pairwise_features(df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the feature matrix (Level C) for all retrieved candidate pairs."""

    # Join A data
    df_a = df.add_suffix('_a')
    feat = pairs_df.merge(df_a, left_on='id_a', right_on='product_idx_a', how='left')

    # Join B data
    df_b = df.add_suffix('_b')
    feat = feat.merge(df_b, left_on='id_b', right_on='product_idx_b', how='left')

    features = pd.DataFrame()
    features['id_a'] = feat['id_a']
    features['id_b'] = feat['id_b']
    features['cosine_sim'] = feat['score']  # from FAISS

    features['unit_missing_a'] = feat['unit_value_a'].isna().astype(int)
    features['unit_missing_b'] = feat['unit_value_b'].isna().astype(int)
    features['both_units_present'] = (feat['unit_value_a'].notna() & feat['unit_value_b'].notna()).astype(int)

    # Pack quantities — treat NaN as 1 (single unit)
    pq_a = np.where(feat['pack_quantity_a'].isna(), 1.0, feat['pack_quantity_a'].fillna(1).clip(lower=1))
    pq_b = np.where(feat['pack_quantity_b'].isna(), 1.0, feat['pack_quantity_b'].fillna(1).clip(lower=1))

    uv_a = feat['unit_value_a'].fillna(0).values.astype(float)
    uv_b = feat['unit_value_b'].fillna(0).values.astype(float)

    both_present = feat['unit_value_a'].notna().values & feat['unit_value_b'].notna().values

    # Multi-interpretation size delta (handles one-sided multipack reporting)
    raw_delta = _best_delta_size(uv_a, pq_a, uv_b, pq_b)
    features['delta_size'] = np.where(both_present, raw_delta, np.nan)

    # Total-only size delta: relative difference between uv_a and uv_b directly.
    # Since unit_value stores the TOTAL package amount in this dataset, this is
    # the correct measure for "same physical product" — d1 (per-unit) collapses
    # 1x750ml vs 4x750ml to zero, which incorrectly admits multipack/single pairs.
    # delta_size (multi-interp) stays as the model feature for flexibility; hard
    # gates use delta_size_total.
    hi = np.maximum(np.abs(uv_a), np.abs(uv_b))
    total_delta = np.where(hi > 1e-5, np.abs(uv_a - uv_b) / hi, 0.0)
    features['delta_size_total'] = np.where(both_present, total_delta, np.nan)

    features['same_unit_type'] = np.where(
        feat['unit_type_a'].notna() & feat['unit_type_b'].notna(),
        (feat['unit_type_a'] == feat['unit_type_b']).astype(int),
        np.nan,
    )

    # Metadata Agreement
    features['same_brand'] = np.where(
        feat['known_brand_clean_a'].notna() & feat['known_brand_clean_b'].notna(),
        (feat['known_brand_clean_a'] == feat['known_brand_clean_b']).astype(int),
        np.nan,
    )
    features['same_category'] = (feat['category_a'] == feat['category_b']).astype(int)
    features['is_own_brand_a'] = _own_brand_int(feat['own_brand_a'])
    features['is_own_brand_b'] = _own_brand_int(feat['own_brand_b'])

    # Truncation flags — True when the product name was source-truncated (Morrisons).
    # Lets the model discount low fuzz_sort values that arise from truncation, not
    # genuine name dissimilarity.
    if 'is_truncated_a' in feat.columns:
        features['is_truncated_a'] = feat['is_truncated_a'].fillna(False).astype(int)
        features['is_truncated_b'] = feat['is_truncated_b'].fillna(False).astype(int)
    else:
        features['is_truncated_a'] = 0
        features['is_truncated_b'] = 0

    # Text Similarities
    names_a = feat['normalized_name_a'].fillna('')
    names_b = feat['normalized_name_b'].fillna('')

    features['fuzz_sort'] = [fuzz.token_sort_ratio(a, b) for a, b in zip(names_a, names_b, strict=True)]
    features['fuzz_set'] = [fuzz.token_set_ratio(a, b) for a, b in zip(names_a, names_b, strict=True)]
    features['fuzz_partial'] = [fuzz.partial_ratio(a, b) for a, b in zip(names_a, names_b, strict=True)]
    features['hard_conflict'] = [check_hard_conflict(a, b) for a, b in zip(names_a, names_b, strict=True)]

    return features
