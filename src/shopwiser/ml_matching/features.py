"""Feature engineering for product pairs."""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Ported from your original config - excellent heuristics to keep as ML features!
FLAVOR_NAMED_TOKENS = frozenset({'ginger', 'mint', 'raspberry', 'lemon', 'orange', 'cherry', 'strawberry', 'blueberry', 'mango', 'blackcurrant', 'blackberry', 'elderflower', 'rhubarb', 'lime', 'peach', 'apricot', 'vanilla', 'caramel', 'toffee', 'honey', 'maple', 'cinnamon', 'cola', 'lychee', 'basil', 'chilli', 'banana', 'syrup', 'kiwi', 'berry', 'melon', 'grape', 'pear', 'pineapple', 'pomegranate', 'watermelon', 'passion', 'fig', 'plum', 'lamb', 'ham', 'pork', 'beef', 'chicken', 'turkey', 'duck', 'venison', 'bacon', 'salmon', 'tuna', 'cod', 'haddock', 'prawn', 'shrimp', 'crab', 'mackerel', 'trout', 'sardine', 'anchovy'})
ONE_SIDED_CONFLICT_TOKENS = frozenset({'baby', 'reduced', 'granary', 'buttons', 'rose', 'light', 'lite', 'decaf', 'blonde', 'zero', 'soup', 'jam', 'juice', 'eyed', 'plum', 'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'salmon', 'tuna', 'cod', 'bacon', 'prawn'})


def check_hard_conflict(name_a: str, name_b: str) -> int:
    """Returns 1 if a hard semantic conflict is detected, else 0."""
    toks_a = set(str(name_a).lower().split())
    toks_b = set(str(name_b).lower().split())

    # 1. Flavor / Meat Clash (Only if both have a flavor token)
    flav_a = toks_a & FLAVOR_NAMED_TOKENS
    flav_b = toks_b & FLAVOR_NAMED_TOKENS
    if flav_a and flav_b and not (flav_a & flav_b):
        return 1

    # 2. One-sided descriptors (e.g. one is 'Diet', other is not)
    for tok in ONE_SIDED_CONFLICT_TOKENS:
        if (tok in toks_a) != (tok in toks_b):
            return 1

    return 0


def _own_brand_int(series: pd.Series) -> pd.Series:
    """CSV/bool-safe 0/1 flags for own-brand columns."""
    return series.astype(str).str.lower().isin(('true', '1', 'yes')).astype(int)


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

    # Base Unit Size Calculation (Fixes the Multipack Math Problem)
    pq_a = np.where(feat['pack_quantity_a'].isna(), 1.0, feat['pack_quantity_a'])
    pq_b = np.where(feat['pack_quantity_b'].isna(), 1.0, feat['pack_quantity_b'])
    base_size_a = feat['unit_value_a'] / pq_a
    base_size_b = feat['unit_value_b'] / pq_b

    # Numeric Agreement
    max_size = np.maximum(base_size_a, base_size_b)
    min_size = np.minimum(base_size_a, base_size_b)
    features['delta_size'] = np.where(
        feat['unit_value_a'].notna() & feat['unit_value_b'].notna(),
        (max_size - min_size) / np.maximum(max_size, 1e-5),
        np.nan,
    )
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

    # Text Similarities
    names_a = feat['normalized_name_a'].fillna('')
    names_b = feat['normalized_name_b'].fillna('')

    features['fuzz_sort'] = [fuzz.token_sort_ratio(a, b) for a, b in zip(names_a, names_b, strict=True)]
    features['fuzz_set'] = [fuzz.token_set_ratio(a, b) for a, b in zip(names_a, names_b, strict=True)]
    features['hard_conflict'] = [check_hard_conflict(a, b) for a, b in zip(names_a, names_b, strict=True)]

    return features
