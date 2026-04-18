"""Level C: Bootstrapping Silver Labels and Training the GBDT Ranker."""

import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import LGBM_NUM_BOOST_ROUNDS, LGBM_PARAMS, SIZE_GATE_TOLERANCE

# Rows per predict() chunk so tqdm can show scoring progress on ~1M+ rows.
_PREDICT_CHUNK_ROWS = 65_536

# Feature columns used by the LGBM model — exported so other modules can build
# compatible feature vectors for model-based scoring (e.g. model-guided retrieval).
FEATURE_COLS = [
    'cosine_sim', 'delta_size', 'same_unit_type', 'same_brand',
    'same_category', 'is_own_brand_a', 'is_own_brand_b',
    'fuzz_sort', 'fuzz_set', 'fuzz_partial', 'hard_conflict',
]


def _try_import_lightgbm():
    import lightgbm as lgb
    return lgb


def generate_silver_labels(feat_df: pd.DataFrame) -> pd.DataFrame:
    print('Generating Silver Labels for training...')
    df = feat_df.copy()

    # 1. Silver Positives
    ds = df['delta_size']
    size_tight  = ds.isna() | ((ds >= 0) & (ds <= 0.05))   # ≤5% or missing
    size_medium = ds.isna() | ((ds >= 0) & (ds <= 0.10))   # ≤10% or missing

    # Case A: Same brand + tight size + text similarity
    case_a = (
        (df['same_brand'] == 1)
        & size_tight
        & (df['fuzz_sort'] >= 75)
        & (df['hard_conflict'] == 0)
    )

    # Case B: Both own-brand + same category + tight size + text similarity
    case_b = (
        (df['is_own_brand_a'] == 1) & (df['is_own_brand_b'] == 1)
        & (df['same_category'] == 1)
        & size_tight
        & (df['fuzz_sort'] >= 75)
        & (df['hard_conflict'] == 0)
    )

    # Case C: High text + size agreement — brand-detection-failure-tolerant.
    # Catches valid matches where one SM's brand wasn't extracted (same_brand=NaN).
    # Requires same unit type and same branded status to guard against cross-brand FPs.
    # Deliberately broader than Cases A/B so the model learns from asymmetric-brand pairs.
    case_c = (
        (df['fuzz_sort'] >= 85)
        & size_medium
        & (df['hard_conflict'] == 0)
        & (df['same_unit_type'] == 1)
        & (df['is_own_brand_a'] == df['is_own_brand_b'])
    )

    pos_mask = case_a | case_b | case_c

    # 2. Silver Negatives (Contradictions)
    neg_mask = (
        ((df['same_brand'] == 1) & (df['delta_size'] > 0.20))
        | ((df['fuzz_sort'] > 90) & (df['same_brand'] == 0) & (df['is_own_brand_a'] == 0))
        | (df['hard_conflict'] == 1)
        | (df['is_own_brand_a'] != df['is_own_brand_b'])
    )

    df['label'] = -1
    df.loc[pos_mask, 'label'] = 1
    df.loc[neg_mask, 'label'] = 0

    labeled_df = df[df['label'] != -1]
    print(f"  -> Generated {len(labeled_df[labeled_df['label'] == 1]):,} Positives")
    print(f"  -> Generated {len(labeled_df[labeled_df['label'] == 0]):,} Negatives")

    return labeled_df


def _predict_probs_chunked_lgb(model, X: np.ndarray, n_rows: int) -> np.ndarray:
    with tqdm(
        total=n_rows,
        desc='LightGBM predict',
        unit='rows',
        unit_scale=True,
        mininterval=0.5,
    ) as pbar:
        out = model.predict(X, num_threads=1)
        pbar.update(n_rows)
    return out


def train_and_score(features_df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Trains the GBDT on silver labels and predicts probabilities for all pairs.

    Returns
    -------
    scored_pairs : DataFrame with match_prob column added
    model        : trained LightGBM Booster, or None if the sklearn fallback was used
    """

    # 1. Apply Hard Gating (Level B2)
    # Pass pairs where: unit data is missing on either side (delta_size is NaN)
    # OR size difference is within tolerance.
    ds = features_df['delta_size']
    valid_pairs = features_df[
        ds.isna() | (ds <= SIZE_GATE_TOLERANCE)
    ].copy()

    print(f'Level B Gating dropped {len(features_df) - len(valid_pairs):,} implausible pairs.')

    # 2. Get training data
    train_data = generate_silver_labels(valid_pairs)

    # 3. Train Model (fallback if labels are too sparse for a binary classifier)
    if len(train_data) < 20 or train_data['label'].nunique() < 2:
        print(
            'Insufficient silver labels for LightGBM; using cosine similarity '
            'as match probability (clip to [0, 1]).',
        )
        valid_pairs['match_prob'] = valid_pairs['cosine_sim'].clip(0.0, 1.0)
        return valid_pairs, None

    X_train = train_data[FEATURE_COLS]
    y_train = train_data['label']

    try:
        lgb = _try_import_lightgbm()
    except (OSError, ImportError) as err:
        warnings.warn(
            f'LightGBM unavailable ({err!r}); using HistGradientBoostingClassifier. '
            'Install OpenMP (e.g. brew install libomp on macOS) for LightGBM.',
            stacklevel=2,
        )
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=100, random_state=42, class_weight='balanced',
        )
        X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
        with tqdm(total=1, desc='HistGradientBoosting train', bar_format='{desc}: [{elapsed}]') as pbar:
            clf.fit(X_train_np, y_train.to_numpy())
            pbar.update(1)
        X_score_np = np.ascontiguousarray(valid_pairs[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False))
        n_score = X_score_np.shape[0]
        out = np.empty(n_score, dtype=np.float64)
        for start in range(0, n_score, _PREDICT_CHUNK_ROWS):
            end = min(start + _PREDICT_CHUNK_ROWS, n_score)
            out[start:end] = clf.predict_proba(X_score_np[start:end])[:, 1]
        valid_pairs['match_prob'] = out
        return valid_pairs, None

    train_dataset = lgb.Dataset(X_train, label=y_train)
    with tqdm(
        total=LGBM_NUM_BOOST_ROUNDS,
        desc='LightGBM train',
        unit='round',
        mininterval=0.2,
    ) as pbar:
        def _on_iteration(_env) -> None:
            pbar.update(1)

        model = lgb.train(
            LGBM_PARAMS,
            train_dataset,
            num_boost_round=LGBM_NUM_BOOST_ROUNDS,
            callbacks=[_on_iteration],
        )

    X_score_np = np.ascontiguousarray(valid_pairs[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False))
    valid_pairs['match_prob'] = _predict_probs_chunked_lgb(model, X_score_np, len(valid_pairs))
    return valid_pairs, model
