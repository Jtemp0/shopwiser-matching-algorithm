"""Level C: Bootstrapping Silver Labels and Training the GBDT Ranker."""

import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import LGBM_NUM_BOOST_ROUNDS, LGBM_PARAMS, SIZE_GATE_TOLERANCE

# Rows per predict() chunk so tqdm can show scoring progress on ~1M+ rows.
_PREDICT_CHUNK_ROWS = 65_536


def _try_import_lightgbm():
    import lightgbm as lgb

    return lgb


def generate_silver_labels(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Creates highly-confident positive and negative labels from the candidates."""

    print('Generating Silver Labels for training...')
    df = feat_df.copy()

    # 1. Silver Positives (strict rules; includes own-brand↔own-brand so the model
    #    does not learn to reject every own-brand pair)
    brand_or_own_both = (df['same_brand'] == 1) | (
        (df['is_own_brand_a'] == 1) & (df['is_own_brand_b'] == 1)
    )
    pos_mask = (
        (df['delta_size'] >= 0) & (df['delta_size'] <= 0.02)
        & (df['fuzz_sort'] >= 85)
        & (df['same_category'] == 1)
        & (df['hard_conflict'] == 0)
        & brand_or_own_both
    )

    # 2. Silver Negatives (Contradictions)
    neg_mask = (
        ((df['same_brand'] == 1) & (df['delta_size'] > 0.15))
        | ((df['fuzz_sort'] > 85) & (df['same_brand'] == 0) & (df['is_own_brand_a'] == 0))
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


def _predict_probs_chunked_sklearn(clf, X: np.ndarray, n_rows: int) -> np.ndarray:
    """Score in chunks so the bar reflects progress on large matrices."""
    out = np.empty(n_rows, dtype=np.float64)
    with tqdm(
        total=n_rows,
        desc='HistGradientBoosting predict',
        unit='rows',
        unit_scale=True,
        mininterval=0.5,
    ) as pbar:
        for start in range(0, n_rows, _PREDICT_CHUNK_ROWS):
            end = min(start + _PREDICT_CHUNK_ROWS, n_rows)
            out[start:end] = clf.predict_proba(X[start:end])[:, 1]
            pbar.update(end - start)
    return out


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



def _fit_and_predict_probs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_score: pd.DataFrame,
) -> np.ndarray:
    """Prefer LightGBM; fall back to sklearn if OpenMP/lib missing (common on macOS)."""
    X_score_np = np.ascontiguousarray(X_score.to_numpy(dtype=np.float32, copy=False))
    n_score = X_score_np.shape[0]

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
            max_iter=100,
            random_state=42,
            class_weight='balanced',
        )
        X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
        y_train_np = y_train.to_numpy()
        with tqdm(
            total=1,
            desc='HistGradientBoosting train',
            bar_format='{desc}: [{elapsed}]',
        ) as pbar:
            clf.fit(X_train_np, y_train_np)
            pbar.update(1)
        return _predict_probs_chunked_sklearn(clf, X_score_np, n_score)

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

    return _predict_probs_chunked_lgb(model, X_score_np, n_score)


def train_and_score(features_df: pd.DataFrame) -> pd.DataFrame:
    """Trains the GBDT on silver labels and predicts probabilities for all pairs."""

    # 1. Apply Hard Gating (Level B2)
    valid_pairs = features_df[
        (features_df['delta_size'] == -1.0)
        | (features_df['delta_size'] <= SIZE_GATE_TOLERANCE)
    ].copy()

    print(f'Level B Gating dropped {len(features_df) - len(valid_pairs):,} implausible pairs.')

    feature_cols = [
        'cosine_sim', 'delta_size', 'same_unit_type', 'same_brand',
        'same_category', 'is_own_brand_a', 'is_own_brand_b',
        'fuzz_sort', 'fuzz_set', 'hard_conflict',
    ]

    # 2. Get training data
    train_data = generate_silver_labels(valid_pairs)

    # 3. Train Model (fallback if labels are too sparse for a binary classifier)
    if len(train_data) < 20 or train_data['label'].nunique() < 2:
        print(
            'Insufficient silver labels for LightGBM; using cosine similarity '
            'as match probability (clip to [0, 1]).',
        )
        valid_pairs['match_prob'] = valid_pairs['cosine_sim'].clip(0.0, 1.0)
        return valid_pairs

    X_train = train_data[feature_cols]
    y_train = train_data['label']

    valid_pairs['match_prob'] = _fit_and_predict_probs(X_train, y_train, valid_pairs[feature_cols])
    return valid_pairs
