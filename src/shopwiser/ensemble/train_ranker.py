"""Train an ensemble-specific LightGBM ranker using r4 clusters as silver labels.

Positives: product pairs that belong to the same ensemble_cluster_id in r4.csv
           (our highest-precision validated baseline, 96.2% UB on 5,854 × 4-way).
Negatives: cross-cluster pairs with high cosine similarity (hard negatives only;
           random cross-SM pairs are too easy and would teach the model trivial
           separability).

The trained model is persisted to data/outputs/ensemble/ranker_model.pkl so that
`recomplete_ml.py` and `rescore_rb.py` can load it and replace their hand-tuned
`cosine * 0.55 + fuzz * 0.45` linear score with `model.predict(features)`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from shopwiser.rule_matcher.data_prep import load_prepared_dataframe
from shopwiser.ml_matcher.config import EMBEDDING_MODEL, LGBM_NUM_BOOST_ROUNDS, LGBM_PARAMS
from shopwiser.ml_matcher.features import build_pairwise_features
from shopwiser.ml_matcher.ranker import FEATURE_COLS
from shopwiser.ml_matcher.retrieval import create_embedding_text
from shopwiser.paths import DATA_OUTPUTS

R4_CSV = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_r4.csv'
MODEL_OUT = DATA_OUTPUTS / 'ensemble' / 'ranker_model.pkl'

# Hard-negative mining: per anchor, consider top-K nearest cross-SM neighbours
# outside the anchor's cluster.
HARD_NEG_TOP_K = 20
HARD_NEG_PER_POSITIVE = 3        # cap 3 hard negatives per positive pair
HARD_NEG_MIN_COS = 0.40          # only mine negatives with real overlap
RNG_SEED = 42


def _build_embeddings(allp: pd.DataFrame) -> np.ndarray:
    print(f'Loading {EMBEDDING_MODEL}...')
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = allp.apply(create_embedding_text, axis=1).tolist()
    print(f'Encoding {len(texts):,} products...')
    embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=256)
    return np.asarray(embs, dtype=np.float32)


def _build_positives(clusters: pd.DataFrame) -> list[tuple[int, int, float]]:
    """All unordered cross-SM pairs within each ensemble_cluster_id."""
    pos: list[tuple[int, int, float]] = []
    for _cid, g in clusters.groupby('ensemble_cluster_id'):
        idxs = g['product_idx'].astype(int).tolist()
        sms = g['supermarket'].tolist()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if sms[i] == sms[j]:
                    continue
                a, b = sorted((idxs[i], idxs[j]))
                pos.append((a, b, 1.0))
    return pos


def _build_hard_negatives(
    clusters: pd.DataFrame,
    allp: pd.DataFrame,
    embeddings: np.ndarray,
    n_positives: int,
) -> list[tuple[int, int, float]]:
    """Mine cross-cluster high-cosine pairs as silver negatives.

    For every cluster member, retrieve its top-K nearest neighbours in each *other*
    supermarket using per-SM FAISS indices.  Keep only neighbours that belong to a
    different ensemble_cluster_id (or no cluster at all) — these are the false-
    positive candidates the model must learn to reject.
    """
    rng = np.random.default_rng(RNG_SEED)

    member_to_cid: dict[int, int] = {
        int(r['product_idx']): int(r['ensemble_cluster_id'])
        for _, r in clusters.iterrows()
    }
    cluster_members = set(member_to_cid)

    # Per-SM FAISS indices over ALL products (we want neighbours anywhere)
    print('Building per-SM FAISS indices for hard-negative mining...')
    sm_indices: dict[str, faiss.IndexFlatIP] = {}
    sm_idmap: dict[str, np.ndarray] = {}
    for sm, g in allp.groupby('supermarket'):
        idxs = g['product_idx'].astype(int).values
        embs = embeddings[idxs]
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        sm_indices[sm] = idx
        sm_idmap[sm] = idxs

    sm_of = dict(zip(allp['product_idx'].astype(int), allp['supermarket']))

    negs: list[tuple[int, int, float]] = []
    seen_neg: set[tuple[int, int]] = set()
    per_anchor_cap = max(1, HARD_NEG_PER_POSITIVE)
    target = int(n_positives * 1.2)  # slight oversample, later pruned

    print(f'Mining hard negatives (target ≈ {target:,})...')
    cluster_members_list = list(cluster_members)
    rng.shuffle(cluster_members_list)

    for anchor in cluster_members_list:
        if len(negs) >= target:
            break
        anchor_cid = member_to_cid[anchor]
        anchor_sm = sm_of.get(anchor)
        if anchor_sm is None:
            continue
        q = embeddings[anchor : anchor + 1]

        added_this_anchor = 0
        for sm, idx in sm_indices.items():
            if sm == anchor_sm or added_this_anchor >= per_anchor_cap:
                continue
            k = min(HARD_NEG_TOP_K, idx.ntotal)
            scores, local_ids = idx.search(q, k)
            for cos, lid in zip(scores[0], local_ids[0]):
                if lid < 0 or cos < HARD_NEG_MIN_COS:
                    break
                tgt = int(sm_idmap[sm][lid])
                # Same cluster → not a negative
                if member_to_cid.get(tgt, -1) == anchor_cid:
                    continue
                pair = tuple(sorted((anchor, tgt)))
                if pair in seen_neg:
                    continue
                seen_neg.add(pair)
                negs.append((pair[0], pair[1], 0.0))
                added_this_anchor += 1
                if added_this_anchor >= per_anchor_cap:
                    break

    print(f'  mined {len(negs):,} hard negatives.')
    return negs


def main() -> None:
    print(f'Loading r4: {R4_CSV}')
    clusters = pd.read_csv(R4_CSV)
    allp = load_prepared_dataframe(sample=False)
    # load_prepared_dataframe already sets product_idx = index

    # Size prior on clusters
    sizes = clusters.groupby('ensemble_cluster_id').size()
    print(f'  clusters: {len(sizes):,}  (4w={int((sizes==4).sum())}, '
          f'3w={int((sizes==3).sum())}, 2w={int((sizes==2).sum())})')

    embeddings = _build_embeddings(allp)

    positives = _build_positives(clusters)
    print(f'\nPositives (cross-SM pairs inside clusters): {len(positives):,}')

    negatives = _build_hard_negatives(clusters, allp, embeddings, len(positives))

    # Build pairs_df with cosine_sim from embeddings (what FAISS would return)
    pairs_rows: list[dict] = []
    for a, b, lbl in positives + negatives:
        cos = float(np.dot(embeddings[a], embeddings[b]))
        pairs_rows.append({'id_a': a, 'id_b': b, 'score': cos, 'label': lbl})
    pairs_df = pd.DataFrame(pairs_rows)
    pairs_df = pairs_df.drop_duplicates(subset=['id_a', 'id_b'])

    # build_pairwise_features expects the normalized_products frame to have
    # product_idx as a column — we already added it above.
    print(f'\nBuilding features for {len(pairs_df):,} labelled pairs...')
    feat_df = build_pairwise_features(allp, pairs_df[['id_a', 'id_b', 'score']])
    feat_df = feat_df.merge(pairs_df[['id_a', 'id_b', 'label']], on=['id_a', 'id_b'])

    n_pos = int((feat_df['label'] == 1).sum())
    n_neg = int((feat_df['label'] == 0).sum())
    print(f'  positives: {n_pos:,}  negatives: {n_neg:,}')

    # Impute NaN features — LightGBM handles them but some (same_brand/delta_size)
    # can be NaN when unit data or brand is missing; LGBM's default handling is fine.
    X = feat_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = feat_df['label'].to_numpy(dtype=np.int32)

    # Simple 80/20 split for diagnostic — not used for early stopping here
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, va = perm[:split], perm[split:]

    import lightgbm as lgb
    dtrain = lgb.Dataset(X[tr], label=y[tr])
    dval = lgb.Dataset(X[va], label=y[va], reference=dtrain)

    print('\nTraining LightGBM ranker...')
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=LGBM_NUM_BOOST_ROUNDS,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(period=25)],
    )

    # Save FIRST — sklearn import on Python 3.14 / macOS can segfault via joblib.
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open('wb') as fh:
        pickle.dump({'model': model, 'feature_cols': FEATURE_COLS}, fh)
    print(f'\nWrote: {MODEL_OUT}')

    # Diagnostic — threshold sweep without sklearn (avoid joblib segfault)
    p_val = model.predict(X[va])
    # AUC without sklearn: sort-and-integrate ROC
    order = np.argsort(-p_val)
    y_s = y[va][order]
    n_pos = int(y_s.sum())
    n_neg = len(y_s) - n_pos
    if n_pos and n_neg:
        tp_cum = np.cumsum(y_s)
        fp_cum = np.cumsum(1 - y_s)
        tpr = tp_cum / n_pos
        fpr = fp_cum / n_neg
        auc = float(np.trapz(tpr, fpr))
        print(f'\nValidation AUC: {auc:.4f}')
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (p_val >= thr).astype(int)
        tp = int(((pred == 1) & (y[va] == 1)).sum())
        fp = int(((pred == 1) & (y[va] == 0)).sum())
        fn = int(((pred == 0) & (y[va] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f'  thr={thr:.2f}  P={prec:.3f}  R={rec:.3f}  TP={tp}  FP={fp}  FN={fn}')


if __name__ == '__main__':
    main()
