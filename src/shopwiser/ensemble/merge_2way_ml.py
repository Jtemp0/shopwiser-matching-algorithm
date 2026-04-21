"""ML-enhanced 2-way × 2-way cluster merger.

For each pair of 2-way clusters whose supermarkets are disjoint (so their union
would be a valid 4-way), score the match via cosine similarity between their
embedding centroids, plus hard gates on category / brand / size / hard-conflict.

Accepts the best-scoring compatible pair greedily; each cluster used at most once.

Also handles 3-way + singleton merges (3-way × 1-way) where the 1-way cluster
is a product already in a lone 2-way that becomes 3-way after one pass.

Writes ensemble_clusters_merged_ml.csv.
"""

from __future__ import annotations

# ml_scorer pulls lightgbm in FIRST to avoid faiss/lightgbm OpenMP segfault on macOS.
from shopwiser.ensemble.ml_scorer import load_model, score_cluster_pair

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from shopwiser.clustering.data_prep import load_prepared_dataframe
from shopwiser.ml_matching.features import check_hard_conflict
from shopwiser.ml_matching.retrieval import create_embedding_text
from shopwiser.ml_matching.config import EMBEDDING_MODEL
from shopwiser.paths import DATA_OUTPUTS

ALL_SMS = frozenset(('ASDA', 'Morrisons', 'Sains', 'Tesco'))
SIZE_TOL = 0.15
CENTROID_MIN = 0.60      # minimum centroid-centroid cosine for a merge proposal
MERGE_ACCEPT = 0.62      # combined centroid_cos*0.6 + min_member_fuzz/100*0.4
MODEL_MERGE_ACCEPT = 0.55  # match_prob gate when the trained LGBM ranker is used

IN_CSV  = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_ml2.csv'
OUT_CSV = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_merged_ml.csv'

from rapidfuzz import fuzz as rfuzz


def _brand_token(b: object) -> str:
    if not isinstance(b, str): return ''
    b = b.strip().lower()
    return b.split()[0] if b else ''

def _norm_brand_set(s: pd.Series) -> set[str]:
    t = {_brand_token(b) for b in s.dropna()}; t.discard(''); return t

def _smart_delta(uv_a, pq_a, uv_b, pq_b) -> float:
    if pd.isna(uv_a) or pd.isna(uv_b): return 0.0
    def rd(x, y):
        h = max(abs(x), abs(y)); return 0.0 if h < 1e-6 else abs(x-y)/h
    pu_a = float(uv_a)/float(pq_a) if pd.notna(pq_a) and pq_a else float(uv_a)
    pu_b = float(uv_b)/float(pq_b) if pd.notna(pq_b) and pq_b else float(uv_b)
    return min(rd(float(uv_a),float(uv_b)), rd(pu_a,pu_b),
               rd(float(uv_a),pu_b), rd(pu_a,float(uv_b)))


def _clusters_compat(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Hard-gate check for cluster merge."""
    if set(a['supermarket']) & set(b['supermarket']): return False
    if len(a) + len(b) > 4: return False
    # Category
    ca = set(a['category'].dropna().astype(str)); cb = set(b['category'].dropna().astype(str))
    if ca and cb and not (ca & cb): return False
    # Brand conflict
    ba = _norm_brand_set(a['known_brand_clean']); bb = _norm_brand_set(b['known_brand_clean'])
    if ba and bb and not (ba & bb): return False
    # Hard-conflict between any cross-pair
    na = a['normalized_name'].fillna('').astype(str).tolist()
    nb = b['normalized_name'].fillna('').astype(str).tolist()
    for x in na:
        for y in nb:
            if check_hard_conflict(x, y) == 1: return False
    # Size: all cross-pairs within tolerance
    for _, ra in a.iterrows():
        for _, rb in b.iterrows():
            if pd.notna(ra.get('unit_value')) and pd.notna(rb.get('unit_value')):
                if _smart_delta(ra['unit_value'], ra.get('pack_quantity'),
                                rb['unit_value'], rb.get('pack_quantity')) > SIZE_TOL:
                    return False
    return True


def _min_cross_fuzz(a: pd.DataFrame, b: pd.DataFrame) -> float:
    na = a['normalized_name'].fillna('').astype(str).tolist()
    nb = b['normalized_name'].fillna('').astype(str).tolist()
    return min(rfuzz.token_set_ratio(x, y) / 100.0 for x in na for y in nb)


def main() -> None:
    print(f'Loading: {IN_CSV}')
    df = pd.read_csv(IN_CSV)
    allp = load_prepared_dataframe(sample=False)

    model_bundle = load_model()
    if model_bundle is not None:
        print(f'Loaded trained LGBM ranker (merge gate = {MODEL_MERGE_ACCEPT})')
    else:
        print('No trained model found — using linear cosine*fuzz fallback')

    sizes = df.groupby('ensemble_cluster_id').size()
    incomplete = sizes[sizes < 4].index
    inc_df = df[df['ensemble_cluster_id'].isin(incomplete)]

    print(f'  incomplete clusters: {len(incomplete):,}')
    sizes_inc = inc_df.groupby('ensemble_cluster_id').size()
    print(f'  2-way: {int((sizes_inc==2).sum()):,}  3-way: {int((sizes_inc==3).sum()):,}')

    print(f'\nLoading embedding model {EMBEDDING_MODEL}...')
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    texts = allp.apply(create_embedding_text, axis=1).tolist()
    print(f'Encoding {len(texts):,} products...')
    embeddings = st_model.encode(texts, show_progress_bar=True,
                                  normalize_embeddings=True, batch_size=256)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Compute centroids for each incomplete cluster
    cid_list = []
    centroid_list = []
    for cid, g in inc_df.groupby('ensemble_cluster_id'):
        idxs = g['product_idx'].astype(int).values
        c = embeddings[idxs].mean(axis=0)
        n = np.linalg.norm(c)
        c = c / n if n > 1e-6 else c
        cid_list.append(int(cid))
        centroid_list.append(c)

    centroids = np.asarray(centroid_list, dtype=np.float32)  # (N_inc, D)
    cid_arr = np.array(cid_list)

    # Build FAISS index of all incomplete cluster centroids
    idx_all = faiss.IndexFlatIP(centroids.shape[1])
    idx_all.add(centroids)

    # Group clusters by their SM set for fast disjoint-SM lookup
    cid_to_sms: dict[int, frozenset] = {
        cid: frozenset(g['supermarket'])
        for cid, g in inc_df.groupby('ensemble_cluster_id')
    }
    cid_to_df: dict[int, pd.DataFrame] = dict(list(inc_df.groupby('ensemble_cluster_id')))

    print(f'\nSearching for compatible merge pairs...')
    proposals: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    k = min(50, len(cid_list))

    for i, (cid_a, cent_a) in enumerate(zip(cid_list, centroid_list)):
        sms_a = cid_to_sms[cid_a]
        if len(sms_a) + 2 > 4 and len(sms_a) != 2:
            continue  # only want 2+2 or 2+1 or 3+1
        q = cent_a.reshape(1, -1).astype(np.float32)
        scores_arr, local_ids = idx_all.search(q, k)
        for cos, lid in zip(scores_arr[0], local_ids[0]):
            if lid < 0 or cos < CENTROID_MIN or lid == i:
                continue
            cid_b = int(cid_arr[lid])
            pair = (min(cid_a, cid_b), max(cid_a, cid_b))
            if pair in seen:
                continue
            seen.add(pair)
            sms_b = cid_to_sms[cid_b]
            if sms_a & sms_b:
                continue  # overlapping SMs
            if len(sms_a) + len(sms_b) > 4:
                continue
            ga, gb = cid_to_df[cid_a], cid_to_df[cid_b]
            if not _clusters_compat(ga, gb):
                continue
            if model_bundle is not None:
                score = score_cluster_pair(model_bundle, ga, gb, float(cos), allp)
                gate = MODEL_MERGE_ACCEPT
            else:
                mf = _min_cross_fuzz(ga, gb)
                score = 0.6 * float(cos) + 0.4 * mf
                gate = MERGE_ACCEPT
            if score >= gate:
                proposals.append((score, cid_a, cid_b))

    proposals.sort(key=lambda p: -p[0])
    claimed: set[int] = set()
    merges: list[tuple[int, int, float]] = []
    for score, ca, cb in proposals:
        if ca in claimed or cb in claimed:
            continue
        claimed.add(ca); claimed.add(cb)
        merges.append((ca, cb, score))

    print(f'  accepted merges: {len(merges):,}')

    if merges:
        remap = {cb: ca for ca, cb, _ in merges}
        df = df.copy()
        df['ensemble_cluster_id'] = df['ensemble_cluster_id'].replace(remap)
        df['cluster_size'] = df['ensemble_cluster_id'].map(df.groupby('ensemble_cluster_id').size())
        df = df.sort_values(['ensemble_cluster_id', 'supermarket']).reset_index(drop=True)

    sizes2 = df.groupby('ensemble_cluster_id').size()
    print(f'\nResult: 4w={int((sizes2==4).sum()):,}  3w={int((sizes2==3).sum()):,}  '
          f'2w={int((sizes2==2).sum()):,}  total={int(sizes2.shape[0]):,}')
    df.to_csv(OUT_CSV, index=False)
    print(f'Wrote: {OUT_CSV}')


if __name__ == '__main__':
    main()
