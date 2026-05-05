"""ML-enhanced cluster completion via sentence-transformer embeddings + FAISS.

Mirrors the cluster-guided retrieval in ml_matching.main, but applied to
already-built ensemble clusters:

  1. Embed all products (cluster members + singletons) with all-mpnet-base-v2.
  2. Per missing-SM: build a FAISS index of that SM's singletons.
  3. For each incomplete cluster, query FAISS with the cluster centroid.
  4. Hard gates: same category, no hard-conflict, no brand conflict, size ≤ tol.
  5. Score = 0.55 × cosine_sim + 0.45 × fuzz_set/100.
  6. Accept top candidate if score ≥ ACCEPT_THRESHOLD.
  7. Greedy claim resolution: larger cluster wins ties, each product used once.

Processes both 3-way → 4-way AND 2-way → 3-way (then a second pass catches
any new 3-ways → 4-way).

Writes ensemble_clusters_ml.csv alongside a log.
"""

from __future__ import annotations

# ml_scorer pulls lightgbm in FIRST to avoid faiss/lightgbm OpenMP segfault on macOS.
from shopwiser.ensemble.ml_scorer import load_model, score_cluster_candidate

import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from shopwiser.rule_matcher.data_prep import load_prepared_dataframe
from shopwiser.ml_matcher.features import check_hard_conflict
from shopwiser.ml_matcher.retrieval import create_embedding_text
from shopwiser.ml_matcher.config import EMBEDDING_MODEL
from shopwiser.paths import DATA_OUTPUTS

ALL_SMS = ('ASDA', 'Morrisons', 'Sains', 'Tesco')
SIZE_TOL = 0.15
COSINE_MIN = 0.52          # minimum cosine similarity to consider
FUZZ_MIN = 55
ACCEPT_THRESHOLD = 0.65    # combined score gate (cosine*0.55 + fuzz/100*0.45)
COSINE_W = 0.55
FUZZ_W = 0.45
MODEL_ACCEPT = 0.55        # match_prob gate when the trained LGBM ranker is used

IN_CSV = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_ml_fuzz.csv'
OUT_CSV = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_ml2.csv'
LOG_CSV = DATA_OUTPUTS / 'ensemble' / 'recomplete_ml_log.csv'


def _brand_token(b: object) -> str:
    if not isinstance(b, str):
        return ''
    b = b.strip().lower()
    return b.split()[0] if b else ''


def _norm_brand_set(series: pd.Series) -> set[str]:
    toks = {_brand_token(b) for b in series.dropna()}
    toks.discard('')
    return toks


def _smart_size_delta(uv_a: float, pq_a: float, uv_b: float, pq_b: float) -> float:
    if pd.isna(uv_a) or pd.isna(uv_b):
        return 0.0

    def _rd(x: float, y: float) -> float:
        hi = max(abs(x), abs(y))
        return 0.0 if hi < 1e-6 else abs(x - y) / hi

    pu_a = float(uv_a) / float(pq_a) if pd.notna(pq_a) and pq_a else float(uv_a)
    pu_b = float(uv_b) / float(pq_b) if pd.notna(pq_b) and pq_b else float(uv_b)
    return min(_rd(float(uv_a), float(uv_b)), _rd(pu_a, pu_b),
               _rd(float(uv_a), pu_b), _rd(pu_a, float(uv_b)))


def _passes_hard_gates(cluster_df: pd.DataFrame, cand: pd.Series) -> bool:
    """Category, brand-conflict, hard-conflict, size gates."""
    # Category
    cl_cats = set(cluster_df['category'].dropna().astype(str).unique())
    cc = cand.get('category')
    if cl_cats and (not isinstance(cc, str) or cc not in cl_cats):
        return False

    # Brand conflict: reject only if candidate has a DIFFERENT known brand
    cl_brands = _norm_brand_set(cluster_df['known_brand_clean'])
    cb = _brand_token(cand.get('known_brand_clean'))
    cn_toks = set(str(cand.get('normalized_name', '')).split())
    if cl_brands and cb and cb not in cl_brands and not (cl_brands & cn_toks):
        return False

    # Hard-conflict
    cn = str(cand.get('normalized_name') or '')
    for nm in cluster_df['normalized_name'].fillna('').astype(str):
        if check_hard_conflict(nm, cn) == 1:
            return False

    # Size: ALL members within tolerance (stricter — matches merge_2way's rule).
    # Previously "at least one member within" which let trained ranker sneak in
    # size-outliers via any single matching member.
    for _, m in cluster_df.iterrows():
        if pd.notna(m.get('unit_value')) and pd.notna(cand.get('unit_value')):
            if _smart_size_delta(
                m.get('unit_value'), m.get('pack_quantity'),
                cand.get('unit_value'), cand.get('pack_quantity'),
            ) > SIZE_TOL:
                return False

    return True


def _combined_score(cosine_sim: float, cluster_names: list[str], cand_name: str) -> float:
    fs = max(fuzz.token_set_ratio(cand_name, n) for n in cluster_names) / 100.0
    return COSINE_W * cosine_sim + FUZZ_W * fs


def embed_all(allp: pd.DataFrame) -> np.ndarray:
    print(f'  Loading embedding model {EMBEDDING_MODEL}...')
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    texts = allp.apply(create_embedding_text, axis=1).tolist()
    print(f'  Encoding {len(texts):,} products...')
    embs = st_model.encode(texts, show_progress_bar=True, normalize_embeddings=True,
                           batch_size=256)
    return np.asarray(embs, dtype=np.float32)


def run_recomplete(
    clusters: pd.DataFrame,
    allp: pd.DataFrame,
    embeddings: np.ndarray,
    claimed_globally: set[int],
    model_bundle: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """One pass of ML-enhanced recomplete. Returns (additions_df, log)."""
    clustered = set(clusters['product_idx'].astype(int))
    free = allp[~allp['product_idx'].isin(clustered) &
                ~allp['product_idx'].isin(claimed_globally)]

    # Build per-SM FAISS indices of free singletons
    sm_free: dict[str, pd.DataFrame] = {}
    sm_index: dict[str, faiss.IndexFlatIP] = {}
    sm_idmap: dict[str, np.ndarray] = {}

    print(f'  Building FAISS indices for {len(free):,} singletons...')
    for sm, g in free.groupby('supermarket'):
        idxs = g['product_idx'].astype(int).values
        embs = embeddings[idxs]
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        sm_free[sm] = g
        sm_index[sm] = idx
        sm_idmap[sm] = idxs

    proposals: list[tuple[float, int, str, int, int]] = []

    incomplete = [(cid, g) for cid, g in clusters.groupby('ensemble_cluster_id')
                  if set(g['supermarket']) < set(ALL_SMS)]
    print(f'  Scoring {len(incomplete):,} incomplete clusters...')

    for cid, g in incomplete:
        present = set(g['supermarket'])
        missing = [sm for sm in ALL_SMS if sm not in present]

        member_idxs = g['product_idx'].astype(int).values
        centroid = embeddings[member_idxs].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-6:
            centroid = centroid / norm
        centroid_f32 = centroid.reshape(1, -1).astype(np.float32)

        cluster_names = g['normalized_name'].fillna('').astype(str).tolist()

        for sm in missing:
            if sm not in sm_index:
                continue
            index = sm_index[sm]
            idmap = sm_idmap[sm]
            pool = sm_free[sm]

            k = min(75, index.ntotal)
            scores_arr, local_ids = index.search(centroid_f32, k)
            scores_arr = scores_arr[0]
            local_ids = local_ids[0]

            best: tuple[float, int] | None = None
            accept_gate = MODEL_ACCEPT if model_bundle else ACCEPT_THRESHOLD
            for rank, (cos, lid) in enumerate(zip(scores_arr, local_ids)):
                if lid < 0 or cos < COSINE_MIN:
                    break
                pidx = int(idmap[lid])
                cand = pool.loc[pool['product_idx'] == pidx].iloc[0]
                if not _passes_hard_gates(g, cand):
                    continue
                if model_bundle is not None:
                    score = score_cluster_candidate(model_bundle, g, cand, float(cos), allp)
                else:
                    cn = str(cand.get('normalized_name') or '')
                    score = _combined_score(float(cos), cluster_names, cn)
                if score >= accept_gate and (best is None or score > best[0]):
                    best = (score, pidx)

            if best:
                proposals.append((best[0], int(cid), sm, best[1], len(g)))

    # Greedy resolution
    proposals.sort(key=lambda p: (-p[0], -p[4]))
    claimed_idx: set[int] = set()
    claimed_cs: set[tuple[int, str]] = set()
    accepted: list[dict] = []
    for score, cid, sm, pidx, csize in proposals:
        if pidx in claimed_idx or (cid, sm) in claimed_cs or pidx in claimed_globally:
            continue
        claimed_idx.add(pidx)
        claimed_cs.add((cid, sm))
        accepted.append({
            'ensemble_cluster_id': cid,
            'product_idx': pidx,
            'supermarket': sm,
            'score': round(score, 4),
        })

    claimed_globally.update(claimed_idx)
    return pd.DataFrame(accepted), accepted


def _apply_additions(clusters: pd.DataFrame, adds: pd.DataFrame, allp_idxed: pd.DataFrame) -> pd.DataFrame:
    if adds.empty:
        return clusters
    extra = allp_idxed.loc[adds['product_idx'].values].copy()
    extra['ensemble_cluster_id'] = adds['ensemble_cluster_id'].values
    for c in clusters.columns:
        if c not in extra.columns and c != 'ensemble_cluster_id':
            extra[c] = pd.NA
    extra = extra[clusters.columns]
    merged = pd.concat([clusters, extra], ignore_index=True)
    merged['cluster_size'] = merged['ensemble_cluster_id'].map(
        merged.groupby('ensemble_cluster_id').size(),
    )
    return merged


def _summary(df: pd.DataFrame, label: str) -> None:
    sizes = df.groupby('ensemble_cluster_id').size()
    print(f'  [{label}] 4w={int((sizes==4).sum()):,}  '
          f'3w={int((sizes==3).sum()):,}  2w={int((sizes==2).sum()):,}  '
          f'total={int(sizes.shape[0]):,}')


def main() -> None:
    print(f'Loading: {IN_CSV}')
    clusters = pd.read_csv(IN_CSV)
    allp = load_prepared_dataframe(sample=False)
    allp_idxed = allp.set_index('product_idx', drop=False)

    _summary(clusters, 'input')

    print('\nComputing embeddings for all products...')
    embeddings = embed_all(allp)

    model_bundle = load_model()
    if model_bundle is not None:
        print(f'Loaded trained LGBM ranker (accept gate = {MODEL_ACCEPT})')
    else:
        print('No trained model found — using linear cosine*fuzz fallback')

    claimed_globally: set[int] = set()
    log_all: list[dict] = []

    # Two passes: first adds 3→4 and 2→3, second catches new 3→4
    for it in range(1, 3):
        print(f'\n=== Pass {it} ===')
        adds_df, log = run_recomplete(clusters, allp, embeddings, claimed_globally, model_bundle)
        print(f'  accepted: {len(adds_df):,}')
        if adds_df.empty:
            print('  converged.')
            break
        log_all.extend([{**r, 'pass': it} for r in log])
        clusters = _apply_additions(clusters, adds_df, allp_idxed)
        _summary(clusters, f'after pass {it}')

    clusters = clusters.sort_values(['ensemble_cluster_id', 'supermarket']).reset_index(drop=True)
    clusters.to_csv(OUT_CSV, index=False)
    pd.DataFrame(log_all).to_csv(LOG_CSV, index=False)
    print(f'\nWrote: {OUT_CSV}')
    print(f'Log:   {LOG_CSV}')


if __name__ == '__main__':
    main()
