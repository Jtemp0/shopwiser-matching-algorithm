"""Code-based audit of ensemble R3 clusters — no LLM calls.

Flags clusters that violate structural rules or semantic heuristics:

  1.  one-per-SM invariant  (fatal if violated)
  2.  hard-conflict tokens  (from features.py — flavor/meat/variant clashes)
  3.  size mismatches       (delta between unit_values in same cluster)
  4.  brand mismatches      (≥2 distinct known brands in one cluster)
  5.  branded↔own_brand mix with no shared brand-token

For each rule, prints incidence, a sample of offenders, and an estimated
precision bound (upper: clusters without any hard flag).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shopwiser.ml_matching.features import (
    FLAVOR_NAMED_TOKENS,
    ONE_SIDED_CONFLICT_TOKENS,
    check_hard_conflict,
)

ENS = Path('data/outputs/ensemble/ensemble_clusters_merged_ml.csv')
SIZE_TOL = 0.15


def _names_of(g: pd.DataFrame) -> list[str]:
    return g['normalized_name'].fillna('').astype(str).tolist()


def _has_hard_conflict(g: pd.DataFrame) -> bool:
    names = _names_of(g)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if check_hard_conflict(a, b) == 1:
                return True
    return False


def _size_mismatch(g: pd.DataFrame) -> float:
    """Max pairwise smart_size_delta: min of raw, per-unit, and cross interpretations."""
    import numpy as np

    rows = g[['unit_value', 'pack_quantity']].values.tolist()

    def _rd(x, y):
        hi = max(abs(x), abs(y))
        return 0.0 if hi < 1e-6 else abs(x - y) / hi

    def _delta(uv_a, pq_a, uv_b, pq_b):
        if pd.isna(uv_a) or pd.isna(uv_b):
            return 0.0
        pu_a = float(uv_a) / float(pq_a) if pd.notna(pq_a) and pq_a else float(uv_a)
        pu_b = float(uv_b) / float(pq_b) if pd.notna(pq_b) and pq_b else float(uv_b)
        return min(
            _rd(float(uv_a), float(uv_b)),
            _rd(pu_a, pu_b),
            _rd(float(uv_a), pu_b),
            _rd(pu_a, float(uv_b)),
        )

    worst = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = _delta(rows[i][0], rows[i][1], rows[j][0], rows[j][1])
            if d > worst:
                worst = d
    return worst


def _brand_mismatch(g: pd.DataFrame) -> bool:
    brands = g['known_brand_clean'].dropna().astype(str)
    brands = [b.strip().lower() for b in brands if b.strip()]
    return len(set(brands)) >= 2


def _branded_vs_own_no_brand_token(g: pd.DataFrame) -> bool:
    types = g['product_type'].dropna().unique()
    if len(set(types) & {'branded', 'own_brand'}) < 2:
        return False
    known = g.loc[g['product_type'] == 'branded', 'known_brand_clean'].dropna()
    if known.empty:
        return False
    primary = str(known.iloc[0]).strip().split()[0].lower()
    if len(primary) < 3:
        return False
    # Every row in the cluster should carry this primary token in normalized_name.
    names = _names_of(g)
    for nm in names:
        toks = set(nm.split())
        if primary not in toks:
            return True
    return False


def main() -> None:
    df = pd.read_csv(ENS)
    print(f'Loaded: {len(df):,} rows, {df["ensemble_cluster_id"].nunique():,} clusters')
    sizes = df.groupby('ensemble_cluster_id').size()
    for sz in (2, 3, 4):
        print(f'  {sz}-way: {int((sizes == sz).sum()):,}')
    print()

    # 1. One-per-SM check
    sm_dup = df.groupby('ensemble_cluster_id').apply(
        lambda g: len(g) != g['supermarket'].nunique(),
    )
    print(f'[1] one-per-SM violations: {int(sm_dup.sum()):,}')

    # 2. Hard-conflict
    hc_rows = []
    for cid, g in df.groupby('ensemble_cluster_id'):
        if _has_hard_conflict(g):
            hc_rows.append(cid)
    print(f'[2] hard-conflict (FLAVOR/ONE_SIDED): {len(hc_rows):,}')

    # 3. Size mismatch
    sz_deltas = df.groupby('ensemble_cluster_id').apply(_size_mismatch)
    size_bad = sz_deltas[sz_deltas > SIZE_TOL].index.tolist()
    print(f'[3] size mismatch (>{int(SIZE_TOL*100)}%):     {len(size_bad):,}')

    # 4. Brand mismatch
    bm = df.groupby('ensemble_cluster_id').apply(_brand_mismatch)
    bm_ids = bm[bm].index.tolist()
    print(f'[4] ≥2 distinct known brands:        {len(bm_ids):,}')

    # 5. Branded↔own_brand without shared brand-token
    bob = df.groupby('ensemble_cluster_id').apply(_branded_vs_own_no_brand_token)
    bob_ids = bob[bob].index.tolist()
    print(f'[5] branded↔own_brand w/o brand tok: {len(bob_ids):,}')

    # Union of flagged clusters (any suspicion)
    flagged = set(hc_rows) | set(size_bad) | set(bm_ids) | set(bob_ids)
    flagged |= set(sm_dup[sm_dup].index.tolist())
    total = df['ensemble_cluster_id'].nunique()
    clean = total - len(flagged)
    print()
    print('=' * 60)
    print(f'Clusters with NO flag (upper-bound precision): {clean:,} / {total:,} = {clean/total*100:.1f}%')
    print(f'Clusters with ≥1 flag (likely-FP upper bound): {len(flagged):,} = {len(flagged)/total*100:.1f}%')
    print('=' * 60)

    # By cluster size
    flagged_by_size = (
        df[df['ensemble_cluster_id'].isin(flagged)]
        .drop_duplicates('ensemble_cluster_id')
        .groupby(df.loc[df['ensemble_cluster_id'].isin(flagged)].drop_duplicates('ensemble_cluster_id')['ensemble_cluster_id'].map(sizes))
    )
    print()
    print('Flagged clusters by size:')
    flagged_ids_df = df[df['ensemble_cluster_id'].isin(flagged)].drop_duplicates('ensemble_cluster_id')
    flagged_ids_df = flagged_ids_df.assign(_size=flagged_ids_df['ensemble_cluster_id'].map(sizes))
    for sz in (2, 3, 4):
        n_flag = int((flagged_ids_df['_size'] == sz).sum())
        n_tot = int((sizes == sz).sum())
        pct = n_flag / n_tot * 100 if n_tot else 0.0
        print(f'  {sz}-way: {n_flag:,} / {n_tot:,} ({pct:.1f}%) flagged')

    # Sample offenders
    def _sample_offenders(name: str, ids: list[int], n: int = 4) -> None:
        if not ids:
            return
        print(f'\n--- Sample of "{name}" (showing {min(n, len(ids))} of {len(ids)}) ---')
        chosen = ids[: n * 3]  # take front slice — deterministic
        import random

        rng = random.Random(20260419)
        sample = rng.sample(chosen if len(chosen) >= n else ids, min(n, len(ids)))
        for cid in sample:
            g = df[df['ensemble_cluster_id'] == cid]
            print(f'\n  cluster {int(cid)} (size={len(g)}):')
            for _, r in g.iterrows():
                brand = r.get('known_brand_clean') or ''
                bs = f' [{brand}]' if isinstance(brand, str) and brand else ''
                uv = r.get('unit_value')
                ut = r.get('unit_type')
                wt = f' — {uv:g}{ut}' if pd.notna(uv) and isinstance(ut, str) else ''
                pt = r.get('product_type')
                pt_s = pt if isinstance(pt, str) else ''
                print(f'    {r["supermarket"]:10s} {pt_s:10s} {str(r["names"])[:80]}{bs}{wt}')

    _sample_offenders('hard_conflict', hc_rows, 5)
    _sample_offenders('size_mismatch', size_bad, 5)
    _sample_offenders('brand_mismatch', bm_ids, 5)
    _sample_offenders('branded_vs_own_no_token', bob_ids, 5)


if __name__ == '__main__':
    main()
