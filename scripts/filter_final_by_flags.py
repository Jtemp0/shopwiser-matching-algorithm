"""Post-filter ensemble_clusters_final.csv: drop clusters with precision-killing flags.

Three flags reliably indicate wrong matches:
  - size_mismatch:   any pairwise smart_size_delta > 0.15 inside the cluster
  - brand_mismatch:  ≥2 distinct known_brand_clean values (space-normalised)
  - hard_conflict:   any pair of names sharing a one-sided flavor/variant token
                     (e.g. "jasmine" green tea mixed with plain green tea,
                     "Dark Roast" mixed with "Gold" of the same brand)

branded_vs_own_no_token is KEPT (many are legitimate retailer own-brand
equivalents priced against a national brand).

Writes back to ensemble_clusters_final.csv (in-place); pre-filter data
preserved in ensemble_clusters_merged_ml.csv upstream.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from shopwiser.ml_matching.features import check_hard_conflict

ENS = Path('data/outputs/ensemble/ensemble_clusters_final.csv')
SIZE_TOL = 0.15

# Tiers that are meaningfully different from each other.  NONE/null means the
# tier was not detected — it does not count as a tier and will not trigger a
# mismatch.  Two own-brand products with different KNOWN tiers in the same
# cluster are different product lines (e.g. "Savers" vs "Extra Special") and
# should not be presented as price-comparable equivalents.
_KNOWN_TIERS = frozenset({'value', 'standard', 'premium', 'dietary'})


def _size_mismatch(g: pd.DataFrame) -> float:
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
    brands = [re.sub(r'\s+', '', b.strip().lower()) for b in brands if b.strip()]
    return len(set(brands)) >= 2


def _tier_mismatch(g: pd.DataFrame) -> bool:
    """True when own-brand/unbranded products in the cluster carry 2+ distinct
    known tiers (e.g. 'value' and 'premium').  Products with null/unknown tier
    are ignored — they cannot cause a mismatch on their own."""
    ob = g[g['product_type'].isin(['own_brand', 'unbranded'])]
    tiers = {
        str(t).lower()
        for t in ob['tier_type'].dropna()
        if str(t).lower() in _KNOWN_TIERS
    }
    return len(tiers) >= 2


def _has_hard_conflict(g: pd.DataFrame) -> bool:
    names = g['normalized_name'].fillna('').astype(str).tolist()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if check_hard_conflict(a, b) == 1:
                return True
    return False


def main() -> None:
    df = pd.read_csv(ENS)
    sizes_before = df.groupby('ensemble_cluster_id').size()
    print(f'Loaded: {len(df):,} rows, {sizes_before.shape[0]:,} clusters')
    print(f'  4w={int((sizes_before==4).sum()):,}  '
          f'3w={int((sizes_before==3).sum()):,}  '
          f'2w={int((sizes_before==2).sum()):,}')

    # Compute flags per cluster
    sz_deltas = df.groupby('ensemble_cluster_id').apply(_size_mismatch)
    size_bad = set(sz_deltas[sz_deltas > SIZE_TOL].index)

    bm = df.groupby('ensemble_cluster_id').apply(_brand_mismatch)
    brand_bad = set(bm[bm].index)

    hc_ids = set()
    tier_bad = set()
    for cid, g in df.groupby('ensemble_cluster_id'):
        if _has_hard_conflict(g):
            hc_ids.add(cid)
        if _tier_mismatch(g):
            tier_bad.add(cid)

    drop = size_bad | brand_bad | hc_ids | tier_bad
    print(f'\nDropping {len(drop):,} flagged clusters:')
    print(f'  size_mismatch:  {len(size_bad):,}')
    print(f'  brand_mismatch: {len(brand_bad):,}')
    print(f'  hard_conflict:  {len(hc_ids):,}')
    print(f'  tier_mismatch:  {len(tier_bad):,}')
    print(f'  union:          {len(drop):,}')

    # Per-size breakdown of dropped
    dropped_sizes = sizes_before.loc[sizes_before.index.isin(drop)]
    print(f'  dropped by size: '
          f'4w={int((dropped_sizes==4).sum()):,}  '
          f'3w={int((dropped_sizes==3).sum()):,}  '
          f'2w={int((dropped_sizes==2).sum()):,}')

    df_clean = df[~df['ensemble_cluster_id'].isin(drop)].reset_index(drop=True)
    sizes_after = df_clean.groupby('ensemble_cluster_id').size()
    df_clean['cluster_size'] = df_clean['ensemble_cluster_id'].map(sizes_after)

    print(f'\nAfter filter:')
    print(f'  {len(df_clean):,} rows, {sizes_after.shape[0]:,} clusters')
    print(f'  4w={int((sizes_after==4).sum()):,}  '
          f'3w={int((sizes_after==3).sum()):,}  '
          f'2w={int((sizes_after==2).sum()):,}')

    df_clean.to_csv(ENS, index=False)
    print(f'\nWrote: {ENS}')


if __name__ == '__main__':
    main()
