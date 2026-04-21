"""Phase 3 ensemble: union ML-matching and rule-based clusters.

Both pipelines emit one cluster_id per product_idx.  Each cluster with
size ≥ 2 induces a complete subgraph of intra-cluster pairs; we score those
pairs with the cluster-level ``avg_pairwise_score`` and combine into a
single edge list, deduping per (min,max) pair by keeping the max score.

Final clusters are built by Kruskal-style score-ordered union-find under
two hard constraints:

    1.  at most one product per supermarket
    2.  at most 4 members per cluster

Rule-based and ML edges flow through the same sort, so wherever they
disagree, the higher-confidence source wins the tie — and either pipeline
can contribute unique 4th members that the other missed.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from shopwiser.paths import DATA_OUTPUTS, cluster_outputs_path, ml_matching_outputs_path

MAX_CLUSTER_SIZE = 4


def _pairs_from_clusters(
    df: pd.DataFrame,
    source: str,
    *,
    score_col: str = 'avg_pairwise_score',
) -> pd.DataFrame:
    """Emit one row per (id_a, id_b) intra-cluster edge with that cluster's score."""
    rows: list[tuple[int, int, float, str]] = []
    grouped = df.dropna(subset=['cluster_id']).groupby('cluster_id')
    for _cid, g in grouped:
        if len(g) < 2:
            continue
        score = float(g[score_col].iloc[0]) if score_col in g.columns and pd.notna(g[score_col].iloc[0]) else 0.0
        idxs = g['product_idx'].astype(int).tolist()
        for u, v in combinations(idxs, 2):
            a, b = (u, v) if u < v else (v, u)
            rows.append((a, b, score, source))
    return pd.DataFrame(rows, columns=['id_a', 'id_b', 'score', 'source'])


def _kruskal_one_per_sm(
    edges: pd.DataFrame,
    sm_map: dict[int, str],
    *,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
) -> dict[int, int]:
    """Greedy score-ordered union-find.  Returns product_idx → root."""
    parent: dict[int, int] = {}
    cluster_sms: dict[int, set[str]] = {}
    cluster_size: dict[int, int] = {}

    def ensure(x: int) -> None:
        if x not in parent:
            parent[x] = x
            cluster_sms[x] = {sm_map[x]}
            cluster_size[x] = 1

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for row in edges.itertuples(index=False):
        u, v = int(row.id_a), int(row.id_b)
        ensure(u)
        ensure(v)
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        merged_sms = cluster_sms[ru] | cluster_sms[rv]
        merged_n = cluster_size[ru] + cluster_size[rv]
        if len(merged_sms) != merged_n:  # SM collision
            continue
        if merged_n > max_cluster_size:
            continue
        parent[rv] = ru
        cluster_sms[ru] = merged_sms
        cluster_size[ru] = merged_n

    return {node: find(node) for node in parent}


def _assign_cluster_ids(root_map: dict[int, int]) -> dict[int, int]:
    """Map root → sequential cluster_id starting at 0."""
    roots = sorted(set(root_map.values()))
    return {root: cid for cid, root in enumerate(roots)}


def run_ensemble(
    *,
    sample: bool = False,
    rule_csv: Path | None = None,
    ml_csv: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    rule_csv = rule_csv or (cluster_outputs_path(sample=sample) / 'clusters.csv')
    ml_csv = ml_csv or (ml_matching_outputs_path(sample=sample) / 'ml_clusters.csv')
    out_dir = out_dir or (DATA_OUTPUTS / ('ensemble_sample' if sample else 'ensemble'))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading rule-based clusters: {rule_csv}')
    rule_df = pd.read_csv(rule_csv)
    print(f'Loading ML clusters:         {ml_csv}')
    ml_df = pd.read_csv(ml_csv)

    print('Extracting edges from each source...')
    rule_edges = _pairs_from_clusters(rule_df, source='rule')
    ml_edges = _pairs_from_clusters(ml_df, source='ml')
    print(f'  rule edges: {len(rule_edges):,}')
    print(f'  ml   edges: {len(ml_edges):,}')

    edges = pd.concat([rule_edges, ml_edges], ignore_index=True)
    edges = edges.sort_values('score', ascending=False).drop_duplicates(
        subset=['id_a', 'id_b'], keep='first',
    )
    print(f'  union edges (deduped): {len(edges):,}')

    sm_map = dict(zip(ml_df['product_idx'].astype(int), ml_df['supermarket']))
    for pid, sm in zip(rule_df['product_idx'].astype(int), rule_df['supermarket']):
        sm_map.setdefault(pid, sm)

    print('Running Kruskal union-find with one-per-SM constraint...')
    root_map = _kruskal_one_per_sm(edges, sm_map, max_cluster_size=MAX_CLUSTER_SIZE)
    cid_map = _assign_cluster_ids(root_map)

    # Build output cluster assignments.  Only products that ended up in a
    # multi-product cluster are kept; singletons drop out.
    members: dict[int, int] = {pid: cid_map[root] for pid, root in root_map.items()}

    # Cluster sizes
    size_counts: dict[int, int] = {}
    for cid in members.values():
        size_counts[cid] = size_counts.get(cid, 0) + 1
    multi_members = {pid: cid for pid, cid in members.items() if size_counts[cid] >= 2}

    # Join to full product metadata from the ML output (it carries full cols).
    base = ml_df.copy()
    base['ensemble_cluster_id'] = base['product_idx'].astype(int).map(multi_members)
    out = base.dropna(subset=['ensemble_cluster_id']).copy()
    out['ensemble_cluster_id'] = out['ensemble_cluster_id'].astype(int)
    out['cluster_size'] = out['ensemble_cluster_id'].map(
        out.groupby('ensemble_cluster_id').size(),
    )
    out = out.sort_values(['ensemble_cluster_id', 'supermarket']).reset_index(drop=True)

    clusters_path = out_dir / 'ensemble_clusters.csv'
    out.to_csv(clusters_path, index=False)
    print(f'Saved {clusters_path}  ({len(out):,} rows)')

    # Summary per cluster
    summary_rows: list[dict] = []
    for cid, g in out.groupby('ensemble_cluster_id'):
        summary_rows.append({
            'ensemble_cluster_id': int(cid),
            'cluster_size': len(g),
            'n_supermarkets': g['supermarket'].nunique(),
            'supermarkets_present': '|'.join(sorted(g['supermarket'].unique())),
            'category': g['category'].mode().iat[0] if g['category'].notna().any() else '',
            'known_brand': g['known_brand_clean'].mode().iat[0] if g['known_brand_clean'].notna().any() else '',
            'product_type': g['product_type'].mode().iat[0] if g['product_type'].notna().any() else '',
            'core_product_name_consensus': g['core_product_name'].mode().iat[0] if g['core_product_name'].notna().any() else '',
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / 'ensemble_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved {summary_path}  ({len(summary_df):,} rows)')

    # Stats
    size_dist = summary_df['cluster_size'].value_counts().sort_index()
    stats = {
        'total_clusters': len(summary_df),
        'size_dist': {int(k): int(v) for k, v in size_dist.items()},
        '4_way': int((summary_df['cluster_size'] == 4).sum()),
        '3_way': int((summary_df['cluster_size'] == 3).sum()),
        '2_way': int((summary_df['cluster_size'] == 2).sum()),
        'products_matched': len(out),
    }
    print()
    print('=' * 60)
    print('Ensemble cluster distribution:')
    for sz, n in stats['size_dist'].items():
        print(f'  {sz}-way: {n:,}')
    print(f'  Total multi-product clusters: {stats["total_clusters"]:,}')
    print(f'  Total products matched:       {stats["products_matched"]:,}')
    print('=' * 60)

    return stats


if __name__ == '__main__':
    run_ensemble()
