"""Sample 80 ensemble clusters (stratified by size) and run the LLM audit."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))

from shopwiser.audit.v5 import run_audit  # noqa: E402
from shopwiser.paths import DATA_OUTPUTS  # noqa: E402

SEED = 20260418
N_TOTAL = 80


def main() -> None:
    ens_dir = DATA_OUTPUTS / 'ensemble'
    clusters_path = ens_dir / 'ensemble_clusters.csv'
    sample_path = ens_dir / 'audit_sample_80.csv'
    results_path = ens_dir / 'audit_results_80.csv'

    df = pd.read_csv(clusters_path)
    if 'cluster_id' in df.columns:
        df = df.drop(columns=['cluster_id'])
    df = df.rename(columns={'ensemble_cluster_id': 'cluster_id'})
    # columns the audit expects
    df['match_type'] = df.get('product_type', 'ensemble')
    df['avg_pairwise_score'] = 0.0

    unique = df[['cluster_id', 'cluster_size']].drop_duplicates('cluster_id')
    # Stratified: proportional across 2-way/3-way/4-way, weighted toward 4-way.
    buckets = {2: 15, 3: 25, 4: 40}
    picks: list[int] = []
    rng_state = 0

    def _sample(size: int, n: int) -> list[int]:
        pool = unique.loc[unique['cluster_size'] == size, 'cluster_id']
        if len(pool) <= n:
            return list(pool)
        return list(pool.sample(n=n, random_state=SEED + size))

    for sz, n in buckets.items():
        picks.extend(_sample(sz, n))

    sample_df = df[df['cluster_id'].isin(picks)].copy().sort_values(['cluster_id', 'supermarket'])
    ens_dir.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    print(f'Wrote audit sample: {sample_path}  ({len(sample_df)} rows, {len(picks)} clusters)')

    run_audit(input_csv=sample_path, output_csv=results_path)


if __name__ == '__main__':
    main()
