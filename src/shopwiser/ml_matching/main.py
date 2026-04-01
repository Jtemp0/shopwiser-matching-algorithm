"""Main orchestrator for ML Matching Pipeline."""

import networkx as nx
import pandas as pd

from shopwiser.clustering.data_prep import load_prepared_dataframe

from .config import ACCEPT_THRESHOLD, MARGIN_THRESHOLD, OUTPUT_DIR, configure_paths
from .features import build_pairwise_features
from .ranker import train_and_score
from .retrieval import retrieve_candidates


def build_final_clusters(df: pd.DataFrame, scored_pairs: pd.DataFrame) -> None:
    """Section 2.6: Argmax selection and final output construction."""
    print('\nApplying Argmax Selection...')

    # We need to map scored_pairs back to directed (Anchor -> Target) to do Argmax
    # scored_pairs currently has (id_a < id_b). We duplicate to make it bidrectional.
    edges_forward = scored_pairs[['id_a', 'id_b', 'match_prob']].rename(
        columns={'id_a': 'anchor', 'id_b': 'target'},
    )
    edges_backward = scored_pairs[['id_b', 'id_a', 'match_prob']].rename(
        columns={'id_b': 'anchor', 'id_a': 'target'},
    )
    directed_edges = pd.concat([edges_forward, edges_backward])

    # Add supermarket info to know which retailer the target belongs to
    sm_map = df['supermarket'].to_dict()
    directed_edges['target_sm'] = directed_edges['target'].map(sm_map)

    accepted_edges = []

    # For each item i and retailer sb:
    grouped = directed_edges.groupby(['anchor', 'target_sm'])
    for (anchor_id, _target_sm), group in grouped:
        # Sort by match probability descending
        sorted_group = group.sort_values('match_prob', ascending=False)

        top_score = sorted_group.iloc[0]['match_prob']
        top_target = sorted_group.iloc[0]['target']

        # 1. Must beat acceptance threshold
        if top_score < ACCEPT_THRESHOLD:
            continue

        # 2. Must beat runner-up by margin
        if len(sorted_group) > 1:
            runner_up_score = sorted_group.iloc[1]['match_prob']
            if (top_score - runner_up_score) < MARGIN_THRESHOLD:
                continue  # Ambiguous, abstain

        accepted_edges.append((int(anchor_id), int(top_target), float(top_score)))

    print(f'Accepted {len(accepted_edges):,} highly confident directed edges.')

    # Build Graph for clustering
    G = nx.Graph()
    G.add_nodes_from(df.index)
    for u, v, weight in accepted_edges:
        G.add_edge(u, v, weight=weight)

    clusters = list(nx.connected_components(G))

    # Assign cluster IDs
    cluster_map = {}
    for cluster_id, node_set in enumerate(sorted(clusters, key=len, reverse=True)):
        for node in node_set:
            cluster_map[node] = cluster_id

    df['cluster_id'] = df.index.map(cluster_map)

    # Diagnostics
    cluster_sizes = df['cluster_id'].value_counts()
    n_multi = (cluster_sizes >= 2).sum()
    n_4way = (df.groupby('cluster_id')['supermarket'].nunique() == 4).sum()

    print('\n' + '=' * 50)
    print('FINAL ML CLUSTERING RESULTS')
    print('=' * 50)
    print(f'Total Clusters: {len(clusters):,}')
    print(f'Multi-product Clusters (>=2): {n_multi:,}')
    print(f'Perfect 4-way Clusters: {n_4way:,}')
    print('=' * 50)

    # Save output
    output_path = OUTPUT_DIR / 'ml_clusters.csv'
    df.to_csv(output_path, index=False)
    print(f'\nSaved final clusters to {output_path}')


def run_ml_matching(*, sample: bool = False) -> None:
    """Load prepared products, run FAISS retrieval → features → LightGBM → graph clusters."""
    configure_paths(sample=sample)

    print('Loading normalized data...')
    df = load_prepared_dataframe(sample=sample)

    # Clean apostrophes (Crucial fix from Phase 1 assessment)
    df['normalized_name'] = df['normalized_name'].str.replace("'", '', regex=False)

    # Ensure contiguous integer index aligned with retrieval indices
    df = df.reset_index(drop=True)
    df['product_idx'] = df.index

    # 1. Level A: Retrieval
    candidate_pairs = retrieve_candidates(df)

    # 2. Build Features
    features_df = build_pairwise_features(df, candidate_pairs)

    # 3. Level B & C: Gating, Training, and Scoring
    scored_pairs = train_and_score(features_df)

    # 4. Argmax and Final Output
    build_final_clusters(df, scored_pairs)


def main(*, sample: bool = False) -> None:
    """CLI-compatible entry (same signature as ``shopwiser.clustering.main.main``)."""
    run_ml_matching(sample=sample)


if __name__ == '__main__':
    import argparse

    _p = argparse.ArgumentParser(description='Run ML matching (FAISS + LightGBM) on normalized products CSV.')
    _p.add_argument(
        '--sample',
        action='store_true',
        help='Use normalized_products_sample.csv and write under data/outputs/ml_clusters_sample/',
    )
    _args = _p.parse_args()
    main(sample=_args.sample)
