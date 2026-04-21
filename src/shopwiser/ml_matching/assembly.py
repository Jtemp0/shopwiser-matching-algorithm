"""One-per-supermarket cluster assembly.

After all edge-addition passes, connected components of the match graph may
contain multiple products from the same supermarket.  This happens when weak
transitive edges bridge disparate products (e.g. size-similar Cadbury bars
chained into one 548-product blob).

`enforce_one_per_sm` rebuilds each violating component using a score-ordered
Kruskal-style union-find with two hard constraints:

    1.  The merged cluster must have **at most one product per supermarket**.
    2.  The merged cluster must have **at most ``max_cluster_size`` members**
        (==4 by default — same as the number of retailers).

Edges are ordered by (match_prob, fuzz_sort) descending so the strongest
pairs win when a blob decomposes into multiple valid sub-clusters.
Components already satisfying the invariants are left untouched.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def _build_pair_score_lookup(
    scored_pairs: pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Return (min_id, max_id) → (match_prob, fuzz_sort, delta_size)."""
    merged = scored_pairs[['id_a', 'id_b', 'match_prob']].merge(
        features_df[['id_a', 'id_b', 'fuzz_sort', 'delta_size']],
        on=['id_a', 'id_b'], how='left',
    )
    lookup: dict[tuple[int, int], tuple[float, float, float]] = {}
    for row in merged.itertuples(index=False):
        u, v = int(row.id_a), int(row.id_b)
        key = (min(u, v), max(u, v))
        mp = float(row.match_prob)
        fz = float(row.fuzz_sort) if pd.notna(row.fuzz_sort) else 0.0
        ds = float(row.delta_size) if pd.notna(row.delta_size) else -1.0
        prev = lookup.get(key)
        if prev is None or mp > prev[0]:
            lookup[key] = (mp, fz, ds)
    return lookup


def _edge_weight(
    u: int,
    v: int,
    pair_scores: dict[tuple[int, int], tuple[float, float, float]],
    guided_meta: dict[tuple[int, int], tuple],
) -> tuple[float, float]:
    """Primary = match_prob; secondary = fuzz_sort.  Guided-only edges map
    cosine_sim and fuzz into comparable scales so they slot into the same
    sort without dominating true model-scored pairs."""
    key = (min(u, v), max(u, v))
    ps = pair_scores.get(key)
    if ps is not None:
        return (ps[0], ps[1])
    gm = guided_meta.get(key)
    if gm is not None:
        sim, fuzz_val = gm[0], gm[1]
        # Map cosine sim (typically 0.55–0.90) into a match_prob-scale proxy.
        # Cap at 0.5 so a strong scored-pair (match_prob ≥ 0.5) always outranks
        # a guided-only edge: we prefer model evidence when both exist.
        return (min(0.5, sim * 0.5 + fuzz_val / 200.0), fuzz_val)
    return (0.0, 0.0)


def enforce_one_per_sm(
    G: nx.Graph,
    df: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    features_df: pd.DataFrame,
    guided_edge_meta: dict[tuple[int, int], tuple],
    *,
    max_cluster_size: int = 4,
) -> dict:
    """Rebuild G in-place so every component has at most one product per SM.

    Returns a stats dict.
    """
    sm_map = df['supermarket'].to_dict()
    pair_scores = _build_pair_score_lookup(scored_pairs, features_df)

    comps = list(nx.connected_components(G))
    comps_clean = 0
    comps_rebuilt = 0
    nodes_touched = 0
    edges_dropped = 0
    edges_kept = 0

    for comp in comps:
        members = list(comp)
        sms = [sm_map[n] for n in members]

        clean_sms = len(set(sms)) == len(sms)
        clean_size = len(members) <= max_cluster_size
        if clean_sms and clean_size:
            comps_clean += 1
            continue

        comps_rebuilt += 1
        nodes_touched += len(members)

        # Collect intra-component edges with weights.
        sub_edges: list[tuple[tuple[float, float], int, int]] = []
        for u, v in G.subgraph(members).edges():
            w = _edge_weight(u, v, pair_scores, guided_edge_meta)
            sub_edges.append((w, u, v))
        sub_edges.sort(key=lambda x: x[0], reverse=True)

        # Drop every existing edge in this component; re-add the survivors.
        G.remove_edges_from(list(G.subgraph(members).edges()))

        parent: dict[int, int] = {n: n for n in members}
        cluster_sms: dict[int, set[str]] = {n: {sm_map[n]} for n in members}
        cluster_size: dict[int, int] = {n: 1 for n in members}

        def find(x: int) -> int:
            # Iterative path compression (safe against deep chains).
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for _w, u, v in sub_edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                # Edge within the same sub-cluster already; drop it.
                # (Prevents cycles — our final clusters only need a spanning
                # tree of edges within each sub-cluster.)
                edges_dropped += 1
                continue
            merged_sms = cluster_sms[ru] | cluster_sms[rv]
            merged_n = cluster_size[ru] + cluster_size[rv]
            if len(merged_sms) != merged_n:
                edges_dropped += 1
                continue  # duplicate SM — skip merge
            if merged_n > max_cluster_size:
                edges_dropped += 1
                continue  # over-size — skip merge
            # Accept the merge.
            parent[rv] = ru
            cluster_sms[ru] = merged_sms
            cluster_size[ru] = merged_n
            G.add_edge(u, v)
            edges_kept += 1

    stats = {
        'components_clean': comps_clean,
        'components_rebuilt': comps_rebuilt,
        'nodes_touched': nodes_touched,
        'edges_kept': edges_kept,
        'edges_dropped': edges_dropped,
    }
    return stats
