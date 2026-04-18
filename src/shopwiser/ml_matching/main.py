"""Main orchestrator for ML Matching Pipeline."""

import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from shopwiser.clustering.data_prep import load_prepared_dataframe

from .config import (
    ACCEPT_SIZE_GATE,
    ACCEPT_THRESHOLD,
    BLOB_SPLIT_THRESHOLD,
    CLUSTER_GUIDED_MAX_DELTA,
    CLUSTER_GUIDED_MIN_FUZZ,
    CLUSTER_GUIDED_MIN_SIM,
    CLUSTER_GUIDED_TOP_K,
    COMPLETION_THRESHOLD,
    MAX_CLUSTER_SIZE,
    OUTPUT_DIR,
    REVERSE_THRESHOLD,
    configure_paths,
)
from .features import build_pairwise_features, check_hard_conflict
from .ranker import FEATURE_COLS, train_and_score
from .retrieval import retrieve_candidates


def _parse_uv_pq(unit_value, pack_quantity) -> tuple[float, float] | None:
    """Parse unit_value and pack_quantity; returns None when unit_value is missing."""
    try:
        uv = float(unit_value)
        pq_raw = pack_quantity
        if isinstance(pq_raw, float) and np.isnan(pq_raw):
            pq = 1.0
        else:
            pq = max(float(pq_raw), 1.0)
        return uv, pq
    except (TypeError, ValueError):
        return None


def _best_delta_size_scalar(
    uv_a: float, pq_a: float, uv_b: float, pq_b: float
) -> float:
    """Scalar version of _best_delta_size: minimum delta across 3 size interpretations."""
    def _rd(x: float, y: float) -> float:
        hi = max(abs(x), abs(y))
        return abs(x - y) / hi if hi > 1e-5 else 0.0

    d1 = _rd(uv_a / pq_a, uv_b / pq_b)   # per-unit vs per-unit
    d2 = _rd(uv_a, uv_b * pq_b)           # A total vs B total-from-pack
    d3 = _rd(uv_a * pq_a, uv_b)           # A total-from-pack vs B total
    return min(d1, d2, d3)


def _delta_size_multi(
    parsed_a: tuple[float, float] | None,
    parsed_b: tuple[float, float] | None,
) -> float:
    """Multi-interpretation relative size delta; returns -1.0 when either side has no unit."""
    if parsed_a is None or parsed_b is None:
        return -1.0
    uv_a, pq_a = parsed_a
    uv_b, pq_b = parsed_b
    return _best_delta_size_scalar(uv_a, pq_a, uv_b, pq_b)


def _run_guided_retrieval(
    G: 'nx.Graph',
    df: pd.DataFrame,
    embeddings: np.ndarray,
    indices: dict,
    id_maps: dict,
    all_sms: set,
    seen: set,
    guided_edge_meta: dict,
    *,
    label: str = 'Cluster-guided retrieval',
) -> int:
    """Query missing-SM FAISS indices for every incomplete cluster in G.

    Returns the number of bridging edges added.  Updates G, seen and
    guided_edge_meta in-place.
    """
    print(f'Running {label}...')

    sm_map = df['supermarket'].to_dict()
    unit_value_map = df['unit_value'].to_dict()
    pack_qty_map = df['pack_quantity'].to_dict()
    norm_name_map = df['normalized_name'].fillna('').to_dict()
    brand_map = df['known_brand_clean'].fillna('').to_dict()
    # is_truncated: True for Morrisons products with source-truncated names.
    # token_set_ratio is used instead of token_sort_ratio for these candidates
    # because the shorter truncated name is typically a prefix of the full name.
    is_truncated_map = df['is_truncated'].fillna(False).astype(bool).to_dict() if 'is_truncated' in df.columns else {}

    # Build component snapshot from current G
    components_now: dict[int, int] = {}
    for cid_g, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            components_now[node] = cid_g

    comp_sms_g: dict[int, set[str]] = {}
    for node, cid_g in components_now.items():
        comp_sms_g.setdefault(cid_g, set()).add(sm_map[node])

    comp_members_g: dict[int, list[int]] = {}
    for node, cid_g in components_now.items():
        comp_members_g.setdefault(cid_g, []).append(node)

    guided_added = 0

    for cid_g, sms in list(comp_sms_g.items()):
        if len(sms) < 2 or len(sms) == len(all_sms):
            continue

        # 3-way clusters: use CLUSTER_GUIDED_MIN_FUZZ - 3 (extra tolerance given the 3-SM prior).
        # 2-way clusters: use CLUSTER_GUIDED_MIN_FUZZ + 6 (tighter — less evidence).
        guided_min_fuzz = CLUSTER_GUIDED_MIN_FUZZ - 3 if len(sms) >= 3 else CLUSTER_GUIDED_MIN_FUZZ
        # 3-way clusters have 3-SM agreement as prior — cast a wider FAISS net
        # while keeping quality gates (fuzz, size, brand, conflict) the same.
        guided_top_k = CLUSTER_GUIDED_TOP_K + 25
        # Lower cosine-sim threshold for 3-way clusters: the 3-SM prior is strong
        # evidence; a 4th-SM product that scores 0.58–0.62 still passes fuzz/brand/
        # size checks and is more likely genuine than for a 2-way cluster.
        guided_min_sim = CLUSTER_GUIDED_MIN_SIM - 0.04 if len(sms) >= 3 else CLUSTER_GUIDED_MIN_SIM

        members = comp_members_g.get(cid_g, [])
        if not members:
            continue

        missing_sms = all_sms - sms

        member_embs = embeddings[members]
        mean_emb = member_embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm < 1e-8:
            continue
        mean_emb = (mean_emb / norm).astype(np.float32).reshape(1, -1)

        # Build a deduplicated set of query embeddings: the mean + each member's
        # own embedding.  Searching with individual member embeddings recovers
        # candidates that are closer to one specific member than to the centroid,
        # which happens when SM-specific naming conventions shift the embedding.
        query_embs: list[np.ndarray] = [mean_emb]
        for m_emb in member_embs:
            m_emb_norm = m_emb / (np.linalg.norm(m_emb) + 1e-12)
            query_embs.append(m_emb_norm.astype(np.float32).reshape(1, -1))

        member_parsed_sizes = [_parse_uv_pq(unit_value_map.get(m), pack_qty_map.get(m)) for m in members]
        member_names = [norm_name_map.get(m, '') for m in members]
        member_brands = [b for m in members if (b := brand_map.get(m, ''))]
        cluster_brand = max(set(member_brands), key=member_brands.count) if member_brands else ''
        member_is_truncated = any(is_truncated_map.get(m, False) for m in members)

        for missing_sm in list(missing_sms):
            if missing_sm not in indices:
                continue

            # Aggregate candidates from all queries; keep best score per target
            candidate_scores: dict[int, float] = {}
            for q_emb in query_embs:
                q_scores, q_local_ids = indices[missing_sm].search(q_emb, guided_top_k)
                for k in range(guided_top_k):
                    t_local = int(q_local_ids[0, k])
                    if t_local == -1:
                        continue
                    t_id = int(id_maps[missing_sm][t_local])
                    t_sim = float(q_scores[0, k])
                    if t_sim < guided_min_sim:
                        break
                    if t_id not in candidate_scores or t_sim > candidate_scores[t_id]:
                        candidate_scores[t_id] = t_sim

            best_target: int | None = None
            best_sim: float = -1.0
            best_fuzz_for_meta: float = 0.0

            for target_id, sim in sorted(candidate_scores.items(), key=lambda x: -x[1]):
                if sim < guided_min_sim:
                    break

                if components_now.get(target_id) == cid_g:
                    continue

                t_brand = brand_map.get(target_id, '')
                if cluster_brand and t_brand and cluster_brand != t_brand:
                    continue

                t_parsed = _parse_uv_pq(unit_value_map.get(target_id), pack_qty_map.get(target_id))
                if not any(
                    _delta_size_multi(m_ps, t_parsed) <= CLUSTER_GUIDED_MAX_DELTA
                    for m_ps in member_parsed_sizes
                ):
                    continue

                # Fuzz check: for truncated targets (Morrisons source-truncated names),
                # use token_set_ratio — the short truncated name is often a proper
                # subset of the full name, making set_ratio a much better measure.
                t_name = norm_name_map.get(target_id, '')
                t_truncated = is_truncated_map.get(target_id, False)
                if t_truncated or member_is_truncated:
                    best_fuzz_candidate = max(fuzz.token_set_ratio(m_name, t_name) for m_name in member_names)
                    fuzz_threshold = max(guided_min_fuzz - 5, 55)
                else:
                    best_fuzz_candidate = max(fuzz.token_sort_ratio(m_name, t_name) for m_name in member_names)
                    fuzz_threshold = guided_min_fuzz
                if best_fuzz_candidate < fuzz_threshold:
                    continue

                if any(check_hard_conflict(m_name, t_name) for m_name in member_names):
                    continue

                if sim > best_sim:
                    best_sim = sim
                    best_target = target_id
                    best_fuzz_for_meta = best_fuzz_candidate

            if best_target is None:
                continue

            anchor = members[0]
            pair = (min(anchor, best_target), max(anchor, best_target))
            if pair in seen:
                continue

            G.add_edge(anchor, best_target)
            seen.add(pair)
            guided_added += 1
            guided_edge_meta[pair] = (best_sim, best_fuzz_for_meta, guided_min_sim)

            # Update local bookkeeping
            merged_sms = sms | {missing_sm}
            t_cid = components_now.get(best_target)
            if t_cid is not None and t_cid != cid_g:
                merged_sms |= comp_sms_g.get(t_cid, set())
                for node, c in list(components_now.items()):
                    if c == t_cid:
                        components_now[node] = cid_g
                comp_sms_g.pop(t_cid, None)
                if t_cid in comp_members_g:
                    comp_members_g[cid_g] = comp_members_g.get(cid_g, []) + comp_members_g.pop(t_cid)
            components_now[best_target] = cid_g
            comp_sms_g[cid_g] = merged_sms
            sms = merged_sms

    print(f'{label} added {guided_added:,} bridging edges.')
    return guided_added


def _run_model_guided_retrieval(
    G: 'nx.Graph',
    df: pd.DataFrame,
    embeddings: np.ndarray,
    indices: dict,
    id_maps: dict,
    all_sms: set,
    seen: set,
    model,
    *,
    min_sim: float = 0.35,
    top_k: int = 250,
    model_threshold_3way: float = 0.08,
    model_threshold_2way: float = 0.20,
    label: str = 'Model-guided retrieval',
) -> int:
    """For 2-way and 3-way incomplete clusters: score FAISS candidates with the trained LGBM model.

    Unlike the hard-gate guided retrieval, the model combines all 11 features to accept
    candidates where individual gates (fuzz, size) are weak but the combined signal is
    strong.  3-way clusters use a lower threshold (stronger SM prior); 2-way clusters
    use a stricter threshold to compensate for weaker evidence.

    Returns the number of bridging edges added.
    """
    if model is None:
        return 0

    print(f'Running {label}...')

    sm_map = df['supermarket'].to_dict()
    norm_name_map = df['normalized_name'].fillna('').to_dict()
    brand_map = df['known_brand_clean'].fillna('').to_dict()
    is_truncated_map = df['is_truncated'].fillna(False).astype(bool).to_dict() if 'is_truncated' in df.columns else {}

    # Build component snapshot
    components_now: dict[int, int] = {}
    for cid_g, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            components_now[node] = cid_g

    comp_sms_g: dict[int, set[str]] = {}
    for node, cid_g in components_now.items():
        comp_sms_g.setdefault(cid_g, set()).add(sm_map[node])

    comp_members_g: dict[int, list[int]] = {}
    for node, cid_g in components_now.items():
        comp_members_g.setdefault(cid_g, []).append(node)

    # Collect all candidate pairs across all incomplete 3-way clusters
    # (anchor_id, target_id, cosine_sim, cid_g, missing_sm)
    raw_cands: list[tuple[int, int, float, int, str]] = []
    cluster_info: dict[int, tuple[list[int], str]] = {}  # cid_g → (members, cluster_brand)

    for cid_g, sms in comp_sms_g.items():
        if len(sms) < 2 or len(sms) == len(all_sms):  # 2-way and 3-way only
            continue

        members = comp_members_g.get(cid_g, [])
        if not members:
            continue

        missing_sms = all_sms - sms
        member_embs = embeddings[members]
        mean_emb = member_embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm < 1e-8:
            continue
        mean_emb = (mean_emb / norm).astype(np.float32).reshape(1, -1)

        query_embs = [mean_emb]
        for m_emb in member_embs:
            m_emb_norm = m_emb / (np.linalg.norm(m_emb) + 1e-12)
            query_embs.append(m_emb_norm.astype(np.float32).reshape(1, -1))

        member_brands = [b for m in members if (b := brand_map.get(m, ''))]
        cluster_brand = max(set(member_brands), key=member_brands.count) if member_brands else ''
        cluster_info[cid_g] = (members, cluster_brand)

        anchor = members[0]

        for missing_sm in missing_sms:
            if missing_sm not in indices:
                continue
            candidate_scores: dict[int, float] = {}
            for q_emb in query_embs:
                q_scores, q_local_ids = indices[missing_sm].search(q_emb, top_k)
                for k in range(top_k):
                    t_local = int(q_local_ids[0, k])
                    if t_local == -1:
                        continue
                    t_id = int(id_maps[missing_sm][t_local])
                    t_sim = float(q_scores[0, k])
                    if t_sim < min_sim:
                        break
                    if t_id not in candidate_scores or t_sim > candidate_scores[t_id]:
                        candidate_scores[t_id] = t_sim

            for t_id, sim in candidate_scores.items():
                if components_now.get(t_id) == cid_g:
                    continue
                t_brand = brand_map.get(t_id, '')
                if cluster_brand and t_brand and cluster_brand != t_brand:
                    continue
                member_names = [norm_name_map.get(m, '') for m in members]
                t_name = norm_name_map.get(t_id, '')
                if any(check_hard_conflict(mn, t_name) for mn in member_names):
                    continue
                raw_cands.append((anchor, t_id, sim, cid_g, missing_sm))

    if not raw_cands:
        print(f'{label} added 0 bridging edges.')
        return 0

    # Build a DataFrame of unique (anchor, target) pairs with best cosine_sim
    anchor_arr = np.array([c[0] for c in raw_cands], dtype=np.int64)
    target_arr = np.array([c[1] for c in raw_cands], dtype=np.int64)
    sim_arr = np.array([c[2] for c in raw_cands], dtype=np.float32)
    cid_arr = np.array([c[3] for c in raw_cands], dtype=np.int64)

    id_a = np.minimum(anchor_arr, target_arr)
    id_b = np.maximum(anchor_arr, target_arr)
    cands_df = pd.DataFrame({'id_a': id_a, 'id_b': id_b, 'score': sim_arr})
    # Keep best sim per unique pair
    cands_df = cands_df.sort_values('score', ascending=False).drop_duplicates(subset=['id_a', 'id_b'])

    # Compute features and score with model
    feat_df = build_pairwise_features(df, cands_df)
    X = feat_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    probs = model.predict(X, num_threads=1)
    feat_df = feat_df.assign(match_prob=probs)

    # Build lookup: (anchor, target) → match_prob
    prob_lookup: dict[tuple[int, int], float] = {}
    for row in feat_df.itertuples(index=False):
        prob_lookup[(int(row.id_a), int(row.id_b))] = float(row.match_prob)

    # Per-cluster: find best model-scored candidate above threshold.
    # 2-way clusters use a stricter threshold (weaker prior = higher bar).
    # 3-way clusters use the lower threshold (3-SM agreement is strong evidence).
    from collections import defaultdict
    cluster_candidates: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    cluster_n_sms: dict[int, int] = {cid_g: len(sms) for cid_g, sms in comp_sms_g.items()}
    for anchor, t_id, sim, cid_g, _msm in raw_cands:
        pair = (min(anchor, t_id), max(anchor, t_id))
        prob = prob_lookup.get(pair, 0.0)
        n_sms = cluster_n_sms.get(cid_g, 2)
        threshold = model_threshold_3way if n_sms >= 3 else model_threshold_2way
        if prob >= threshold:
            cluster_candidates[cid_g].append((prob, t_id, anchor))

    added = 0
    for cid_g, candidates in cluster_candidates.items():
        members, _ = cluster_info[cid_g]
        for prob, t_id, anchor in sorted(candidates, reverse=True):
            if components_now.get(t_id) == cid_g:
                continue
            pair = (min(anchor, t_id), max(anchor, t_id))
            if pair in seen:
                continue
            G.add_edge(anchor, t_id)
            seen.add(pair)
            added += 1
            # Update component tracking
            t_cid = components_now.get(t_id)
            if t_cid is not None and t_cid != cid_g:
                merged_sms = comp_sms_g.get(cid_g, set()) | comp_sms_g.get(t_cid, set())
                for node, c in list(components_now.items()):
                    if c == t_cid:
                        components_now[node] = cid_g
                comp_sms_g.pop(t_cid, None)
                if t_cid in comp_members_g:
                    comp_members_g[cid_g] = comp_members_g.get(cid_g, []) + comp_members_g.pop(t_cid)
                comp_sms_g[cid_g] = merged_sms
            components_now[t_id] = cid_g
            comp_sms_g.setdefault(cid_g, set()).add(sm_map[t_id])
            break  # one bridge per cluster per pass

    print(f'{label} added {added:,} bridging edges.')
    return added


def build_final_clusters(
    df: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    features_df: pd.DataFrame,
    embeddings: np.ndarray,
    indices: dict,
    id_maps: dict,
    model=None,
) -> None:
    """Soft mutual argmax → completion pass → cluster-guided retrieval for missing SM links."""
    print('\nApplying Soft Mutual Argmax Selection...')

    # Include delta_size so we can hard-gate on size at acceptance time.
    # scored_pairs already has delta_size (it's a filtered copy of features_df).
    _sp_cols = ['id_a', 'id_b', 'match_prob']
    if 'delta_size' in scored_pairs.columns:
        _sp_cols.append('delta_size')

    edges_f = scored_pairs[_sp_cols].rename(columns={'id_a': 'anchor', 'id_b': 'target'})
    edges_b = scored_pairs[_sp_cols].rename(columns={'id_b': 'anchor', 'id_a': 'target'})
    directed = pd.concat([edges_f, edges_b])

    sm_map = df['supermarket'].to_dict()
    directed['target_sm'] = directed['target'].map(sm_map)

    # Step 1: Argmax — each anchor's single best match per target SM, above threshold.
    best_matches = directed.sort_values(
        ['anchor', 'target_sm', 'match_prob'],
        ascending=[True, True, False],
    )
    best_matches = best_matches.groupby(['anchor', 'target_sm'], sort=False).head(1)
    best_matches = best_matches[best_matches['match_prob'] >= ACCEPT_THRESHOLD]

    # Hard size gate: pairs where both unit values are known and the relative size
    # difference exceeds ACCEPT_SIZE_GATE are rejected regardless of model score.
    # This prevents same-brand, different-size transitive chains from forming blobs
    # (e.g. 250 g → 400 g → 700 g chains that cluster all Heinz Ketchup sizes together).
    if 'delta_size' in best_matches.columns:
        size_safe = best_matches['delta_size'].isna() | (best_matches['delta_size'] <= ACCEPT_SIZE_GATE)
        n_size_rejected = (~size_safe).sum()
        if n_size_rejected > 0:
            print(f'Hard size gate rejected {n_size_rejected:,} edges at acceptance (delta_size > {ACCEPT_SIZE_GATE}).')
        best_matches = best_matches[size_safe]
    argmax_set = set(zip(best_matches['anchor'].astype(int), best_matches['target'].astype(int)))

    # Step 2: Reverse confirmation set — uses REVERSE_THRESHOLD (lower than ACCEPT)
    # to handle asymmetric scoring where one direction is slightly weaker due to
    # brand-extraction failures or name-length differences.
    high_conf_set = set(
        zip(
            directed.loc[directed['match_prob'] >= REVERSE_THRESHOLD, 'anchor'].astype(int),
            directed.loc[directed['match_prob'] >= REVERSE_THRESHOLD, 'target'].astype(int),
        )
    )

    # Step 3: Accept (u,v) if u chose v as argmax AND reverse pair is above threshold.
    seen: set[tuple[int, int]] = set()
    final_edges: list[tuple[int, int]] = []
    for u, v in argmax_set:
        pair = (min(u, v), max(u, v))
        if pair not in seen:
            if (v, u) in high_conf_set:
                final_edges.append(pair)
                seen.add(pair)

    print(f'Accepted {len(final_edges):,} soft-symmetric, high-confidence edges.')

    # ── Scored-pairs completion pass ──────────────────────────────────────────
    # For clusters already holding 2–3 SMs, bridge to a missing SM using already-
    # scored pairs (match_prob ≥ COMPLETION_THRESHOLD).  One argmax bridge per
    # missing SM keeps false-positive risk low.
    print('Running scored-pairs completion pass...')
    G = nx.Graph()
    G.add_nodes_from(df.index)
    G.add_edges_from(final_edges)

    components_before = {node: cid for cid, comp in enumerate(nx.connected_components(G)) for node in comp}
    comp_sms: dict[int, set[str]] = {}
    for node, cid in components_before.items():
        comp_sms.setdefault(cid, set()).add(sm_map[node])

    all_sms = set(df['supermarket'].unique())
    incomplete_comps = {cid for cid, sms in comp_sms.items() if 1 < len(sms) < len(all_sms)}

    completion_candidates = scored_pairs[scored_pairs['match_prob'] >= COMPLETION_THRESHOLD].copy()
    completion_candidates['comp_a'] = completion_candidates['id_a'].map(components_before)
    completion_candidates['comp_b'] = completion_candidates['id_b'].map(components_before)
    completion_candidates['sm_a'] = completion_candidates['id_a'].map(sm_map)
    completion_candidates['sm_b'] = completion_candidates['id_b'].map(sm_map)

    sp_edges_added = 0
    for row in (
        completion_candidates[
            (completion_candidates['comp_a'].isin(incomplete_comps) | completion_candidates['comp_b'].isin(incomplete_comps))
            & (completion_candidates['comp_a'] != completion_candidates['comp_b'])
        ]
        .sort_values('match_prob', ascending=False)
        .itertuples(index=False)
    ):
        u, v = int(row.id_a), int(row.id_b)
        cu, cv = components_before.get(u), components_before.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        sm_u, sm_v = sm_map[u], sm_map[v]

        u_missing = sm_v not in comp_sms.get(cu, set())
        v_missing = sm_u not in comp_sms.get(cv, set())
        if not (u_missing or v_missing):
            continue

        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        G.add_edge(u, v)
        seen.add(pair)
        sp_edges_added += 1

        merged_sms = comp_sms.get(cu, set()) | comp_sms.get(cv, set())
        keep_cid = cu if len(comp_sms.get(cu, set())) >= len(comp_sms.get(cv, set())) else cv
        drop_cid = cv if keep_cid == cu else cu
        comp_sms[keep_cid] = merged_sms
        comp_sms.pop(drop_cid, None)
        for node, cid in list(components_before.items()):
            if cid == drop_cid:
                components_before[node] = keep_cid
        if len(merged_sms) == len(all_sms):
            incomplete_comps.discard(keep_cid)

    print(f'Scored-pairs completion added {sp_edges_added:,} bridging edges.')

    # ── Cluster-guided retrieval (pass 1) ─────────────────────────────────────
    # For clusters still missing an SM after the scored-pairs pass, query that
    # SM's FAISS index directly using the cluster's mean embedding.
    guided_edge_meta: dict[tuple[int, int], tuple[float, float, float]] = {}  # pair → (sim, fuzz, min_sim)
    _run_guided_retrieval(
        G, df, embeddings, indices, id_maps, all_sms, seen, guided_edge_meta,
        label='Cluster-guided retrieval (pass 1)',
    )
    # ── End completion ────────────────────────────────────────────────────────

    # ── Blob splitting ────────────────────────────────────────────────────────
    # Clusters larger than MAX_CLUSTER_SIZE are almost certainly blobs formed by
    # transitive closure of weak (0.20-threshold) edges — e.g., all Cadbury or
    # all Magnum products ending up in one giant cluster.  We rebuild each blob
    # from scratch using only scored pairs above the stricter BLOB_SPLIT_THRESHOLD,
    # which retains genuine same-product links while severing false-positive chains.
    #
    # Guided retrieval edges (validated by FAISS + fuzz + size) are NOT in scored_pairs,
    # so they'd be lost during blob splitting.  We preserve them separately and re-add
    # any that survive the fuzz and size criteria.
    blobs = [comp for comp in nx.connected_components(G) if len(comp) > MAX_CLUSTER_SIZE]
    if blobs:
        print(f'Splitting {len(blobs):,} blob clusters (>{MAX_CLUSTER_SIZE} products)...')

        # Build a flat lookup: (min_id, max_id) → (match_prob, fuzz_sort, delta_size)
        # Using features_df which has fuzz_sort and delta_size alongside match_prob.
        scored_with_feats = scored_pairs[['id_a', 'id_b', 'match_prob']].merge(
            features_df[['id_a', 'id_b', 'fuzz_sort', 'delta_size']],
            on=['id_a', 'id_b'], how='left',
        )
        blob_lookup: dict[tuple[int, int], tuple[float, float, float]] = {}
        for row in scored_with_feats.itertuples(index=False):
            key = (min(int(row.id_a), int(row.id_b)), max(int(row.id_a), int(row.id_b)))
            if key not in blob_lookup or row.match_prob > blob_lookup[key][0]:
                fuzz_score = float(row.fuzz_sort) if pd.notna(row.fuzz_sort) else 0.0
                ds = float(row.delta_size) if pd.notna(row.delta_size) else -1.0
                blob_lookup[key] = (row.match_prob, fuzz_score, ds)

        split_removed = 0
        split_added = 0
        guided_preserved = 0
        for blob in blobs:
            blob_edges = list(G.subgraph(blob).edges())
            G.remove_edges_from(blob_edges)
            split_removed += len(blob_edges)

            blob_set = frozenset(blob)
            for (u, v), (prob, fuzz_score, ds) in blob_lookup.items():
                if u not in blob_set or v not in blob_set:
                    continue
                # Strict gate: require strong model confidence AND high text similarity
                # AND tight size agreement.  This breaks size-chain transitive blobs
                # (28g→100g→250g→400g) while preserving genuine same-product pairs.
                size_ok = (ds < 0) or (ds <= 0.15)  # ds<0 means missing unit data
                if prob >= BLOB_SPLIT_THRESHOLD and fuzz_score >= 82 and size_ok:
                    G.add_edge(u, v)
                    split_added += 1

            # Re-add valid guided retrieval edges that fell into this blob.
            # These were validated during retrieval (FAISS sim + fuzz + size + conflict)
            # but lack a model match_prob, so they don't appear in blob_lookup.
            for (u, v), meta in guided_edge_meta.items():
                if u not in blob_set or v not in blob_set:
                    continue
                sim, g_fuzz, edge_min_sim = meta[0], meta[1], meta[2] if len(meta) > 2 else CLUSTER_GUIDED_MIN_SIM
                if g_fuzz >= 72 and sim >= edge_min_sim:
                    G.add_edge(u, v)
                    guided_preserved += 1

        print(
            f'Blob splitting: removed {split_removed:,} weak edges, '
            f're-added {split_added:,} high-confidence + {guided_preserved:,} guided edges.'
        )
    # ── End blob splitting ────────────────────────────────────────────────────

    # ── Cluster-guided retrieval (passes 2–N) ────────────────────────────────
    # Blob splitting creates fresh 2-way / 3-way clusters.  Run guided retrieval
    # iteratively until convergence (no new edges added) or max 5 passes total.
    for pass_n in range(2, 6):
        added = _run_guided_retrieval(
            G, df, embeddings, indices, id_maps, all_sms, seen, guided_edge_meta,
            label=f'Cluster-guided retrieval (pass {pass_n})',
        )
        if added == 0:
            break

    # ── Model-guided retrieval (iterative, 2-way and 3-way clusters) ─────────
    # Uses the trained LGBM model to score FAISS candidates at a low cosine
    # threshold (0.35) so products whose combined feature vector is strong but
    # whose cosine sim alone is borderline are still admitted.
    # Runs iteratively: pass 1 can convert 2-way→3-way, which pass 2 converts
    # to 4-way.  3-way uses model_threshold=0.08; 2-way uses stricter 0.20.
    for mgr_pass in range(1, 5):
        mgr_added = _run_model_guided_retrieval(
            G, df, embeddings, indices, id_maps, all_sms, seen, model,
            label=f'Model-guided retrieval (pass {mgr_pass})',
        )
        if mgr_added == 0:
            break

    # ── Final scored-pairs completion pass ───────────────────────────────────
    # After blob splitting and guided retrieval, cluster structure has changed.
    # Run a second scored-pairs completion pass using the updated components so
    # clusters formed or re-shaped during the above steps can be bridged by pairs
    # that were scored by the model but not yet used (e.g. they crossed a blob
    # boundary that no longer exists).
    print('Running final scored-pairs completion pass...')
    final_comps = {
        node: cid
        for cid, comp in enumerate(nx.connected_components(G))
        for node in comp
    }
    final_comp_sms: dict[int, set[str]] = {}
    for node, cid in final_comps.items():
        final_comp_sms.setdefault(cid, set()).add(sm_map[node])

    final_incomplete = {cid for cid, sms in final_comp_sms.items() if 1 < len(sms) < len(all_sms)}
    completion_candidates2 = scored_pairs[scored_pairs['match_prob'] >= COMPLETION_THRESHOLD].copy()
    completion_candidates2['comp_a'] = completion_candidates2['id_a'].map(final_comps)
    completion_candidates2['comp_b'] = completion_candidates2['id_b'].map(final_comps)

    sp2_edges_added = 0
    for row in (
        completion_candidates2[
            (completion_candidates2['comp_a'].isin(final_incomplete) | completion_candidates2['comp_b'].isin(final_incomplete))
            & (completion_candidates2['comp_a'] != completion_candidates2['comp_b'])
        ]
        .sort_values('match_prob', ascending=False)
        .itertuples(index=False)
    ):
        u, v = int(row.id_a), int(row.id_b)
        cu, cv = final_comps.get(u), final_comps.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        sm_u, sm_v = sm_map[u], sm_map[v]

        u_missing = sm_v not in final_comp_sms.get(cu, set())
        v_missing = sm_u not in final_comp_sms.get(cv, set())
        if not (u_missing or v_missing):
            continue

        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        G.add_edge(u, v)
        seen.add(pair)
        sp2_edges_added += 1

        merged_sms = final_comp_sms.get(cu, set()) | final_comp_sms.get(cv, set())
        keep_cid = cu if len(final_comp_sms.get(cu, set())) >= len(final_comp_sms.get(cv, set())) else cv
        drop_cid = cv if keep_cid == cu else cu
        final_comp_sms[keep_cid] = merged_sms
        final_comp_sms.pop(drop_cid, None)
        for node, cid in list(final_comps.items()):
            if cid == drop_cid:
                final_comps[node] = keep_cid
        if len(merged_sms) == len(all_sms):
            final_incomplete.discard(keep_cid)

    print(f'Final scored-pairs completion added {sp2_edges_added:,} bridging edges.')
    # ── End final scored-pairs completion ─────────────────────────────────────

    clusters = list(nx.connected_components(G))
    cluster_map = {}
    for cluster_id, node_set in enumerate(sorted(clusters, key=len, reverse=True)):
        for node in node_set:
            cluster_map[node] = cluster_id

    df['cluster_id'] = df.index.map(cluster_map)
    df['cluster_size'] = df.groupby('cluster_id')['product_idx'].transform('count')

    multi_clusters = df[df['cluster_size'] >= 2].groupby('cluster_id')
    n_multi = multi_clusters.ngroups
    n_4way = sum(1 for _cid, g in multi_clusters if g['supermarket'].nunique() == 4)

    print('\n' + '=' * 50)
    print('FINAL ML CLUSTERING RESULTS')
    print('=' * 50)
    print(f'Total Products: {len(df):,}')
    print(f'Multi-product Clusters: {n_multi:,}')
    print(f'Perfect 4-way Clusters: {n_4way:,}')
    print('=' * 50)

    output_path = OUTPUT_DIR / 'ml_clusters.csv'
    df.to_csv(output_path, index=False)
    print(f'\nSaved final clusters to {output_path}')


def run_ml_matching(*, sample: bool = False) -> None:
    """Load prepared products, run FAISS retrieval → features → LightGBM → graph clusters."""
    configure_paths(sample=sample)

    print('Loading normalized data...')
    df = load_prepared_dataframe(sample=sample)

    df['normalized_name'] = df['normalized_name'].str.replace("'", '', regex=False)

    df = df.reset_index(drop=True)
    df['product_idx'] = df.index

    candidate_pairs, embeddings, indices, id_maps = retrieve_candidates(df)
    features_df = build_pairwise_features(df, candidate_pairs)
    scored_pairs, model = train_and_score(features_df)
    build_final_clusters(df, scored_pairs, features_df, embeddings, indices, id_maps, model)


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
