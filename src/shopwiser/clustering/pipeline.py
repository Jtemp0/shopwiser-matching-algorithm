"""Clustering passes: blocking, union-find, completion, export."""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from .blocking import _get_tokens, _unit_bucket, build_multi_blocks, run_pass
from . import config
from .config import *
from .similarity import _hard_conflict_check, _strip_brand, compute_similarity, compute_similarity_pass4
from .union_find import UnionFind


def run_clustering(df: pd.DataFrame) -> None:
    # ============================================================
    # PASS 1 — BRANDED
    # ============================================================

    branded_df = df[df['product_type'] == 'branded'].copy()
    print(f'\nPass 1 — Branded: {len(branded_df):,} products')

    def _branded_key_a(row):
        brand = str(row['known_brand_clean']).lower().strip()
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        return f'{brand}||{cat}||{utype}'

    def _branded_key_b(row):
        brand = str(row['known_brand_clean']).lower().strip()
        cat   = str(row['cat_norm']).lower().strip()
        return f'{brand}||{cat}'

    def _branded_key_c(row):
        brand = str(row['known_brand_clean']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        return f'{brand}||{utype}'

    print('Building multi-key blocks for branded...')
    pairs_branded = build_multi_blocks(branded_df, [_branded_key_a, _branded_key_b, _branded_key_c])
    print('Running comparisons...')
    matches_branded = run_pass(branded_df, pairs_branded, 'branded', UNIT_TOLERANCE_BRANDED)
    print(f'Pass 1 complete: {len(matches_branded):,} matches')

    # ============================================================
    # PASS 2 — OWN-BRAND
    # ============================================================

    own_brand_df = df[df['product_type'] == 'own_brand'].copy()
    print(f'\nPass 2 — Own-brand: {len(own_brand_df):,} products')

    def _own_key_a(row):
        tier  = str(row['tier_type'] or 'standard').lower().strip()
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        toks  = _get_tokens(row['normalized_name'])
        tok1  = toks[0] if toks else 'unknown'
        return f'{tier}||{cat}||{utype}||{tok1}'

    def _own_key_b(row):
        tier = str(row['tier_type'] or 'standard').lower().strip()
        cat  = str(row['cat_norm']).lower().strip()
        toks = _get_tokens(row['normalized_name'])
        tok1 = toks[0] if toks else 'unknown'
        return f'{tier}||{cat}||{tok1}'

    def _own_key_c(row):
        tier  = str(row['tier_type'] or 'standard').lower().strip()
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        toks  = _get_tokens(row['normalized_name'])
        tok2  = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
        return f'{tier}||{cat}||{utype}||{tok2}'

    def _own_key_d(row):
        tier = str(row['tier_type'] or 'standard').lower().strip()
        cat  = str(row['cat_norm']).lower().strip()
        toks = _get_tokens(row['normalized_name'])
        tok2 = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
        return f'{tier}||{cat}||{tok2}'

    def _own_key_e(row):
        """v7: no-tier key — catches products where tier classification differs between stores."""
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        ubucket = _unit_bucket(row['unit_value'], bucket_size=100)
        toks  = _get_tokens(row['normalized_name'])
        tok1  = toks[0] if toks else 'unknown'
        return f'{cat}||{utype}||{ubucket}||{tok1}'

    def _own_key_f(row):
        """v9: no-tier, no-bucket key — relies on sub-blocking (500/1000g) for wider unit coverage.
        Helps when unit values differ slightly between SMs for the same product."""
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        toks  = _get_tokens(row['normalized_name'])
        tok1  = toks[0] if toks else 'unknown'
        return f'{cat}||{utype}||{tok1}'

    def _own_key_g(row):
        """v9: no-tier, no-bucket, tok2 — covers products where first token differs but second matches."""
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        toks  = _get_tokens(row['normalized_name'])
        tok2  = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
        return f'{cat}||{utype}||{tok2}'

    print('Building multi-key blocks for own-brand...')
    pairs_own = build_multi_blocks(own_brand_df, [_own_key_a, _own_key_b, _own_key_c, _own_key_d, _own_key_e, _own_key_f, _own_key_g])
    print('Running comparisons...')
    matches_own = run_pass(own_brand_df, pairs_own, 'own_brand', UNIT_TOLERANCE_OWN_BRAND)
    print(f'Pass 2 complete: {len(matches_own):,} matches')

    # ============================================================
    # PASS 3 — UNBRANDED
    # ============================================================

    unbranded_df = df[df['product_type'] == 'unbranded'].copy()
    print(f'\nPass 3 — Unbranded: {len(unbranded_df):,} products')

    def _unb_key_a(row):
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        toks  = _get_tokens(row['normalized_name'])
        sig   = sorted(toks[:2])
        tok_key = '_'.join(sig) if sig else 'unknown'
        return f'{cat}||{utype}||{tok_key}'

    def _unb_key_b(row):
        cat  = str(row['cat_norm']).lower().strip()
        toks = _get_tokens(row['normalized_name'])
        sig  = sorted(toks[:2])
        tok_key = '_'.join(sig) if sig else 'unknown'
        return f'{cat}||{tok_key}'

    def _unb_key_c(row):
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        toks  = _get_tokens(row['normalized_name'])
        picks = [toks[0]] if toks else []
        if len(toks) > 2:
            picks.append(toks[2])
        elif len(toks) > 1:
            picks.append(toks[1])
        tok_key = '_'.join(sorted(picks)) if picks else 'unknown'
        return f'{cat}||{utype}||{tok_key}'

    def _unb_key_d(row):
        cat   = str(row['cat_norm']).lower().strip()
        utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        toks  = _get_tokens(row['normalized_name'])
        picks = []
        if len(toks) > 1:
            picks.append(toks[1])
        if len(toks) > 2:
            picks.append(toks[2])
        tok_key = '_'.join(sorted(picks)) if picks else 'unknown'
        return f'{cat}||{utype}||{tok_key}'

    def _unb_key_e(row):
        """
        v5: Cross-token-window key — cat + utype + unit_bucket(25g) + tok3.
        Catches products whose tok1/tok2 differ but share a later token
        (e.g. 'jacket potato' vs 'baking potato' — both have 'potato' but at
        different positions; key_c/d don't help because tok3 is also absent).
        Using 25g buckets + tok3 keeps blocks small.
        """
        cat    = str(row['cat_norm']).lower().strip()
        utype  = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        ubucket = _unit_bucket(row['unit_value'], bucket_size=25)
        toks   = _get_tokens(row['normalized_name'])
        # Prefer tok2 (index 1) — most products have ≥2 tokens
        tok2 = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
        return f'{cat}||{utype}||{ubucket}||{tok2}'

    def _unb_key_f(row):
        """
        v7: tok3-based key — catches products where tok1/tok2 are category descriptors
        (e.g. 'thick cut' / 'oven ready') but the core product word is tok3
        (e.g. 'chips' / 'burgers').  Using a 50g bucket keeps block sizes manageable.
        """
        cat    = str(row['cat_norm']).lower().strip()
        utype  = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        ubucket = _unit_bucket(row['unit_value'], bucket_size=50)
        toks   = _get_tokens(row['normalized_name'])
        tok3   = toks[2] if len(toks) > 2 else (toks[-1] if toks else 'unknown')
        return f'{cat}||{utype}||{ubucket}||{tok3}'

    print('Building multi-key blocks for unbranded (keys a–f)...')
    pairs_unbranded_all = build_multi_blocks(
        unbranded_df, [_unb_key_a, _unb_key_b, _unb_key_c, _unb_key_d, _unb_key_e, _unb_key_f]
    )

    print('Running comparisons...')
    matches_unbranded = run_pass(unbranded_df, pairs_unbranded_all, 'unbranded', UNIT_TOLERANCE_UNBRANDED)
    print(f'Pass 3 complete: {len(matches_unbranded):,} matches')

    # ============================================================
    # PASS 4 — CROSS-BUCKET CATCH-ALL (v5 new)
    # ============================================================

    # Tentative singletons: products with no direct match in passes 1–3
    matched_idx_so_far = set()
    for ia, ib, _ in matches_branded + matches_own + matches_unbranded:
        matched_idx_so_far.add(ia)
        matched_idx_so_far.add(ib)

    singleton_df = df[~df.index.isin(matched_idx_so_far)].copy()
    print(f'\nPass 4 — Cross-bucket: {len(singleton_df):,} tentative singletons')

    def _pass4_key_a(row):
        """First significant token key — catches direct brand-detection failures."""
        cat    = str(row['cat_norm']).lower().strip()
        utype  = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        ubucket = _unit_bucket(row['unit_value'], bucket_size=100)
        toks = _get_tokens(row['normalized_name'])
        tok1 = toks[0] if toks else 'unknown'
        return f'{cat}||{utype}||{ubucket}||{tok1}'

    def _pass4_key_b(row):
        """Second token key — catches tok1-differs but tok2-matches cases."""
        cat    = str(row['cat_norm']).lower().strip()
        utype  = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
        ubucket = _unit_bucket(row['unit_value'], bucket_size=100)
        toks = _get_tokens(row['normalized_name'])
        tok2 = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
        return f'{cat}||{utype}||{ubucket}||{tok2}'

    print('Building blocks for pass 4...')
    pairs_pass4 = build_multi_blocks(singleton_df, [_pass4_key_a, _pass4_key_b], max_block_size=80)

    print('Running comparisons...')
    matches_pass4 = []
    for ia, ib in tqdm(pairs_pass4, desc='  Pass=cross_bucket', leave=True):
        is_match, score = compute_similarity_pass4(singleton_df.loc[ia], singleton_df.loc[ib])
        if is_match:
            matches_pass4.append((ia, ib, score))
    print(f'Pass 4 complete: {len(matches_pass4):,} matches')

    # ============================================================
    # UNION-FIND ASSEMBLY
    # ============================================================

    print('\nAssembling Union-Find...')
    uf = UnionFind(len(df))

    pair_scores = {}
    all_matches = (
        [('branded',      m) for m in matches_branded] +
        [('own_brand',    m) for m in matches_own] +
        [('unbranded',    m) for m in matches_unbranded] +
        [('cross_bucket', m) for m in matches_pass4]
    )

    for _pass, (ia, ib, score) in all_matches:
        uf.union(ia, ib)
        pair_scores[(min(ia, ib), max(ia, ib))] = score

    df['raw_cluster_id'] = df['product_idx'].apply(uf.find)
    raw_cluster_sizes = df.groupby('raw_cluster_id')['product_idx'].count()
    n_raw = df['raw_cluster_id'].nunique()
    print(f'Raw clusters: {n_raw:,}  singletons: {(raw_cluster_sizes==1).sum():,}  multi: {(raw_cluster_sizes>1).sum():,}')

    # ============================================================
    # PASS 5 — CLUSTER COMPLETION (v8)
    # ============================================================
    # Many 3-way clusters exist because the 4th supermarket's matching product
    # was never placed in the same blocking bucket (blocking miss).  This pass
    # bypasses blocking entirely: for every 3-way cluster we directly compare
    # against same-brand / tier+category products from the missing SM.

    _ALL_SMS_COMPLETION = {'ASDA', 'Tesco', 'Sains', 'Morrisons'}
    _COMPLETION_MAX_CANDS = {'branded': 200, 'own_brand': 80, 'unbranded': 50}


    def _get_completion_candidates(members: pd.DataFrame, df_full: pd.DataFrame,
                                   missing_sm: str):
        """Return (candidates_df, pass_type) for a 3-way cluster's missing SM."""
        ptype = members['product_type'].mode()[0]
        pool  = df_full[df_full['supermarket'] == missing_sm]

        if ptype == 'branded':
            brand = members['known_brand_clean'].dropna()
            if brand.empty:
                return pd.DataFrame(), ptype
            cands = pool[
                (pool['product_type'] == 'branded') &
                (pool['known_brand_clean'] == brand.iloc[0])
            ]

        elif ptype == 'own_brand':
            cands = pool[pool['product_type'] == 'own_brand']
            tier  = members['tier_type'].dropna()
            cat   = members['cat_norm'].dropna()
            if not tier.empty:
                cands = cands[cands['tier_type'] == tier.iloc[0]]
            if not cat.empty:
                cands = cands[cands['cat_norm'] == cat.iloc[0]]
            # v10: apply frozen/fresh_food storage-condition guard
            # (same logic as compute_similarity own_brand guard)
            _member_cats = members['category'].dropna().str.lower()
            _cluster_frozen = any('frozen' in c for c in _member_cats)
            if _cluster_frozen:
                cands = cands[cands['category'].str.lower().str.contains('frozen', na=False)]
            else:
                cands = cands[~cands['category'].str.lower().str.contains('frozen', na=True)]

        else:  # unbranded
            cands = pool[pool['product_type'].isin(['unbranded', 'branded'])]
            cat   = members['cat_norm'].dropna()
            utype = members['unit_type'].dropna()
            if not cat.empty:
                cands = cands[cands['cat_norm'] == cat.iloc[0]]
            if not utype.empty:
                cands = cands[cands['unit_type'] == utype.iloc[0]]
            avg_uv = members['unit_value'].mean()
            if pd.notna(avg_uv) and avg_uv > 0:
                lo, hi = avg_uv * 0.80, avg_uv * 1.25
                cands = cands[
                    cands['unit_value'].isna() |
                    ((cands['unit_value'] >= lo) & (cands['unit_value'] <= hi))
                ]

        return cands, ptype


    # Identify 3-way clusters using current UF state
    df['_comp_rc'] = df['product_idx'].apply(uf.find)
    _rc_sms   = df.groupby('_comp_rc')['supermarket'].agg(frozenset)
    _rc_sizes = df.groupby('_comp_rc').size()
    _three_way_roots = [
        root for root in _rc_sizes.index
        if _rc_sizes[root] == 3 and len(_rc_sms[root]) == 3
    ]
    print(f'\nPass 5 — Cluster completion: {len(_three_way_roots):,} 3-way clusters to extend')

    matches_completion: list = []
    _n_upgraded = 0

    for root in tqdm(_three_way_roots, desc='  Pass=completion', leave=True):
        members   = df[df['_comp_rc'] == root]
        present   = set(members['supermarket'])
        missing_l = list(_ALL_SMS_COMPLETION - present)
        if not missing_l:
            continue
        missing_sm = missing_l[0]

        cands, pass_type = _get_completion_candidates(members, df, missing_sm)
        if cands.empty:
            continue
        if len(cands) > _COMPLETION_MAX_CANDS.get(pass_type, 25):
            continue

        best_score = 0.0
        best_pair: tuple | None = None

        _unit_tol = {
            'branded':   UNIT_TOLERANCE_BRANDED,
            'own_brand': UNIT_TOLERANCE_OWN_BRAND,
            'unbranded': UNIT_TOLERANCE_UNBRANDED,
        }.get(pass_type, UNIT_TOLERANCE_BRANDED)

        for member_idx in members.index:
            m_row = members.loc[member_idx]
            for cand_idx in cands.index:
                if uf.find(member_idx) == uf.find(cand_idx):
                    continue  # already merged
                is_match, score = compute_similarity(m_row, cands.loc[cand_idx],
                                                     pass_type=pass_type,
                                                     unit_tolerance=_unit_tol)
                if is_match and score > best_score:
                    best_score = score
                    best_pair  = (member_idx, cand_idx)

        if best_pair:
            ia, ib = best_pair
            uf.union(ia, ib)
            matches_completion.append((ia, ib, best_score))
            pair_scores[(min(ia, ib), max(ia, ib))] = best_score
            _n_upgraded += 1

    df.drop(columns=['_comp_rc'], inplace=True, errors='ignore')
    print(f'Pass 5 complete: {len(matches_completion):,} new matches  ({_n_upgraded:,} clusters extended)')

    # ============================================================
    # POST-CLUSTER VALIDATION — prevent transitive-link artifacts
    # ============================================================

    print('\nPost-cluster validation (direct-edge rebuild)...')

    direct_pair_set = set(pair_scores.keys())
    validated_clusters = []
    transitive_breaks  = 0

    uf_components = uf.components()

    for raw_root, members in tqdm(uf_components.items(), desc='  Validating', leave=True):
        if len(members) == 1:
            validated_clusters.append(members)
            continue

        adj = defaultdict(set)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ia, ib = members[i], members[j]
                key = (min(ia, ib), max(ia, ib))
                if key in direct_pair_set:
                    adj[ia].add(ib)
                    adj[ib].add(ia)

        visited = set()
        n_before, n_after = 1, 0
        for start in members:
            if start in visited:
                continue
            comp, queue = [], [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                queue.extend(adj[node] - visited)
            validated_clusters.append(comp)
            n_after += 1

        if n_after > n_before:
            transitive_breaks += 1

    print(f'  Transitive links broken: {transitive_breaks:,}')
    print(f'  Validated clusters: {len(validated_clusters):,}')

    # ============================================================
    # POST-PROCESSING — same-SM and cross-tier violations
    # ============================================================

    def fix_same_supermarket_violation(group_df, pair_scores):
        """
        v5 SCORE-BASED fix (replaces v4 greedy iteration).

        For clusters with multiple products from the same supermarket:
        - For each duplicate SM, keep the product with the highest
          max cross-SM pair score as the representative.
        - Excluded products are recycled into sub-clusters if they have
          direct pairs between them; otherwise they become singletons.

        This preserves genuine 4-way clusters that the greedy approach
        destroyed by arbitrary iteration-order selection.
        """
        by_sm = {sm: list(g.index) for sm, g in group_df.groupby('supermarket')}

        # Fast path: no duplicate SMs
        if not any(len(idxs) > 1 for idxs in by_sm.values()):
            return [group_df]

        indices = list(group_df.index)

        def best_cross_sm_score(idx, my_sm):
            """Max pair score from this product to any product from a different SM."""
            best = 0.0
            for other in indices:
                if group_df.loc[other, 'supermarket'] == my_sm:
                    continue
                key = (min(idx, other), max(idx, other))
                s = pair_scores.get(key, 0.0)
                if s > best:
                    best = s
            return best

        primary_indices  = []
        excluded_indices = []

        for sm, idxs in by_sm.items():
            if len(idxs) == 1:
                primary_indices.append(idxs[0])
            else:
                # Pick the representative with the highest cross-SM score
                ranked = sorted(idxs, key=lambda x: best_cross_sm_score(x, sm), reverse=True)
                primary_indices.append(ranked[0])
                excluded_indices.extend(ranked[1:])

        result = [group_df.loc[primary_indices]]

        if not excluded_indices:
            return result

        # Try to form valid sub-clusters from excluded products
        if len(excluded_indices) >= 2:
            excl_df = group_df.loc[excluded_indices]
            excl_by_sm = excl_df.groupby('supermarket').size()
            if len(excl_by_sm) >= 2:
                # Only form a sub-cluster if there are direct cross-SM pairs
                has_valid_pair = any(
                    (min(ia, ib), max(ia, ib)) in pair_scores
                    for ia, ib in combinations(excluded_indices, 2)
                    if excl_df.loc[ia, 'supermarket'] != excl_df.loc[ib, 'supermarket']
                )
                if has_valid_pair:
                    # Recurse to handle any remaining SM duplicates in the sub-group
                    result.extend(fix_same_supermarket_violation(excl_df, pair_scores))
                    return result

        # No valid sub-clusters — excluded become singletons
        for idx in excluded_indices:
            result.append(group_df.loc[[idx]])

        return result


    def fix_cross_tier_violation(group_df):
        own = group_df[group_df['product_type'] == 'own_brand']
        if own.empty or own['tier_type'].nunique() <= 1:
            return [group_df]
        sub_clusters = []
        non_own = group_df[group_df['product_type'] != 'own_brand']
        for tier, tier_group in own.groupby('tier_type'):
            sub_clusters.append(pd.concat([tier_group, non_own]))
        return sub_clusters


    def fix_unit_type_violation(group_df):
        uts = group_df['unit_type'].dropna().unique()
        if len(uts) <= 1:
            return [group_df]
        no_unit = group_df[group_df['unit_type'].isna()]
        sub_clusters = []
        for ut in uts:
            sub = group_df[group_df['unit_type'] == ut]
            if not no_unit.empty:
                sub = pd.concat([sub, no_unit])
            sub_clusters.append(sub)
        return sub_clusters


    def _purge_hard_conflicts(group_df):
        """
        v5.2: Post-cluster hard-conflict purge.

        Transitive union-find chains can link products via a "bridge" product that
        has no flavor/packaging tokens, creating a cluster where two members would
        directly fail _hard_conflict_check (e.g. 'caramel' product bridged to a
        'chilli' product through a plain 'milk chocolate bar').

        Greedily removes the member with the most pairwise hard-conflicts until the
        cluster is internally consistent.  Removed products are returned as a list of
        product indices (each will become a singleton or seed a new sub-cluster).
        """
        indices = list(group_df.index)
        names   = {idx: str(group_df.loc[idx, 'normalized_name'] or '') for idx in indices}
        removed = []

        changed = True
        while changed and len(indices) >= 2:
            changed = False
            conflict_counts: dict = defaultdict(int)
            for ia, ib in combinations(indices, 2):
                if _hard_conflict_check(names[ia], names[ib]):
                    conflict_counts[ia] += 1
                    conflict_counts[ib] += 1
            if conflict_counts:
                # Remove the product with the most conflicts; tiebreak by higher idx
                # (later-indexed product is typically the "intruder" in most cases)
                worst = max(conflict_counts, key=lambda x: (conflict_counts[x], x))
                indices.remove(worst)
                removed.append(worst)
                changed = True

        return group_df.loc[indices], removed


    print('\nRunning violation fixes...')
    final_clusters        = []
    sm_violations_fixed   = 0
    tier_violations_fixed = 0

    for members in tqdm(validated_clusters, desc='  Post-processing', leave=True):
        group = df.loc[members]
        if len(group) == 1:
            final_clusters.append(group)
            continue
        sub_clusters = fix_same_supermarket_violation(group, pair_scores)
        if len(sub_clusters) > 1:
            sm_violations_fixed += 1
        for sc in sub_clusters:
            ut_subs = fix_unit_type_violation(sc)
            for ut_sc in ut_subs:
                tier_subs = fix_cross_tier_violation(ut_sc)
                if len(tier_subs) > 1:
                    tier_violations_fixed += 1
                final_clusters.extend(tier_subs)

    unit_type_violations_fixed = sum(
        1 for members in validated_clusters
        if len(members) > 1 and df.loc[members, 'unit_type'].dropna().nunique() > 1
    )
    print(f'  Same-SM violations fixed:      {sm_violations_fixed:,}')
    print(f'  Unit-type violations fixed:    {unit_type_violations_fixed:,}')
    print(f'  Cross-tier violations fixed:   {tier_violations_fixed:,}')

    # v5.2: Post-cluster hard-conflict purge — split clusters where a transitive
    # union-find bridge created internally inconsistent flavor/packaging pairs.
    print('Running post-cluster hard-conflict purge...')
    purge_products_removed = 0
    purged_clusters: list = []
    for cl in final_clusters:
        if len(cl) <= 1:
            purged_clusters.append(cl)
            continue
        clean_cl, conflict_idxs = _purge_hard_conflicts(cl)
        purged_clusters.append(clean_cl)
        for bad_idx in conflict_idxs:
            purged_clusters.append(cl.loc[[bad_idx]])   # singleton
        purge_products_removed += len(conflict_idxs)

    final_clusters = purged_clusters
    print(f'  Products purged from clusters: {purge_products_removed:,}')
    print(f'  Final cluster count:           {len(final_clusters):,}')

    # ============================================================
    # PASS 5B — POST-PROCESSING CLUSTER COMPLETION (v8)
    # ============================================================
    # The raw-cluster completion pass (Pass 5) misses 3-way clusters that
    # emerge from splitting larger raw clusters during post-processing.
    # This pass operates on the FINAL cluster list and merges qualifying
    # singletons into 3-way clusters, bypassing blocking misses entirely.
    #
    # Optimisation: pre-build candidate lookup dicts so that per-cluster
    # filtering is an O(1) dict look-up instead of an O(N) row-scan.
    print('\nPass 5B — Post-processing cluster completion...')

    # Index singletons: product_idx → position in final_clusters
    _sing_pos: dict = {}     # product_idx → index in final_clusters list
    for _i, _cl in enumerate(final_clusters):
        if len(_cl) == 1:
            _sing_pos[_cl.index[0]] = _i

    print(f'  Available singletons: {len(_sing_pos):,}')

    # ── Pre-build candidate dicts from the singleton pool ──────────────────
    # Key = (supermarket, ...) → list of product_idx
    # Only singletons are eligible candidates; pre-filter by product_type.

    _branded_idx:   dict = defaultdict(list)  # (sm, brand_lower)    → [idx]
    _own_brand_idx: dict = defaultdict(list)  # (sm, tier, cat)       → [idx]
    _unbranded_idx: dict = defaultdict(list)  # (sm, cat, unit_type)  → [idx]

    _sing_df = df.loc[list(_sing_pos.keys())]   # one DataFrame of all singletons

    for _pidx, _row in _sing_df.iterrows():
        _sm   = _row['supermarket']
        _pt   = _row['product_type']
        _cat  = _row['cat_norm']
        if _pt == 'branded':
            _brand_l = str(_row['known_brand_clean'] or '').lower()
            if _brand_l:
                _branded_idx[(_sm, _brand_l)].append(_pidx)
        elif _pt == 'own_brand':
            _tier = _row['tier_type']
            _own_brand_idx[(_sm, _tier, _cat)].append(_pidx)
        else:
            _ut = _row['unit_type'] if pd.notna(_row['unit_type']) else ''
            _unbranded_idx[(_sm, _cat, _ut)].append(_pidx)
            _unbranded_idx[(_sm, _cat, '')].append(_pidx)   # loose key (any unit_type)

    # Remove duplicates in the loose key
    for _k in _unbranded_idx:
        _unbranded_idx[_k] = list(dict.fromkeys(_unbranded_idx[_k]))

    # ── Shared state used across Pass 5C and Pass 5B ───────────────────────
    _sing_used: set = set()   # product_idxs consumed by completion passes


    _P5_PRE_UV_TOL = 0.30   # coarse pre-filter — exact check happens in compute_similarity


    def _p5_uv_prefilter(filt, uv3):
        """Keep candidates whose unit_value is within _P5_PRE_UV_TOL of uv3, or is NaN."""
        if not (pd.notna(uv3) and uv3 > 0) or not filt:
            return filt
        kept = [p for p in filt
                if not pd.notna(df.at[p, 'unit_value'])
                or (max(df.at[p, 'unit_value'], uv3) / min(df.at[p, 'unit_value'], uv3)
                    <= 1.0 + _P5_PRE_UV_TOL)]
        return kept if kept else filt   # fall back to unfiltered if over-aggressive


    def _p5_lookup(ptype, miss_sm, brand3, tier3, cat3, ut3, uv3, used):
        """Return candidate product_idx list for a completion pass lookup."""
        if ptype == 'branded':
            if brand3.empty:
                return []
            brand_l = str(brand3.iloc[0]).lower()
            filt = [p for p in _branded_idx.get((miss_sm, brand_l), []) if p not in used]
            filt = _p5_uv_prefilter(filt, uv3)
        elif ptype == 'own_brand':
            tier_v = tier3.iloc[0] if not tier3.empty else np.nan
            cat_v  = cat3.iloc[0]  if not cat3.empty  else np.nan
            filt = [p for p in _own_brand_idx.get((miss_sm, tier_v, cat_v), []) if p not in used]
            if not filt and not cat3.empty:
                _fb = [p for p in _own_brand_idx.get((miss_sm, np.nan, cat_v), []) if p not in used]
                for _tb in ['standard', 'value', 'premium']:
                    _fb.extend(p for p in _own_brand_idx.get((miss_sm, _tb, cat_v), []) if p not in used)
                filt = list(dict.fromkeys(_fb))
            filt = _p5_uv_prefilter(filt, uv3)
        else:
            cat_v = cat3.iloc[0] if not cat3.empty else np.nan
            ut_v  = ut3.iloc[0]  if not ut3.empty  else ''
            filt  = [p for p in _unbranded_idx.get((miss_sm, cat_v, ut_v), []) if p not in used]
            if pd.notna(uv3) and uv3 > 0 and filt:
                filt = [p for p in filt
                        if not pd.notna(df.at[p, 'unit_value']) or
                           (1.0 / (1.0 + _P5_PRE_UV_TOL)
                            <= df.at[p, 'unit_value'] / uv3
                            <= 1.0 + _P5_PRE_UV_TOL)]
        return filt


    def _p5_name_trim(cl_df, filt, hard_cap):
        """
        When the candidate list exceeds hard_cap, pre-sort by name similarity
        against the cluster's first member and take the top hard_cap candidates.
        This replaces the old "skip entirely" logic and gives the best-scoring
        names a fair chance even in large brand blocks.
        """
        if len(filt) <= hard_cap:
            return filt
        rep = str(cl_df['normalized_name'].iloc[0])
        filt_sorted = sorted(
            filt,
            key=lambda p: -fuzz.token_sort_ratio(rep, str(df.at[p, 'normalized_name']))
        )
        return filt_sorted[:hard_cap]


    _p5_score_diag = {'hard_zero': 0, 'low_score': 0, 'max_low': 0.0}


    def _p5_best_match(cl_df, filt, ptype, unit_tol):
        """Return (best_cand_idx, best_score, best_mem_idx) over all members × candidates.

        v9: When compute_similarity gives a non-zero score that still falls below _comp_thresh,
        we try a token_set_ratio blended fallback.  In the completion-pass context the
        brand/category/unit constraints are already very tight, so a partial-token overlap
        (e.g. "Cream of Tomato Soup" vs "Tomato Soup") is usually a valid match.
        Hard-conflict checks (FLAVOR_NAMED_TOKENS, ONE_SIDED_CONFLICT_TOKENS, …) have
        already fired and returned score=0 for genuine variant mismatches, so only
        same-category pairs enter this path.
        """
        _comp_thresh = COMPLETION_THRESHOLD.get(ptype, FUZZY_THRESHOLD_COMPLETION)
        best_score = 0.0
        best_cand  = None
        best_mem   = None
        cluster_max_score = 0.0
        cluster_all_zero  = True
        for _m in cl_df.index:
            _mr = cl_df.loc[_m]
            for _cidx in filt:
                _cr = df.loc[_cidx]
                _ok, _s = compute_similarity(_mr, _cr, pass_type=ptype, unit_tolerance=unit_tol)
                if _s > 0:
                    cluster_all_zero = False
                    cluster_max_score = max(cluster_max_score, _s)
                if not _ok and _s > 0:
                    # token_set_ratio fallback for same-type pairs that just miss the threshold
                    _na = str(_mr.get('normalized_name', ''))
                    _nb = str(_cr.get('normalized_name', ''))
                    if ptype == 'branded':
                        _brand = str(_mr.get('known_brand_clean', '') or '').lower()
                        _na = _strip_brand(_na, _brand)
                        _nb = _strip_brand(_nb, _brand)
                    _tset = fuzz.token_set_ratio(_na, _nb) / 100.0
                    _s2 = 0.55 * _s + 0.45 * _tset
                    if _s2 >= _comp_thresh:
                        _ok = True
                        _s = _s2
                elif not _ok and _s >= _comp_thresh:
                    _ok = True
                if _ok and _s > best_score:
                    best_score = _s
                    best_cand  = _cidx
                    best_mem   = _m
        if best_cand is None:
            if cluster_all_zero:
                _p5_score_diag['hard_zero'] += 1
            else:
                _p5_score_diag['low_score'] += 1
                _p5_score_diag['max_low'] = max(_p5_score_diag['max_low'], cluster_max_score)
        return best_cand, best_score, best_mem


    # ============================================================
    # PASS 5C — 2-WAY CLUSTER EXTENSION
    # ============================================================
    # Extend 2-way clusters to 3-way or 4-way by finding singletons
    # from the 2 missing supermarkets (or just 1).  Newly created
    # 3-way clusters will be picked up by Pass 5B below.
    print('\nPass 5C — 2-way cluster extension...')

    _final_2way = [
        (i, cl) for i, cl in enumerate(final_clusters)
        if len(cl) == 2 and cl['supermarket'].nunique() == 2
    ]
    print(f'  2-way final clusters: {len(_final_2way):,}')

    _p5c_to4 = 0; _p5c_to3 = 0
    _p5c_old: set = set(); _p5c_sing: set = set(); _p5c_new: list = []

    for _2pos, _cl2 in tqdm(_final_2way, desc='  Pass=5C', leave=True):
        _present2 = set(_cl2['supermarket'])
        _miss2    = list(_ALL_SMS_COMPLETION - _present2)   # exactly 2 SMs
        if len(_miss2) != 2:
            continue

        _ptype2 = _cl2['product_type'].mode()[0]
        _brand2 = _cl2['known_brand_clean'].dropna()
        _tier2  = _cl2['tier_type'].dropna()
        _cat2   = _cl2['cat_norm'].dropna()
        _ut2    = _cl2['unit_type'].dropna()
        _uv2    = _cl2['unit_value'].mean()

        _utol2 = COMPLETION_UNIT_TOL.get(_ptype2, UNIT_TOLERANCE_BRANDED)

        _res = {}  # sm → (cand_idx, score, mem_idx)
        for _msm in _miss2:
            _filt = _p5_lookup(_ptype2, _msm, _brand2, _tier2, _cat2, _ut2, _uv2, _sing_used)
            if not _filt:
                continue
            _hard_cap = _COMPLETION_MAX_CANDS.get(_ptype2, 200) * 3
            _filt = _p5_name_trim(_cl2, _filt, _hard_cap)
            _bc, _bs, _bm = _p5_best_match(_cl2, _filt, _ptype2, _utol2)
            if _bc is not None:
                _res[_msm] = (_bc, _bs, _bm)

        if not _res:
            continue

        if len(_res) == 2:
            # Try to build a 4-way cluster
            _cands = [_res[sm][0] for sm in _miss2]
            _cl4 = pd.concat([_cl2, df.loc[_cands]])
            _cl4_clean, _ = _purge_hard_conflicts(_cl4)
            if len(_cl4_clean) == 4:
                # Check the 2 new candidates are mutually compatible too
                _ca, _cb = _cands
                _ra, _rb = df.loc[_ca], df.loc[_cb]
                _ok_ab, _ = compute_similarity(_ra, _rb, pass_type=_ptype2, unit_tolerance=_utol2)
                if _ok_ab or True:   # cross-SM pair check is best-effort
                    for _cidx in _cands:
                        _sing_used.add(_cidx)
                        _p5c_sing.add(_sing_pos[_cidx])
                    for _bm, _cidx, _bs in [(_res[sm][2], _res[sm][0], _res[sm][1]) for sm in _miss2]:
                        matches_completion.append((_bm, _cidx, _bs))
                        pair_scores[(min(_bm, _cidx), max(_bm, _cidx))] = _bs
                    _p5c_old.add(_2pos)
                    _p5c_new.append(_cl4_clean)
                    _p5c_to4 += 1
                    continue

        # Fall through: try to add just one (best-scoring) missing SM
        _best_sm = max(_res.keys(), key=lambda sm: _res[sm][1])
        _bc, _bs, _bm = _res[_best_sm]
        _cl3 = pd.concat([_cl2, df.loc[[_bc]]])
        _cl3_clean, _ = _purge_hard_conflicts(_cl3)
        if len(_cl3_clean) >= 3:
            _sing_used.add(_bc)
            _p5c_sing.add(_sing_pos[_bc])
            matches_completion.append((_bm, _bc, _bs))
            pair_scores[(min(_bm, _bc), max(_bm, _bc))] = _bs
            _p5c_old.add(_2pos)
            _p5c_new.append(_cl3_clean)
            _p5c_to3 += 1

    # Rebuild final_clusters after Pass 5C
    _p5c_remove = _p5c_old | _p5c_sing
    final_clusters = [cl for i, cl in enumerate(final_clusters) if i not in _p5c_remove]
    final_clusters.extend(_p5c_new)
    print(f'Pass 5C complete: {_p5c_to4:,} to 4-way, {_p5c_to3:,} to 3-way')

    # IMPORTANT: rebuild _sing_pos after Pass 5C mutates final_clusters
    # (Pass 5B uses _sing_pos to locate singletons for removal)
    _sing_pos = {_cl.index[0]: _i
                 for _i, _cl in enumerate(final_clusters) if len(_cl) == 1}

    # ── Collect final 3-way clusters ───────────────────────────────────────
    _final_3way = [
        (i, cl) for i, cl in enumerate(final_clusters)
        if len(cl) == 3 and cl['supermarket'].nunique() == 3
    ]
    print(f'  3-way final clusters: {len(_final_3way):,}')

    _p5b_new_pairs:    list = []
    _p5b_old_positions: set = set()
    _p5b_sing_positions: set = set()
    _p5b_new_clusters:  list = []
    _p5b_upgraded = 0
    # _sing_used is shared with Pass 5C (declared above)

    # Debug counters
    _p5b_dbg = {'no_brand': 0, 'no_cand': 0, 'too_many': 0,
                 'sim_checked': 0, 'sim_fail': 0, 'conflict_purge': 0}

    for _3pos, _cl3 in tqdm(_final_3way, desc='  Pass=5B', leave=True):
        _present3 = set(_cl3['supermarket'])
        _miss_list = list(_ALL_SMS_COMPLETION - _present3)
        if not _miss_list:
            continue
        _miss_sm = _miss_list[0]

        _ptype3  = _cl3['product_type'].mode()[0]
        _brand3  = _cl3['known_brand_clean'].dropna()
        _tier3   = _cl3['tier_type'].dropna()
        _cat3    = _cl3['cat_norm'].dropna()
        _ut3     = _cl3['unit_type'].dropna()
        _uv3     = _cl3['unit_value'].mean()

        # ── Look up pre-built index (shared helper) ──────────────────────
        if _ptype3 == 'branded' and _brand3.empty:
            _p5b_dbg['no_brand'] += 1
            continue
        _filt = _p5_lookup(_ptype3, _miss_sm, _brand3, _tier3, _cat3, _ut3, _uv3, _sing_used)

        if not _filt:
            _p5b_dbg['no_cand'] += 1
            continue
        _hard_cap = _COMPLETION_MAX_CANDS.get(_ptype3, 200) * 3
        if len(_filt) > _hard_cap:
            _p5b_dbg['too_many'] += 1
            _filt = _p5_name_trim(_cl3, _filt, _hard_cap)

        _unit_tol3 = COMPLETION_UNIT_TOL.get(_ptype3, UNIT_TOLERANCE_BRANDED)

        _p5b_dbg['sim_checked'] += 1
        _best3_cand, _best3_score, _best3_mem = _p5_best_match(_cl3, _filt, _ptype3, _unit_tol3)

        if _best3_cand is None:
            _p5b_dbg['sim_fail'] += 1
            continue

        if _best3_cand is not None:
            _cl4 = pd.concat([_cl3, df.loc[[_best3_cand]]])
            _cl4_clean, _bad_idxs = _purge_hard_conflicts(_cl4)
            if len(_cl4_clean) < 4:
                _p5b_dbg['conflict_purge'] += 1
                continue

            _p5b_new_pairs.append((_best3_mem, _best3_cand, _best3_score))
            _p5b_old_positions.add(_3pos)
            _p5b_sing_positions.add(_sing_pos[_best3_cand])
            _p5b_new_clusters.append(_cl4_clean)
            _sing_used.add(_best3_cand)
            pair_scores[(min(_best3_mem, _best3_cand), max(_best3_mem, _best3_cand))] = _best3_score
            _p5b_upgraded += 1

    # Rebuild final_clusters
    _remove_positions = _p5b_old_positions | _p5b_sing_positions
    final_clusters = [cl for i, cl in enumerate(final_clusters) if i not in _remove_positions]
    final_clusters.extend(_p5b_new_clusters)
    matches_completion.extend(_p5b_new_pairs)

    print(f'Pass 5B complete: {_p5b_upgraded:,} clusters extended to 4-way')
    print(f'  no_brand={_p5b_dbg["no_brand"]}  no_cand={_p5b_dbg["no_cand"]}  '
          f'too_many={_p5b_dbg["too_many"]}  sim_checked={_p5b_dbg["sim_checked"]}  '
          f'sim_fail={_p5b_dbg["sim_fail"]}  conflict_purge={_p5b_dbg["conflict_purge"]}')
    print(f'  sim_fail breakdown: hard_zero={_p5_score_diag["hard_zero"]}  '
          f'low_score={_p5_score_diag["low_score"]}  '
          f'max_low_score={_p5_score_diag["max_low"]:.3f}')

    # Assign sequential cluster IDs (largest first)
    final_clusters.sort(key=lambda g: -len(g))
    cluster_id_map = {}
    match_type_map = {}

    # Build Pass-4 / Pass-5 pair key sets for diagnostic tracking
    pass4_pair_keys      = {(min(ia, ib), max(ia, ib)) for ia, ib, _ in matches_pass4}
    completion_pair_keys = {(min(ia, ib), max(ia, ib)) for ia, ib, _ in matches_completion}

    for cid, group in enumerate(final_clusters):
        for idx in group.index:
            cluster_id_map[idx] = cid
        types = group['product_type'].value_counts()
        match_type_map[cid] = types.index[0] if len(types) else 'unknown'

    # Override match_type to 'cross_bucket' or 'completion' where applicable
    for cid, group in enumerate(final_clusters):
        idxs = list(group.index)
        for ia, ib in combinations(idxs, 2):
            key = (min(ia, ib), max(ia, ib))
            if key in completion_pair_keys:
                match_type_map[cid] = 'completion'
                break
            if key in pass4_pair_keys:
                match_type_map[cid] = 'cross_bucket'
                break

    df['cluster_id'] = df.index.map(cluster_id_map)

    final_sizes = df.groupby('cluster_id')['product_idx'].count()
    print(f'\nFinal size distribution:')
    for size, count in final_sizes.value_counts().sort_index().items():
        if size <= 8:
            print(f'  size {size}: {count:,}')

    # ============================================================
    # OUTPUT GENERATION
    # ============================================================

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cluster_avg_scores = defaultdict(list)
    for (ia, ib), score in pair_scores.items():
        cid_a = cluster_id_map.get(ia)
        cid_b = cluster_id_map.get(ib)
        if cid_a == cid_b and cid_a is not None:
            cluster_avg_scores[cid_a].append(score)

    avg_score_map = {cid: np.mean(scores) for cid, scores in cluster_avg_scores.items()}
    min_score_map = {cid: np.min(scores)  for cid, scores in cluster_avg_scores.items()}

    clusters_df = df[[
        'cluster_id', 'product_idx', 'supermarket', 'names', 'category',
        'known_brand_clean', 'own_brand', 'tier_type', 'unit_value', 'unit_type',
        'pack_quantity', 'core_product_name', 'normalized_name',
        'prices_(£)', 'prices_unit_(£)', 'product_type',
    ]].copy()

    clusters_df['cluster_size']       = df.groupby('cluster_id')['product_idx'].transform('count')
    clusters_df['n_supermarkets']     = df.groupby('cluster_id')['supermarket'].transform('nunique')
    clusters_df['match_type']         = clusters_df['cluster_id'].map(match_type_map)
    clusters_df['avg_pairwise_score'] = clusters_df['cluster_id'].map(avg_score_map)

    clusters_df.to_csv(config.OUTPUT_DIR / 'clusters.csv', index=False)
    print(f'\nSaved clusters.csv  ({len(clusters_df):,} rows)')

    # Cluster summary
    summary_rows = []
    for cid, group in tqdm(clusters_df.groupby('cluster_id'), desc='Building summary'):
        names_avail = group['core_product_name'].dropna()
        consensus   = names_avail.loc[names_avail.str.len().idxmin()] if len(names_avail) else ''
        summary_rows.append({
            'cluster_id':                  cid,
            'cluster_size':                len(group),
            'n_supermarkets':              group['supermarket'].nunique(),
            'supermarkets_present':        '|'.join(sorted(group['supermarket'].unique())),
            'category':                    group['category'].mode()[0] if len(group) else None,
            'match_type':                  match_type_map.get(cid, 'unknown'),
            'known_brand':                 group['known_brand_clean'].dropna().iloc[0] if group['known_brand_clean'].notna().any() else None,
            'tier_type':                   group['tier_type'].dropna().iloc[0] if group['tier_type'].notna().any() else None,
            'unit_value':                  group['unit_value'].dropna().mean() if group['unit_value'].notna().any() else None,
            'unit_type':                   group['unit_type'].dropna().iloc[0] if group['unit_type'].notna().any() else None,
            'pack_quantity':               group['pack_quantity'].dropna().mean() if group['pack_quantity'].notna().any() else None,
            'core_product_name_consensus': consensus,
            'avg_pairwise_score':          avg_score_map.get(cid),
            'min_pairwise_score':          min_score_map.get(cid),
        })

    cluster_summary = pd.DataFrame(summary_rows)
    cluster_summary.to_csv(config.OUTPUT_DIR / 'cluster_summary.csv', index=False)
    print(f'Saved cluster_summary.csv  ({len(cluster_summary):,} rows)')

    singletons = clusters_df[clusters_df['cluster_size'] == 1]
    singletons.to_csv(config.OUTPUT_DIR / 'singletons.csv', index=False)
    print(f'Saved singletons.csv  ({len(singletons):,} rows)')

    # Audit sample
    multi = cluster_summary[cluster_summary['cluster_size'] >= 2]
    bp    = multi[multi['match_type'] == 'branded']
    op    = multi[multi['match_type'] == 'own_brand']
    up    = multi[multi['match_type'] == 'unbranded']
    xp    = multi[multi['match_type'] == 'cross_bucket']

    audit_ids = pd.concat([
        bp.sample(min(20, len(bp)), random_state=RANDOM_SEED),
        op.sample(min(15, len(op)), random_state=RANDOM_SEED),
        up.sample(min(10, len(up)), random_state=RANDOM_SEED),
        xp.sample(min(5,  len(xp)), random_state=RANDOM_SEED) if len(xp) else pd.DataFrame(),
    ])['cluster_id'].tolist()

    audit_df = clusters_df[clusters_df['cluster_id'].isin(audit_ids)].copy()
    audit_df = audit_df.sort_values(['cluster_id', 'supermarket'])
    audit_df['AUDIT_same_core_product'] = ''
    audit_df['AUDIT_weight_ok']         = ''
    audit_df['AUDIT_tier_ok']           = ''
    audit_df['AUDIT_notes']             = ''
    audit_df.to_csv(config.OUTPUT_DIR / 'audit_sample_50.csv', index=False)
    print(f'Saved audit_sample_50.csv  ({len(audit_df)} rows, {len(audit_ids)} clusters)')

    # ============================================================
    # DIAGNOSTIC REPORT
    # ============================================================

    non_singleton = cluster_summary[cluster_summary['cluster_size'] >= 2]
    n_multi = len(non_singleton)

    print('\n' + '=' * 60)
    print('DIAGNOSTIC REPORT v10')
    print('=' * 60)
    print(f'Total clusters (incl singletons): {len(cluster_summary):,}')
    print(f'Singletons:                       {(cluster_summary["cluster_size"]==1).sum():,}')
    print(f'Multi-product clusters (≥2):      {n_multi:,}')
    print(f'  4-way: {(non_singleton["n_supermarkets"]==4).sum():,}')
    print(f'  3-way: {(non_singleton["n_supermarkets"]==3).sum():,}')
    print(f'  2-way: {(non_singleton["n_supermarkets"]==2).sum():,}')

    print(f'\nBy match type:')
    for mt in ['branded', 'own_brand', 'unbranded', 'cross_bucket']:
        n = (non_singleton['match_type'] == mt).sum()
        if n:
            print(f'  {mt:14s}: {n:,}')

    print(f'\nPass 4 contribution:')
    print(f'  Cross-bucket matches found:  {len(matches_pass4):,}')
    print(f'  Cross-bucket clusters:       {(non_singleton["match_type"]=="cross_bucket").sum():,}')

    print(f'\nProduct coverage:')
    in_cluster = clusters_df[clusters_df['cluster_size'] >= 2]
    for sm in sorted(df['supermarket'].unique()):
        total   = len(df[df['supermarket'] == sm])
        matched = len(in_cluster[in_cluster['supermarket'] == sm])
        print(f'  {sm:12s}: {matched:,}/{total:,} = {matched/total*100:.1f}%')

    print(f'\nQuality scores (avg_pairwise_score):')
    scores_avail = non_singleton.dropna(subset=['avg_pairwise_score'])
    for mt in ['branded', 'own_brand', 'unbranded', 'cross_bucket']:
        sub = scores_avail[scores_avail['match_type'] == mt]['avg_pairwise_score']
        if len(sub):
            print(f'  {mt:14s}: mean={sub.mean():.3f}  p5={sub.quantile(0.05):.3f}  p25={sub.quantile(0.25):.3f}')

    print(f'\nAutomated validation checks:')
    violations = {'same_sm': 0, 'unit_type_mixed': 0, 'tier_mixed': 0, 'weight_high': 0, 'score_low': 0}
    for cid, group in clusters_df[clusters_df['cluster_size'] >= 2].groupby('cluster_id'):
        if group['supermarket'].value_counts().max() > 1:
            violations['same_sm'] += 1
        if group['unit_type'].dropna().nunique() > 1:
            violations['unit_type_mixed'] += 1
        own = group[group['product_type'] == 'own_brand']
        if len(own) >= 2 and own['tier_type'].nunique() > 1:
            violations['tier_mixed'] += 1
        uv = group['unit_value'].dropna()
        if len(uv) >= 2 and uv.min() > 0 and uv.max() / uv.min() > 1.10:
            violations['weight_high'] += 1
        avg_s = avg_score_map.get(cid)
        if avg_s is not None and avg_s < 0.75:
            violations['score_low'] += 1

    print(f'  Same-SM violations:     {violations["same_sm"]:,}  (should be 0)')
    print(f'  Mixed unit type:        {violations["unit_type_mixed"]:,}  (should be 0)')
    print(f'  Tier mixing:            {violations["tier_mixed"]:,}  (should be 0)')
    print(f'  Weight variance >10%:   {violations["weight_high"]:,}  (flag)')
    print(f'  Score below 0.75:       {violations["score_low"]:,}  (flag)')

    print(f'\nTarget range: 10,000–20,000 multi-clusters')
    print(f'Actual:       {n_multi:,}')
    print(f'In range:     {"YES ✓" if 10_000 <= n_multi <= 20_000 else "NO — review thresholds"}')
    print('=' * 60)

