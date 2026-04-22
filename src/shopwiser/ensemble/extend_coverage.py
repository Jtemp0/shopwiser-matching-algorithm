"""Extend cluster coverage with two rule-based passes for unmatched singletons.

Pass A — Own-brand / unbranded:
    Groups singletons by (cat_norm, unit_type). Within each group finds
    cross-SM product pairs satisfying:
        - size_delta ≤ SIZE_TOL_OB (15 %)
        - token_set_ratio(normalized_name) ≥ OB_FUZZ_MIN (90)
        - no hard-conflict tokens
    Accepts the highest-scoring candidate per product (greedy union-find,
    max 1 product per supermarket per cluster).

Pass B — Branded same-brand:
    Groups branded singletons by (known_brand_clean, unit_type). Within
    each group finds cross-SM product pairs satisfying:
        - size_delta ≤ SIZE_TOL_BRAND (10 %)
        - token_set_ratio(normalized_name) ≥ BRAND_FUZZ_MIN (75)
        - no hard-conflict tokens
    Same one-per-SM greedy clustering.

New clusters are appended to ensemble_clusters_final.csv with fresh
ensemble_cluster_id values.  Pre-extension data is preserved in
ensemble_clusters_preextend.csv.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from shopwiser.ml_matching.features import check_hard_conflict
from shopwiser.paths import DATA_OUTPUTS

# ── paths ─────────────────────────────────────────────────────────────────────
FINAL_CSV   = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_final.csv'
ML_CSV      = DATA_OUTPUTS / 'ml_clusters' / 'ml_clusters.csv'
BACKUP_CSV  = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_preextend.csv'

# ── thresholds ────────────────────────────────────────────────────────────────
SIZE_TOL_OB    = 0.15   # ±15 % for own-brand / unbranded
SIZE_TOL_BRAND = 0.10   # ±10 % for branded (same brand → tighter)
OB_FUZZ_MIN    = 90     # token_set_ratio floor for own-brand matching
BRAND_FUZZ_MIN = 80     # token_set_ratio floor for branded matching
# Require character-level ratio ≥ this in BOTH passes.  Prevents truncated/
# short names ("galaxy", "cadbury", "broccoli") from matching any product that
# merely *contains* that token — a known failure mode of token_set_ratio's
# subset logic.
RATIO_MIN      = 70     # fuzz.ratio floor (character-level) for both passes
# Pass C: cluster completion.  Lower TSR threshold is safe because the check
# is unanimous — the candidate must meet it against EVERY existing member.
COMPLETION_TSR_MIN = 70

ALL_SMS = ('ASDA', 'Morrisons', 'Sains', 'Tesco')

# Tiers that are meaningfully distinct.  A product with an unknown/null tier
# is treated as compatible with any tier (it may just be undetected).
_KNOWN_TIERS = frozenset({'value', 'standard', 'premium', 'dietary'})


# ── helpers ───────────────────────────────────────────────────────────────────

def _canon_brand(b: object) -> str:
    if not isinstance(b, str):
        return ''
    s = b.strip().lower()
    s = re.sub(r"['\u2019\.\-]", '', s)
    s = re.sub(r'\s+', '', s)
    return s


def _size_delta(uv_a: float, uv_b: float) -> float:
    hi = max(abs(uv_a), abs(uv_b))
    if hi < 1e-6:
        return 0.0
    return abs(uv_a - uv_b) / hi


def _score(name_a: str, name_b: str) -> float:
    """Token-set fuzz ratio in [0, 1]."""
    return fuzz.token_set_ratio(name_a, name_b) / 100.0


def _tiers_compatible(tier_a: object, tier_b: object) -> bool:
    """True when the two tiers can co-exist in the same cluster.

    Two products are compatible unless both carry a KNOWN tier and those tiers
    differ.  A null/unknown tier is always compatible with anything — the tier
    may simply not have been detected.
    """
    ta = str(tier_a).lower() if tier_a and not (isinstance(tier_a, float) and tier_a != tier_a) else ''
    tb = str(tier_b).lower() if tier_b and not (isinstance(tier_b, float) and tier_b != tier_b) else ''
    if ta not in _KNOWN_TIERS or tb not in _KNOWN_TIERS:
        return True
    return ta == tb


# ── union-find ────────────────────────────────────────────────────────────────

class _UF:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[rb] = ra
        return True


def _build_clusters(
    pairs: list[tuple[float, int, int, str, str]],
    idx_to_sm: dict[int, str],
) -> dict[int, list[int]]:
    """Greedy cluster assignment.  Returns {root_idx: [member_idx, ...]}."""
    uf = _UF()
    root_sm: dict[int, set[str]] = {}  # root -> set of SMs already in cluster

    pairs.sort(key=lambda p: -p[0])  # descending score

    for score, idx_a, idx_b, sm_a, sm_b in pairs:
        ra = uf.find(idx_a)
        rb = uf.find(idx_b)
        if ra == rb:
            continue  # already in same cluster
        sms_a = root_sm.get(ra, {sm_a})
        sms_b = root_sm.get(rb, {sm_b})
        # Conflict: merged cluster would have two products from same SM
        if sms_a & sms_b:
            continue
        # Limit to 4 (one per SM)
        if len(sms_a) + len(sms_b) > 4:
            continue
        uf.union(ra, rb)
        new_root = uf.find(ra)
        root_sm[new_root] = sms_a | sms_b

    clusters: dict[int, list[int]] = {}
    for idx in idx_to_sm:
        r = uf.find(idx)
        clusters.setdefault(r, []).append(idx)
    # Only keep multi-product clusters
    return {r: members for r, members in clusters.items() if len(members) >= 2}


# ── Pass A: own-brand / unbranded ─────────────────────────────────────────────

def _pass_a(singletons: pd.DataFrame) -> list[tuple[float, int, int, str, str]]:
    """Return scored pairs for own-brand / unbranded singletons."""
    ob = singletons[singletons['product_type'].isin(['own_brand', 'unbranded'])].copy()
    ob = ob[ob['unit_value'].notna() & (ob['unit_value'] > 0)]

    pairs: list[tuple[float, int, int, str, str]] = []

    for (cat, ut), grp in ob.groupby(['cat_norm', 'unit_type']):
        grp_s = grp.sort_values('unit_value').reset_index(drop=True)
        rows = grp_s[['supermarket', 'normalized_name', 'unit_value', 'product_idx', 'tier_type']].values.tolist()
        n = len(rows)
        for i in range(n):
            sm_i, name_i, uv_i, idx_i, tier_i = rows[i]
            name_i_s = str(name_i) if isinstance(name_i, str) else ''
            for j in range(i + 1, n):
                sm_j, name_j, uv_j, idx_j, tier_j = rows[j]
                if uv_j > uv_i * (1 + SIZE_TOL_OB):
                    break
                if sm_i == sm_j:
                    continue
                if _size_delta(uv_i, uv_j) > SIZE_TOL_OB:
                    continue
                if not _tiers_compatible(tier_i, tier_j):
                    continue
                name_j_s = str(name_j) if isinstance(name_j, str) else ''
                sc = _score(name_i_s, name_j_s)
                if sc < OB_FUZZ_MIN / 100.0:
                    continue
                # Guard against short / truncated names matching via subset
                if fuzz.ratio(name_i_s, name_j_s) < RATIO_MIN:
                    continue
                if check_hard_conflict(name_i_s, name_j_s) == 1:
                    continue
                pairs.append((sc, int(idx_i), int(idx_j), sm_i, sm_j))

    return pairs


# ── Pass B: branded same-brand ────────────────────────────────────────────────

def _pass_b(singletons: pd.DataFrame) -> list[tuple[float, int, int, str, str]]:
    """Return scored pairs for branded singletons with matching brand."""
    branded = singletons[singletons['product_type'] == 'branded'].copy()
    branded = branded[branded['unit_value'].notna() & (branded['unit_value'] > 0)]
    branded['brand_canon'] = branded['known_brand_clean'].apply(_canon_brand)
    branded = branded[branded['brand_canon'] != '']

    pairs: list[tuple[float, int, int, str, str]] = []

    for (brand, ut), grp in branded.groupby(['brand_canon', 'unit_type']):
        if grp['supermarket'].nunique() < 2:
            continue
        grp_s = grp.sort_values('unit_value').reset_index(drop=True)
        rows = grp_s[['supermarket', 'normalized_name', 'unit_value', 'product_idx']].values.tolist()
        n = len(rows)
        for i in range(n):
            sm_i, name_i, uv_i, idx_i = rows[i]
            name_i_s = str(name_i) if isinstance(name_i, str) else ''
            for j in range(i + 1, n):
                sm_j, name_j, uv_j, idx_j = rows[j]
                if uv_j > uv_i * (1 + SIZE_TOL_BRAND):
                    break
                if sm_i == sm_j:
                    continue
                if _size_delta(uv_i, uv_j) > SIZE_TOL_BRAND:
                    continue
                name_j_s = str(name_j) if isinstance(name_j, str) else ''
                sc = _score(name_i_s, name_j_s)
                if sc < BRAND_FUZZ_MIN / 100.0:
                    continue
                # Guard against truncated brand-only names ("Galaxy 268g" → "galaxy")
                if fuzz.ratio(name_i_s, name_j_s) < RATIO_MIN:
                    continue
                if check_hard_conflict(name_i_s, name_j_s) == 1:
                    continue
                pairs.append((sc, int(idx_i), int(idx_j), sm_i, sm_j))

    return pairs


# ── Pass C: complete new incomplete clusters ──────────────────────────────────

def _pass_c(
    new_clusters: dict[int, list[int]],
    used_idxs: set[int],
    singletons: pd.DataFrame,
    idx_to_sm: dict[int, str],
    idx_to_row: dict[int, pd.Series],
) -> list[tuple[float, int, int]]:
    """Cluster-completion pass for incomplete clusters from Passes A/B.

    Finds singletons from missing supermarkets using a unanimous-vote rule:
    the candidate must score ≥ COMPLETION_TSR_MIN against EVERY existing
    cluster member and must not hard-conflict with any of them.  The lower
    individual threshold is safe because the unanimity requirement is strictly
    harder to satisfy than a single pairwise match.

    Returns a list of (avg_score, cluster_root, singleton_idx) tuples for
    the best-scoring candidate per (cluster, missing_sm), to be applied
    greedily in descending score order.
    """
    available = set(idx_to_sm.keys()) - used_idxs

    # Build index: (cat_norm, supermarket) → [product_idx, ...]
    sing_index: dict[tuple, list[int]] = {}
    for idx in available:
        row = idx_to_row.get(idx)
        if row is None:
            continue
        cat = row.get('cat_norm') if hasattr(row, 'get') else row['cat_norm']
        sm  = idx_to_sm[idx]
        if pd.notna(cat):
            sing_index.setdefault((str(cat), sm), []).append(idx)

    scored: list[tuple[float, int, int]] = []

    for root, members in new_clusters.items():
        if len(members) >= 4:
            continue

        present_sms  = {idx_to_sm[m] for m in members}
        missing_sms  = set(ALL_SMS) - present_sms

        m_rows = [idx_to_row[m] for m in members if m in idx_to_row]
        if not m_rows:
            continue

        def _get(row, col):
            return row[col] if not hasattr(row, 'get') else row.get(col)

        is_branded  = any(str(_get(r, 'product_type') or '') == 'branded' for r in m_rows)
        size_tol    = SIZE_TOL_BRAND if is_branded else SIZE_TOL_OB

        cat = next((str(_get(r, 'cat_norm')) for r in m_rows
                    if pd.notna(_get(r, 'cat_norm'))), None)
        if not cat:
            continue

        m_uvs   = [float(_get(r, 'unit_value'))
                   for r in m_rows
                   if pd.notna(_get(r, 'unit_value')) and float(_get(r, 'unit_value') or 0) > 0]
        m_names = [str(_get(r, 'normalized_name') or '') for r in m_rows]

        if not m_uvs:
            continue

        # Cluster brand / type constraints
        cluster_types = {str(_get(r, 'product_type') or '') for r in m_rows}
        cluster_branded = 'branded' in cluster_types
        cluster_brands: set[str] = set()
        if cluster_branded:
            for r in m_rows:
                b = _get(r, 'known_brand_clean')
                if b and pd.notna(b):
                    cluster_brands.add(_canon_brand(str(b)))

        for missing_sm in missing_sms:
            cand_idxs = sing_index.get((cat, missing_sm), [])
            best_score, best_idx = -1, None

            for cidx in cand_idxs:
                crow = idx_to_row.get(cidx)
                if crow is None:
                    continue

                # Type compatibility: don't mix branded and own-brand
                ctype = str(_get(crow, 'product_type') or '')
                if cluster_branded and ctype in ('own_brand', 'unbranded'):
                    continue
                if not cluster_branded and ctype == 'branded':
                    continue

                # Tier compatibility: candidate tier must be compatible with all members
                if not cluster_branded:
                    cand_tier = _get(crow, 'tier_type')
                    m_tiers = [_get(r, 'tier_type') for r in m_rows]
                    if not all(_tiers_compatible(cand_tier, mt) for mt in m_tiers):
                        continue

                # Brand compatibility: if cluster is branded, candidate must share brand
                if cluster_branded and cluster_brands:
                    cb = _get(crow, 'known_brand_clean')
                    if not cb or pd.isna(cb):
                        continue
                    if _canon_brand(str(cb)) not in cluster_brands:
                        continue

                cuv = float(_get(crow, 'unit_value') or 0)
                if cuv <= 0:
                    continue
                if any(_size_delta(cuv, muv) > size_tol for muv in m_uvs):
                    continue
                cname = str(_get(crow, 'normalized_name') or '')
                tsr_scores = [fuzz.token_set_ratio(cname, mn) for mn in m_names]
                if min(tsr_scores) < COMPLETION_TSR_MIN:
                    continue
                if any(check_hard_conflict(cname, mn) == 1 for mn in m_names):
                    continue
                avg = sum(tsr_scores) / len(tsr_scores)
                if avg > best_score:
                    best_score, best_idx = avg, cidx

            if best_idx is not None:
                scored.append((best_score, root, best_idx))

    return scored


def _apply_completions(
    clusters: dict[int, list[int]],
    completions: list[tuple[float, int, int]],
    idx_to_sm: dict[int, str],
) -> dict[int, list[int]]:
    """Greedily apply Pass C completions, one product per SM per cluster."""
    completions.sort(key=lambda x: -x[0])
    used: set[int] = set()
    cluster_sms: dict[int, set[str]] = {
        root: {idx_to_sm[m] for m in members}
        for root, members in clusters.items()
    }
    result = {root: list(members) for root, members in clusters.items()}

    for score, root, cidx in completions:
        if cidx in used:
            continue
        new_sm = idx_to_sm.get(cidx, '')
        if not new_sm or new_sm in cluster_sms.get(root, set()):
            continue
        if len(result.get(root, [])) >= 4:
            continue
        result.setdefault(root, []).append(cidx)
        cluster_sms[root].add(new_sm)
        used.add(cidx)

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def extend_coverage() -> None:
    # ── load existing clusters + all products
    print(f'Loading final clusters: {FINAL_CSV.name}')
    final = pd.read_csv(FINAL_CSV)
    print(f'  {len(final):,} rows, {final["ensemble_cluster_id"].nunique():,} clusters')
    eid_sizes = final.groupby('ensemble_cluster_id').size()
    for sz in (2, 3, 4):
        print(f'    {sz}-way: {int((eid_sizes == sz).sum()):,}')

    print(f'\nLoading all products: {ML_CSV.name}')
    all_prods = pd.read_csv(ML_CSV)
    matched_idxs = set(final['product_idx'].tolist())
    singletons = all_prods[~all_prods['product_idx'].isin(matched_idxs)].copy()
    print(f'  Singletons: {len(singletons):,} / {len(all_prods):,}')

    # ── build candidate pairs
    print('\nPass A — own-brand / unbranded matching...')
    pairs_a = _pass_a(singletons)
    print(f'  candidate pairs: {len(pairs_a):,}')

    print('Pass B — branded same-brand matching...')
    pairs_b = _pass_b(singletons)
    print(f'  candidate pairs: {len(pairs_b):,}')

    all_pairs = pairs_a + pairs_b
    print(f'  total pairs: {len(all_pairs):,}')

    # ── cluster the pairs
    idx_to_sm = {int(r['product_idx']): r['supermarket']
                 for _, r in singletons.iterrows()}
    clusters = _build_clusters(all_pairs, idx_to_sm)

    new_4way  = sum(1 for m in clusters.values() if len(m) == 4)
    new_3way  = sum(1 for m in clusters.values() if len(m) == 3)
    new_2way  = sum(1 for m in clusters.values() if len(m) == 2)
    print(f'\nAfter A+B: {len(clusters):,} clusters  '
          f'(4-way: {new_4way:,}, 3-way: {new_3way:,}, 2-way: {new_2way:,})')

    if not clusters:
        print('Nothing to add.')
        return

    # ── build idx_to_row lookup (needed by Pass C)
    idx_to_row = {int(r['product_idx']): r for _, r in all_prods.iterrows()}

    # ── Pass C: complete new incomplete clusters ───────────────────────────────
    used_idxs = {idx for members in clusters.values() for idx in members}
    print('\nPass C — cluster completion (unanimous TSR ≥ 70)...')
    completions = _pass_c(clusters, used_idxs, singletons, idx_to_sm, idx_to_row)
    print(f'  completions found: {len(completions):,}')

    clusters = _apply_completions(clusters, completions, idx_to_sm)

    # Update extended singletons lookup with Pass C additions
    c_used = {cidx for _, _, cidx in completions}
    for cidx in c_used:
        row = idx_to_row.get(cidx)
        if row is not None:
            idx_to_sm[cidx] = row['supermarket']

    post_4way = sum(1 for m in clusters.values() if len(m) == 4)
    post_3way = sum(1 for m in clusters.values() if len(m) == 3)
    post_2way = sum(1 for m in clusters.values() if len(m) == 2)
    post_total = sum(len(m) for m in clusters.values())
    print(f'After  C: {len(clusters):,} clusters  '
          f'(4-way: {post_4way:,}, 3-way: {post_3way:,}, 2-way: {post_2way:,})')
    print(f'Total new products: {post_total:,}')

    # ── build new rows dataframe
    max_eid = int(final['ensemble_cluster_id'].max())
    new_rows = []
    for k, (root, members) in enumerate(sorted(clusters.items())):
        new_eid = max_eid + 1 + k
        sz = len(members)
        for idx in members:
            row = idx_to_row[idx].copy()
            row['ensemble_cluster_id'] = new_eid
            row['cluster_size'] = sz
            new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    # ── back up and write
    print(f'\nBacking up → {BACKUP_CSV.name}')
    import shutil
    shutil.copy(FINAL_CSV, BACKUP_CSV)

    extended = pd.concat([final, new_df], ignore_index=True)
    eid_sizes_ext = extended.groupby('ensemble_cluster_id').size()
    print(f'\nExtended file: {len(extended):,} rows, '
          f'{eid_sizes_ext.shape[0]:,} clusters')
    for sz in (2, 3, 4):
        print(f'  {sz}-way: {int((eid_sizes_ext == sz).sum()):,}')

    extended.to_csv(FINAL_CSV, index=False)
    print(f'\nWrote: {FINAL_CSV}')


if __name__ == '__main__':
    extend_coverage()
