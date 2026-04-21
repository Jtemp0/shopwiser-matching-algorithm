"""Rule-based cleanup + completion + 2-way merge — iterates until stable.

Phases per iteration:
  A. SPLIT — evict members that hard-conflict / brand-mismatch / size-mismatch.
  B. RECOMPLETE — for each incomplete cluster, score singletons from missing
     SMs deterministically. Accept top candidate if score ≥ ACCEPT_THRESHOLD
     and it passes all hard gates.
  C. MERGE — pair incomplete clusters (2-way↔2-way, 2-way↔3-way) whose SMs
     are disjoint and whose members all score compatibly.

Smart size comparison handles pack-count variants: compares BOTH the raw
unit_value AND the per-unit_value (unit_value / pack_quantity), accepting
whichever is tighter. Keeps "20x36g" and "40x72g" together where they are
the same core product.

Writes ``ensemble_clusters_final.csv`` + per-iteration counts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from shopwiser.ml_matching.features import check_hard_conflict
from shopwiser.paths import DATA_OUTPUTS, normalized_products_path

ALL_SMS = ('ASDA', 'Morrisons', 'Sains', 'Tesco')
SIZE_TOL = 0.15
ACCEPT_THRESHOLD = 60
FUZZ_MIN = 60
MERGE_THRESHOLD = 90  # stricter: merging touches more members
MAX_ITER = 1

IN_CSV = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_ml.csv'
OUT_CSV = DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_ml_fuzz.csv'
LOG_CSV = DATA_OUTPUTS / 'ensemble' / 'rescore_rb_log.csv'


def _brand_token(b: object) -> str:
    if not isinstance(b, str):
        return ''
    b = b.strip().lower()
    return b.split()[0] if b else ''


def _norm_brand_set(series: pd.Series) -> set[str]:
    toks = {_brand_token(b) for b in series.dropna()}
    toks.discard('')
    return toks


def _per_unit(uv: float, pq: float) -> float:
    if pd.isna(uv):
        return np.nan
    if pd.isna(pq) or pq == 0:
        return float(uv)
    return float(uv) / float(pq)


def _smart_size_delta(uv_a: float, pq_a: float, uv_b: float, pq_b: float) -> float:
    """Delta under both raw and per-unit interpretations (take min)."""
    if pd.isna(uv_a) or pd.isna(uv_b):
        return 0.0

    def _rd(x: float, y: float) -> float:
        hi = max(abs(x), abs(y))
        return 0.0 if hi < 1e-6 else abs(x - y) / hi

    # Per-unit vs per-unit (treats missing pq as 1)
    pu_a = _per_unit(uv_a, pq_a)
    pu_b = _per_unit(uv_b, pq_b)

    raw = _rd(float(uv_a), float(uv_b))
    per = _rd(pu_a, pu_b)
    # Also handle "A stores total, B stores per-unit" one-sided-multipack case
    cross_1 = _rd(float(uv_a), pu_b)
    cross_2 = _rd(pu_a, float(uv_b))
    return min(raw, per, cross_1, cross_2)


def _size_ok_pair(a: pd.Series, b: pd.Series, tol: float = SIZE_TOL) -> bool:
    return _smart_size_delta(
        a.get('unit_value'), a.get('pack_quantity'),
        b.get('unit_value'), b.get('pack_quantity'),
    ) <= tol


def _evict_brand_minority(g: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Rule 1: ≥2 distinct known brands → keep dominant.

    Rule 2 (drift check): if a cluster has ≥2 branded members agreeing on a
    dominant brand, evict any OTHER member (unbranded / own_brand) whose
    normalized_name doesn't contain the dominant-brand prefix. This catches
    the 'Jack Daniel's + Sainsbury's Apple Juice' drift without being
    overeager on single-branded clusters where name normalization may drop
    accents (Nescafé / nescafe).
    """
    toks = [_brand_token(b) for b in g['known_brand_clean']]
    names = g['normalized_name'].fillna('').astype(str).tolist()
    counts = Counter(t for t in toks if t)

    if not counts:
        return g, []

    dominant, dom_count = counts.most_common(1)[0]
    keep_mask = [t == dominant or t == '' for t in toks]

    # Rule 2 only when ≥2 members confirm the brand (avoids single-branded
    # clusters where accent stripping may hide the token in the normalized name)
    if dom_count >= 2 and len(dominant) >= 4:
        for i, (km, nm, tk) in enumerate(zip(keep_mask, names, toks)):
            if not km or tk == dominant:
                continue
            # Unbranded/own_brand row: require some fragment of the dominant
            # brand (first 4 chars) to appear in the name
            prefix = dominant[:4]
            if prefix not in nm:
                keep_mask[i] = False

    kept = g.loc[keep_mask]
    evicted = g.loc[~np.array(keep_mask), 'product_idx'].astype(int).tolist()
    return kept, evicted


def _evict_size_outliers(g: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Evict members whose smart_size_delta to the cluster median is >SIZE_TOL."""
    if len(g) < 2:
        return g, []
    rows = list(g.itertuples(index=False))
    # Use member-to-member smart delta to find true outliers. A member is an
    # outlier if it disagrees with >50% of the cluster.
    uvs = g['unit_value'].tolist()
    pqs = g['pack_quantity'].tolist() if 'pack_quantity' in g.columns else [np.nan] * len(g)
    n = len(g)
    disagree = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = _smart_size_delta(uvs[i], pqs[i], uvs[j], pqs[j])
            if d > SIZE_TOL:
                disagree[i] += 1
                disagree[j] += 1
    keep = [d < (n - 1) / 2 + 1e-6 for d in disagree]  # tolerate ties
    # If everyone disagrees with everyone, keep the median-size row
    if not any(keep):
        med = float(pd.Series(uvs).median()) if pd.notna(pd.Series(uvs).median()) else 0
        # keep row closest to median
        diffs = [abs((x or 0) - med) for x in uvs]
        keep_idx = int(np.argmin(diffs))
        keep = [i == keep_idx for i in range(n)]
    kept = g.iloc[keep]
    evicted = g.iloc[[not k for k in keep]]['product_idx'].astype(int).tolist()
    return kept, evicted


def _evict_hard_conflicts(g: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    names = g['normalized_name'].fillna('').astype(str).tolist()
    idxs = g['product_idx'].astype(int).tolist()
    if len(names) < 2:
        return g, []
    evicted: list[int] = []
    active = list(range(len(names)))
    while True:
        conflict_counts = [0] * len(names)
        has_conflict = False
        for i in active:
            for j in active:
                if j <= i:
                    continue
                if check_hard_conflict(names[i], names[j]) == 1:
                    conflict_counts[i] += 1
                    conflict_counts[j] += 1
                    has_conflict = True
        if not has_conflict:
            break
        worst = max(active, key=lambda k: (conflict_counts[k], -idxs[k]))
        active.remove(worst)
        evicted.append(idxs[worst])
        if len(active) < 2:
            break
    kept = g[g['product_idx'].astype(int).isin({idxs[a] for a in active})]
    return kept, evicted


def phase_a_split(df: pd.DataFrame, *, first_pass: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Split flagged clusters.

    first_pass=True (run on the original R3 CSV): only evict hard-conflict
    members — the LLM already validated brand/size so we trust those edges.

    Subsequent passes: also evict brand minority and size outliers, since new
    additions were placed by the rule-based scorer (not LLM).
    """
    rows_out: list[pd.DataFrame] = []
    log: list[dict] = []
    for cid, g in df.groupby('ensemble_cluster_id'):
        original_size = len(g)
        cur = g
        # Hard-conflict is reliable — always evict
        cur, ev_conf = _evict_hard_conflicts(cur) if len(cur) >= 2 else (cur, [])

        ev_brand: list[int] = []
        ev_size: list[int] = []
        if not first_pass and len(cur) >= 2:
            cur, ev_brand = _evict_brand_minority(cur)
            cur, ev_size = _evict_size_outliers(cur) if len(cur) >= 2 else (cur, [])

        if ev_brand or ev_size or ev_conf:
            log.append({
                'ensemble_cluster_id': int(cid),
                'original_size': original_size,
                'kept': len(cur),
                'evicted_brand': len(ev_brand),
                'evicted_size': len(ev_size),
                'evicted_conflict': len(ev_conf),
            })

        if len(cur) >= 2:
            rows_out.append(cur)

    cleaned = pd.concat(rows_out, ignore_index=True) if rows_out else df.iloc[0:0]
    return cleaned, log


def _score_candidate(cluster_df: pd.DataFrame, cand: pd.Series) -> int:
    """Mixed gate/penalty scoring.

    Hard gates (return -1000):
      - semantic hard-conflict (FLAVOR / ONE_SIDED)
      - category mismatch
      - brand CONFLICT: cluster has brand X, candidate has a DIFFERENT brand Y
      - size: ALL members disagree with candidate by >SIZE_TOL

    Soft penalties/bonuses (adjust score):
      - brand match bonus
      - size quality bonus
    """
    cluster_names = cluster_df['normalized_name'].fillna('').astype(str).tolist()
    cn = str(cand.get('normalized_name') or '')
    if not cn or not cluster_names:
        return -1_000

    # Hard-conflict gate
    for nm in cluster_names:
        if check_hard_conflict(nm, cn) == 1:
            return -1_000

    # Category gate
    cl_cats = set(cluster_df['category'].dropna().astype(str).unique())
    cc = cand.get('category')
    if cl_cats and (not isinstance(cc, str) or cc not in cl_cats):
        return -1_000

    # Brand conflict gate: only reject if candidate has a DIFFERENT known brand
    cl_brands = _norm_brand_set(cluster_df['known_brand_clean'])
    cb = _brand_token(cand.get('known_brand_clean'))
    cn_toks = set(cn.split())
    brand_conflict = bool(cl_brands) and bool(cb) and cb not in cl_brands and not (cl_brands & cn_toks)
    if brand_conflict:
        return -1_000

    # Size gate: reject only when ALL members disagree
    deltas = []
    for _, m in cluster_df.iterrows():
        if pd.notna(m.get('unit_value')) and pd.notna(cand.get('unit_value')):
            deltas.append(_smart_size_delta(
                m.get('unit_value'), m.get('pack_quantity'),
                cand.get('unit_value'), cand.get('pack_quantity'),
            ))
    if deltas and min(deltas) > SIZE_TOL:
        return -1_000

    score = max(fuzz.token_set_ratio(cn, x) for x in cluster_names)
    if score < FUZZ_MIN:
        return score

    # Bonuses
    if cl_brands and cb in cl_brands:
        score += 10
    elif cl_brands and bool(cl_brands & cn_toks):
        score += 5  # brand in name but not in known_brand field
    if deltas and min(deltas) <= SIZE_TOL / 2:
        score += 5
    return int(score)


def phase_b_recomplete(
    clusters: pd.DataFrame,
    singles_by_sm: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    proposals: list[tuple[int, int, str, int, int]] = []
    for cid, g in clusters.groupby('ensemble_cluster_id'):
        present = set(g['supermarket'])
        missing = [sm for sm in ALL_SMS if sm not in present]
        if not missing:
            continue
        cl_cats = set(g['category'].dropna().astype(str).unique())

        for sm in missing:
            pool = singles_by_sm.get(sm)
            if pool is None or pool.empty:
                continue
            pf = pool[pool['category'].isin(cl_cats)] if cl_cats else pool

            if pf.empty:
                continue
            cn_list = g['normalized_name'].fillna('').astype(str).tolist()

            def _fuzz(nm: str, cn_list=cn_list) -> int:
                nm = nm or ''
                return max(fuzz.token_set_ratio(nm, c) for c in cn_list)

            pf = pf.assign(_f=pf['normalized_name'].fillna('').map(_fuzz))
            pf = pf[pf['_f'] >= FUZZ_MIN].sort_values('_f', ascending=False).head(10)
            if pf.empty:
                continue

            best: tuple[int, int] | None = None
            for _, cand in pf.iterrows():
                s = _score_candidate(g, cand)
                if s >= ACCEPT_THRESHOLD and (best is None or s > best[0]):
                    best = (s, int(cand['product_idx']))
            if best:
                proposals.append((best[0], int(cid), sm, best[1], len(g)))

    proposals.sort(key=lambda p: (-p[0], -p[4]))
    claimed_idx: set[int] = set()
    claimed_cs: set[tuple[int, str]] = set()
    accepted: list[dict] = []
    for score, cid, sm, pidx, csize in proposals:
        if pidx in claimed_idx or (cid, sm) in claimed_cs:
            continue
        claimed_idx.add(pidx)
        claimed_cs.add((cid, sm))
        accepted.append({
            'ensemble_cluster_id': cid,
            'product_idx': pidx,
            'supermarket': sm,
            'score': score,
        })
    return pd.DataFrame(accepted)


def _clusters_compat(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, int]:
    """Can clusters a and b be merged? Returns (ok, min_pair_score)."""
    if set(a['supermarket']) & set(b['supermarket']):
        return False, 0
    if len(a) + len(b) > 4:
        return False, 0

    # Category match
    ca = set(a['category'].dropna().astype(str).unique())
    cb = set(b['category'].dropna().astype(str).unique())
    if ca and cb and not (ca & cb):
        return False, 0

    # Brand compat
    ba = _norm_brand_set(a['known_brand_clean'])
    bb = _norm_brand_set(b['known_brand_clean'])
    if ba and bb and not (ba & bb):
        return False, 0

    # Hard-conflict between any pair
    names_a = a['normalized_name'].fillna('').astype(str).tolist()
    names_b = b['normalized_name'].fillna('').astype(str).tolist()
    for na in names_a:
        for nb in names_b:
            if check_hard_conflict(na, nb) == 1:
                return False, 0

    # Size compat: all pairs within tolerance under smart delta
    rows_a = list(a.itertuples(index=False))
    rows_b = list(b.itertuples(index=False))
    for ra in rows_a:
        for rb in rows_b:
            uv_a = getattr(ra, 'unit_value', np.nan)
            pq_a = getattr(ra, 'pack_quantity', np.nan)
            uv_b = getattr(rb, 'unit_value', np.nan)
            pq_b = getattr(rb, 'pack_quantity', np.nan)
            if pd.notna(uv_a) and pd.notna(uv_b):
                if _smart_size_delta(uv_a, pq_a, uv_b, pq_b) > SIZE_TOL:
                    return False, 0

    # Min fuzz across cross-pairs
    min_fuzz = min(
        fuzz.token_set_ratio(na, nb)
        for na in names_a for nb in names_b
    )
    return min_fuzz >= MERGE_THRESHOLD, min_fuzz


def phase_c_merge(clusters: pd.DataFrame) -> list[tuple[int, int, int]]:
    """Find compatible (cid_small, cid_big, score) merges. Greedy by score."""
    incomplete = {
        cid: g for cid, g in clusters.groupby('ensemble_cluster_id') if len(g) < 4
    }
    # Index clusters by (category, brand_token) to narrow candidate pairs
    bucket: dict[tuple[str, str], list[int]] = {}
    for cid, g in incomplete.items():
        cats = list(g['category'].dropna().astype(str).unique()) or ['']
        brands = _norm_brand_set(g['known_brand_clean'])
        brand_keys = list(brands) if brands else ['']
        for c in cats:
            for b in brand_keys:
                bucket.setdefault((c, b), []).append(int(cid))

    proposals: list[tuple[int, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for cids in bucket.values():
        for i, ca_id in enumerate(cids):
            for cb_id in cids[i + 1:]:
                pair = (min(ca_id, cb_id), max(ca_id, cb_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                a = incomplete[ca_id]
                b = incomplete[cb_id]
                ok, score = _clusters_compat(a, b)
                if ok:
                    proposals.append((score, pair[0], pair[1]))

    proposals.sort(key=lambda p: -p[0])
    claimed: set[int] = set()
    accepted: list[tuple[int, int, int]] = []
    for score, ca_id, cb_id in proposals:
        if ca_id in claimed or cb_id in claimed:
            continue
        claimed.add(ca_id)
        claimed.add(cb_id)
        accepted.append((ca_id, cb_id, score))
    return accepted


def _apply_merges(clusters: pd.DataFrame, merges: list[tuple[int, int, int]]) -> pd.DataFrame:
    if not merges:
        return clusters
    remap = {}
    for ca, cb, _ in merges:
        # Merge cb into ca (lower id wins)
        remap[cb] = ca
    clusters = clusters.copy()
    clusters['ensemble_cluster_id'] = clusters['ensemble_cluster_id'].replace(remap)
    return clusters


def _recompute_cluster_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['cluster_size'] = df['ensemble_cluster_id'].map(
        df.groupby('ensemble_cluster_id').size(),
    )
    return df


def _summary(df: pd.DataFrame, label: str) -> dict:
    sizes = df.groupby('ensemble_cluster_id').size()
    s = {
        'label': label,
        'rows': len(df),
        'clusters': int(sizes.shape[0]),
        '4way': int((sizes == 4).sum()),
        '3way': int((sizes == 3).sum()),
        '2way': int((sizes == 2).sum()),
    }
    print(f'  [{label}] clusters={s["clusters"]:,} | 4w={s["4way"]:,} 3w={s["3way"]:,} 2w={s["2way"]:,}')
    return s


def main() -> None:
    print(f'Loading: {IN_CSV}')
    df = pd.read_csv(IN_CSV)
    allp = pd.read_csv(normalized_products_path(sample=False))
    allp['product_idx'] = allp.index
    print(f'  rows={len(df):,} clusters={df["ensemble_cluster_id"].nunique():,}')
    _summary(df, 'baseline')

    history = [_summary(df, 'R3 input')]
    split_log_all: list[dict] = []
    prev_4way = -1
    for it in range(1, MAX_ITER + 1):
        print(f'\n=== Iteration {it} ===')
        # A. Split — skip on it1 to preserve all LLM-validated R3 edges;
        #    apply on subsequent iterations to clean rule-based additions only.
        if it == 1:
            log_a = []
            history.append(_summary(df, f'it{it} split (skipped)'))
        else:
            df, log_a = phase_a_split(df, first_pass=False)
            split_log_all.extend([{**e, 'iter': it} for e in log_a])
            history.append(_summary(df, f'it{it} split'))

        # B. Recomplete
        clustered = set(df['product_idx'].astype(int))
        singles = allp[~allp['product_idx'].isin(clustered)]
        singles_by_sm = {sm: g for sm, g in singles.groupby('supermarket')}

        adds = phase_b_recomplete(df, singles_by_sm)
        if len(adds):
            allp_idxed = allp.set_index('product_idx', drop=False)
            extra = allp_idxed.loc[adds['product_idx'].values].copy()
            extra['ensemble_cluster_id'] = adds['ensemble_cluster_id'].values
            for c in df.columns:
                if c not in extra.columns and c != 'ensemble_cluster_id':
                    extra[c] = pd.NA
            extra = extra[df.columns]
            df = pd.concat([df, extra], ignore_index=True)
        df = _recompute_cluster_size(df)
        history.append(_summary(df, f'it{it} recomplete (+{len(adds):,})'))

        # C. Merge
        merges = phase_c_merge(df)
        if merges:
            df = _apply_merges(df, merges)
            df = _recompute_cluster_size(df)
        history.append(_summary(df, f'it{it} merge (+{len(merges):,} pairs)'))

        cur_4way = int((df.groupby('ensemble_cluster_id').size() == 4).sum())
        if cur_4way == prev_4way:
            print(f'  converged at 4-way={cur_4way:,}')
            break
        prev_4way = cur_4way

    df = df.sort_values(['ensemble_cluster_id', 'supermarket']).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    pd.DataFrame(split_log_all).to_csv(LOG_CSV, index=False)

    print('\n=== History ===')
    for h in history:
        print(f'  {h["label"]:30s} 4w={h["4way"]:,} 3w={h["3way"]:,} 2w={h["2way"]:,} total={h["clusters"]:,}')
    print(f'\nFinal: {OUT_CSV}')
    print(f'Log:   {LOG_CSV}')


if __name__ == '__main__':
    main()
