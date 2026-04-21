"""Fuse complementary 2-way (and 2-way↔3-way) clusters into 4-way via LLM.

Two incomplete clusters can be merged when:
  1. Their supermarket sets are disjoint (so a merge preserves one-per-SM).
  2. The merged size does not exceed 4.
  3. Both describe the same core product.

The ensemble's Kruskal pass does not connect such pairs when the cross-cluster
edges weren't present in either source pipeline.  Here we retrieve candidate
partners by fuzz similarity within the same category and complementary-SM
constraint, then let Claude Haiku judge whether to merge.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz

from shopwiser.paths import DATA_OUTPUTS

ALL_SMS: frozenset[str] = frozenset({'ASDA', 'Morrisons', 'Sains', 'Tesco'})
MODEL = 'claude-haiku-4-5-20251001'
MAX_PARTNERS = 4  # per anchor cluster
FUZZ_MIN = 60
SIZE_TOL = 0.15
MAX_WORKERS = 16

SYSTEM = (
    'You are a grocery matching expert. You decide whether two partial clusters '
    'from UK supermarkets describe the SAME core product and should be merged.'
)

INSTRUCTIONS = (
    'Rules:\n'
    '- SAME core product = same brand (if branded), same variety/flavour, same weight/volume, same pack count.\n'
    '- Own-brand across SMs is allowed IFF the core unbranded product is identical.\n'
    '- DIFFERENT variety/flavour/weight/pack count/brand → NOT the same.\n'
    '- When in doubt, answer NO.\n\n'
    'Respond with JSON only: {"merge": "YES" or "NO", "confidence": 0.0-1.0, "reason": "short"}'
)

_JSON_RE = re.compile(r'\{[^{}]*\}', re.S)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {'merge': 'NO', 'confidence': 0.0, 'reason': f'parse_error: {raw[:80]}'}


def _cluster_text(df: pd.DataFrame) -> str:
    lines = []
    for _, r in df.iterrows():
        brand = r.get('known_brand_clean') or r.get('known_brand') or ''
        brand_s = f' [{brand}]' if isinstance(brand, str) and brand else ''
        uv = r.get('unit_value')
        ut = r.get('unit_type')
        wt = f' — {uv:g}{ut}' if pd.notna(uv) and isinstance(ut, str) else ''
        lines.append(f'  {r["supermarket"]}: {r["names"]}{brand_s}{wt}')
    return '\n'.join(lines)


def _build_prompt(a_df: pd.DataFrame, b_df: pd.DataFrame) -> str:
    return (
        f'Cluster A ({len(a_df)} SMs):\n{_cluster_text(a_df)}\n\n'
        f'Cluster B ({len(b_df)} SMs):\n{_cluster_text(b_df)}\n\n'
        'Do Clusters A and B describe the SAME core product (and should merge into one 4-way)?\n'
        + INSTRUCTIONS
    )


def _call_llm(client, prompt: str, retries: int = 2) -> dict:
    last_err = ''
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=160,
                system=SYSTEM,
                messages=[{'role': 'user', 'content': prompt}],
            )
            return _parse_json(resp.content[0].text)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    return {'merge': 'NO', 'confidence': 0.0, 'reason': f'api_error: {last_err[:100]}'}


def _cluster_size_median(df: pd.DataFrame) -> float | None:
    vals = df['unit_value'].dropna()
    return float(vals.median()) if len(vals) else None


def _size_compat(a_sz: float | None, b_sz: float | None) -> bool:
    if a_sz is None or b_sz is None:
        return True
    if a_sz == 0 or b_sz == 0:
        return True
    return abs(a_sz - b_sz) / max(abs(a_sz), abs(b_sz)) <= SIZE_TOL


def _fuzz_between(a_df: pd.DataFrame, b_df: pd.DataFrame) -> int:
    a_names = a_df['normalized_name'].fillna('').tolist()
    b_names = b_df['normalized_name'].fillna('').tolist()
    best = 0
    for an in a_names:
        for bn in b_names:
            s = fuzz.token_set_ratio(an, bn)
            if s > best:
                best = s
    return best


def _iter_anchor_tasks(
    incomplete: pd.DataFrame,
    max_partners: int,
) -> list[tuple[int, int, pd.DataFrame, pd.DataFrame, int]]:
    """Generate (anchor_cid, partner_cid, a_df, b_df, fuzz) tasks.

    Only pairs with disjoint SMs, merged size ≤4, same category,
    size-compatible, and fuzz ≥ FUZZ_MIN are emitted.
    Deduped: (min, max) ordering.
    """
    tasks: list[tuple[int, int, pd.DataFrame, pd.DataFrame, int]] = []
    cluster_groups: dict[int, pd.DataFrame] = {
        int(cid): g for cid, g in incomplete.groupby('ensemble_cluster_id')
    }
    meta_rows = []
    for cid, g in cluster_groups.items():
        cat = g['category'].iloc[0] if g['category'].notna().any() else ''
        sms = frozenset(g['supermarket'].unique())
        meta_rows.append({
            'cid': cid,
            'category': cat,
            'sms': sms,
            'size': len(g),
            'unit_value': _cluster_size_median(g),
            'brand': (
                g['known_brand'].dropna().astype(str).str.lower().iloc[0]
                if g['known_brand'].dropna().size else ''
            ),
        })
    meta = pd.DataFrame(meta_rows).set_index('cid')

    print(f'  Building partner index over {len(meta):,} incomplete clusters...')
    # Group by category for speed.
    by_cat: dict[str, list[int]] = {}
    for cid, row in meta.iterrows():
        by_cat.setdefault(row['category'], []).append(int(cid))

    pair_seen: set[tuple[int, int]] = set()
    for cat, cids in by_cat.items():
        if len(cids) < 2:
            continue
        # Per-anchor: keep top-MAX_PARTNERS by fuzz among candidates.
        for anchor in cids:
            a = meta.loc[anchor]
            a_sms = a['sms']
            a_size = a['size']
            if a_size >= 4:
                continue
            a_df = cluster_groups[anchor]
            cands: list[tuple[int, int]] = []  # (fuzz, partner_cid)
            for partner in cids:
                if partner == anchor:
                    continue
                key = (min(anchor, partner), max(anchor, partner))
                if key in pair_seen:
                    continue
                b = meta.loc[partner]
                if a_size + b['size'] > 4:
                    continue
                if a_sms & b['sms']:
                    continue  # SM overlap breaks one-per-SM
                if not _size_compat(a['unit_value'], b['unit_value']):
                    continue
                # Fast brand pre-check — if both have distinct brands, skip.
                ab, bb = a['brand'], b['brand']
                if ab and bb and ab != bb:
                    continue
                b_df = cluster_groups[partner]
                fz = _fuzz_between(a_df, b_df)
                if fz < FUZZ_MIN:
                    continue
                cands.append((fz, partner))
            cands.sort(reverse=True)
            for fz, partner in cands[:max_partners]:
                key = (min(anchor, partner), max(anchor, partner))
                if key in pair_seen:
                    continue
                pair_seen.add(key)
                tasks.append((anchor, partner, cluster_groups[anchor], cluster_groups[partner], fz))
    return tasks


def merge_twoways(
    *,
    ensemble_csv: Path | None = None,
    out_dir: Path | None = None,
    max_partners: int = MAX_PARTNERS,
    max_workers: int = MAX_WORKERS,
    min_confidence: float = 0.70,
    dry_run_limit: int | None = None,
) -> dict:
    load_dotenv(override=True)
    if not os.getenv('ANTHROPIC_API_KEY'):
        raise RuntimeError('ANTHROPIC_API_KEY not set (after load_dotenv)')

    from anthropic import Anthropic

    client = Anthropic()
    ensemble_csv = ensemble_csv or (DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters_completed.csv')
    out_dir = out_dir or (DATA_OUTPUTS / 'ensemble')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading ensemble: {ensemble_csv}')
    ens = pd.read_csv(ensemble_csv)
    size_map = ens.groupby('ensemble_cluster_id').size()
    incomplete_cids = size_map[size_map < 4].index
    incomplete = ens[ens['ensemble_cluster_id'].isin(incomplete_cids)].copy()
    print(f'  Incomplete clusters: {len(incomplete_cids):,} ({len(incomplete):,} rows)')

    tasks = _iter_anchor_tasks(incomplete, max_partners=max_partners)
    if dry_run_limit:
        tasks = tasks[:dry_run_limit]
    print(f'  Merge tasks (fuzz ≥ {FUZZ_MIN}): {len(tasks):,}')

    if not tasks:
        print('No candidate pairs — writing input unchanged.')
        ens.to_csv(out_dir / 'ensemble_clusters_merged.csv', index=False)
        return {'merged_pairs': 0}

    results = []
    lock = threading.Lock()
    done = 0
    start = time.time()

    def _work(task):
        a_cid, b_cid, a_df, b_df, fz = task
        prompt = _build_prompt(a_df, b_df)
        parsed = _call_llm(client, prompt)
        merge = str(parsed.get('merge', 'NO')).strip().upper()
        conf = float(parsed.get('confidence', 0.0) or 0.0)
        reason = str(parsed.get('reason', ''))[:160]
        return {
            'a_cid': a_cid,
            'b_cid': b_cid,
            'a_size': len(a_df),
            'b_size': len(b_df),
            'fuzz': fz,
            'merge': merge,
            'confidence': conf,
            'reason': reason,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_work, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                results.append(r)
                done += 1
                if done % 200 == 0 or done == len(tasks):
                    rate = done / max(time.time() - start, 1e-3)
                    eta = (len(tasks) - done) / max(rate, 1e-3)
                    print(f'  [{done}/{len(tasks)}] {rate:.1f}/s  eta={eta/60:.1f}min')

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / 'merge_twoways_log.csv', index=False)

    # Union-find merges: accept in confidence-desc order; skip if either cluster
    # already merged with something else in this round.
    accepted = res_df[(res_df['merge'] == 'YES') & (res_df['confidence'] >= min_confidence)].copy()
    accepted['priority'] = (accepted['a_size'] + accepted['b_size']) * 10 + accepted['confidence']
    accepted = accepted.sort_values('priority', ascending=False)

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Track merged cluster SM sets to enforce one-per-SM after chaining.
    ens_sms = ens.groupby('ensemble_cluster_id')['supermarket'].agg(frozenset).to_dict()
    ens_sz = ens.groupby('ensemble_cluster_id').size().to_dict()
    root_sms: dict[int, frozenset] = {int(cid): ens_sms[cid] for cid in ens_sms}
    root_sz: dict[int, int] = {int(cid): int(v) for cid, v in ens_sz.items()}

    merged_pairs = 0
    for _, r in accepted.iterrows():
        a, b = int(r['a_cid']), int(r['b_cid'])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        merged_sms = root_sms[ra] | root_sms[rb]
        merged_sz = root_sz[ra] + root_sz[rb]
        if len(merged_sms) != merged_sz:
            continue  # SM collision after chain
        if merged_sz > 4:
            continue
        parent[rb] = ra
        root_sms[ra] = merged_sms
        root_sz[ra] = merged_sz
        merged_pairs += 1

    # Apply remapping to ens
    new_cid = ens['ensemble_cluster_id'].apply(lambda c: find(int(c)))
    ens2 = ens.copy()
    ens2['ensemble_cluster_id'] = new_cid.astype(int)
    # Re-normalize to contiguous cluster_ids (optional — keep roots for traceability)
    ens2['cluster_size'] = ens2['ensemble_cluster_id'].map(
        ens2.groupby('ensemble_cluster_id').size(),
    )
    ens2 = ens2.sort_values(['ensemble_cluster_id', 'supermarket']).reset_index(drop=True)

    out_path = out_dir / 'ensemble_clusters_merged.csv'
    ens2.to_csv(out_path, index=False)
    print(f'Saved {out_path}  ({len(ens2):,} rows)')

    size_dist = ens2.groupby('ensemble_cluster_id').size().value_counts().sort_index()
    stats = {
        'total_clusters': int(ens2['ensemble_cluster_id'].nunique()),
        '4_way': int((ens2.groupby('ensemble_cluster_id').size() == 4).sum()),
        '3_way': int((ens2.groupby('ensemble_cluster_id').size() == 3).sum()),
        '2_way': int((ens2.groupby('ensemble_cluster_id').size() == 2).sum()),
        'merged_pairs': int(merged_pairs),
        'llm_calls': int(len(res_df)),
        'yes_rate': float((res_df['merge'] == 'YES').mean()) if len(res_df) else 0.0,
    }
    print()
    print('=' * 60)
    print('Post-merge distribution:')
    for sz, n in size_dist.items():
        print(f'  {int(sz)}-way: {int(n):,}')
    print(f'  Total multi-product clusters: {stats["total_clusters"]:,}')
    print(f'  Merges applied:               {stats["merged_pairs"]:,}')
    print(f'  LLM calls:                    {stats["llm_calls"]:,}')
    print(f'  YES rate:                     {stats["yes_rate"]*100:.1f}%')
    print('=' * 60)
    return stats


if __name__ == '__main__':
    merge_twoways()
