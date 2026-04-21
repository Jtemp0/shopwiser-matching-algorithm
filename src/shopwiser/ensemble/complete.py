"""LLM-gated completion: promote 2/3-way clusters to 4-way.

For each incomplete cluster:
  1. Identify missing SMs (of {ASDA, Morrisons, Sains, Tesco}).
  2. For each missing SM, pull candidate singletons that clear hard gates
     (same category, size within tolerance, brand compatible).
  3. Rank by fuzzy name similarity to any cluster member; keep the top-K.
  4. Ask Claude Haiku to pick the matching candidate (or NONE).
  5. Resolve claims: larger clusters get priority, then confidence.

Adds edges only where the LLM responds with a concrete letter choice.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz

from shopwiser.paths import DATA_OUTPUTS, normalized_products_path

ALL_SMS: set[str] = {'ASDA', 'Morrisons', 'Sains', 'Tesco'}
MODEL = 'claude-haiku-4-5-20251001'
MAX_CANDIDATES = 5
SIZE_TOL = 0.15
FUZZ_MIN = 55
MAX_WORKERS = 12
LETTERS = 'ABCDEFGHIJKLMNOP'

SYSTEM = (
    'You are a grocery matching expert. You decide whether a candidate product '
    'represents the SAME core product as a cluster of listings from UK supermarkets.'
)

INSTRUCTIONS = (
    'Rules:\n'
    '- SAME core product = same brand (if branded), same variety/flavour, same weight/volume, same pack count.\n'
    '- DIFFERENT variety, flavour, weight, pack count, or brand → NOT the same.\n'
    '- Own-brand of different SMs is allowed IFF the core unbranded product is identical.\n'
    '- When in doubt, respond NONE.\n\n'
    'Respond with JSON only (no markdown, no prose outside JSON):\n'
    '{"match": "<letter>" or "NONE", "confidence": 0.0-1.0, "reason": "short"}'
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
    return {'match': 'NONE', 'confidence': 0.0, 'reason': f'parse_error: {raw[:80]}'}


def _size_ok(cluster_sz: float | None, cand_sz: float | None) -> bool:
    if cluster_sz is None or cand_sz is None or pd.isna(cluster_sz) or pd.isna(cand_sz):
        return True  # admit when missing
    if cluster_sz == 0 or cand_sz == 0:
        return True
    delta = abs(cluster_sz - cand_sz) / max(abs(cluster_sz), abs(cand_sz))
    return delta <= SIZE_TOL


def _brand_ok(cluster_brands: set[str], cand_brand: str | None) -> bool:
    if not cluster_brands:
        return True
    if cand_brand is None or not isinstance(cand_brand, str) or cand_brand == '':
        return True
    return cand_brand.lower() in cluster_brands


def _prefilter(cluster_df: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    cats = set(cluster_df['category'].dropna().unique())
    if cats:
        cands = cands[cands['category'].isin(cats)]
    if cands.empty:
        return cands

    sz_vals = cluster_df['unit_value'].dropna()
    cluster_sz = float(sz_vals.median()) if len(sz_vals) else None
    if cluster_sz is not None:
        cands = cands[cands['unit_value'].isna() | cands['unit_value'].apply(
            lambda v: _size_ok(cluster_sz, v),
        )]
    if cands.empty:
        return cands

    cluster_brands = {
        b.lower() for b in cluster_df['known_brand'].dropna().astype(str).unique() if b
    }
    cands = cands[cands['known_brand'].apply(lambda b: _brand_ok(cluster_brands, b))]
    return cands


def _rank(cluster_df: pd.DataFrame, cands: pd.DataFrame, k: int) -> pd.DataFrame:
    cluster_names = cluster_df['normalized_name'].fillna('').tolist()
    if not cluster_names or cands.empty:
        return cands.iloc[0:0]

    def _score(nm: str) -> int:
        nm = nm or ''
        return max(fuzz.token_set_ratio(nm, cn) for cn in cluster_names)

    cands = cands.assign(_fuzz=cands['normalized_name'].fillna('').map(_score))
    cands = cands[cands['_fuzz'] >= FUZZ_MIN].sort_values('_fuzz', ascending=False)
    return cands.head(k)


def _build_prompt(cluster_df: pd.DataFrame, cands: pd.DataFrame, sm: str) -> str:
    cluster_lines = []
    for _, r in cluster_df.iterrows():
        brand = r.get('known_brand_clean') or r.get('known_brand') or ''
        brand_s = f' [{brand}]' if isinstance(brand, str) and brand else ''
        uv = r.get('unit_value')
        ut = r.get('unit_type')
        wt = f' — {uv:g}{ut}' if pd.notna(uv) and isinstance(ut, str) else ''
        cluster_lines.append(f'  {r["supermarket"]}: {r["names"]}{brand_s}{wt}')
    cand_lines = []
    for i, (_, r) in enumerate(cands.iterrows()):
        brand = r.get('known_brand') or ''
        brand_s = f' [{brand}]' if isinstance(brand, str) and brand else ''
        uv = r.get('unit_value')
        ut = r.get('unit_type')
        wt = f' — {uv:g}{ut}' if pd.notna(uv) and isinstance(ut, str) else ''
        cand_lines.append(f'  {LETTERS[i]}. {r["names"]}{brand_s}{wt}')
    return (
        f'Cluster (same core product, {len(cluster_df)} SMs present):\n'
        + '\n'.join(cluster_lines)
        + f'\n\nCandidates from {sm}:\n'
        + '\n'.join(cand_lines)
        + f'\n\nWhich candidate (A-{LETTERS[len(cands) - 1]}) is the SAME core product as the cluster?\n'
        + INSTRUCTIONS
    )


def _call_llm(client, prompt: str, retries: int = 6) -> dict:
    last_err = ''
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=160,
                system=SYSTEM,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = resp.content[0].text
            return _parse_json(raw)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            msg = str(e)
            # Exponential backoff with jitter.  Longer for rate limits.
            if '429' in msg or 'rate_limit' in msg or 'overloaded' in msg.lower():
                delay = min(60.0, 2.0 ** attempt + (attempt * 0.5))
            else:
                delay = min(8.0, 0.8 * (attempt + 1))
            time.sleep(delay)
    return {'match': 'NONE', 'confidence': 0.0, 'reason': f'api_error: {last_err[:100]}'}


def _iter_tasks(
    ens: pd.DataFrame,
    singles_by_sm: dict[str, pd.DataFrame],
) -> list[tuple[int, str, pd.DataFrame, pd.DataFrame]]:
    tasks: list[tuple[int, str, pd.DataFrame, pd.DataFrame]] = []
    for cid, cluster_df in ens.groupby('ensemble_cluster_id'):
        present_sms = set(cluster_df['supermarket'])
        missing = ALL_SMS - present_sms
        if not missing:
            continue
        for sm in missing:
            pool = singles_by_sm.get(sm)
            if pool is None or pool.empty:
                continue
            filt = _prefilter(cluster_df, pool)
            if filt.empty:
                continue
            ranked = _rank(cluster_df, filt, MAX_CANDIDATES)
            if ranked.empty:
                continue
            tasks.append((int(cid), sm, cluster_df, ranked))
    return tasks


def complete_clusters(
    *,
    ensemble_csv: Path | None = None,
    normalised_csv: Path | None = None,
    out_dir: Path | None = None,
    max_workers: int = MAX_WORKERS,
    min_confidence: float = 0.70,
    dry_run_limit: int | None = None,
    fuzz_min: int = FUZZ_MIN,
    size_tol: float = SIZE_TOL,
    max_candidates: int = MAX_CANDIDATES,
    output_name: str = 'ensemble_clusters_completed.csv',
    log_name: str = 'completion_llm_log.csv',
    additions_name: str = 'completion_additions.csv',
) -> dict:
    global FUZZ_MIN, SIZE_TOL, MAX_CANDIDATES
    FUZZ_MIN = fuzz_min
    SIZE_TOL = size_tol
    MAX_CANDIDATES = max_candidates
    load_dotenv(override=True)
    if not os.getenv('ANTHROPIC_API_KEY'):
        raise RuntimeError('ANTHROPIC_API_KEY not set (after load_dotenv)')

    from anthropic import Anthropic

    client = Anthropic()
    ensemble_csv = ensemble_csv or (DATA_OUTPUTS / 'ensemble' / 'ensemble_clusters.csv')
    normalised_csv = normalised_csv or normalized_products_path(sample=False)
    out_dir = out_dir or (DATA_OUTPUTS / 'ensemble')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading ensemble:   {ensemble_csv}')
    ens = pd.read_csv(ensemble_csv)
    print(f'Loading normalised: {normalised_csv}')
    allp = pd.read_csv(normalised_csv)
    allp['product_idx'] = allp.index

    clustered_idx = set(ens['product_idx'].astype(int))
    singles = allp[~allp['product_idx'].isin(clustered_idx)].copy()
    print(f'  Clustered products: {len(clustered_idx):,}')
    print(f'  Singleton pool:     {len(singles):,}')

    singles_by_sm = {sm: g for sm, g in singles.groupby('supermarket')}

    print('Building candidate tasks...')
    tasks = _iter_tasks(ens, singles_by_sm)
    if dry_run_limit:
        tasks = tasks[:dry_run_limit]
    print(f'  LLM tasks: {len(tasks):,}')

    # Parallel LLM validation
    results: list[dict] = []
    lock = threading.Lock()
    done = 0
    start = time.time()

    def _work(task):
        cid, sm, cluster_df, cands = task
        prompt = _build_prompt(cluster_df, cands, sm)
        parsed = _call_llm(client, prompt)
        match = str(parsed.get('match', 'NONE')).strip().upper()
        conf = float(parsed.get('confidence', 0.0) or 0.0)
        reason = str(parsed.get('reason', ''))[:160]

        picked_idx: int | None = None
        if match != 'NONE' and len(match) == 1 and match in LETTERS[: len(cands)]:
            row_i = LETTERS.index(match)
            picked_idx = int(cands.iloc[row_i]['product_idx'])

        return {
            'ensemble_cluster_id': cid,
            'cluster_size': len(cluster_df),
            'missing_sm': sm,
            'match_letter': match,
            'picked_idx': picked_idx,
            'confidence': conf,
            'reason': reason,
            'n_candidates': len(cands),
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
    res_df.to_csv(out_dir / log_name, index=False)

    # Resolve claims: larger clusters first, then higher confidence.
    # This preserves the already-strong 3-way edges before we try 2-ways.
    accepted = res_df[
        res_df['picked_idx'].notna()
        & (res_df['confidence'] >= min_confidence)
    ].copy()
    accepted['priority'] = accepted['cluster_size'] * 10 + accepted['confidence']
    accepted = accepted.sort_values('priority', ascending=False)

    claimed_idx: set[int] = set()
    claimed_cluster_sm: set[tuple[int, str]] = set()
    to_add: list[dict] = []
    for _, r in accepted.iterrows():
        idx = int(r['picked_idx'])
        cid = int(r['ensemble_cluster_id'])
        sm = r['missing_sm']
        if idx in claimed_idx:
            continue
        if (cid, sm) in claimed_cluster_sm:
            continue
        claimed_idx.add(idx)
        claimed_cluster_sm.add((cid, sm))
        to_add.append({
            'ensemble_cluster_id': cid,
            'product_idx': idx,
            'supermarket': sm,
            'source': 'llm_completion',
            'confidence': float(r['confidence']),
            'reason': r['reason'],
        })

    add_df = pd.DataFrame(to_add)
    add_df.to_csv(out_dir / additions_name, index=False)
    print(f'\nAdditions accepted: {len(add_df):,} products')

    # Rebuild ensemble clusters with additions
    new_rows = []
    if not add_df.empty:
        allp_idxed = allp.set_index('product_idx', drop=False)
        extra = allp_idxed.loc[add_df['product_idx'].values].copy()
        extra['ensemble_cluster_id'] = add_df['ensemble_cluster_id'].values
        # Align columns with ens
        for c in ens.columns:
            if c not in extra.columns and c != 'ensemble_cluster_id':
                extra[c] = pd.NA
        extra = extra[ens.columns]
        new_rows.append(extra)

    merged = pd.concat([ens] + new_rows, ignore_index=True)
    # Recompute cluster_size
    merged['cluster_size'] = merged['ensemble_cluster_id'].map(
        merged.groupby('ensemble_cluster_id').size(),
    )
    merged = merged.sort_values(['ensemble_cluster_id', 'supermarket']).reset_index(drop=True)

    out_clusters = out_dir / output_name
    merged.to_csv(out_clusters, index=False)
    print(f'Saved {out_clusters}  ({len(merged):,} rows)')

    # Stats
    size_dist = merged.groupby('ensemble_cluster_id').size().value_counts().sort_index()
    stats = {
        'total_clusters': int(merged['ensemble_cluster_id'].nunique()),
        '4_way': int((merged.groupby('ensemble_cluster_id').size() == 4).sum()),
        '3_way': int((merged.groupby('ensemble_cluster_id').size() == 3).sum()),
        '2_way': int((merged.groupby('ensemble_cluster_id').size() == 2).sum()),
        'products_matched': len(merged),
        'llm_calls': len(res_df),
        'additions': len(add_df),
    }
    print()
    print('=' * 60)
    print('Post-completion distribution:')
    for sz, n in size_dist.items():
        print(f'  {int(sz)}-way: {int(n):,}')
    print(f'  Total multi-product clusters: {stats["total_clusters"]:,}')
    print(f'  Total products matched:       {stats["products_matched"]:,}')
    print(f'  LLM calls:                    {stats["llm_calls"]:,}')
    print(f'  Additions accepted:           {stats["additions"]:,}')
    print('=' * 60)

    return stats


if __name__ == '__main__':
    complete_clusters()
