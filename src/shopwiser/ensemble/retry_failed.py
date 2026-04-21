"""Re-run only the rate-limited or parse-error tasks from a prior completion log.

Reads ``completion_r2_log.csv`` (or any completion log), pulls rows with
``api_error``/``parse_error`` reason, reconstructs each task from the ensemble
CSV + normalised products, and retries with Claude Haiku using a lower
concurrency and longer backoff.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from shopwiser.ensemble import complete as C
from shopwiser.paths import DATA_OUTPUTS, normalized_products_path

ALL_SMS = frozenset({'ASDA', 'Morrisons', 'Sains', 'Tesco'})


def retry_failed(
    *,
    prior_log: Path | None = None,
    ensemble_csv: Path | None = None,
    out_dir: Path | None = None,
    workers: int = 6,
    min_confidence: float = 0.65,
    fuzz_min: int = 42,
    size_tol: float = 0.25,
    max_candidates: int = 8,
) -> dict:
    load_dotenv(override=True)
    if not os.getenv('ANTHROPIC_API_KEY'):
        raise RuntimeError('ANTHROPIC_API_KEY not set')

    from anthropic import Anthropic

    client = Anthropic()
    out_dir = out_dir or (DATA_OUTPUTS / 'ensemble')
    prior_log = prior_log or (out_dir / 'completion_r2_log.csv')
    ensemble_csv = ensemble_csv or (out_dir / 'ensemble_clusters_r2.csv')

    # Push wider params into module globals (used by _prefilter/_rank).
    C.FUZZ_MIN = fuzz_min
    C.SIZE_TOL = size_tol
    C.MAX_CANDIDATES = max_candidates

    print(f'Prior log:   {prior_log}')
    print(f'Ensemble:    {ensemble_csv}')
    log = pd.read_csv(prior_log)
    failed = log[log['reason'].str.contains('api_error|parse_error', na=False)].copy()
    print(f'Failed tasks (api/parse): {len(failed):,}')

    ens = pd.read_csv(ensemble_csv)
    allp = pd.read_csv(normalized_products_path(sample=False))
    allp['product_idx'] = allp.index
    clustered = set(ens['product_idx'].astype(int))
    singles = allp[~allp['product_idx'].isin(clustered)]
    singles_by_sm = {sm: g for sm, g in singles.groupby('supermarket')}

    # Rebuild tasks only for (cid, missing_sm) pairs in the failed set.
    need = {(int(r['ensemble_cluster_id']), r['missing_sm']) for _, r in failed.iterrows()}
    tasks = []
    for cid, cluster_df in ens.groupby('ensemble_cluster_id'):
        cid = int(cid)
        missing = ALL_SMS - set(cluster_df['supermarket'])
        for sm in missing:
            if (cid, sm) not in need:
                continue
            pool = singles_by_sm.get(sm)
            if pool is None or pool.empty:
                continue
            filt = C._prefilter(cluster_df, pool)
            if filt.empty:
                continue
            ranked = C._rank(cluster_df, filt, C.MAX_CANDIDATES)
            if ranked.empty:
                continue
            tasks.append((cid, sm, cluster_df, ranked))

    print(f'Rebuilt tasks to retry: {len(tasks):,}')

    results: list[dict] = []
    lock = threading.Lock()
    done = 0
    start = time.time()

    def _work(task):
        cid, sm, cluster_df, cands = task
        prompt = C._build_prompt(cluster_df, cands, sm)
        parsed = C._call_llm(client, prompt, retries=8)
        match = str(parsed.get('match', 'NONE')).strip().upper()
        conf = float(parsed.get('confidence', 0.0) or 0.0)
        reason = str(parsed.get('reason', ''))[:160]
        picked_idx = None
        if match != 'NONE' and len(match) == 1 and match in C.LETTERS[: len(cands)]:
            picked_idx = int(cands.iloc[C.LETTERS.index(match)]['product_idx'])
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

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                results.append(r)
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    rate = done / max(time.time() - start, 1e-3)
                    eta = (len(tasks) - done) / max(rate, 1e-3)
                    print(f'  [{done}/{len(tasks)}] {rate:.1f}/s  eta={eta/60:.1f}min')

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / 'completion_r3_log.csv', index=False)

    accepted = res_df[
        res_df['picked_idx'].notna() & (res_df['confidence'] >= min_confidence)
    ].copy()
    accepted['priority'] = accepted['cluster_size'] * 10 + accepted['confidence']
    accepted = accepted.sort_values('priority', ascending=False)

    claimed_idx: set[int] = set()
    claimed_cs: set[tuple[int, str]] = set()
    to_add: list[dict] = []
    for _, r in accepted.iterrows():
        idx = int(r['picked_idx'])
        cid = int(r['ensemble_cluster_id'])
        sm = r['missing_sm']
        if idx in claimed_idx or (cid, sm) in claimed_cs:
            continue
        claimed_idx.add(idx)
        claimed_cs.add((cid, sm))
        to_add.append({
            'ensemble_cluster_id': cid,
            'product_idx': idx,
            'supermarket': sm,
            'source': 'llm_r3_retry',
            'confidence': float(r['confidence']),
            'reason': r['reason'],
        })

    add_df = pd.DataFrame(to_add)
    add_df.to_csv(out_dir / 'completion_r3_additions.csv', index=False)
    print(f'\nR3 additions accepted: {len(add_df):,} products')

    # Merge additions into ens
    if add_df.empty:
        ens.to_csv(out_dir / 'ensemble_clusters_r3.csv', index=False)
    else:
        allp_idxed = allp.set_index('product_idx', drop=False)
        extra = allp_idxed.loc[add_df['product_idx'].values].copy()
        extra['ensemble_cluster_id'] = add_df['ensemble_cluster_id'].values
        for c in ens.columns:
            if c not in extra.columns and c != 'ensemble_cluster_id':
                extra[c] = pd.NA
        extra = extra[ens.columns]
        merged = pd.concat([ens, extra], ignore_index=True)
        merged['cluster_size'] = merged['ensemble_cluster_id'].map(
            merged.groupby('ensemble_cluster_id').size(),
        )
        merged = merged.sort_values(['ensemble_cluster_id', 'supermarket']).reset_index(drop=True)
        merged.to_csv(out_dir / 'ensemble_clusters_r3.csv', index=False)

    # Stats
    merged = pd.read_csv(out_dir / 'ensemble_clusters_r3.csv')
    sizes = merged.groupby('ensemble_cluster_id').size()
    stats = {
        'total_clusters': int(sizes.shape[0]),
        '4_way': int((sizes == 4).sum()),
        '3_way': int((sizes == 3).sum()),
        '2_way': int((sizes == 2).sum()),
        'retried': len(res_df),
        'additions': len(add_df),
    }
    print()
    print('=' * 60)
    print(f'Post-R3 distribution (4-way: {stats["4_way"]:,}, 3-way: {stats["3_way"]:,}, 2-way: {stats["2_way"]:,})')
    print(f'Retries: {stats["retried"]:,}, Additions: {stats["additions"]:,}')
    print('=' * 60)
    return stats


if __name__ == '__main__':
    retry_failed()
