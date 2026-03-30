"""
LLM cluster audit (Claude): evaluates ``audit_sample_50.csv`` per cluster.

Run via: ``uv run python main.py audit [--sample]`` (requires ``ANTHROPIC_API_KEY``).
"""

import json
import time

import pandas as pd
from anthropic import Anthropic

from shopwiser.paths import cluster_outputs_path

MODEL = 'claude-sonnet-4-5'
PASS_THRESHOLD = 0.90

SYSTEM_PROMPT = """You are a grocery product matching expert.
Your job is to review a set of product listings from different UK supermarkets
and decide whether they all represent the SAME core product.

Rules:
- Same core product = same brand (if branded), same flavour/variety, same weight/volume, same pack count
- Minor name differences between retailers are fine (abbreviations, punctuation)
- DIFFERENT product = different flavour, different variety, different weight, or different core product altogether
- Own-brand vs branded = DIFFERENT (unless explicitly asked to allow)
- If ANY product in the cluster is clearly a different product, the cluster FAILS

Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{
  "verdict": "PASS" or "FAIL",
  "confidence": 0.0-1.0,
  "reason": "one sentence explanation"
}"""


def build_prompt(cluster_df: pd.DataFrame) -> str:
    lines = []
    for _, row in cluster_df.iterrows():
        brand = row.get('known_brand_clean', '')
        brand_str = f" [{brand}]" if pd.notna(brand) and brand else ''
        weight = f"{row['unit_value']}{row['unit_type']}" if pd.notna(row.get('unit_value')) else 'unknown weight'
        pack = f" x{int(row['pack_quantity'])}" if pd.notna(row.get('pack_quantity')) and row['pack_quantity'] > 1 else ''
        tier = row.get('tier_type', '')
        tier_str = f" ({tier})" if pd.notna(tier) and tier else ''
        lines.append(
            f"  • {row['supermarket']}: {row['names']}{brand_str} — {weight}{pack}{tier_str}"
        )
    products_block = '\n'.join(lines)
    cluster_id = cluster_df['cluster_id'].iloc[0]
    match_type  = cluster_df['match_type'].iloc[0]
    avg_score   = cluster_df['avg_pairwise_score'].iloc[0]
    return (
        f"Cluster {cluster_id} ({match_type}, avg_score={avg_score:.3f}, "
        f"{len(cluster_df)} products from {cluster_df['supermarket'].nunique()} supermarkets):\n"
        f"{products_block}\n\n"
        f"Are ALL of these the same core grocery product?"
    )


def run_audit(*, sample: bool = False) -> None:
    """Read ``audit_sample_50.csv`` from full or sample cluster output dir."""
    out_root = cluster_outputs_path(sample=sample)
    input_csv = out_root / 'audit_sample_50.csv'
    output_csv = out_root / 'audit_results_v5.csv'
    client = Anthropic()

    print(f'Audit: {"sample" if sample else "full"} → {out_root}')
    df = pd.read_csv(input_csv)
    cluster_ids = sorted(df['cluster_id'].unique())
    print(f'Loaded {len(df)} rows, {len(cluster_ids)} clusters')

    results = []
    passes  = 0
    fails   = 0
    errors  = 0

    for i, cid in enumerate(cluster_ids):
        cluster_df = df[df['cluster_id'] == cid].copy()
        prompt = build_prompt(cluster_df)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            verdict    = parsed.get('verdict', 'ERROR').upper()
            confidence = float(parsed.get('confidence', 0.0))
            reason     = parsed.get('reason', '')
        except Exception as e:
            verdict    = 'ERROR'
            confidence = 0.0
            reason     = str(e)
            errors    += 1

        if verdict == 'PASS':
            passes += 1
        elif verdict == 'FAIL':
            fails += 1

        for idx in cluster_df.index:
            results.append({
                'cluster_id':   cid,
                'product_idx':  df.loc[idx, 'product_idx'],
                'supermarket':  df.loc[idx, 'supermarket'],
                'names':        df.loc[idx, 'names'],
                'unit_value':   df.loc[idx, 'unit_value'],
                'unit_type':    df.loc[idx, 'unit_type'],
                'match_type':   df.loc[idx, 'match_type'],
                'avg_pairwise_score': df.loc[idx, 'avg_pairwise_score'],
                'cluster_size': df.loc[idx, 'cluster_size'],
                'verdict':      verdict,
                'confidence':   confidence,
                'reason':       reason,
            })

        status_icon = '✓' if verdict == 'PASS' else ('✗' if verdict == 'FAIL' else '?')
        print(f'[{i+1:2d}/{len(cluster_ids)}] Cluster {cid:5d} {status_icon}  conf={confidence:.2f}  {reason[:80]}')

        time.sleep(0.3)

    total_scored = passes + fails
    pass_rate    = passes / total_scored if total_scored else 0.0

    print('\n' + '='*60)
    print('AUDIT RESULTS')
    print('='*60)
    print(f'  Clusters evaluated : {len(cluster_ids)}')
    print(f'  PASS               : {passes}')
    print(f'  FAIL               : {fails}')
    print(f'  ERROR              : {errors}')
    print(f'  Pass rate          : {pass_rate*100:.1f}%  (target ≥{PASS_THRESHOLD*100:.0f}%)')
    print(f'  Result             : {"✅ TARGET MET" if pass_rate >= PASS_THRESHOLD else "❌ BELOW TARGET"}')
    print('='*60)

    if fails:
        print('\nFailed clusters:')
        fail_rows = [r for r in results if r['verdict'] == 'FAIL']
        seen = set()
        for r in fail_rows:
            if r['cluster_id'] not in seen:
                seen.add(r['cluster_id'])
                print(f"  Cluster {r['cluster_id']:5d} ({r['match_type']:12s} score={r['avg_pairwise_score']:.3f}): {r['reason']}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f'\nSaved detailed results → {output_csv}')
