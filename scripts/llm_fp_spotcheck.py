"""Spot-check random clusters with Claude Haiku 4.5 — same-SKU yes/no.

Reads fp_candidates.csv + ensemble CSV (defaults match scripts/export_fp_candidates.py), takes a random sample,
batches them to save tokens, and outputs a clean CSV.

Requires ANTHROPIC_API_KEY in the environment or repo-root ``.env``.

Usage:
  uv run python scripts/llm_fp_spotcheck.py --sample 500 --batch-size 25
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MODEL = "claude-haiku-4-5-20251001"
DEFAULT_CLUSTERS = REPO / "data/outputs/ensemble/ensemble_clusters.csv"
DEFAULT_FP = REPO / "data/outputs/fp_analys/fp_candidates.csv"
DEFAULT_OUT = REPO / "data/outputs/fp_analys/llm_spotcheck_latest.csv"

def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k:
            os.environ[k] = v

def _cluster_block(df: pd.DataFrame, cid: int) -> str:
    g = df[df["ensemble_cluster_id"] == cid].sort_values("supermarket")
    lines = []
    for _, r in g.iterrows():
        nm = str(r.get("names", ""))[:120]
        lines.append(f"- {r['supermarket']}: {nm}")
    return f"Cluster {cid} ({len(g)} products):\n" + "\n".join(lines)

def main() -> None:
    _load_env_file(REPO / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping LLM spot-check.")
        sys.exit(0)

    p = argparse.ArgumentParser()
    p.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    p.add_argument("--fp", type=Path, default=DEFAULT_FP)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from anthropic import Anthropic
    client = Anthropic()

    print(f"Loading {args.fp.name} and {args.clusters.name}...")
    fp = pd.read_csv(args.fp, low_memory=False)
    df = pd.read_csv(args.clusters, low_memory=False)
    
    # Sample random clusters
    sampled_ids = fp["ensemble_cluster_id"].sample(n=min(args.sample, len(fp)), random_state=args.seed).tolist()
    print(f"Sampled {len(sampled_ids)} clusters for review.")

    results = []
    
    # Process in batches
    for i in range(0, len(sampled_ids), args.batch_size):
        batch_ids = sampled_ids[i:i + args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1}/{(len(sampled_ids) + args.batch_size - 1)//args.batch_size} ({len(batch_ids)} clusters)...")
        
        user = (
            "For each cluster below, decide if ALL products are the SAME grocery SKU "
            "(same brand line + variant + comparable pack size), only sold at different "
            "UK supermarkets. \n"
            "IMPORTANT RULES:\n"
            "1. For branded products, they must be the exact same brand and variant. (IPA vs Pale Ale = DIFFERENT. Different beer brands = DIFFERENT).\n"
            "2. For supermarket own-brand products (e.g. ASDA Potato Slices vs Morrisons Potato Slices), they MUST BE CONSIDERED THE SAME SKU if they are the same core product, same tier (e.g. both standard or both premium), and comparable pack size. This is a price comparison app that matches own-brand equivalents.\n"
            "Reply with ONE JSON object containing a 'verdicts' array: "
            '{"verdicts":[{"cluster_id":int,"same_sku":bool,"reason":"one short phrase"}]} '
            f"covering these cluster ids in order: {batch_ids}.\n\n"
        )
        for cid in batch_ids:
            user += _cluster_block(df, cid) + "\n\n"

        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                temperature=0,
                system="You only output valid JSON. No markdown fences.",
                messages=[{"role": "user", "content": user}],
            )
            text = msg.content[0].text if msg.content else ""
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                payload = json.loads(m.group())
                verdicts = payload.get("verdicts", [])
                results.extend(verdicts)
            else:
                print(f"Failed to parse JSON for batch {i//args.batch_size + 1}")
        except Exception as e:
            print(f"API Error on batch {i//args.batch_size + 1}: {e}")
            
        time.sleep(1) # small delay to avoid rate limits

    # Build final CSV
    if not results:
        print("No results collected.")
        return

    res_df = pd.DataFrame(results)
    # Merge with FP metadata for context
    final_df = res_df.merge(
        fp[["ensemble_cluster_id", "cluster_size", "supermarkets", "titles_preview", "flags", "risk_bucket"]],
        left_on="cluster_id",
        right_on="ensemble_cluster_id",
        how="left"
    ).drop(columns=["ensemble_cluster_id"])
    
    # Reorder columns
    cols = ["cluster_id", "same_sku", "reason", "cluster_size", "supermarkets", "risk_bucket", "flags", "titles_preview"]
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.out, index=False)
    print(f"\nDone! Wrote {len(final_df)} verdicts to {args.out}")
    print("\nSummary of same_sku:")
    print(final_df["same_sku"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
