"""Code-based audit of any cluster CSV — no LLM, no API calls.

Replaces the old `audit_*_code.py` family (one script per pipeline stage)
with a single CLI tool that takes ``--input``.

Flags clusters that violate structural rules or semantic heuristics:

  1. one-per-supermarket invariant  (fatal if violated)
  2. hard-conflict tokens           (FLAVOR / ONE_SIDED / packaging / preparation)
  3. size mismatches                (smart pairwise unit_value delta)
  4. brand mismatches               (≥2 distinct known brands in one cluster)
  5. branded↔own_brand mix          (without a shared brand-token)

For each rule prints incidence, a sample of offenders, and an upper bound on
precision (clusters with no flag at all).

Usage
-----
    uv run python scripts/audit_clusters.py
    uv run python scripts/audit_clusters.py --input data/outputs/ensemble/ensemble_clusters.csv
    uv run python scripts/audit_clusters.py --input <path> --size-tol 0.20 --samples 8
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shopwiser.conflict_tokens import check_hard_conflict  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data/outputs/ensemble/ensemble_clusters.csv"


# ---------------------------------------------------------------------------
# Per-cluster checks
# ---------------------------------------------------------------------------

def _names(g: pd.DataFrame) -> list[str]:
    return g["normalized_name"].fillna("").astype(str).tolist()


def _has_hard_conflict(g: pd.DataFrame) -> bool:
    names = _names(g)
    return any(
        check_hard_conflict(a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    )


def _smart_pair_delta(uv_a, pq_a, uv_b, pq_b) -> float:
    if pd.isna(uv_a) or pd.isna(uv_b):
        return 0.0

    def _rd(x: float, y: float) -> float:
        hi = max(abs(x), abs(y))
        return 0.0 if hi < 1e-6 else abs(x - y) / hi

    pu_a = float(uv_a) / float(pq_a) if pd.notna(pq_a) and pq_a else float(uv_a)
    pu_b = float(uv_b) / float(pq_b) if pd.notna(pq_b) and pq_b else float(uv_b)
    return min(
        _rd(float(uv_a), float(uv_b)),
        _rd(pu_a, pu_b),
        _rd(float(uv_a), pu_b),
        _rd(pu_a, float(uv_b)),
    )


def _max_size_mismatch(g: pd.DataFrame) -> float:
    rows = g[["unit_value", "pack_quantity"]].values.tolist()
    worst = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            worst = max(worst, _smart_pair_delta(*rows[i], *rows[j]))
    return worst


def _has_brand_mismatch(g: pd.DataFrame) -> bool:
    brands = {
        b.strip().lower()
        for b in g["known_brand_clean"].dropna().astype(str)
        if b.strip()
    }
    return len(brands) >= 2


def _branded_own_brand_no_shared_token(g: pd.DataFrame) -> bool:
    types = set(g["product_type"].dropna().unique())
    if not {"branded", "own_brand"} <= types:
        return False
    branded_known = g.loc[g["product_type"] == "branded", "known_brand_clean"].dropna()
    if branded_known.empty:
        return False
    primary = str(branded_known.iloc[0]).strip().split()[0].lower()
    if len(primary) < 3:
        return False
    return any(primary not in set(nm.split()) for nm in _names(g))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_offenders(
    df: pd.DataFrame, label: str, ids: list, n: int = 5, seed: int = 20260419
) -> None:
    if not ids:
        return
    rng = random.Random(seed)
    sample = rng.sample(ids, min(n, len(ids)))
    print(f'\n--- "{label}" sample ({len(sample)} of {len(ids)}) ---')
    for cid in sample:
        g = df[df["ensemble_cluster_id"] == cid]
        print(f"\n  cluster {int(cid)} (size={len(g)}):")
        for _, r in g.iterrows():
            brand = r.get("known_brand_clean") or ""
            uv = r.get("unit_value")
            ut = r.get("unit_type")
            wt = f" — {uv:g}{ut}" if pd.notna(uv) and isinstance(ut, str) else ""
            bs = f" [{brand}]" if isinstance(brand, str) and brand else ""
            print(f"    {r['supermarket']:10s} {str(r['names'])[:80]}{bs}{wt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help=f"cluster CSV (default: {DEFAULT_INPUT.name})")
    parser.add_argument("--size-tol", type=float, default=0.15,
                        help="Max allowed pairwise size delta (default 0.15)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Sample offenders per check (default 5)")
    args = parser.parse_args()

    in_path = Path(args.input)
    print(f"Loading {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    n_clusters = df["ensemble_cluster_id"].nunique()
    sizes = df.groupby("ensemble_cluster_id").size()
    print(f"  {len(df):,} rows / {n_clusters:,} clusters")
    for sz in (2, 3, 4):
        print(f"    {sz}-way: {int((sizes == sz).sum()):,}")
    print()

    # 1. One per supermarket
    sm_dup = df.groupby("ensemble_cluster_id").apply(
        lambda g: len(g) != g["supermarket"].nunique()
    )
    sm_dup_ids = sm_dup[sm_dup].index.tolist()
    print(f"[1] one-per-supermarket violations    : {len(sm_dup_ids):>6,}")

    # 2. Hard conflict
    hc_ids = [cid for cid, g in df.groupby("ensemble_cluster_id") if _has_hard_conflict(g)]
    print(f"[2] hard conflict (FLAVOR/ONE_SIDED)  : {len(hc_ids):>6,}")

    # 3. Size mismatch
    deltas = df.groupby("ensemble_cluster_id").apply(_max_size_mismatch)
    size_ids = deltas[deltas > args.size_tol].index.tolist()
    print(f"[3] size mismatch >{int(args.size_tol*100)}%               : {len(size_ids):>6,}")

    # 4. Brand mismatch
    bm = df.groupby("ensemble_cluster_id").apply(_has_brand_mismatch)
    bm_ids = bm[bm].index.tolist()
    print(f"[4] ≥2 distinct known brands          : {len(bm_ids):>6,}")

    # 5. Branded↔own_brand without shared token
    bob = df.groupby("ensemble_cluster_id").apply(_branded_own_brand_no_shared_token)
    bob_ids = bob[bob].index.tolist()
    print(f"[5] branded↔own_brand w/o shared tok  : {len(bob_ids):>6,}")

    flagged = set(sm_dup_ids) | set(hc_ids) | set(size_ids) | set(bm_ids) | set(bob_ids)
    clean = n_clusters - len(flagged)
    print()
    print("=" * 60)
    print(f"  Clean clusters (no flag): {clean:>6,} / {n_clusters:,} = {clean/n_clusters*100:.1f}%")
    print(f"  Flagged ≥1 issue        : {len(flagged):>6,} = {len(flagged)/n_clusters*100:.1f}%")
    print("=" * 60)

    flagged_meta = (
        df[df["ensemble_cluster_id"].isin(flagged)]
        .drop_duplicates("ensemble_cluster_id")
        .assign(_sz=lambda d: d["ensemble_cluster_id"].map(sizes))
    )
    print("\nFlagged by cluster size:")
    for sz in (2, 3, 4):
        n_flag = int((flagged_meta["_sz"] == sz).sum())
        n_tot = int((sizes == sz).sum())
        pct = n_flag / n_tot * 100 if n_tot else 0.0
        print(f"  {sz}-way: {n_flag:>5,} / {n_tot:>5,}  ({pct:.1f}%)")

    _print_offenders(df, "hard_conflict", hc_ids, n=args.samples)
    _print_offenders(df, "size_mismatch", size_ids, n=args.samples)
    _print_offenders(df, "brand_mismatch", bm_ids, n=args.samples)
    _print_offenders(df, "branded_vs_own_no_shared_token", bob_ids, n=args.samples)


if __name__ == "__main__":
    main()
