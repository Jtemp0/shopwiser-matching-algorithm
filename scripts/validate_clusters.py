"""
Validation script for ensemble_clusters_final.csv.

Tests agreement clause 4.2 success criteria across N random samples
of 50 four-way clusters, each drawn with a different seed.

Usage:
    python scripts/validate_clusters.py               # 10 samples, default seeds
    python scripts/validate_clusters.py --n 20        # 20 samples
    python scripts/validate_clusters.py --n 5 --base-seed 999  # custom seed offset
    python scripts/validate_clusters.py --csv path/to/other.csv
    python scripts/validate_clusters.py --sample-size 100      # 100 clusters per sample
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or scripts/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shopwiser.ml_matching.features import check_hard_conflict  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CSV = REPO_ROOT / "data/outputs/ensemble/ensemble_clusters_final.csv"
KNOWN_TIERS = frozenset({"value", "standard", "premium", "dietary"})
PASS_THRESHOLD = 0.90  # 90 % of sampled clusters must pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canon_brand(s: object) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", "", re.sub(r"[''\u2019.\-]", "", s.strip().lower()))


def _size_delta(a: float, b: float) -> float:
    hi = max(abs(a), abs(b))
    return abs(a - b) / hi if hi > 1e-6 else 0.0


def _q1_pass(g: pd.DataFrame) -> bool:
    """Q1: All items are the same core product (no hard conflict, single brand)."""
    names = g["normalized_name"].fillna("").tolist()
    brands = {_canon_brand(b) for b in g["known_brand_clean"].fillna("") if _canon_brand(b)}
    if len(brands) >= 2:
        return False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if check_hard_conflict(names[i], names[j]) == 1:
                return False
    return True


def _q2_pass(g: pd.DataFrame, max_delta: float = 0.15) -> bool:
    """Q2: All items within acceptable weight / size variance (<= 15 % by default)."""
    uvs = [u for u in g["unit_value"].tolist() if isinstance(u, (int, float)) and u > 0]
    if len(uvs) < 2:
        return True
    return all(
        _size_delta(uvs[i], uvs[j]) <= max_delta
        for i in range(len(uvs))
        for j in range(i + 1, len(uvs))
    )


def _q3_pass(g: pd.DataFrame) -> bool:
    """Q3: Own-brand/unbranded items are all in the same product tier."""
    ob = g[g["product_type"].isin(["own_brand", "unbranded"])]
    tiers = {
        str(t).lower()
        for t in ob["tier_type"].dropna()
        if str(t).lower() in KNOWN_TIERS
    }
    return len(tiers) < 2


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def validate_sample(
    df: pd.DataFrame,
    four_way_ids: list,
    seed: int,
    sample_size: int,
) -> dict:
    """Draw `sample_size` 4-way clusters at `seed` and return per-Q counts."""
    import random
    rng = random.Random(seed)
    sample_ids = rng.sample(four_way_ids, min(sample_size, len(four_way_ids)))

    passes = q1_fails = q2_fails = q3_fails = 0
    for cid in sample_ids:
        g = df[df["ensemble_cluster_id"] == cid]
        q1 = _q1_pass(g)
        q2 = _q2_pass(g)
        q3 = _q3_pass(g)
        if not q1:
            q1_fails += 1
        if not q2:
            q2_fails += 1
        if not q3:
            q3_fails += 1
        if q1 and q2 and q3:
            passes += 1

    total = len(sample_ids)
    pass_rate = passes / total
    return {
        "seed": seed,
        "total": total,
        "passes": passes,
        "pass_rate": pass_rate,
        "verdict": "PASS" if pass_rate >= PASS_THRESHOLD else "FAIL",
        "q1_fails": q1_fails,
        "q2_fails": q2_fails,
        "q3_fails": q3_fails,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ensemble clusters against agreement clause 4.2."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to the clusters CSV (default: data/outputs/ensemble/ensemble_clusters_final.csv)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of independent samples to draw (default: 10)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of 4-way clusters per sample (default: 50)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=100,
        help="Seeds used are base-seed, base-seed+100, base-seed+200, ... (default: 100)",
    )
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)

    size_map = df.groupby("ensemble_cluster_id").size()
    four_way_ids = list(size_map[size_map == 4].index)

    print(f"Pool : {len(four_way_ids):,} four-way clusters")
    print(f"Runs : {args.n} samples of {args.sample_size} clusters each")
    print(f"Seeds: {args.base_seed}, {args.base_seed + 100}, ...\n")

    seeds = [args.base_seed + i * 100 for i in range(args.n)]
    results = [validate_sample(df, four_way_ids, seed, args.sample_size) for seed in seeds]

    # --- table header ---
    col = f"passes/{args.sample_size}"
    w = max(len(col), 8)
    header = (
        f"{'seed':>6}  {col:>{w}}  {'rate':>6}  "
        f"{'Q1 fails':>8}  {'Q2 fails':>8}  {'Q3 fails':>8}  verdict"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    all_verdicts = []
    for r in results:
        print(
            f"{r['seed']:>6}  "
            f"{r['passes']:>{w - len(str(args.sample_size)) - 1}}/{args.sample_size}  "
            f"{r['pass_rate']:>5.1%}  "
            f"{r['q1_fails']:>8}  "
            f"{r['q2_fails']:>8}  "
            f"{r['q3_fails']:>8}  "
            f"{r['verdict']}"
        )
        all_verdicts.append(r["verdict"])

    # --- summary ---
    print(sep)
    rates = [r["pass_rate"] for r in results]
    n_pass = sum(1 for v in all_verdicts if v == "PASS")
    n_fail = args.n - n_pass

    print(
        f"\nSummary: {n_pass}/{args.n} samples PASS  |  "
        f"min {min(rates):.1%}  avg {sum(rates)/len(rates):.1%}  max {max(rates):.1%}"
    )

    overall = "ALL PASS" if n_fail == 0 else f"{n_fail} FAIL"
    print(f"Overall: {overall}  (threshold per sample: {PASS_THRESHOLD:.0%})\n")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
