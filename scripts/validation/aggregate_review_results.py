"""
Aggregate the reviewer CSVs from the 4 reviewers into the acceptance
pass-rate measure.

A cluster is deemed to "pass" if all three Yes/No questions
receive "Yes" (or N/A for Q3 when no own-brand items present). Per clause
4.4 the algorithm satisfies the success criteria if at least 90% of the 50
sampled clusters pass.

Aggregation rule across the 4 reviewers: a cluster is treated as passed
when the majority (≥3 of 4) of reviewers marked it as passed. This is robust
to a single outlier reviewer.

Usage
-----
    uv run python scripts/validation/aggregate_review_results.py \
        --in data/validation/reviews/

The script reads every *.csv in the directory and assumes each came from a
different reviewer. Empty directory just prints the schema reminder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = REPO_ROOT / "data/validation/reviews"
OUT = REPO_ROOT / "data/validation/review_aggregated.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", default=str(DEFAULT_DIR),
                        help="Directory containing reviewer CSVs")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        print(f"No reviewer CSVs found in {in_dir}.")
        print("Put one CSV per reviewer in that directory, then re-run.")
        return

    print(f"Found {len(csvs)} reviewer file(s):")
    for p in csvs:
        print(f"  - {p.name}")

    frames = [pd.read_csv(p) for p in csvs]
    all_reviews = pd.concat(frames, ignore_index=True)
    print(f"\n{len(all_reviews):,} answer rows across all reviewers.\n")

    by_cluster = all_reviews.groupby("cluster_id").agg(
        n_reviewers=("reviewer", "nunique"),
        n_pass=("cluster_passed", lambda s: (s == "yes").sum()),
        n_fail=("cluster_passed", lambda s: (s == "no").sum()),
    ).reset_index()

    n_total_reviewers = all_reviews["reviewer"].nunique()
    majority_threshold = (n_total_reviewers // 2) + 1
    by_cluster["majority_pass"] = by_cluster["n_pass"] >= majority_threshold

    pass_rate = by_cluster["majority_pass"].mean() * 100

    print(f"Total reviewers          : {n_total_reviewers}")
    print(f"Majority threshold       : {majority_threshold} of {n_total_reviewers}")
    print(f"Clusters reviewed        : {len(by_cluster)}")
    print(f"Clusters passing majority: {int(by_cluster['majority_pass'].sum())}")
    print(f"Pass rate                : {pass_rate:.1f}%")
    print("Acceptance threshold    : 90.0%")
    print(f"Result                   : {'PASS' if pass_rate >= 90 else 'BELOW THRESHOLD'}")

    by_cluster.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
