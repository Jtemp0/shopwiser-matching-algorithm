"""
Audit the deliverable for flavour-variant failures of the Pringles type.

For every cluster, tokenise each item's name and check whether the new
expanded FLAVOR_NAMED_TOKENS / ONE_SIDED_CONFLICT_TOKENS would have flagged
a conflict the existing pipeline missed. Output the offending clusters so we
can quantify the impact of the v11 vocab expansion before re-running the
full pipeline.

Output
------
  data/outputs/improvements/flavour_failures.csv
      ensemble_cluster_id, cluster_size, conflict_type, conflict_token,
      retailer_a, retailer_b, name_a, name_b

Usage
-----
    uv run python scripts/improvements/audit_flavour_failures.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shopwiser.clustering.config import (  # noqa: E402
    FLAVOR_NAMED_TOKENS,
    HARD_CONFLICT_NORM,
    ONE_SIDED_CONFLICT_TOKENS,
)

CLUSTERS = REPO_ROOT / "data/outputs/ensemble/ensemble_clusters_final.csv"
OUT = REPO_ROOT / "data/outputs/improvements/flavour_failures.csv"


def tokenise(name: str) -> frozenset[str]:
    s = re.sub(r"[^a-zA-Z' ]", " ", str(name).lower())
    raw = [t for t in s.split() if len(t) > 2]
    norm = frozenset(HARD_CONFLICT_NORM.get(t, t) for t in raw)
    return norm


def main() -> None:
    print(f"Loading {CLUSTERS}")
    df = pd.read_csv(CLUSTERS, low_memory=False)
    print(f"  {len(df):,} rows / {df['ensemble_cluster_id'].nunique():,} clusters\n")

    df["_tok"] = df["names"].apply(tokenise)

    failures: list[dict] = []

    for cid, g in df.groupby("ensemble_cluster_id"):
        items = g[["supermarket", "names", "_tok"]].reset_index(drop=True)
        n = len(items)
        if n < 2:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                t_a = items.at[i, "_tok"]
                t_b = items.at[j, "_tok"]

                # Named flavour clash: both have named tokens, none overlap
                f_a = t_a & FLAVOR_NAMED_TOKENS
                f_b = t_b & FLAVOR_NAMED_TOKENS
                if f_a and f_b and not (f_a & f_b):
                    failures.append({
                        "ensemble_cluster_id": cid,
                        "cluster_size": n,
                        "conflict_type": "named_flavour_clash",
                        "conflict_token": f"{','.join(sorted(f_a))} vs {','.join(sorted(f_b))}",
                        "retailer_a": items.at[i, "supermarket"],
                        "retailer_b": items.at[j, "supermarket"],
                        "name_a": items.at[i, "names"],
                        "name_b": items.at[j, "names"],
                    })
                    continue

                # One-sided conflict
                for tok in ONE_SIDED_CONFLICT_TOKENS:
                    if (tok in t_a) ^ (tok in t_b):
                        failures.append({
                            "ensemble_cluster_id": cid,
                            "cluster_size": n,
                            "conflict_type": "one_sided",
                            "conflict_token": tok,
                            "retailer_a": items.at[i, "supermarket"],
                            "retailer_b": items.at[j, "supermarket"],
                            "name_a": items.at[i, "names"],
                            "name_b": items.at[j, "names"],
                        })
                        break

    out = pd.DataFrame(failures)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    n_clusters_with_failure = out["ensemble_cluster_id"].nunique() if len(out) else 0
    print(f"Found {len(out):,} pairwise conflicts spanning "
          f"{n_clusters_with_failure:,} clusters.\n")

    if len(out):
        print("Breakdown by conflict type:")
        print(out["conflict_type"].value_counts().to_string())
        print("\nTop conflict tokens:")
        print(out["conflict_token"].value_counts().head(15).to_string())
        print("\nFirst 10 examples:")
        print(out[["conflict_type", "conflict_token", "name_a", "name_b"]].head(10).to_string(index=False))

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
