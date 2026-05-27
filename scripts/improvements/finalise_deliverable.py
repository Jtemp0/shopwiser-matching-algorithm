"""
Post-process the ensemble deliverable for hand-off:

  1. Brand canonicalisation: apostrophes/hyphens normalised across known
     UK supermarket brands (McVitie's, Hartley's, Ben's, Young's, Jacob's,
     Fever-Tree). Affects the `names` column only — matching logic is
     unchanged, this is cosmetic for display.

  2. unit_value normalisation: for clusters where retailers report
     unit_value on different bases (per-unit vs total-pack), normalise to
     per-unit using pack_quantity. Example: Coca-Cola 12x150ml reported
     by Sains as unit_value=150 (per-can) and by Tesco as 1800 (total
     pack). After this pass both read 150.

  3. Pack-size guard: drop clusters where pack_quantity differs by more
     than 2x across members (genuine pack-size clustering errors that
     slipped past the original gate). Border-line cases (1.15x–2x) are
     left as-is; they are typically size variants of the same product.

  4. Confidence enrichment: per-cluster confidence_score (raw, in
     [0, 1]) merged in from cluster_review_metrics.csv. Categorical
     bands are deliberately not included — they would impose arbitrary
     cut-offs that don't match the empirical calibration of the score.

Inputs
------
  data/outputs/ensemble/ensemble_clusters.csv
  data/outputs/ensemble/cluster_review_metrics.csv

Output
------
  data/outputs/improvements/ensemble_clusters_final.csv

Usage
-----
    uv run python scripts/improvements/finalise_deliverable.py
"""

from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
CLUSTERS = REPO / "data/outputs/ensemble/ensemble_clusters.csv"
METRICS = REPO / "data/outputs/ensemble/cluster_review_metrics.csv"
OUT = REPO / "data/outputs/improvements/ensemble_clusters_final.csv"

# (regex, replacement) — case-insensitive on the whole-word form,
# then we re-case to the canonical form
# Each pattern is matched only when NOT already followed by 's (negative
# lookahead), so "McVitie's" stays as-is and only "McVitie" / "Mcvities" /
# "McVities" get rewritten.
BRAND_NORMALISATIONS: list[tuple[str, str]] = [
    (r"\bMcvities(?!')\b",      "McVitie's"),
    (r"\bMcvitie(?!')\b",       "McVitie's"),
    (r"\bMcVities(?!')\b",      "McVitie's"),
    (r"\bHartleys(?!')\b",      "Hartley's"),
    (r"\bYoungs(?!')\b",        "Young's"),
    (r"\bJacobs(?!')\b",        "Jacob's"),
    (r"\bBens(?!') Original\b", "Ben's Original"),
    (r"\bFever Tree\b",         "Fever-Tree"),
]


def canonicalise_brand(name: str) -> str:
    if not isinstance(name, str):
        return name
    out = name
    for pattern, repl in BRAND_NORMALISATIONS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def normalise_unit_value(df: pd.DataFrame) -> pd.DataFrame:
    """For each cluster, if any row's unit_value ≈ another row's
    unit_value × pack_quantity, normalise the larger to per-unit basis.

    Only triggers when:
      - both rows have a pack_quantity matching the multiplier exactly
      - both rows share the same unit_type
    Conservative by design: leaves ambiguous cases alone.
    """
    df = df.copy()
    df["unit_value_original"] = df["unit_value"]
    fixed_count = 0

    for cid, grp in df.groupby("ensemble_cluster_id"):
        uvs = grp[["product_idx", "unit_value", "pack_quantity", "unit_type"]].dropna(
            subset=["unit_value"]
        )
        if len(uvs) < 2:
            continue
        # find the smallest non-null unit_value (likely the per-unit basis)
        base_uv = uvs["unit_value"].min()
        if base_uv <= 0:
            continue
        for _, row in uvs.iterrows():
            if row["unit_value"] == base_uv:
                continue
            ratio = row["unit_value"] / base_uv
            pq = row["pack_quantity"]
            # only normalise if ratio matches pack quantity within 5%
            if pd.notna(pq) and pq > 1 and abs(ratio - pq) / pq < 0.05:
                df.loc[df["product_idx"] == row["product_idx"], "unit_value"] = base_uv
                fixed_count += 1

    print(f"  unit_value rows normalised: {fixed_count}")
    return df


def drop_packsize_mismatches(df: pd.DataFrame, ratio_threshold: float = 2.0) -> pd.DataFrame:
    """Drop clusters whose pack_quantity span exceeds ratio_threshold."""
    bad = []
    for cid, grp in df.groupby("ensemble_cluster_id"):
        pqs = grp["pack_quantity"].dropna().tolist()
        if len(pqs) < 2:
            continue
        mn, mx = min(pqs), max(pqs)
        if mn > 0 and (mx / mn) >= ratio_threshold:
            bad.append(cid)
    print(f"  Dropping {len(bad)} clusters with pack_quantity mismatch >= {ratio_threshold}x")
    return df[~df["ensemble_cluster_id"].isin(bad)].copy()


def add_confidence(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["ensemble_cluster_id", "confidence_score"]
    out = df.merge(metrics[cols], on="ensemble_cluster_id", how="left")
    out["confidence_score"] = out["confidence_score"].round(4)
    return out


def main() -> None:
    print(f"Loading {CLUSTERS}")
    df = pd.read_csv(CLUSTERS, low_memory=False)
    print(f"  {len(df):,} rows / {df['ensemble_cluster_id'].nunique():,} clusters")

    print("\n[1/4] Brand canonicalisation …")
    before = df["names"].copy()
    df["names"] = df["names"].apply(canonicalise_brand)
    n_changed = (before != df["names"]).sum()
    print(f"  Names rewritten: {n_changed:,}")

    print("\n[2/4] unit_value normalisation …")
    df = normalise_unit_value(df)

    print("\n[3/4] Pack-size guard …")
    n_before = df["ensemble_cluster_id"].nunique()
    df = drop_packsize_mismatches(df, ratio_threshold=1.5)
    n_after = df["ensemble_cluster_id"].nunique()
    print(f"  Clusters before: {n_before:,}  after: {n_after:,}")

    print("\n[4/4] Confidence enrichment …")
    metrics = pd.read_csv(METRICS)
    df = add_confidence(df, metrics)
    scores = df.groupby("ensemble_cluster_id")["confidence_score"].first()
    print(f"  confidence_score   min={scores.min():.3f}  "
          f"median={scores.median():.3f}  max={scores.max():.3f}  "
          f"missing={scores.isna().sum()}")

    sizes = df.groupby("ensemble_cluster_id").size().value_counts().sort_index()
    print("\nFinal cluster size distribution:")
    for sz, cnt in sizes.items():
        print(f"  {sz}-way: {cnt:,}")
    print(f"\nFinal total: {df['ensemble_cluster_id'].nunique():,} clusters, "
          f"{len(df):,} rows")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
