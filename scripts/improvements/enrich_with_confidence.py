"""
Enrich ensemble_clusters.csv with per-cluster confidence_score columns.

Inputs
------
  data/outputs/ensemble/ensemble_clusters.csv             one row per product
  data/outputs/ensemble/cluster_review_metrics.csv        one row per cluster
                                                          (produced by scripts/review_metrics.py)

Output
------
  data/outputs/improvements/ensemble_clusters_with_confidence.csv
      same rows + columns:
        confidence_score  in [0, 1]
        confidence_band   one of {high, medium, low}
        likely_good       boolean from the structural probe

Confidence bands (calibrated against the per-size pass rates so that
each band roughly matches a downstream operating point):
  high    : score >= 0.75   (recommend ship as-is)
  medium  : 0.55–0.75       (surface in UI with caveat)
  low     : <0.55           (route to human review or hide)

Usage
-----
    uv run python scripts/improvements/enrich_with_confidence.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLUSTERS = REPO_ROOT / "data/outputs/ensemble/ensemble_clusters.csv"
METRICS = REPO_ROOT / "data/outputs/ensemble/cluster_review_metrics.csv"
OUT = REPO_ROOT / "data/outputs/improvements/ensemble_clusters_with_confidence.csv"


def _band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def main() -> None:
    print(f"Loading clusters: {CLUSTERS}")
    clusters = pd.read_csv(CLUSTERS, low_memory=False)
    print(f"  {len(clusters):,} rows")

    print(f"Loading metrics : {METRICS}")
    metrics = pd.read_csv(METRICS)
    print(f"  {len(metrics):,} clusters scored")

    keep = ["ensemble_cluster_id", "confidence_score", "likely_good"]
    enriched = clusters.merge(metrics[keep], on="ensemble_cluster_id", how="left")

    enriched["confidence_score"] = enriched["confidence_score"].round(4)
    enriched["confidence_band"] = enriched["confidence_score"].apply(
        lambda s: _band(s) if pd.notna(s) else "unknown"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(enriched):,} rows)")

    band_counts = enriched.groupby("ensemble_cluster_id")["confidence_band"].first().value_counts()
    print("\nConfidence bands (one count per cluster):")
    for b in ("high", "medium", "low", "unknown"):
        print(f"  {b:<8}  {int(band_counts.get(b, 0)):>6,}")

    sizes = enriched.groupby("ensemble_cluster_id").size()
    by_band = enriched.groupby("ensemble_cluster_id").first()
    by_band["sz"] = sizes
    print("\nBy cluster size × band:")
    print(by_band.groupby(["sz", "confidence_band"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
