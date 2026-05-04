"""
Precision-vs-Coverage curve for the cluster deliverable.

Treats the per-cluster confidence_score (from cluster_review_metrics.csv) as the
operating-point dial. Walking the threshold from low to high trades coverage
for cleaner clusters. The output is the data Jack/Alex need to choose where
the MVP cuts.

Outputs
-------
  data/outputs/improvements/precision_coverage_curve.csv
      threshold, n_clusters_kept, n_4way, n_3way, n_2way,
      probe_pass_rate, mean_confidence

  data/outputs/improvements/precision_coverage_curve.png  (if matplotlib present)

Usage
-----
    uv run python scripts/improvements/precision_coverage_curve.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS = REPO_ROOT / "data/outputs/ensemble/cluster_review_metrics.csv"
OUT_CSV = REPO_ROOT / "data/outputs/improvements/precision_coverage_curve.csv"
OUT_PNG = REPO_ROOT / "data/outputs/improvements/precision_coverage_curve.png"

THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]


def main() -> None:
    print(f"Loading {METRICS}")
    m = pd.read_csv(METRICS)
    print(f"  {len(m):,} clusters\n")

    rows = []
    for t in THRESHOLDS:
        kept = m[m["confidence_score"] >= t]
        if len(kept) == 0:
            continue
        rows.append({
            "threshold": t,
            "n_clusters_kept": len(kept),
            "kept_pct_of_full": round(len(kept) / len(m) * 100, 1),
            "n_4way": int((kept["cluster_size"] == 4).sum()),
            "n_3way": int((kept["cluster_size"] == 3).sum()),
            "n_2way": int((kept["cluster_size"] == 2).sum()),
            "probe_pass_rate_pct": round(kept["likely_good"].mean() * 100, 1),
            "mean_confidence": round(kept["confidence_score"].mean(), 3),
        })

    curve = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT_CSV, index=False)

    print("Precision-vs-coverage table:")
    print(curve.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(curve["threshold"], curve["n_clusters_kept"], "o-", color="#2E86AB", label="Clusters kept")
        ax1.plot(curve["threshold"], curve["n_4way"], "s--", color="#118AB2", label="4-way clusters")
        ax1.set_xlabel("Confidence threshold")
        ax1.set_ylabel("Cluster count")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(curve["threshold"], curve["probe_pass_rate_pct"], "^-", color="#E63946", label="Probe pass %")
        ax2.set_ylabel("Probe pass rate (%)", color="#E63946")
        ax2.tick_params(axis="y", labelcolor="#E63946")
        ax2.set_ylim(0, 100)

        plt.title("ShopWiser: Precision vs Coverage by confidence threshold")
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        print(f"Wrote {OUT_PNG}")
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
