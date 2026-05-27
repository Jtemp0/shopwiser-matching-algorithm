"""
Precision-vs-Coverage curve — v2 (LLM-filter simulation).

Uses the existing LLM filter verdicts (llm_filter_report.csv) combined with
per-cluster min-pair Jaccard scores to simulate what the deliverable looks
like at any Jaccard auto-pass threshold, without re-running the API.

Methodology
-----------
At threshold T:
  - Clusters with min_jaccard >= T  → auto-pass (kept)
  - Clusters with min_jaccard <  T  → would be LLM-verified; use stored
    verdict from the existing llm_filter_report.csv

This works because the filter report contains LLM verdicts for ALL 3,953
borderline clusters (those with min_jaccard < 0.65 in our current run).
For the auto-pass tier (min_jaccard >= 0.65) we assume all pass (the
contract validation at T=0.65 confirms 95.1% on those combined).

The current operating point (T=0.65, precision=95.1%) is marked explicitly.

Outputs
-------
  data/outputs/improvements/precision_coverage_curve_v2.csv
  data/outputs/improvements/precision_coverage_curve_v2.png

Usage
-----
    uv run python scripts/improvements/precision_coverage_curve_v2.py
"""

from __future__ import annotations
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(__file__).resolve().parent.parent.parent
ENSEMBLE_CSV   = REPO / "data/outputs/ensemble/ensemble_clusters.csv"
REPORT_CSV     = REPO / "data/outputs/fp_analys/llm_filter_report.csv"
RULE_CSV       = REPO / "data/outputs/clusters/clusters.csv"
ML_CSV         = REPO / "data/outputs/ml_clusters/ml_clusters.csv"
OUT_CSV        = REPO / "data/outputs/improvements/precision_coverage_curve_v2.csv"
OUT_PNG        = REPO / "data/outputs/improvements/precision_coverage_curve_v2.png"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
CURRENT_T  = 0.65
CURRENT_PRECISION = 95.1   # from 22-seed clause 4.2 validation

_STOP = frozenset({
    "the","a","of","and","to","in","on","for","with","by","from",
    "pack","x","g","kg","ml","l","litre","litres","grams",
    "large","small","medium","extra","special","best","finest",
    "essentials","just","tesco","sains","sainsbury","sainsburys",
    "asda","morrisons","morrison","pcs","count","ct","co","home",
    "each","new","original","everyday","our","more","plus",
    "quality","value","range",
})

def _toks(name: str) -> set[str]:
    s = re.sub(r"[^a-zA-Z' ]", " ", str(name).lower())
    return {t for t in s.split() if t not in _STOP and len(t) > 2}

def _jaccard(a: set[str], b: set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def min_pair_jaccard(names: list[str]) -> float:
    tsets = [_toks(n) for n in names if pd.notna(n) and str(n).strip()]
    if len(tsets) < 2:
        return 1.0
    return min(_jaccard(a, b) for a, b in combinations(tsets, 2))


def build_pre_filter_ensemble() -> pd.DataFrame:
    """Rebuild the full 11,618-cluster ensemble (before LLM filter) using
    the ensemble step logic — just reads from the already-built CSVs."""
    from shopwiser.ensemble.main import (
        _pairs_from_clusters, _kruskal_one_per_sm,
        _assign_cluster_ids, build_validator, MAX_CLUSTER_SIZE,
    )
    rule_df = pd.read_csv(RULE_CSV)
    ml_df   = pd.read_csv(ML_CSV)

    rule_edges = _pairs_from_clusters(rule_df, source="rule")
    ml_edges   = _pairs_from_clusters(ml_df,   source="ml")
    edges = pd.concat([rule_edges, ml_edges], ignore_index=True)
    edges = edges.sort_values("score", ascending=False).drop_duplicates(
        subset=["id_a","id_b"], keep="first"
    )

    sm_map = dict(zip(ml_df["product_idx"].astype(int), ml_df["supermarket"]))
    for pid, sm in zip(rule_df["product_idx"].astype(int), rule_df["supermarket"]):
        sm_map.setdefault(pid, sm)

    meta_df  = ml_df.set_index("product_idx")[
        ["normalized_name","names","unit_value","pack_quantity",
         "known_brand_clean","product_type","tier_type"]
    ]
    meta_map = meta_df.to_dict("index")
    validator = build_validator(meta_map)

    root_map = _kruskal_one_per_sm(edges, sm_map,
                                   is_valid_cluster=validator,
                                   max_cluster_size=MAX_CLUSTER_SIZE)
    cid_map  = _assign_cluster_ids(root_map)
    members  = {pid: cid_map[root] for pid, root in root_map.items()}

    size_counts: dict[int,int] = {}
    for cid in members.values():
        size_counts[cid] = size_counts.get(cid, 0) + 1
    multi_members = {pid: cid for pid, cid in members.items() if size_counts[cid] >= 2}

    base = ml_df.copy()
    base["ensemble_cluster_id"] = base["product_idx"].astype(int).map(multi_members)
    out = base.dropna(subset=["ensemble_cluster_id"]).copy()
    out["ensemble_cluster_id"] = out["ensemble_cluster_id"].astype(int)
    return out


def main() -> None:
    print("Rebuilding pre-filter ensemble …")
    pre = build_pre_filter_ensemble()
    print(f"  Pre-filter clusters: {pre['ensemble_cluster_id'].nunique():,}")

    print("Computing min-pair Jaccard per cluster …")
    jacc: dict[int, float] = {}
    size: dict[int, int]   = {}
    for cid, grp in pre.groupby("ensemble_cluster_id"):
        names = grp["normalized_name"].dropna().tolist()
        jacc[int(cid)] = min_pair_jaccard(names)
        size[int(cid)] = len(grp)

    print("Loading LLM filter report …")
    report = pd.read_csv(REPORT_CSV)
    verdict = dict(zip(report["cluster_id"].astype(int), report["llm_pass"].astype(bool)))

    print("Simulating thresholds …")
    rows = []
    for T in THRESHOLDS:
        kept = []
        for cid, j in jacc.items():
            if j >= T:
                kept.append(cid)            # auto-pass
            else:
                if verdict.get(cid, False): # LLM-approved
                    kept.append(cid)

        kept_set    = set(kept)
        n_clusters  = len(kept_set)
        cluster_sizes = {cid: size[cid] for cid in kept_set}
        n4 = sum(1 for s in cluster_sizes.values() if s == 4)
        n3 = sum(1 for s in cluster_sizes.values() if s == 3)
        n2 = sum(1 for s in cluster_sizes.values() if s == 2)

        rows.append({
            "threshold":   T,
            "n_clusters":  n_clusters,
            "n_4way":      n4,
            "n_3way":      n3,
            "n_2way":      n2,
            "n_products":  sum(cluster_sizes.values()),
        })
        print(f"  T={T:.2f}  clusters={n_clusters:,}  4-way={n4:,}  3-way={n3:,}  2-way={n2:,}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    col_blue  = "#2E86AB"
    col_dash  = "#1a5276"
    col_mark  = "#e74c3c"

    ax1.plot(df["threshold"], df["n_clusters"], "o-", color=col_blue,
             linewidth=2.2, markersize=7, label="Clusters kept")
    ax1.plot(df["threshold"], df["n_4way"],     "s--", color=col_dash,
             linewidth=2,   markersize=6, label="4-way clusters")
    ax1.set_xlabel("LLM filter Jaccard threshold", fontsize=12)
    ax1.set_ylabel("Cluster count", fontsize=12, color=col_blue)
    ax1.tick_params(axis="y", labelcolor=col_blue)
    ax1.set_ylim(0, df["n_clusters"].max() * 1.12)

    # Mark current operating point
    cur_row = df[df["threshold"] == CURRENT_T].iloc[0]
    ax1.axvline(x=CURRENT_T, color="grey", linestyle=":", linewidth=1.4, alpha=0.7)
    ax1.scatter([CURRENT_T], [cur_row["n_clusters"]], s=120, zorder=5,
                color=col_blue, edgecolors="white", linewidths=2)
    ax1.scatter([CURRENT_T], [cur_row["n_4way"]], s=100, zorder=5,
                color=col_dash, edgecolors="white", linewidths=2, marker="s")

    # Precision annotation at current point
    ax2.scatter([CURRENT_T], [CURRENT_PRECISION], s=160, zorder=6,
                color=col_mark, marker="*", label=f"Precision @ T={CURRENT_T}")
    ax2.annotate(
        f"95.1% precision\n(22-seed validation)\n{int(cur_row['n_clusters']):,} clusters kept",
        xy=(CURRENT_T, CURRENT_PRECISION),
        xytext=(CURRENT_T - 0.22, CURRENT_PRECISION - 18),
        fontsize=9, color=col_mark,
        arrowprops=dict(arrowstyle="->", color=col_mark, lw=1.2),
    )

    ax2.set_ylabel("Clause 4.2 precision (%)", fontsize=12, color=col_mark)
    ax2.tick_params(axis="y", labelcolor=col_mark)
    ax2.set_ylim(0, 105)

    # Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    star_patch = mpatches.Patch(color=col_mark, label=f"Validated precision @ T={CURRENT_T}")
    ax1.legend(handles=handles1 + [star_patch], loc="upper right", fontsize=10)

    ax1.set_title("ShopWiser: Coverage vs Precision by LLM filter threshold", fontsize=13, pad=14)
    ax1.set_xticks(THRESHOLDS)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
