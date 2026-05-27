"""
Rigorous precision-vs-coverage analysis for the cluster deliverable.

Produces three panels suitable for a technical (PhD-level) audience:

  (a) Empirical precision-recall curve with bootstrap 95% CI band.
      Constructed by sweeping the confidence threshold over the 500
      human-validated cluster reviews (10 seeds × 50 stratified samples,
      clause 4.2 protocol). Precision at threshold t is the fraction of
      validated clusters with confidence_score >= t that passed all three
      clause 4.2 questions. CI is computed via 2,000 bootstrap resamples
      over the validation set.

  (b) Reliability diagram (calibration plot). Validated clusters are
      grouped into equal-width confidence bins. Each point is the
      empirical pass rate vs the mean predicted confidence in that bin,
      with Wilson 95% CIs for the empirical proportion. The dashed
      diagonal is perfect calibration.

  (c) Coverage curve. Number of clusters retained (and 4-way subset)
      in the FULL deliverable as the confidence threshold is swept.
      This is the operational tradeoff side: how much of the catalogue
      remains at each threshold.

Inputs
------
  data/outputs/fp_analys/contract_validation.csv     500 validated rows
  data/outputs/ensemble/cluster_review_metrics.csv   confidence + size per cluster

Outputs
-------
  data/outputs/improvements/precision_coverage_rigorous.png
  data/outputs/improvements/precision_coverage_rigorous.csv

Usage
-----
    uv run python scripts/improvements/precision_coverage_rigorous.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
VALID = REPO / "data/outputs/fp_analys/contract_validation.csv"
METRICS = REPO / "data/outputs/ensemble/cluster_review_metrics.csv"
FINAL = REPO / "data/outputs/improvements/ensemble_clusters_final.csv"
OUT_PNG = REPO / "data/outputs/improvements/precision_coverage_rigorous.png"
OUT_CSV = REPO / "data/outputs/improvements/precision_coverage_rigorous.csv"

RNG_SEED = 20260527
N_BOOTSTRAP = 2000
THRESHOLDS = np.linspace(0.30, 0.99, 70)
CAL_BINS = np.linspace(0.5, 1.0, 11)  # 10 equal-width bins on [0.5, 1.0]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    valid = pd.read_csv(VALID)
    metrics = pd.read_csv(METRICS)
    # restrict metrics to clusters that survive in the final post-processed deliverable
    final_ids = set(pd.read_csv(FINAL, low_memory=False)["ensemble_cluster_id"].unique())
    metrics = metrics[metrics["ensemble_cluster_id"].isin(final_ids)].reset_index(drop=True)
    print(f"Final deliverable clusters: {len(metrics):,}")
    merged = valid.merge(
        metrics[["ensemble_cluster_id", "confidence_score", "cluster_size"]],
        left_on="cluster_id", right_on="ensemble_cluster_id", how="inner",
    )
    n_val = len(merged)
    print(f"Validation rows: {n_val}  ({merged['run_seed'].nunique()} seeds)")
    print(f"Overall precision: {merged['pass'].mean():.4f}")

    scores = merged["confidence_score"].to_numpy()
    passes = merged["pass"].astype(int).to_numpy()

    # (a) Empirical PR curve with bootstrap CI
    print("\nBootstrapping PR curve …")
    boot_prec = np.full((N_BOOTSTRAP, len(THRESHOLDS)), np.nan)
    boot_cov = np.full((N_BOOTSTRAP, len(THRESHOLDS)), np.nan)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_val, n_val)
        s, p = scores[idx], passes[idx]
        for i, t in enumerate(THRESHOLDS):
            mask = s >= t
            if mask.sum() >= 5:
                boot_prec[b, i] = p[mask].mean()
                boot_cov[b, i] = mask.mean()

    # Point estimates (no resampling) for reporting
    point_prec, point_cov, point_n = [], [], []
    point_ci_lo, point_ci_hi = [], []
    for t in THRESHOLDS:
        mask = scores >= t
        n_kept = int(mask.sum())
        k_pass = int(passes[mask].sum())
        point_n.append(n_kept)
        if n_kept >= 5:
            point_prec.append(k_pass / n_kept)
            point_cov.append(n_kept / n_val)
            lo, hi = wilson_ci(k_pass, n_kept)
            point_ci_lo.append(lo)
            point_ci_hi.append(hi)
        else:
            point_prec.append(np.nan)
            point_cov.append(np.nan)
            point_ci_lo.append(np.nan)
            point_ci_hi.append(np.nan)
    point_prec = np.array(point_prec)
    point_cov = np.array(point_cov)
    point_ci_lo = np.array(point_ci_lo)
    point_ci_hi = np.array(point_ci_hi)
    boot_ci_lo = np.nanpercentile(boot_prec, 2.5, axis=0)
    boot_ci_hi = np.nanpercentile(boot_prec, 97.5, axis=0)

    # (b) Calibration: equal-width bins
    bin_centres, bin_mean_conf, bin_emp_prec, bin_lo, bin_hi, bin_n = [], [], [], [], [], []
    for j in range(len(CAL_BINS) - 1):
        lo, hi = CAL_BINS[j], CAL_BINS[j + 1]
        mask = (scores >= lo) & (scores < hi if j < len(CAL_BINS) - 2 else scores <= hi)
        n = int(mask.sum())
        if n < 5:
            continue
        k = int(passes[mask].sum())
        emp = k / n
        ci_l, ci_h = wilson_ci(k, n)
        bin_centres.append((lo + hi) / 2)
        bin_mean_conf.append(scores[mask].mean())
        bin_emp_prec.append(emp)
        bin_lo.append(ci_l)
        bin_hi.append(ci_h)
        bin_n.append(n)

    # (c) Coverage on the full deliverable
    full_scores = metrics["confidence_score"].to_numpy()
    full_sizes = metrics["cluster_size"].to_numpy()
    cov_n, cov_n4 = [], []
    for t in THRESHOLDS:
        m = full_scores >= t
        cov_n.append(int(m.sum()))
        cov_n4.append(int((m & (full_sizes == 4)).sum()))
    cov_n = np.array(cov_n)
    cov_n4 = np.array(cov_n4)

    # Headline operating point (overall, T=0)
    overall_k = int(passes.sum())
    overall_n = len(passes)
    overall_p = overall_k / overall_n
    overall_lo, overall_hi = wilson_ci(overall_k, overall_n)
    print(f"Headline: precision = {overall_p:.3f} "
          f"(Wilson 95% CI {overall_lo:.3f}, {overall_hi:.3f}), n={overall_n}")

    # Save CSV
    out_df = pd.DataFrame({
        "threshold": THRESHOLDS,
        "n_validated_kept": point_n,
        "empirical_precision": point_prec,
        "wilson_ci_lo": point_ci_lo,
        "wilson_ci_hi": point_ci_hi,
        "bootstrap_ci_lo": boot_ci_lo,
        "bootstrap_ci_hi": boot_ci_hi,
        "validation_coverage": point_cov,
        "full_deliverable_clusters_kept": cov_n,
        "full_deliverable_4way_kept": cov_n4,
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.30)

    col_prec = "#2E86AB"
    col_band = "#a9d0e3"
    col_cal  = "#117a65"
    col_cov  = "#1a5276"
    col_4way = "#d35400"
    col_mark = "#c0392b"

    # (a) PR curve
    ax = axes[0]
    valid_mask = ~np.isnan(point_prec)
    ax.fill_between(THRESHOLDS[valid_mask],
                    boot_ci_lo[valid_mask] * 100,
                    boot_ci_hi[valid_mask] * 100,
                    color=col_band, alpha=0.5,
                    label="95% bootstrap CI (2,000 resamples)")
    ax.plot(THRESHOLDS[valid_mask], point_prec[valid_mask] * 100,
            "-", color=col_prec, linewidth=2.2,
            label="Empirical precision")
    ax.axhline(90, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0.31, 90.6, "90%", fontsize=8, color="grey")
    # mark operating point
    ax.scatter([THRESHOLDS[0]], [overall_p * 100], s=120, zorder=5,
               color=col_mark, marker="*", edgecolors="white", linewidths=1.5,
               label=f"Operating point: {overall_p*100:.1f}% (n={overall_n})")
    ax.set_xlabel("Confidence threshold $t$", fontsize=11)
    ax.set_ylabel("Precision (%)", fontsize=11)
    ax.set_title("(a) Empirical PR curve, clusters with confidence $\\geq t$",
                 fontsize=11)
    ax.set_ylim(60, 102)
    ax.set_xlim(0.30, 1.00)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # (b) Calibration / reliability
    ax = axes[1]
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="grey",
            linewidth=1, label="Perfect calibration")
    err_lo = np.array(bin_emp_prec) - np.array(bin_lo)
    err_hi = np.array(bin_hi) - np.array(bin_emp_prec)
    ax.errorbar(bin_mean_conf, bin_emp_prec,
                yerr=[err_lo, err_hi],
                fmt="o", color=col_cal, capsize=4, markersize=7,
                linewidth=1.5, label="Validation data (Wilson 95% CI)")
    for x, y, n in zip(bin_mean_conf, bin_emp_prec, bin_n):
        ax.annotate(f"n={n}", (x, y),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=8, color=col_cal)
    ax.set_xlabel("Mean predicted confidence in bin", fontsize=11)
    ax.set_ylabel("Empirical pass rate", fontsize=11)
    ax.set_title("(b) Reliability diagram", fontsize=11)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    # (c) Coverage curve
    ax = axes[2]
    total = cov_n[0]
    ax.plot(THRESHOLDS, cov_n, "-", color=col_cov, linewidth=2.2,
            label="Clusters kept")
    ax.plot(THRESHOLDS, cov_n4, "--", color=col_4way, linewidth=2,
            label="4-way subset")
    ax.axvline(THRESHOLDS[0], color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.scatter([THRESHOLDS[0]], [total], s=80, zorder=5,
               color=col_cov, edgecolors="white", linewidths=1.5)
    ax.annotate(f"current: {total:,} total\n({cov_n4[0]:,} 4-way)",
                xy=(THRESHOLDS[0], total),
                xytext=(0.40, total * 0.78),
                fontsize=9, color=col_cov,
                arrowprops=dict(arrowstyle="->", color=col_cov, lw=1))
    ax.set_xlabel("Confidence threshold $t$", fontsize=11)
    ax.set_ylabel("Clusters retained in full deliverable", fontsize=11)
    ax.set_title("(c) Coverage curve", fontsize=11)
    ax.set_xlim(0.30, 1.00)
    ax.set_ylim(0, total * 1.08)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("ShopWiser cluster deliverable: precision, calibration, coverage",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
