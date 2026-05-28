"""
Independent Review Metrics for ShopWiser Cluster Pipeline
==========================================================

Applies the full validation methodology from REVIEW.md (review-of-shopwiser-contractor-work/)
to any cluster deliverable produced by the pipeline.

The seven-part framework:

  Part 1  Basic shape: 4-way / 3-way / 2-way cluster counts, coverage
  Part 2  Pipeline origin: stage-1-pure vs post-hoc-assembled clusters
  Part 3  Structural checks re-run with explicit null handling
  Part 4  Independent quality probe: TF-IDF char cosine + token Jaccard +
          size agreement + brand uniqueness + category uniqueness →
          per-cluster confidence score + likely_good binary
  Part 5  Coverage stats: raw products in / out of clusters
  Part 6  Calibrated precision estimates per cluster size
  Part 7  Attribute-extraction coverage

Outputs:
  console  – labelled sections matching REVIEW.md numbering
  data/intermediate/cluster_review_metrics.csv  – per-cluster scores
  data/intermediate/review_summary.json         – machine-readable headline numbers

Usage:
    uv run python scripts/review_metrics.py
    uv run python scripts/review_metrics.py --clusters path/to/ensemble.csv
    uv run python scripts/review_metrics.py --clusters path/to/ensemble.csv \\
        --raw path/to/raw.csv --out-dir path/to/output/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CLUSTERS = REPO_ROOT / "data/intermediate/ensemble_clusters.csv"
DEFAULT_RAW = REPO_ROOT / "data/input/raw.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data/intermediate"

# Calibration constants from hand-audit of 20 clusters (10 flagged good, 10 bad)
# "Of 10 flagged bad: 8 genuinely wrong (precision of bad label = 80% error rate)
#  Of 10 flagged good: 9 correct (precision of good label = 90%)"
_CAL_GOOD_TRUE = 0.90   # P(correct | probe says good)
_CAL_BAD_TRUE = 0.20    # P(correct | probe says bad)

# Stopwords for Jaccard tokeniser
_STOP = frozenset({
    "the", "a", "of", "and", "to", "in", "on", "for", "with", "by", "from",
    "pack", "x", "g", "kg", "ml", "l", "litre", "litres", "grams",
    "large", "small", "medium", "extra", "special", "best", "finest",
    "essentials", "just", "tesco", "sains", "sainsbury", "sainsburys",
    "asda", "morrisons", "morrison", "pcs", "count", "ct", "co", "home",
    "each", "new", "original", "everyday", "our", "more", "plus",
    "quality", "value", "range",
})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def _tokenise(s: str) -> frozenset[str]:
    s = re.sub(r"[^a-zA-Z' ]", " ", str(s).lower())
    return frozenset(t for t in s.split() if t not in _STOP and len(t) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _size_ratio(vals: np.ndarray) -> float:
    """Max/min ratio of non-zero size values; returns 1.0 when fewer than 2 known."""
    vals = vals[~np.isnan(vals)]
    vals = vals[vals > 0]
    if len(vals) < 2:
        return 1.0
    return float(vals.max() / vals.min())


# ---------------------------------------------------------------------------
# Part 1 – Basic shape
# ---------------------------------------------------------------------------

def part_1_basic_shape(df: pd.DataFrame) -> dict:
    _section("PART 1  Basic shape of the deliverable")

    sizes = df.groupby("ensemble_cluster_id").size()
    n_total = len(df)
    n_clusters = sizes.shape[0]
    cnt_4 = int((sizes == 4).sum())
    cnt_3 = int((sizes == 3).sum())
    cnt_2 = int((sizes == 2).sum())
    cnt_other = int(((sizes != 4) & (sizes != 3) & (sizes != 2)).sum())

    print(f"Total products in deliverable : {n_total:,}")
    print(f"Total clusters                : {n_clusters:,}")
    print()
    print(f"  4-way clusters : {cnt_4:>7,}")
    print(f"  3-way clusters : {cnt_3:>7,}")
    print(f"  2-way clusters : {cnt_2:>7,}")
    if cnt_other:
        print(f"  other sizes    : {cnt_other:>7,}")
    print()
    print(f"By retailer: {df['supermarket'].value_counts().to_dict()}")

    return {
        "n_products_in_deliverable": n_total,
        "n_clusters": n_clusters,
        "n_4way": cnt_4,
        "n_3way": cnt_3,
        "n_2way": cnt_2,
    }


# ---------------------------------------------------------------------------
# Part 2 – Pipeline origin (stage-1-pure vs post-hoc-assembled)
# ---------------------------------------------------------------------------

def part_2_pipeline_origin(df: pd.DataFrame) -> dict:
    _section("PART 2  Pipeline origin (stage-1-pure vs post-hoc-assembled)")

    if "cluster_id" not in df.columns:
        print("  cluster_id column absent — skipping origin analysis.")
        return {}

    by_origin = df.groupby("ensemble_cluster_id")["cluster_id"].apply(
        lambda x: x.dropna().nunique()
    )
    pure = int((by_origin == 1).sum())
    merged = int((by_origin > 1).sum())
    total = pure + merged
    print(f"  Stage-1-only clusters (cluster_id count == 1) : {pure:,}  ({pure/total*100:.1f}%)")
    print(f"  Post-hoc assembled clusters (cluster_id > 1)  : {merged:,}  ({merged/total*100:.1f}%)")

    sizes = df.groupby("ensemble_cluster_id").size()
    pure_ids = set(by_origin[by_origin == 1].index)
    post_ids = set(by_origin[by_origin > 1].index)

    def _size_dist(ids):
        s = sizes.loc[sizes.index.isin(ids)]
        return {
            "4way": int((s == 4).sum()),
            "3way": int((s == 3).sum()),
            "2way": int((s == 2).sum()),
        }

    p_dist = _size_dist(pure_ids)
    m_dist = _size_dist(post_ids)
    print(f"  Stage-1 breakdown  : 4-way={p_dist['4way']:,}  3-way={p_dist['3way']:,}  2-way={p_dist['2way']:,}")
    print(f"  Post-hoc breakdown : 4-way={m_dist['4way']:,}  3-way={m_dist['3way']:,}  2-way={m_dist['2way']:,}")

    return {"stage1_pure": pure, "post_hoc_assembled": merged,
            "stage1_dist": p_dist, "posthoc_dist": m_dist}


# ---------------------------------------------------------------------------
# Part 3 – Structural checks with explicit null handling
# ---------------------------------------------------------------------------

def part_3_structural_checks(df: pd.DataFrame) -> dict:
    _section("PART 3  Structural checks (with explicit null handling)")

    n_clusters = df["ensemble_cluster_id"].nunique()
    print(f"Running 4 structural checks across {n_clusters:,} clusters...\n")

    v1 = v2 = v3 = v5 = 0
    cross_cat = 0

    for _cid, g in df.groupby("ensemble_cluster_id"):
        # Check 1: one product per supermarket
        if g["supermarket"].duplicated().any():
            v1 += 1

        # Check 2: pack-size mismatch > 15%
        sv = g["unit_value"].dropna().values
        if len(sv) >= 2 and (float(sv.max()) / float(sv.min()) - 1) > 0.15:
            v2 += 1

        # Check 3: known_brand values disagree (null-aware — nulls excluded)
        if g["known_brand"].dropna().str.lower().str.strip().nunique() > 1:
            v3 += 1

        # Check 5: branded item (known_brand filled) mixed with own-brand
        ob_mask = g["own_brand"].astype(str).str.lower() == "true"
        has_branded_with_brand = ((~ob_mask) & g["known_brand"].notna()).any()
        if has_branded_with_brand and ob_mask.any():
            v5 += 1

        # Bonus: cross-category clusters (not in original contractor checks)
        if g["cat_norm"].dropna().nunique() > 1:
            cross_cat += 1

    total_viol = v1 + v2 + v3 + v5
    print(f"  Check 1  one product per supermarket per cluster  : {v1:>6,} violations")
    print(f"  Check 2  size mismatch > 15% (null-aware)         : {v2:>6,} violations")
    print(f"  Check 3  known_brand values disagree              : {v3:>6,} violations")
    print(f"  Check 5  branded mixed with own-brand             : {v5:>6,} violations")
    print(f"  Bonus    cross-category clusters                  : {cross_cat:>6,} "
          f"({cross_cat/n_clusters*100:.1f}%)")
    print()
    print(f"  Sum of violations across 4 checks: {total_viol:,}")
    print(f"  (Upper bound on failing clusters — some clusters fail multiple checks)")

    return {
        "check1_duplicate_retailer": v1,
        "check2_size_mismatch_15pct": v2,
        "check3_brand_conflict": v3,
        "check5_branded_vs_own_brand": v5,
        "cross_category_clusters": cross_cat,
        "cross_category_pct": round(cross_cat / n_clusters * 100, 1) if n_clusters else 0,
    }


# ---------------------------------------------------------------------------
# Part 4 – Independent quality probe
# ---------------------------------------------------------------------------

def part_4_quality_probe(df: pd.DataFrame) -> pd.DataFrame:
    _section("PART 4  Independent quality probe (TF-IDF cosine + token Jaccard)")

    print("Building character n-gram TF-IDF index over product names...")
    texts = df["names"].astype(str).tolist()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    print(f"  index: {X.shape[0]:,} items × {X.shape[1]:,} features")

    df = df.copy()
    df["_idx"] = np.arange(len(df))
    print("  pre-tokenising titles...")
    df["_toks"] = df["names"].astype(str).apply(_tokenise)

    print("  scoring clusters...")
    rows = []
    for cid, g in df.groupby("ensemble_cluster_id"):
        idx = g["_idx"].values
        n = len(idx)
        if n < 2:
            continue

        # Char-level cosine (sparse submatrix multiply)
        sub = X[idx]
        sim_mat = (sub @ sub.T).toarray()
        iu = np.triu_indices(n, k=1)
        min_cos = float(sim_mat[iu].min())

        # Word-level Jaccard
        tsets = g["_toks"].tolist()
        jacs = [
            _jaccard(tsets[i], tsets[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        min_jac = float(min(jacs))

        # Size ratio
        size_r = _size_ratio(g["unit_value"].values.astype(float))

        # Brand / category uniqueness
        n_brands = int(g["known_brand"].dropna().str.lower().str.strip().nunique())
        n_cats = int(g["cat_norm"].dropna().nunique())

        rows.append({
            "ensemble_cluster_id": cid,
            "cluster_size": n,
            "min_cos": min_cos,
            "min_jac": min_jac,
            "size_ratio": size_r,
            "n_brands": n_brands,
            "n_cats": n_cats,
        })

    stats = pd.DataFrame(rows)

    # --- Confidence score (weighted combination, higher = better) ---
    def _confidence(r: pd.Series) -> float:
        cos_t = max(0.0, min(1.0, (r["min_cos"] - 0.10) / 0.70))
        jac_t = max(0.0, min(1.0, (r["min_jac"] - 0.10) / 0.60))
        size_t = (1.0 if r["size_ratio"] <= 1.10
                  else 0.6 if r["size_ratio"] <= 1.20
                  else 0.2)
        brand_t = 1.0 if r["n_brands"] <= 1 else 0.3
        cat_t = 1.0 if r["n_cats"] <= 1 else 0.5
        return (0.30 * cos_t + 0.25 * jac_t + 0.20 * size_t
                + 0.15 * brand_t + 0.10 * cat_t)

    stats["confidence_score"] = stats.apply(_confidence, axis=1)

    # --- likely_good: all five signals pass their threshold ---
    stats["likely_good"] = (
        (stats["min_cos"] >= 0.40)
        & (stats["min_jac"] >= 0.25)
        & (stats["size_ratio"] <= 1.20)
        & (stats["n_brands"] <= 1)
        & (stats["n_cats"] <= 1)
    )

    print()
    print(f"  {'size':<8}{'pass':>8}{'fail':>8}{'% pass':>10}{'count':>8}")
    for sz in [2, 3, 4]:
        sub = stats[stats["cluster_size"] == sz]
        passed = int(sub["likely_good"].sum())
        total = len(sub)
        pct = passed / total * 100 if total else 0
        print(f"  {sz}-way   {passed:>8,}{total - passed:>8,}{pct:>9.1f}%{total:>8,}")

    overall_pass = stats["likely_good"].mean() * 100
    mean_conf = stats["confidence_score"].mean() * 100
    print(f"\n  Overall probe pass rate  : {overall_pass:.1f}%")
    print(f"  Mean confidence score    : {mean_conf:.1f}%")

    return stats


# ---------------------------------------------------------------------------
# Part 5 – Coverage stats (raw vs matched)
# ---------------------------------------------------------------------------

def part_5_coverage(df: pd.DataFrame, raw: pd.DataFrame | None) -> dict:
    _section("PART 5  Coverage stats (raw dataset vs deliverable)")

    if raw is None:
        print("  Raw dataset not provided — skipping coverage analysis.")
        return {}

    n_raw = len(raw)
    n_matched = len(df)
    n_unmatched = n_raw - n_matched
    pct_covered = n_matched / n_raw * 100 if n_raw else 0

    print(f"  Raw corpus products  : {n_raw:,}")
    print(f"  In deliverable       : {n_matched:,}  ({pct_covered:.1f}%)")
    print(f"  Not in deliverable   : {n_unmatched:,}  ({100-pct_covered:.1f}%)")
    print()
    print("  By retailer (raw vs matched):")

    raw_counts = raw["supermarket"].value_counts()
    matched_counts = df["supermarket"].value_counts()
    for retailer in sorted(raw_counts.index):
        r_n = int(raw_counts.get(retailer, 0))
        m_n = int(matched_counts.get(retailer, 0))
        pct = m_n / r_n * 100 if r_n else 0
        print(f"    {retailer:<12}  raw={r_n:>6,}  matched={m_n:>6,}  ({pct:.1f}%)")

    return {
        "n_raw": n_raw,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "pct_covered": round(pct_covered, 1),
    }


# ---------------------------------------------------------------------------
# Part 6 – Calibrated precision estimates
# ---------------------------------------------------------------------------

def part_6_calibrated_precision(stats: pd.DataFrame) -> dict:
    _section("PART 6  Calibrated precision estimates per cluster size")

    print("  Calibration constants (from hand-audit of 20 random clusters):")
    print(f"    P(correct | probe says good) = {_CAL_GOOD_TRUE:.0%}")
    print(f"    P(correct | probe says bad)  = {_CAL_BAD_TRUE:.0%}")
    print()
    print(f"  Formula: true_precision ≈ pass_rate × {_CAL_GOOD_TRUE} "
          f"+ fail_rate × {_CAL_BAD_TRUE}")
    print()

    results = {}
    print(f"  {'size':<8}{'probe pass%':>13}{'calibrated%':>14}{'count':>8}")
    for sz in [2, 3, 4]:
        sub = stats[stats["cluster_size"] == sz]
        if len(sub) == 0:
            continue
        pass_rate = sub["likely_good"].mean()
        fail_rate = 1.0 - pass_rate
        calibrated = pass_rate * _CAL_GOOD_TRUE + fail_rate * _CAL_BAD_TRUE
        print(f"  {sz}-way   {pass_rate*100:>12.1f}%{calibrated*100:>13.1f}%{len(sub):>8,}")
        results[f"calibrated_precision_{sz}way"] = round(calibrated * 100, 1)
        results[f"probe_pass_rate_{sz}way"] = round(pass_rate * 100, 1)

    # Overall
    pass_rate_all = stats["likely_good"].mean()
    fail_rate_all = 1.0 - pass_rate_all
    cal_all = pass_rate_all * _CAL_GOOD_TRUE + fail_rate_all * _CAL_BAD_TRUE
    print(f"  {'overall':<8}{pass_rate_all*100:>12.1f}%{cal_all*100:>13.1f}%{len(stats):>8,}")
    results["calibrated_precision_overall"] = round(cal_all * 100, 1)
    results["probe_pass_rate_overall"] = round(pass_rate_all * 100, 1)

    return results


# ---------------------------------------------------------------------------
# Part 7 – Attribute coverage
# ---------------------------------------------------------------------------

def part_7_attribute_coverage(df: pd.DataFrame) -> dict:
    _section("PART 7  Attribute-extraction layer coverage")

    n = len(df)
    fields = [
        ("brand recognised (known_brand)", "known_brand"),
        ("tier recognised (tier_keyword)", "tier_keyword"),
        ("pack size extracted (unit_value)", "unit_value"),
        ("attribute keywords (attributes_keywords)", "attributes_keywords"),
        ("descriptors", "descriptors"),
    ]

    results = {}
    print(f"  {'field':<48}{'filled':>8}   {'%':>6}")
    for label, col in fields:
        if col not in df.columns:
            continue
        nn = int(df[col].notna().sum())
        pct = nn / n * 100
        print(f"  {label:<48}{nn:>8,}  ({pct:>5.1f}%)")
        results[f"coverage_{col}"] = round(pct, 1)

    branded_no_brand = int(
        ((df["own_brand"].astype(str).str.lower() == "false")
         & df["known_brand"].isna()).sum()
    )
    print(f"\n  Branded items with no recognised brand (whitelist gap) : "
          f"{branded_no_brand:,}")
    results["branded_no_known_brand"] = branded_no_brand

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Independent review metrics for ShopWiser cluster deliverables."
    )
    p.add_argument(
        "--clusters",
        default=str(DEFAULT_CLUSTERS),
        help="Path to ensemble clusters CSV (default: ensemble_clusters.csv)",
    )
    p.add_argument(
        "--raw",
        default=str(DEFAULT_RAW),
        help="Path to raw scraped data CSV (default: data/input/raw.csv)",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for output files (default: data/intermediate/)",
    )
    p.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip the TF-IDF quality probe (faster, skips parts 4 & 6)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    clusters_path = Path(args.clusters)
    raw_path = Path(args.raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading clusters: {clusters_path}")
    df = pd.read_csv(clusters_path, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    raw = None
    if raw_path.exists():
        print(f"Loading raw corpus: {raw_path}")
        raw = pd.read_csv(raw_path, low_memory=False)
        print(f"  {len(raw):,} rows loaded")
    else:
        print(f"  Raw CSV not found at {raw_path} — skipping coverage part.")

    summary: dict = {}

    shape = part_1_basic_shape(df)
    summary.update(shape)

    origin = part_2_pipeline_origin(df)
    summary.update(origin)

    checks = part_3_structural_checks(df)
    summary.update(checks)

    stats: pd.DataFrame | None = None
    if not args.skip_probe:
        stats = part_4_quality_probe(df)
        # Save per-cluster scores
        metrics_path = out_dir / "cluster_review_metrics.csv"
        stats.to_csv(metrics_path, index=False)
        print(f"\n  Per-cluster scores saved to {metrics_path}")

    coverage = part_5_coverage(df, raw)
    summary.update(coverage)

    if stats is not None:
        precision = part_6_calibrated_precision(stats)
        summary.update(precision)

    attr = part_7_attribute_coverage(df)
    summary.update(attr)

    # --- Headline summary ---
    _section("SUMMARY  Headline numbers")
    print(f"  Cluster counts     :  4-way={summary.get('n_4way',0):,}  "
          f"3-way={summary.get('n_3way',0):,}  "
          f"2-way={summary.get('n_2way',0):,}")
    if "n_raw" in summary:
        print(f"  Coverage           :  {summary['n_matched']:,} / {summary['n_raw']:,} "
              f"products  ({summary['pct_covered']:.1f}%)")
    if "calibrated_precision_overall" in summary:
        print(f"  Calibrated precision:")
        print(f"    4-way : {summary.get('calibrated_precision_4way', '?')}%  "
              f"(probe pass {summary.get('probe_pass_rate_4way', '?')}%)")
        print(f"    3-way : {summary.get('calibrated_precision_3way', '?')}%  "
              f"(probe pass {summary.get('probe_pass_rate_3way', '?')}%)")
        print(f"    2-way : {summary.get('calibrated_precision_2way', '?')}%  "
              f"(probe pass {summary.get('probe_pass_rate_2way', '?')}%)")
        print(f"    overall: {summary.get('calibrated_precision_overall', '?')}%")
    if "cross_category_clusters" in summary:
        print(f"  Cross-category clusters : {summary['cross_category_clusters']:,} "
              f"({summary['cross_category_pct']:.1f}%)")

    # --- Write summary JSON ---
    summary_path = out_dir / "review_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Machine-readable summary saved to {summary_path}")
    _section("DONE")


if __name__ == "__main__":
    main()
