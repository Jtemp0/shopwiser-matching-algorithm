"""Merge heuristic false-positive signals into one per-cluster CSV.

Reads ensemble CSV + cluster_review_metrics.csv produced by scripts/review_metrics.py
for the same ensemble file.

Usage:
  uv run python scripts/export_fp_candidates.py
  uv run python scripts/export_fp_candidates.py \\
      --clusters data/outputs/ensemble/ensemble_clusters.csv \\
      --metrics data/outputs/ensemble/cluster_review_metrics.csv \\
      --out data/outputs/fp_analys/fp_candidates.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shopwiser.conflict_tokens import check_hard_conflict  # noqa: E402

DEFAULT_CLUSTERS = REPO / "data/outputs/ensemble/ensemble_clusters.csv"
DEFAULT_METRICS = REPO / "data/outputs/ensemble/cluster_review_metrics.csv"
DEFAULT_OUT = REPO / "data/outputs/fp_analys/fp_candidates.csv"


def _names_norm(g: pd.DataFrame) -> list[str]:
    return g["normalized_name"].fillna("").astype(str).tolist()


def _has_hard_conflict(g: pd.DataFrame) -> bool:
    names = _names_norm(g)
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
    return any(primary not in set(nm.split()) for nm in _names_norm(g))


def _struct_brand_known_disagree(g: pd.DataFrame) -> bool:
    """Part-3 style: raw known_brand column differs (not clean)."""
    if "known_brand" not in g.columns:
        return False
    vals = g["known_brand"].dropna().astype(str).str.lower().str.strip()
    return vals.nunique() > 1 if len(vals) else False


def _struct_size_15pct(g: pd.DataFrame) -> bool:
    """Part-3 style: unit_value max/min > 1.15 when ≥2 numeric."""
    sv = g["unit_value"]
    if sv.notna().sum() < 2:
        return False
    v = sv.dropna().astype(float).values
    return len(v) >= 2 and (float(v.max()) / float(v.min()) - 1) > 0.15


def _struct_branded_vs_own(g: pd.DataFrame) -> bool:
    ob = g["own_brand"].astype(str).str.lower() == "true"
    has_brand = ((~ob) & g["known_brand"].notna()).any()
    return bool(has_brand and ob.any())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--size-tol", type=float, default=0.15)
    args = p.parse_args()

    df = pd.read_csv(args.clusters, low_memory=False)
    met = pd.read_csv(args.metrics, low_memory=False)

    rows: list[dict] = []
    for cid, g in df.groupby("ensemble_cluster_id"):
        sz = len(g)
        sm_dup = len(g) != g["supermarket"].nunique()
        hc = _has_hard_conflict(g)
        delta = _max_size_mismatch(g)
        size_audit = delta > args.size_tol
        bm = _has_brand_mismatch(g)
        bob = _branded_own_brand_no_shared_token(g)
        cross_cat = g["cat_norm"].dropna().nunique() > 1 if "cat_norm" in g.columns else False
        multi_core = (
            g["core_product_name"].dropna().astype(str).str.strip().nunique() > 1
            if "core_product_name" in g.columns
            else False
        )
        s_brand = _struct_brand_known_disagree(g)
        s_size = _struct_size_15pct(g)
        s_bo = _struct_branded_vs_own(g)

        # Strong = same family as scripts/audit_clusters.py (actionable mismatches).
        strong: list[str] = []
        if sm_dup:
            strong.append("dup_supermarket")
        if hc:
            strong.append("hard_conflict")
        if size_audit:
            strong.append("size_mismatch_audit")
        if bm:
            strong.append("brand_clean_mismatch")
        if bob:
            strong.append("branded_own_no_token")

        # Weak = taxonomy / extraction noise — useful for review queues, not automatic FP.
        weak: list[str] = []
        if cross_cat:
            weak.append("cross_category")
        if multi_core:
            weak.append("multi_core_product_name")
        if s_brand:
            weak.append("struct_known_brand_disagree")
        if s_size:
            weak.append("struct_unit_value_15pct")
        if s_bo:
            weak.append("struct_branded_vs_ownbrand")

        flags = strong + weak

        rows.append({
            "ensemble_cluster_id": int(cid),
            "cluster_size": sz,
            "strong_flags": "|".join(strong),
            "weak_flags": "|".join(weak),
            "flags": "|".join(flags),
            "n_strong": len(strong),
            "n_weak": len(weak),
            "n_flags": len(flags),
            "max_size_delta_audit": round(delta, 6),
            "supermarkets": "|".join(sorted(g["supermarket"].astype(str).unique())),
            "titles_preview": " || ".join(
                str(x)[:70] for x in g.sort_values("supermarket")["names"].head(4)
            ),
        })

    out = pd.DataFrame(rows).merge(
        met,
        on=["ensemble_cluster_id", "cluster_size"],
        how="left",
    )
    out["probe_likely_bad"] = ~out["likely_good"].fillna(True)
    out["low_confidence"] = out["confidence_score"] < 0.55
    out["very_low_text_sim"] = (out["min_cos"] < 0.35) & (out["min_jac"] < 0.22)

    def _risk(row: pd.Series) -> str:
        bits: list[str] = []
        if row["n_strong"] > 0:
            bits.append("strong_heuristic")
        if row["n_weak"] > 0:
            bits.append("weak_heuristic")
        if row["probe_likely_bad"]:
            bits.append("probe_fail")
        if row["low_confidence"]:
            bits.append("low_conf")
        if row["very_low_text_sim"]:
            bits.append("text_sim_crash")
        return "|".join(bits) if bits else "clean"

    out["risk_bucket"] = out.apply(_risk, axis=1)
    out["suspicious_strong"] = out["n_strong"] > 0
    out["suspicious_union"] = (
        out["suspicious_strong"]
        | out["probe_likely_bad"]
        | out["low_confidence"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(
        ["suspicious_union", "n_strong", "n_weak", "confidence_score"],
        ascending=[False, False, False, True],
    ).to_csv(args.out, index=False)

    n_susp = int(out["suspicious_union"].sum())
    n_str = int(out["suspicious_strong"].sum())
    n_heur = int((out["n_flags"] > 0).sum())
    print(f"Wrote {args.out} ({len(out):,} clusters)")
    print(f"  Strong heuristic (audit-style): {n_str:,}")
    print(f"  Any heuristic (strong+weak)    : {n_heur:,}")
    print(f"  suspicious_union (∪probe∪lowconf): {n_susp:,}")


if __name__ == "__main__":
    main()
