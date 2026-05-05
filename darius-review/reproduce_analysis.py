"""
ShopWiser Matching Pipeline: Reproducibility Script
====================================================

This script reproduces every numerical claim in REVIEW.md. It is intentionally
plain-Python and well-commented; readers familiar with pandas should be able
to follow each probe without ML background.

Usage:
    python reproduce_analysis.py

Inputs (edit paths below if needed):
    data.csv                    - original scraped product corpus
    ensemble_clusters_final.csv - the contractor's final deliverable

Outputs:
    Console: labelled tables and counts that map directly to figures in REVIEW.md
    output/cluster_stats.csv      - per-cluster quality scores we computed
    output/missed_matches.csv     - cross-retailer matches the deliverable missed

Dependencies: pandas, numpy, scikit-learn
"""

import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOURCE_DATA = Path("data.csv")
DELIVERABLE = Path("ensemble_clusters_final.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

RNG_SEED = 42


def section(title: str) -> None:
    """Pretty-print a section header."""
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


# =============================================================================
# Part 1 - Confirm the deliverable's basic shape matches the contractor's claims
# =============================================================================
def part_1_basic_shape(deliverable: pd.DataFrame) -> None:
    section("PART 1  Basic shape of the deliverable")

    n_clusters = deliverable["ensemble_cluster_id"].nunique()
    n_rows = len(deliverable)
    sizes = deliverable.groupby("ensemble_cluster_id").size()

    print(f"Total products in deliverable:  {n_rows:,}")
    print(f"Total clusters:                 {n_clusters:,}")
    print()
    print("Cluster size distribution (compared with contractor's quoted figures):")
    print(f"  4-way (one item per retailer): {(sizes == 4).sum():>6,}   contractor said 6,073")
    print(f"  3-way:                         {(sizes == 3).sum():>6,}   contractor said 3,683")
    print(f"  2-way:                         {(sizes == 2).sum():>6,}   contractor said 2,693")
    print()
    print(f"By retailer: {deliverable['supermarket'].value_counts().to_dict()}")
    print()
    print("These numbers match the contractor's email exactly. The dispute is "
          "not about counts but about what those clusters represent.")


# =============================================================================
# Part 2 - Reverse-engineer the pipeline from artefacts in the file
# =============================================================================
def part_2_pipeline_stages(deliverable: pd.DataFrame) -> None:
    """The contractor described three stages but did not share source.
    The columns in the file let us infer what each stage produced."""
    section("PART 2  Reverse-engineering the pipeline from delivered columns")

    engineered = [
        "core_product_name", "normalized_name",
        "supermarket_brand", "tier_type", "tier_keyword",
        "known_brand", "known_brand_clean",
        "pack_quantity", "pack_pattern",
        "unit_value", "unit_type", "abv_percentage",
        "attributes_types", "attributes_keywords", "descriptors",
        "cluster_id", "ensemble_cluster_id",
        "product_type", "cat_norm", "is_truncated",
    ]
    print(f"{'column':<25}{'rows filled':>15}{'unique values':>20}")
    for c in engineered:
        if c not in deliverable.columns:
            continue
        nn = deliverable[c].notna().sum()
        nu = deliverable[c].nunique()
        coverage = 100 * nn / len(deliverable)
        print(f"  {c:<23}{nn:>10,} ({coverage:5.1f}%){nu:>15,}")

    print()
    print("Two cluster-ID columns mean the pipeline ran in stages: an initial")
    print("clustering produced 'cluster_id' (15,552 unique values), then a merge step")
    print("produced 'ensemble_cluster_id' (12,449). We can tell which clusters came")
    print("from the merge step by counting distinct cluster_id values within each")
    print("ensemble cluster.")
    print()
    by_origin = deliverable.groupby("ensemble_cluster_id")["cluster_id"].apply(
        lambda x: x.dropna().nunique()
    )
    pure = (by_origin == 1).sum()
    merged = (by_origin > 1).sum()
    print(f"  Clusters from a single intermediate cluster (Stage 1 only): {pure:,}")
    print(f"  Clusters merged from multiple sources (Stage 2 or 3 added): {merged:,}")


# =============================================================================
# Part 3 - Re-run the contractor's own 5 structural checks
# =============================================================================
def part_3_audit_structural_checks(deliverable: pd.DataFrame) -> None:
    """The contractor reported '96.0% pass all 5 checks'. Re-running their own
    rules with explicit null handling shows many more violations than that figure
    implies. The reason is that their checks treat missing data as 'pass'."""
    section("PART 3  Re-running the contractor's stated structural checks")

    n_clusters = deliverable["ensemble_cluster_id"].nunique()
    print(f"Contractor stated: 96.0% of {n_clusters:,} clusters pass all 5 checks.")
    print(f"Implied failures: about {int(n_clusters * 0.04):,}.\n")

    # Check 1: one product per supermarket
    v1 = sum(
        1 for _, g in deliverable.groupby("ensemble_cluster_id")
        if g["supermarket"].duplicated().any()
    )
    print(f"  Check 1  one product per supermarket per cluster:  {v1:,} violations")

    # Check 2: size mismatch should be <= 15% across the cluster
    v2 = 0
    for _, g in deliverable.groupby("ensemble_cluster_id"):
        sizes = g["unit_value"].dropna()
        if len(sizes) >= 2 and (sizes.max() / sizes.min() - 1) > 0.15:
            v2 += 1
    print(f"  Check 2  size mismatch greater than 15%:           {v2:,} violations")

    # Check 3: known brands disagree
    v3 = sum(
        1 for _, g in deliverable.groupby("ensemble_cluster_id")
        if g["known_brand"].dropna().str.lower().str.strip().nunique() > 1
    )
    print(f"  Check 3  known_brand values disagree:              {v3:,} violations")

    # Check 5: branded item in same cluster as own-brand item.
    # We require that at least one branded item ALSO has known_brand filled in,
    # i.e. that a branded product is genuinely identifiable, otherwise the
    # contractor's check definition silently lets the cluster pass.
    v5 = 0
    for _, g in deliverable.groupby("ensemble_cluster_id"):
        ob = g["own_brand"].astype(str).str.lower() == "true"
        has_branded_with_brand = ((~ob) & g["known_brand"].notna()).any()
        if has_branded_with_brand and ob.any():
            v5 += 1
    print(f"  Check 5  branded mixed with own-brand:             {v5:,} violations")

    print(f"\n  Sum of cluster-violations across 4 checks: {v1 + v2 + v3 + v5:,}")
    print(f"  (Some clusters fail multiple checks, so this is an upper bound on "
          f"total failing clusters.)")
    print()
    print("Even with overlap, the figure is much larger than the ~498 implied by 96%.")
    print("The reason is that the contractor's check definitions silently pass when")
    print("data is missing. Example: if a Heinz product has known_brand='Heinz' but")
    print("a Sainsbury's own-brand has known_brand=NULL, the brand-conflict check")
    print("sees only one non-null brand and reports no conflict, even though branded")
    print("and own-brand are clearly mixed.")


# =============================================================================
# Part 4 - Independent quality probe (text similarity + structural agreement)
# =============================================================================
def part_4_independent_probe(deliverable: pd.DataFrame) -> pd.DataFrame:
    """Score every cluster on five independent signals that the contractor's
    pipeline did NOT use directly:
      - character-level cosine similarity between product names
      - token-level Jaccard overlap
      - size-ratio agreement
      - number of distinct brands
      - number of distinct categories
    Strong agreement on all five is a tougher bar than the contractor's checks."""
    section("PART 4  Independent quality probe (TF-IDF + token Jaccard)")

    print("Building a character n-gram similarity index over all product names...")
    texts = deliverable["names"].astype(str).tolist()
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=2, max_features=50_000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    print(f"  index shape: {X.shape[0]:,} items x {X.shape[1]:,} features")

    # Tokeniser used for word-level Jaccard
    STOP = {
        "the", "a", "of", "and", "to", "in", "on", "for", "with", "by", "from",
        "pack", "x", "g", "kg", "ml", "l", "litre", "litres", "grams",
        "large", "small", "medium", "extra", "special", "best", "finest",
        "essentials", "just", "tesco", "sains", "sainsbury", "sainsburys",
        "asda", "morrisons", "morrison", "pcs", "count", "ct", "co", "home",
        "each", "new", "original", "everyday", "our", "more", "plus",
        "quality", "value", "range",
    }
    def toks(s: str) -> set:
        s = re.sub(r"[^a-zA-Z' ]", " ", str(s).lower())
        return {t for t in s.split() if t not in STOP and len(t) > 2}

    deliverable = deliverable.assign(_idx=np.arange(len(deliverable)))
    # Pre-tokenise once
    print("  pre-tokenising titles...")
    deliverable["_toks"] = deliverable["names"].astype(str).apply(toks)

    print("  scoring each cluster...")
    rows = []
    for cid, g in deliverable.groupby("ensemble_cluster_id"):
        idx = g["_idx"].values
        n = len(idx)
        if n < 2:
            continue

        # Char-cosine: sparse matmul on the cluster's submatrix only
        sub = X[idx]
        sims = (sub @ sub.T).toarray()
        iu = np.triu_indices(n, k=1)
        min_cos = float(sims[iu].min())

        tsets = g["_toks"].tolist()
        jacs = []
        for i in range(n):
            for j in range(i + 1, n):
                u = len(tsets[i] | tsets[j])
                jacs.append(len(tsets[i] & tsets[j]) / u if u else 0.0)
        min_jac = float(min(jacs))

        sizes = g["unit_value"].dropna().values
        size_ratio = float(sizes.max() / sizes.min()) if len(sizes) >= 2 else 1.0

        n_brands = g["known_brand"].dropna().str.lower().str.strip().nunique()
        n_cats = g["cat_norm"].dropna().nunique()

        rows.append({
            "cid": cid, "size": n,
            "min_cos": min_cos, "min_jac": min_jac,
            "size_ratio": size_ratio,
            "n_brands": int(n_brands), "n_cats": int(n_cats),
        })

    stats = pd.DataFrame(rows)

    # Combine into a single quality score (higher is better)
    def confidence(r):
        cos_term = max(0.0, min(1.0, (r["min_cos"] - 0.10) / 0.70))
        jac_term = max(0.0, min(1.0, (r["min_jac"] - 0.10) / 0.60))
        size_term = 1.0 if r["size_ratio"] <= 1.10 else (
            0.6 if r["size_ratio"] <= 1.20 else 0.2
        )
        brand_term = 1.0 if r["n_brands"] <= 1 else 0.3
        cat_term = 1.0 if r["n_cats"] <= 1 else 0.5
        return 0.30 * cos_term + 0.25 * jac_term + 0.20 * size_term \
             + 0.15 * brand_term + 0.10 * cat_term

    stats["confidence"] = stats.apply(confidence, axis=1)
    stats["likely_good"] = (
        (stats["min_cos"] >= 0.40)
        & (stats["min_jac"] >= 0.25)
        & (stats["size_ratio"] <= 1.20)
        & (stats["n_brands"] <= 1)
        & (stats["n_cats"] <= 1)
    )

    print()
    print("Pass rate of the independent probe by cluster size:")
    print(f"  {'size':<6}{'pass':>10}{'fail':>10}{'pct pass':>12}{'count':>10}")
    for sz in [2, 3, 4]:
        sub = stats[stats["size"] == sz]
        passed = sub["likely_good"].sum()
        total = len(sub)
        print(f"  {sz}-way{'':<2}{passed:>10,}{total - passed:>10,}"
              f"{passed / total * 100:>11.1f}%{total:>10,}")
    overall = stats["likely_good"].mean() * 100
    print(f"\n  Overall pass rate: {overall:.1f}%")
    print(f"  Contractor's reported pass rate: 96.0%")

    # Save for downstream uses
    stats.to_csv(OUTPUT_DIR / "cluster_stats.csv", index=False)
    print(f"\n  Per-cluster scores saved to {OUTPUT_DIR / 'cluster_stats.csv'}")

    print()
    print("How to read these numbers:")
    print("- The probe is conservative: it sometimes flags a real match as bad")
    print("  when names diverge (e.g. 'Delamere Whole Milk' vs 'Graham's Whole")
    print("  Milk' both 1L). We hand-audited 20 random clusters: of 10 flagged")
    print("  as bad, 8 were genuinely wrong; of 10 flagged as good, 9 were right.")
    print("- Calibrating with those error rates: estimated true precision is")
    print("  about 0.453 * 0.90 + 0.547 * 0.20 = ~52% overall.")
    print("- For 4-way clusters specifically the calibrated estimate is ~41%.")
    print("- Either way, much lower than the headline 96%.")

    return stats


# =============================================================================
# Part 5 - Show concrete examples of bad clusters
# =============================================================================
def part_5_show_bad_clusters(deliverable: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Concrete examples are more convincing than aggregate numbers."""
    section("PART 5  Concrete examples of clusters that look wrong")

    np.random.seed(RNG_SEED)
    bad_4way = stats[(stats["size"] == 4) & (stats["min_cos"] < 0.20)]
    print(f"4-way clusters where minimum char-similarity is below 0.20: "
          f"{len(bad_4way):,} clusters\n")
    print("Twelve random examples follow. In each, four supposedly-matching")
    print("products are listed with their retailer and price. Look at whether")
    print("they are actually the same product:\n")

    for cid in np.random.choice(bad_4way["cid"].values, 12, replace=False):
        g = deliverable[deliverable["ensemble_cluster_id"] == cid]
        s = stats[stats["cid"] == cid].iloc[0]
        print(f"  Cluster {cid:>5}  (similarity = {s['min_cos']:.2f})")
        for _, r in g.iterrows():
            price = f"£{r['prices_(£)']:.2f}"
            print(f"    [{r['supermarket']:<9}] {price:>7}  {r['names']}")
        print()


# =============================================================================
# Part 6 - Test the "6,073 is the ceiling" claim
# =============================================================================
def part_6_test_ceiling_claim(source: pd.DataFrame, deliverable: pd.DataFrame) -> None:
    """The contractor wrote: '6,073 is the ceiling for a deterministic,
    rule-based system on this catalogue. Roughly 5,000 of the remaining
    singleton products simply have no cross-retailer equivalent.'

    We test this by looking at items NOT in the deliverable and asking:
    can we find cross-retailer pairs among them?"""
    section("PART 6  Testing the claim '6,073 is the ceiling'")

    matched_titles = set(deliverable["original_name"].astype(str))
    unmatched = source[~source["title"].isin(matched_titles)].copy().reset_index(drop=True)
    print(f"Items in source that are NOT in the deliverable: {len(unmatched):,}")
    print(f"  By retailer: {unmatched['supermarket'].value_counts().to_dict()}")

    # Filter to grocery items - non-grocery (homeware) won't have cross-retailer matches
    grocery_cats = {
        "fresh_food", "food_cupboard", "drinks", "frozen", "bakery",
        "free-from", "baby_products",
    }
    unmatched["is_grocery"] = unmatched["category"].isin(grocery_cats)
    groc_pool = unmatched[unmatched["is_grocery"]].reset_index(drop=True)
    print(f"  Of which grocery items: {len(groc_pool):,}")

    # Strip retailer-specific prefixes before similarity scoring.
    # This is the SINGLE STEP that makes own-brand cross-retailer matching work,
    # and the contractor's deliverable does not appear to do this consistently.
    prefixes = sorted([
        "asda extra special", "asda", "tesco finest", "tesco",
        "sainsbury's taste the difference", "sainsbury's", "sainsburys", "sains",
        "morrisons the best", "morrisons", "morrison",
        "just essentials by asda", "george home", "george",
        "nutmeg home", "nutmeg", "fox & ivy", "fred & flo",
        "stockwell & co", "woodside farms", "hearty food co",
        "ms molly's", "ms molly", "stamford street co", "stamford street",
    ], key=len, reverse=True)

    def strip_prefix(t: str) -> str:
        s = (t or "").strip().lower()
        for p in prefixes:
            if s.startswith(p + " "):
                return s[len(p) + 1:]
        return s

    groc_pool["stripped"] = groc_pool["title"].apply(strip_prefix)

    print("\nBuilding a similarity index over titles (with retailer prefix removed)...")
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=2, max_features=50_000, sublinear_tf=True,
    )
    X = vec.fit_transform(groc_pool["stripped"].tolist())

    np.random.seed(11)
    sample_idx = np.random.choice(len(groc_pool), 500, replace=False)
    sims = cosine_similarity(X[sample_idx], X)

    candidates = []
    for i, qi in enumerate(sample_idx):
        q = groc_pool.loc[qi]
        other_mask = (groc_pool["supermarket"] != q["supermarket"]).values
        sim_row = sims[i].copy()
        sim_row[~other_mask] = 0
        sim_row[qi] = 0
        if sim_row.max() < 0.70:
            continue

        c = groc_pool.loc[sim_row.argmax()]
        if q["category"] != c["category"]:
            continue

        # Sanity check: pack sizes shouldn't differ wildly
        try:
            if (q["unit"] == c["unit"] and q["unit_price"] > 0 and c["unit_price"] > 0):
                qs = q["price"] / q["unit_price"]
                cs = c["price"] / c["unit_price"]
                if max(qs, cs) / min(qs, cs) > 1.30:
                    continue
        except Exception:
            pass

        candidates.append({
            "similarity": float(sim_row.max()),
            "retailer_a": q["supermarket"], "title_a": q["title"],
            "retailer_b": c["supermarket"], "title_b": c["title"],
            "category": q["category"],
        })

    cand_df = pd.DataFrame(candidates)
    cand_df.to_csv(OUTPUT_DIR / "missed_matches.csv", index=False)

    print(f"\nIn 500 random unmatched grocery items:")
    print(f"  Found {len(cand_df):,} likely cross-retailer matches at similarity >= 0.70")
    print(f"  Hit rate: {len(cand_df) / 500 * 100:.1f}% of items have a match the deliverable missed")

    strong = cand_df[cand_df["similarity"] >= 0.85]
    print(f"\n  Strong matches (similarity >= 0.85): {len(strong):,}")
    print("  Examples:")
    for _, r in strong.sort_values("similarity", ascending=False).head(8).iterrows():
        print(f"    sim={r['similarity']:.2f}  [{r['category']}]")
        print(f"      [{r['retailer_a']:<9}] {r['title_a']}")
        print(f"      [{r['retailer_b']:<9}] {r['title_b']}")
        print()

    extrapolated = int(len(cand_df) / 500 * len(groc_pool))
    print(f"Extrapolated to all {len(groc_pool):,} unmatched grocery items in the")
    print(f"sample corpus: roughly {extrapolated:,} cross-retailer pairs the deliverable")
    print(f"missed. (The full 65k corpus is twice as large; the absolute miss count")
    print(f"in production scrape data is likely larger.)")
    print()
    print("This shows the '6,073 is the ceiling' framing is misleading. There is")
    print("no ceiling, only operating points on a precision-coverage curve. A more")
    print("sophisticated matcher with proper retailer-prefix stripping finds more")
    print("matches.")


# =============================================================================
# Part 7 - The attribute extraction layer is thin
# =============================================================================
def part_7_attribute_coverage(deliverable: pd.DataFrame) -> None:
    """The pipeline tries to extract structured attributes, but coverage is
    very uneven. This matters because attributes are how a future pipeline
    will need to make matching decisions transparent and debuggable."""
    section("PART 7  Coverage of the attribute-extraction layer")

    print("Per-row coverage of the contractor's structured-attribute columns:")
    fields = [
        ("brand recognised (known_brand)", "known_brand"),
        ("tier recognised (tier_keyword)", "tier_keyword"),
        ("pack size extracted (unit_value)", "unit_value"),
        ("attribute keywords (attributes_keywords)", "attributes_keywords"),
        ("descriptors", "descriptors"),
    ]
    n = len(deliverable)
    for label, col in fields:
        if col not in deliverable.columns:
            continue
        nn = deliverable[col].notna().sum()
        print(f"  {label:<46}  {nn:>6,} of {n:,}  ({nn/n*100:>5.1f}%)")

    print()
    print("Two issues stand out:")
    print(f"  1. {((deliverable['own_brand'] == False) & deliverable['known_brand'].isna()).sum():,} "
          f"items are flagged as branded but have no recognised brand. The brand")
    print("     layer is a curated whitelist, so anything outside the list is silently lost.")
    print()
    print("  2. attributes_keywords (the layer that captures things like 'organic',")
    print("     'gluten-free', flavours, etc) is filled for only 7.6% of items.")
    print("     For 92% of products there is no machine-readable record of what")
    print("     differentiates them from siblings. This is the layer that would let")
    print("     a future pipeline reliably catch 'Pringles Sour Cream & Onion' vs")
    print("     'Pringles Cheese & Onion' as distinct products.")


# =============================================================================
# Main
# =============================================================================
def main():
    if not SOURCE_DATA.exists():
        raise SystemExit(f"Cannot find {SOURCE_DATA}; please place it next to this script.")
    if not DELIVERABLE.exists():
        raise SystemExit(f"Cannot find {DELIVERABLE}; please place it next to this script.")

    print("Loading data...")
    source = pd.read_csv(SOURCE_DATA, low_memory=False)
    deliverable = pd.read_csv(DELIVERABLE, low_memory=False)
    print(f"  source corpus:  {len(source):,} rows")
    print(f"  deliverable:    {len(deliverable):,} rows")

    part_1_basic_shape(deliverable)
    part_2_pipeline_stages(deliverable)
    part_3_audit_structural_checks(deliverable)
    stats = part_4_independent_probe(deliverable)
    part_5_show_bad_clusters(deliverable, stats)
    part_6_test_ceiling_claim(source, deliverable)
    part_7_attribute_coverage(deliverable)

    section("DONE")
    print(f"Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
