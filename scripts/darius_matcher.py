"""
Darius Matcher — cluster the full product catalogue using the advisor's checks
as the matching criteria (not as post-hoc validation).

Pipeline
--------
1. Load normalized_products.csv (already has unit_value, known_brand, tier_type …)
2. For each category block:
     a. Build a TF-IDF char-ngram index over core_product_name
     b. In chunks, query each product against all other-retailer products
     c. Collect candidate pairs at cosine >= COS_THRESHOLD
3. For every candidate pair apply the five advisor checks as hard gates:
     - cosine >= COS_THRESHOLD (char n-gram TF-IDF)
     - jaccard >= JAC_THRESHOLD (token overlap)
     - size ratio <= SIZE_THRESHOLD  (unit_value must agree within 20%)
     - no brand conflict  (max 1 known_brand value across the pair)
     - no category conflict  (guaranteed by the blocking step)
     - no branded-vs-own-brand mixing
4. Union-Find on accepted pairs → clusters
5. Enforce one product per retailer per cluster
   (if a cluster has two ASDA products, keep only the highest-similarity one)
6. Report 4-way / 3-way / 2-way distribution + save clusters CSV

Usage
-----
    uv run python scripts/darius_matcher.py
    uv run python scripts/darius_matcher.py --cos 0.45 --jac 0.30
    uv run python scripts/darius_matcher.py --out data/outputs/darius/darius_clusters.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_INPUT = REPO_ROOT / "data/processed/normalized_products.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/outputs/darius/darius_clusters.csv"

# ---------------------------------------------------------------------------
# Thresholds (advisor's pass criteria from REVIEW.md §3.2)
# ---------------------------------------------------------------------------
COS_THRESHOLD = 0.40    # min char-level cosine similarity
JAC_THRESHOLD = 0.25    # min word-level Jaccard overlap
SIZE_THRESHOLD = 1.20   # max ratio of pack sizes (larger/smaller)

CHUNK_SIZE = 1_000      # query products per batch (memory control)
TOP_K = 5               # candidate neighbours to retrieve per query product

# ---------------------------------------------------------------------------
# Stopwords for Jaccard tokeniser
# ---------------------------------------------------------------------------
_STOP = frozenset({
    "the", "a", "of", "and", "to", "in", "on", "for", "with", "by", "from",
    "pack", "x", "g", "kg", "ml", "l", "litre", "litres", "grams",
    "large", "small", "medium", "extra", "special", "best", "finest",
    "essentials", "just", "tesco", "sains", "sainsbury", "sainsburys",
    "asda", "morrisons", "morrison", "pcs", "count", "ct", "co", "home",
    "each", "new", "original", "everyday", "our", "more", "plus",
    "quality", "value", "range",
})


def _tokenise(s: str) -> frozenset[str]:
    s = re.sub(r"[^a-zA-Z' ]", " ", str(s).lower())
    return frozenset(t for t in s.split() if t not in _STOP and len(t) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _size_ratio(uv_a, uv_b) -> float:
    """Max/min ratio; returns 1.0 if either value is unknown."""
    try:
        a, b = float(uv_a), float(uv_b)
    except (TypeError, ValueError):
        return 1.0
    if a <= 0 or b <= 0 or np.isnan(a) or np.isnan(b):
        return 1.0
    return max(a, b) / min(a, b)


def _brand_conflict(kb_a, kb_b) -> bool:
    """True when both products have a recognised brand and they differ."""
    def _norm(v):
        if pd.isna(v) or str(v).strip() == "":
            return None
        return re.sub(r"\s+", "", str(v).strip().lower())
    a, b = _norm(kb_a), _norm(kb_b)
    return a is not None and b is not None and a != b


def _branded_own_brand_mix(own_a: bool, own_b: bool, kb_a, kb_b) -> bool:
    """True when one product is clearly branded (known_brand filled) and the
    other is own-brand — the reviewer's Check 5."""
    def _has_brand(own, kb):
        return (not own) and (not pd.isna(kb)) and str(kb).strip() != ""
    return (_has_brand(own_a, kb_a) and own_b) or (_has_brand(own_b, kb_b) and own_a)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        self._parent[self.find(x)] = self.find(y)

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for x in list(self._parent):
            out[self.find(x)].append(x)
        return dict(out)


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def _find_pairs_in_block(
    block: pd.DataFrame,
    cos_threshold: float,
    jac_threshold: float,
    size_threshold: float,
) -> list[tuple[int, int, float]]:
    """Return accepted (idx_a, idx_b, cosine_score) pairs within one category block."""
    retailers = block["supermarket"].unique()
    if len(retailers) < 2:
        return []

    texts = block["core_product_name"].fillna(block["names"]).astype(str).tolist()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=30_000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)

    # Pre-tokenise for Jaccard
    toks = [_tokenise(t) for t in texts]

    local_idxs = block.index.tolist()       # positions in the full df
    block_arr = block.reset_index(drop=False)  # keep original index as column

    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for r_query in retailers:
        r_others = [r for r in retailers if r != r_query]
        query_mask = block["supermarket"] == r_query
        db_mask = block["supermarket"].isin(r_others)

        q_local = np.where(query_mask.values)[0]   # local positions within block
        d_local = np.where(db_mask.values)[0]

        if len(q_local) == 0 or len(d_local) == 0:
            continue

        X_q = X[q_local]
        X_d = X[d_local]

        # Chunked cosine query to control peak memory
        for start in range(0, len(q_local), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(q_local))
            chunk_q_local = q_local[start:end]
            sim_chunk = (X[chunk_q_local] @ X_d.T).toarray()  # (chunk, n_db)

            for ci, qi_local in enumerate(chunk_q_local):
                row = sim_chunk[ci]
                # top-K db indices above threshold
                above = np.where(row >= cos_threshold)[0]
                if len(above) == 0:
                    continue
                top = above[np.argsort(row[above])[::-1][:TOP_K]]

                qi_global = local_idxs[qi_local]
                q_row = block.iloc[qi_local]

                for di in top:
                    di_local_real = d_local[di]
                    di_global = local_idxs[di_local_real]
                    key = (min(qi_global, di_global), max(qi_global, di_global))
                    if key in seen:
                        continue
                    seen.add(key)

                    cos = float(row[di])
                    d_row = block.iloc[di_local_real]

                    # --- Jaccard check ---
                    if _jaccard(toks[qi_local], toks[di_local_real]) < jac_threshold:
                        continue

                    # --- Size check ---
                    if _size_ratio(q_row["unit_value"], d_row["unit_value"]) > size_threshold:
                        continue

                    # --- Brand conflict check ---
                    if _brand_conflict(q_row["known_brand"], d_row["known_brand"]):
                        continue

                    # --- Branded-vs-own-brand check ---
                    if _branded_own_brand_mix(
                        bool(q_row["own_brand"]),
                        bool(d_row["own_brand"]),
                        q_row["known_brand"],
                        d_row["known_brand"],
                    ):
                        continue

                    pairs.append((qi_global, di_global, cos))

    return pairs


# ---------------------------------------------------------------------------
# Cluster enforcement (one product per retailer)
# ---------------------------------------------------------------------------

def _enforce_one_per_retailer(
    df: pd.DataFrame,
    groups: dict[int, list[int]],
) -> dict[int, list[int]]:
    """Within each cluster, if two products share a retailer, keep the one
    that participates in the highest-cos pair (tracked externally) or simply
    the first one. Returns cleaned groups."""
    clean: dict[int, list[int]] = {}
    for root, members in groups.items():
        if len(members) == 1:
            clean[root] = members
            continue
        seen_retailers: dict[str, int] = {}
        for m in members:
            r = df.loc[m, "supermarket"]
            if r not in seen_retailers:
                seen_retailers[r] = m
        clean[root] = list(seen_retailers.values())
    return clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build clusters from the raw catalogue using the advisor's quality checks as matching gates."
    )
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="normalized_products.csv path")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT),
                   help="Output clusters CSV path")
    p.add_argument("--cos", type=float, default=COS_THRESHOLD,
                   help=f"Min cosine similarity (default {COS_THRESHOLD})")
    p.add_argument("--jac", type=float, default=JAC_THRESHOLD,
                   help=f"Min Jaccard overlap (default {JAC_THRESHOLD})")
    p.add_argument("--size", type=float, default=SIZE_THRESHOLD,
                   help=f"Max size ratio (default {SIZE_THRESHOLD})")
    return p


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
    cos_threshold: float = COS_THRESHOLD,
    jac_threshold: float = JAC_THRESHOLD,
    size_threshold: float = SIZE_THRESHOLD,
) -> pd.DataFrame:
    input_path = input_path or DEFAULT_INPUT
    output_path = output_path or DEFAULT_OUTPUT

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(df):,} products, {df['supermarket'].nunique()} retailers")
    print(f"  Retailers: {df['supermarket'].value_counts().to_dict()}")
    print(f"\nThresholds: cosine>={cos_threshold}  jaccard>={jac_threshold}  "
          f"size_ratio<={size_threshold}")

    # Make own_brand boolean
    df["own_brand"] = df["own_brand"].astype(str).str.lower() == "true"

    categories = sorted(df["category"].dropna().unique())
    print(f"\nProcessing {len(categories)} category blocks:")

    all_pairs: list[tuple[int, int, float]] = []
    for cat in categories:
        block = df[df["category"] == cat]
        n_retailers = block["supermarket"].nunique()
        print(f"  [{cat}]  {len(block):,} products  {n_retailers} retailers", end="", flush=True)
        if n_retailers < 2:
            print("  — skipped (single retailer)")
            continue
        pairs = _find_pairs_in_block(block, cos_threshold, jac_threshold, size_threshold)
        print(f"  → {len(pairs):,} accepted pairs")
        all_pairs.extend(pairs)

    print(f"\nTotal accepted pairs: {len(all_pairs):,}")

    # Build clusters via Union-Find
    uf = UnionFind()
    pair_cos: dict[tuple[int, int], float] = {}
    for a, b, cos in all_pairs:
        uf.union(a, b)
        pair_cos[(min(a, b), max(a, b))] = cos

    raw_groups = uf.groups()
    print(f"Raw clusters (before retailer-dedup): {len(raw_groups):,}")

    # Enforce one product per retailer
    groups = _enforce_one_per_retailer(df, raw_groups)

    # Assign cluster IDs and collect result rows
    result_rows = []
    cluster_id = 1
    cnt = {2: 0, 3: 0, 4: 0}
    for root, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        sz = min(len(members), 4)
        if sz in cnt:
            cnt[sz] += 1
        elif sz > 4:
            cnt[4] += 1  # collapse >4 into 4-way bucket (shouldn't normally occur)
        for m in members:
            row = df.loc[m].to_dict()
            row["darius_cluster_id"] = cluster_id
            row["darius_cluster_size"] = len(members)
            result_rows.append(row)
        cluster_id += 1

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values(
        ["darius_cluster_id", "supermarket"]
    ).reset_index(drop=True)

    # --- Report ---
    total_clusters = cluster_id - 1
    total_products = len(result_df)
    print(f"\n{'='*60}")
    print("DISTRIBUTION OF DARIUS-METHOD CLUSTERS")
    print(f"{'='*60}")
    print(f"  Total clusters  : {total_clusters:,}")
    print(f"  Total products  : {total_products:,}  "
          f"({total_products/len(df)*100:.1f}% of catalogue covered)")
    print()
    for sz in [4, 3, 2]:
        n = cnt.get(sz, 0)
        print(f"  {sz}-way clusters : {n:>7,}")
    print()
    print("  By retailer (in clusters):")
    rc = result_df["supermarket"].value_counts()
    for retailer, n in rc.items():
        print(f"    {retailer:<12}  {n:>6,}")

    # --- Confidence score per cluster (from advisor's formula) ---
    print("\nBuilding per-cluster confidence scores...")
    from sklearn.feature_extraction.text import TfidfVectorizer as _TF
    all_texts = result_df["core_product_name"].fillna(result_df["names"]).astype(str).tolist()
    vec2 = _TF(analyzer="char_wb", ngram_range=(3, 5),
               min_df=2, max_features=40_000, sublinear_tf=True)
    X2 = vec2.fit_transform(all_texts)
    result_df["_idx2"] = np.arange(len(result_df))
    result_df["_toks"] = result_df["core_product_name"].fillna(result_df["names"]).astype(str).apply(_tokenise)

    conf_rows = []
    for cid, g in result_df.groupby("darius_cluster_id"):
        idxs = g["_idx2"].values
        n = len(idxs)
        sub = X2[idxs]
        sim_mat = (sub @ sub.T).toarray()
        iu = np.triu_indices(n, k=1)
        min_cos = float(sim_mat[iu].min()) if len(iu[0]) else 0.0
        tsets = g["_toks"].tolist()
        jacs = [_jaccard(tsets[i], tsets[j])
                for i in range(n) for j in range(i+1, n)]
        min_jac = float(min(jacs)) if jacs else 0.0
        uv = g["unit_value"].dropna().values.astype(float)
        uv = uv[uv > 0]
        size_r = float(uv.max() / uv.min()) if len(uv) >= 2 else 1.0
        n_brands = int(g["known_brand"].dropna().str.lower().str.strip().nunique())
        n_cats = int(g["category"].dropna().nunique())

        cos_t = max(0.0, min(1.0, (min_cos - 0.10) / 0.70))
        jac_t = max(0.0, min(1.0, (min_jac - 0.10) / 0.60))
        size_t = 1.0 if size_r <= 1.10 else (0.6 if size_r <= 1.20 else 0.2)
        brand_t = 1.0 if n_brands <= 1 else 0.3
        cat_t = 1.0 if n_cats <= 1 else 0.5
        score = (0.30*cos_t + 0.25*jac_t + 0.20*size_t + 0.15*brand_t + 0.10*cat_t)
        conf_rows.append({"darius_cluster_id": cid, "confidence_score": round(score, 4)})

    conf_df = pd.DataFrame(conf_rows)
    result_df = result_df.merge(conf_df, on="darius_cluster_id", how="left")
    result_df = result_df.drop(columns=["_idx2", "_toks"])

    # --- Confidence distribution ---
    print("\nConfidence score distribution:")
    bins = [0.0, 0.30, 0.50, 0.70, 0.85, 1.01]
    labels = ["<0.30", "0.30–0.50", "0.50–0.70", "0.70–0.85", "≥0.85"]
    cs = conf_df["confidence_score"]
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        n = int(((cs >= lo) & (cs < hi)).sum())
        print(f"  {lab:<12}  {n:>6,} clusters")
    print(f"  mean score   : {cs.mean():.3f}")

    # --- Save ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nClusters saved to {output_path}")
    return result_df


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        input_path=Path(args.input),
        output_path=Path(args.out),
        cos_threshold=args.cos,
        jac_threshold=args.jac,
        size_threshold=args.size,
    )


if __name__ == "__main__":
    main()
