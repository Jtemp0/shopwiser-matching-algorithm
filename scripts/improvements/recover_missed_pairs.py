"""
Recover obvious cross-retailer matches that the deliverable left as singletons.

This addresses the §5.1 finding in the auditor's review: pairs like
'Morrisons Organic Brown Onions' vs 'ASDA Organic Brown Onions' which are
trivially the same product once the retailer prefix is consistently stripped.

Approach
--------
1. Load the raw catalogue and the deliverable.
2. Identify singleton products (in the raw set but not in any cluster).
3. Strip retailer prefixes ('ASDA', 'Tesco', 'Sainsbury's', 'Morrisons',
   'Tesco Finest', 'ASDA Extra Special', 'Just Essentials by ASDA',
   'Morrisons The Best', etc.) from the names.
4. Within each category, find cross-retailer pairs whose stripped names match
   at TF-IDF char-cosine >= 0.85 and have compatible pack sizes.
5. Emit candidate pairs as a CSV the next pipeline run can ingest.

Output
------
  data/outputs/improvements/recovered_pairs.csv
      idx_a, idx_b, retailer_a, retailer_b, name_a, name_b, similarity,
      size_ratio, category

Usage
-----
    uv run python scripts/improvements/recover_missed_pairs.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NORMALIZED = REPO_ROOT / "data/processed/normalized_products.csv"
CLUSTERS = REPO_ROOT / "data/outputs/ensemble/ensemble_clusters_v11.csv"
OUT = REPO_ROOT / "data/outputs/improvements/recovered_pairs_v11.csv"

SIM_THRESHOLD = 0.85
SIZE_RATIO_MAX = 1.20

# Order matters: longer prefixes first so they're stripped before short ones.
RETAILER_PREFIXES = [
    "just essentials by asda",
    "asda extra special",
    "asda organics",
    "sainsbury's taste the difference",
    "sainsburys taste the difference",
    "taste the difference",
    "morrisons the best",
    "morrisons savers",
    "tesco finest",
    "tesco organic",
    "by sainsbury's",
    "by sainsburys",
    "sainsbury's",
    "sainsburys",
    "morrisons",
    "morrison",
    "tesco",
    "asda",
]


def strip_prefixes(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9'& ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    changed = True
    while changed:
        changed = False
        for p in RETAILER_PREFIXES:
            if s.startswith(p + " "):
                s = s[len(p) + 1:].strip()
                changed = True
                break
    return s


def main() -> None:
    print(f"Loading raw normalized: {NORMALIZED}")
    raw = pd.read_csv(NORMALIZED, low_memory=False)
    print(f"  {len(raw):,} products")

    print(f"Loading clusters     : {CLUSTERS}")
    clu = pd.read_csv(CLUSTERS, low_memory=False)
    matched_idx = set(clu["product_idx"].dropna().astype(int))
    print(f"  {len(matched_idx):,} products already in clusters")

    raw = raw.reset_index(drop=False).rename(columns={"index": "product_idx"})
    singles = raw[~raw["product_idx"].isin(matched_idx)].copy()
    print(f"  {len(singles):,} singletons (not yet clustered)")

    singles["stripped"] = singles["names"].apply(strip_prefixes)
    singles = singles[singles["stripped"].str.len() >= 5]

    out_rows: list[dict] = []
    for cat, block in singles.groupby("category"):
        if block["supermarket"].nunique() < 2:
            continue
        if len(block) < 2:
            continue

        texts = block["stripped"].tolist()
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=20_000,
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts)

        idxs = block.index.to_numpy()
        retailers = block["supermarket"].to_numpy()
        unit_vals = block["unit_value"].to_numpy()

        sim = (X @ X.T).toarray()
        np.fill_diagonal(sim, 0.0)

        ii, jj = np.where(sim >= SIM_THRESHOLD)
        for i, j in zip(ii, jj):
            if i >= j:
                continue
            if retailers[i] == retailers[j]:
                continue
            uv_i, uv_j = unit_vals[i], unit_vals[j]
            try:
                a, b = float(uv_i), float(uv_j)
                if a > 0 and b > 0 and not (np.isnan(a) or np.isnan(b)):
                    if max(a, b) / min(a, b) > SIZE_RATIO_MAX:
                        continue
                    sr = round(max(a, b) / min(a, b), 3)
                else:
                    sr = None
            except (TypeError, ValueError):
                sr = None

            r_i = block.loc[idxs[i]]
            r_j = block.loc[idxs[j]]
            out_rows.append({
                "idx_a": int(r_i["product_idx"]),
                "idx_b": int(r_j["product_idx"]),
                "retailer_a": r_i["supermarket"],
                "retailer_b": r_j["supermarket"],
                "name_a": r_i["names"],
                "name_b": r_j["names"],
                "similarity": round(float(sim[i, j]), 3),
                "size_ratio": sr,
                "category": cat,
            })

    df = (pd.DataFrame(out_rows)
            .sort_values("similarity", ascending=False)
            .reset_index(drop=True))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nRecovered {len(df):,} candidate cross-retailer pairs (sim >= {SIM_THRESHOLD}).")
    print(f"  → {OUT}")
    if len(df):
        print("\nTop 15 by similarity:")
        print(df.head(15)[["retailer_a", "retailer_b", "name_a", "name_b", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()
