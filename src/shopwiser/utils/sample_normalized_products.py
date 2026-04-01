"""
Create a sample CSV from the already-normalised products dataset.

Reads:  data/processed/normalized_products.csv
Writes: data/processed/normalized_products_sample.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from shopwiser.paths import normalized_products_path


def main(*, n: int = 1000, seed: int = 42) -> None:
    input_path = normalized_products_path(sample=False)
    output_path = normalized_products_path(sample=True)

    print("=" * 70)
    print("ShopWiser Normalised Products Sampler")
    print("=" * 70)
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Sample: {n} rows (seed={seed})")

    df = pd.read_csv(input_path, low_memory=False)
    total = len(df)

    if total == 0:
        raise RuntimeError(f"Input CSV has no rows: {input_path}")

    if total <= n:
        sample_df = df
    else:
        sample_df = df.sample(n=n, random_state=seed)
        # Keep the original CSV row order for easier diffs/inspection.
        sample_df = sample_df.sort_index()

    sample_df.to_csv(output_path, index=False)
    print(f"\n✓ Wrote {len(sample_df):,} rows → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sample normalized_products.csv → normalized_products_sample.csv")
    p.add_argument("--n", type=int, default=1000, help="Number of rows to sample (default: 1000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = p.parse_args()

    main(n=args.n, seed=args.seed)

