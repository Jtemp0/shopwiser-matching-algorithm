"""Blocking keys and candidate pair generation for clustering passes."""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import *  # noqa: F403
from .similarity import compute_similarity

# ============================================================
# BLOCKING
# ============================================================

STOPWORDS = {'the', 'a', 'an', 'of', 'and', 'with', 'in', 'for', 'to', 'by', '&', 'or', 'no'}


def _unit_bucket(uv, bucket_size=50):
    if pd.isna(uv) or uv <= 0:
        return 'no_unit'
    return str(round(float(uv) / bucket_size) * bucket_size)


def _get_tokens(norm_name: str) -> list:
    return [t for t in str(norm_name or '').lower().split() if t not in STOPWORDS]


def build_blocks(sub_df, key_fn, max_block_size=MAX_BLOCK_SIZE):
    raw_blocks = defaultdict(list)
    for idx, row in sub_df.iterrows():
        raw_blocks[key_fn(row)].append(idx)

    blocks = []
    total_pairs = 0
    for key, indices in raw_blocks.items():
        block_df = sub_df.loc[indices]
        if block_df['supermarket'].nunique() < 2:
            continue
        if len(indices) > max_block_size:
            # Use TWO bucket sizes (500g and 1000g) so products near a 500g
            # boundary aren't siloed into non-overlapping sub-blocks.
            sub_raw: dict = defaultdict(list)
            for idx in indices:
                uv = sub_df.loc[idx, 'unit_value']
                b500  = key + '||' + _unit_bucket(uv, bucket_size=500)
                b1000 = key + '||' + _unit_bucket(uv, bucket_size=1000)
                sub_raw[b500].append(idx)
                if b500 != b1000:   # only add second bucket when they differ
                    sub_raw[b1000].append(idx)
            for sub_key, sub_idx in sub_raw.items():
                # dedup seen pairs within the same parent block
                if sub_df.loc[sub_idx, 'supermarket'].nunique() < 2:
                    continue
                blocks.append((sub_key, sub_idx))
                n = len(sub_idx)
                total_pairs += n * (n - 1) // 2
        else:
            blocks.append((key, indices))
            n = len(indices)
            total_pairs += n * (n - 1) // 2

    block_sizes = [len(b[1]) for b in blocks]
    print(f'    Blocks: {len(blocks):,}  |  Pairs: {total_pairs:,}')
    if block_sizes:
        print(f'    Size — mean: {np.mean(block_sizes):.1f}, max: {max(block_sizes)}, median: {int(np.median(block_sizes))}')
    return blocks


def build_multi_blocks(sub_df, key_fns, max_block_size=MAX_BLOCK_SIZE):
    seen_pairs = set()
    all_pairs  = []
    for key_fn in key_fns:
        blocks = build_blocks(sub_df, key_fn, max_block_size)
        for _key, indices in blocks:
            for ia, ib in combinations(sorted(indices), 2):
                pair = (min(ia, ib), max(ia, ib))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_pairs.append((ia, ib))
    print(f'    Total unique pairs after dedup: {len(all_pairs):,}')
    return all_pairs


def run_pass(sub_df, pairs, pass_type, unit_tolerance):
    matches = []
    for ia, ib in tqdm(pairs, desc=f'  Pass={pass_type}', leave=True):
        is_match, score = compute_similarity(sub_df.loc[ia], sub_df.loc[ib], pass_type, unit_tolerance)
        if is_match:
            matches.append((ia, ib, score))
    return matches
