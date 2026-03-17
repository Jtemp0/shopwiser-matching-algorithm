"""
ShopWiser Clustering v2 — improved algorithm.

Key changes vs v1:
  1. Brand tokens stripped from normalized_name before fuzzy comparison (branded pass)
  2. token_sort_ratio replaces token_set_ratio as primary score
  3. Post-cluster validation: rebuild clusters from DIRECT-match edges only
     (prevents transitive-link artifacts with NaN pairwise scores)
  4. PACK_QTY_MAX_RATIO  4.0 → 2.0
  5. UNIT_TOLERANCE_BRANDED 0.05 → 0.03
  6. FUZZY_THRESHOLD_NOUNIT 0.88 → 0.90
  7. Extended attribute penalties: wine type, fish medium, spirit age, vegan
  8. Short-name guard: only blocks when BOTH names ≤2 tokens (was: either)
"""

import pandas as pd
import numpy as np
import os
import re
import random
import warnings
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from rapidfuzz import fuzz

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

UNIT_TOLERANCE_BRANDED    = 0.03   # ±3%  (was 0.05 — rejects 515g vs 540g = 4.9%)
UNIT_TOLERANCE_OWN_BRAND  = 0.03
UNIT_TOLERANCE_UNBRANDED  = 0.05

FUZZY_THRESHOLD           = 0.82
FUZZY_THRESHOLD_NOUNIT    = 0.90   # was 0.88

PACK_QTY_MAX_RATIO        = 2.0    # was 4.0

MAX_BLOCK_SIZE            = 200

ATTR_PENALTIES = {
    'organic':   0.20,
    'free_from': 0.20,
    'fairtrade': 0.10,
    'diet':      0.10,   # was 0.05
    'vegan':     0.25,
}

# Wine type: presence of any token in one name but different token in other → heavy penalty
WINE_TYPE_TOKENS = {'red', 'white', 'rose', 'rosé', 'sparkling', 'prosecco', 'champagne', 'blush'}

# Fish/seafood preservation medium tokens (word-boundary matched in names)
FISH_MEDIUM_TOKENS = {'brine', 'spring water', 'olive oil', 'sunflower oil', 'tomato', 'springwater'}

BRAND_EXCLUSIONS = {'extra', 'essential', 'basics', 'finest', 'select', 'special'}

OUTPUT_DIR = 'data/clusters'
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print('Configuration loaded (v2).')
print(f'  Fuzzy threshold:       {FUZZY_THRESHOLD} (no-unit: {FUZZY_THRESHOLD_NOUNIT})')
print(f'  Unit tolerance:        ±{UNIT_TOLERANCE_BRANDED*100:.0f}% branded, ±{UNIT_TOLERANCE_OWN_BRAND*100:.0f}% own-brand')
print(f'  Pack ratio max:        {PACK_QTY_MAX_RATIO}')
print(f'  Output directory:      {OUTPUT_DIR}')

# ============================================================
# DATA LOADING
# ============================================================

df = pd.read_csv('data/normalized_products.csv', low_memory=False)
print(f'\nLoaded {len(df):,} products, {df.shape[1]} columns')

REQUIRED_COLS = [
    'supermarket', 'names', 'category', 'own_brand',
    'supermarket_brand', 'tier_type', 'known_brand',
    'pack_quantity', 'unit_value', 'unit_type',
    'attributes_keywords', 'core_product_name', 'normalized_name'
]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
assert not missing, f'Missing columns: {missing}'

df = df.reset_index(drop=True)
df['product_idx'] = df.index

df['known_brand_clean'] = df['known_brand'].where(
    ~df['known_brand'].str.lower().isin(BRAND_EXCLUSIONS), other=None
)

df['product_type'] = np.where(
    df['known_brand_clean'].notna(), 'branded',
    np.where(df['own_brand'].astype(str).str.lower().isin(['true', '1', 'yes']), 'own_brand', 'unbranded')
)

df['has_unit'] = df['unit_value'].notna() & (df['unit_value'] > 0)
df['has_pack'] = df['pack_quantity'].notna() & (df['pack_quantity'] > 0)
df['supermarket'] = df['supermarket'].str.strip()

print(f'\nSupermarket distribution:\n{df["supermarket"].value_counts().to_string()}')
print(f'\nProduct type:\n{df["product_type"].value_counts().to_string()}')

# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def components(self):
        comps = defaultdict(list)
        for i in range(len(self.parent)):
            comps[self.find(i)].append(i)
        return dict(comps)

# ============================================================
# SIMILARITY HELPERS
# ============================================================

def _unit_value_compatible(uv_a, uv_b, tolerance):
    if pd.isna(uv_a) or pd.isna(uv_b) or uv_a <= 0 or uv_b <= 0:
        return True
    return abs(uv_a - uv_b) / max(uv_a, uv_b) <= tolerance


def _unit_type_compatible(ut_a, ut_b):
    if pd.isna(ut_a) or pd.isna(ut_b):
        return True
    return str(ut_a).strip() == str(ut_b).strip()


def _pack_compatible(pq_a, pq_b):
    if pd.isna(pq_a) or pd.isna(pq_b) or pq_a <= 0 or pq_b <= 0:
        return True
    return max(pq_a, pq_b) / min(pq_a, pq_b) <= PACK_QTY_MAX_RATIO


def _strip_brand(name: str, brand: str) -> str:
    """Remove brand tokens from normalized_name for discriminating comparison."""
    if not brand or pd.isna(brand):
        return name
    name_tokens = str(name).lower().split()
    brand_tokens = set(str(brand).lower().split())
    stripped = [t for t in name_tokens if t not in brand_tokens]
    result = ' '.join(stripped)
    # Fall back to original if stripping left too little
    return result if len(stripped) >= 2 else name


def _get_wine_type(text: str) -> set:
    tokens = set(text.lower().split())
    return tokens & WINE_TYPE_TOKENS


def _get_fish_medium(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for m in FISH_MEDIUM_TOKENS:
        if m in text_lower:
            found.add(m)
    return found


_AGE_RE = re.compile(r'(\d+)\s*year', re.IGNORECASE)


def _attribute_penalty(attrs_a, attrs_b, name_a='', name_b='') -> float:
    str_a = (str(attrs_a).lower() if pd.notna(attrs_a) else '') + ' ' + str(name_a).lower()
    str_b = (str(attrs_b).lower() if pd.notna(attrs_b) else '') + ' ' + str(name_b).lower()

    penalty = 0.0

    # Standard keyword attributes
    for attr, p in ATTR_PENALTIES.items():
        if (attr in str_a) != (attr in str_b):
            penalty += p

    # Wine type (red vs white vs rosé etc.)
    wine_a = _get_wine_type(str_a)
    wine_b = _get_wine_type(str_b)
    if wine_a and wine_b and wine_a != wine_b:
        penalty += 0.60   # decisive rejection

    # Fish medium (brine vs olive oil vs spring water)
    med_a = _get_fish_medium(str_a)
    med_b = _get_fish_medium(str_b)
    if med_a and med_b and med_a != med_b:
        penalty += 0.50

    # Spirit age (10 year vs 12 year)
    ages_a = [int(m.group(1)) for m in _AGE_RE.finditer(str_a)]
    ages_b = [int(m.group(1)) for m in _AGE_RE.finditer(str_b)]
    if ages_a and ages_b and set(ages_a) != set(ages_b):
        penalty += 0.60

    return min(penalty, 1.0)


def compute_similarity(row_a, row_b, pass_type, unit_tolerance):
    # Hard constraint 1: same supermarket
    if row_a['supermarket'] == row_b['supermarket']:
        return False, 0.0

    # Hard constraint 2: unit type
    if not _unit_type_compatible(row_a.get('unit_type'), row_b.get('unit_type')):
        return False, 0.0

    # Hard constraint 3: unit value
    if not _unit_value_compatible(row_a.get('unit_value'), row_b.get('unit_value'), unit_tolerance):
        return False, 0.0

    # Hard constraint 4: pack quantity
    if not _pack_compatible(row_a.get('pack_quantity'), row_b.get('pack_quantity')):
        return False, 0.0

    # Pass-specific brand/tier rules
    if pass_type == 'branded':
        if str(row_a['known_brand_clean']).lower() != str(row_b['known_brand_clean']).lower():
            return False, 0.0

    elif pass_type == 'own_brand':
        tier_a = str(row_a.get('tier_type', '')).lower() if pd.notna(row_a.get('tier_type')) else 'standard'
        tier_b = str(row_b.get('tier_type', '')).lower() if pd.notna(row_b.get('tier_type')) else 'standard'
        if tier_a != tier_b:
            return False, 0.0
        if row_a.get('product_type') != row_b.get('product_type'):
            return False, 0.0

    raw_name_a = str(row_a.get('normalized_name', '') or '')
    raw_name_b = str(row_b.get('normalized_name', '') or '')

    # --- BRAND STRIPPING (branded pass only) ---
    # Remove brand tokens so "mr kipling chocolate" vs "mr kipling cherry bakewell"
    # is compared as "chocolate" vs "cherry bakewell" — avoids brand inflation
    if pass_type == 'branded':
        brand = str(row_a.get('known_brand_clean', '') or '').lower()
        name_a = _strip_brand(raw_name_a, brand)
        name_b = _strip_brand(raw_name_b, brand)
    else:
        name_a, name_b = raw_name_a, raw_name_b

    # Short-name guard: only enforce when BOTH names are very short (was: either)
    if len(name_a.split()) <= 2 and len(name_b.split()) <= 2:
        if not (row_a['has_unit'] and row_b['has_unit']):
            return False, 0.0

    # --- FUZZY SCORING ---
    # Primary: token_sort_ratio (order-independent but no set tricks) + partial
    token_sort = fuzz.token_sort_ratio(name_a, name_b) / 100.0
    partial    = fuzz.partial_ratio(name_a, name_b) / 100.0

    # For verbose-name pairs (one name is ≥40% longer), blend in token_set
    # to handle e.g. "roast chicken drumsticks" vs "roast british cooked chicken drumsticks"
    len_a, len_b = len(name_a), len(name_b)
    len_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 1.0

    if len_ratio < 0.65:
        token_set = fuzz.token_set_ratio(name_a, name_b) / 100.0
        score = 0.45 * token_sort + 0.25 * token_set + 0.30 * partial
    else:
        score = 0.70 * token_sort + 0.30 * partial

    # Attribute penalty (extended: wine type, fish medium, spirit age)
    penalty = _attribute_penalty(
        row_a.get('attributes_keywords'), row_b.get('attributes_keywords'),
        name_a=raw_name_a, name_b=raw_name_b
    )
    score -= penalty

    threshold = FUZZY_THRESHOLD_NOUNIT if (not row_a['has_unit'] or not row_b['has_unit']) else FUZZY_THRESHOLD
    return score >= threshold, score

# ============================================================
# SELF-TESTS
# ============================================================

print('\nRunning similarity self-tests...')

def _mk(name, uv, ut, sm, brand='heinz', tier='standard', ptype='branded', pq=None, attrs=None):
    return pd.Series({
        'normalized_name': name, 'unit_value': uv, 'unit_type': ut,
        'supermarket': sm, 'has_unit': pd.notna(uv) and uv > 0,
        'pack_quantity': pq, 'attributes_keywords': attrs,
        'known_brand_clean': brand, 'tier_type': tier, 'product_type': ptype
    })

tests = [
    # (description, row_a, row_b, pass_type, tolerance, expected_match)
    ('Same product diff SM',
     _mk('heinz baked beans tomato sauce', 415, 'g', 'Tesco'),
     _mk('heinz baked beans tomato sauce', 415, 'g', 'ASDA'),
     'branded', UNIT_TOLERANCE_BRANDED, True),

    ('Plum vs Chopped tomatoes',
     _mk('plum tomatoes', 400, 'g', 'Tesco'),
     _mk('chopped tomatoes', 400, 'g', 'ASDA'),
     'unbranded', UNIT_TOLERANCE_UNBRANDED, False),

    ('Same supermarket',
     _mk('whole milk', 2270, 'g', 'Tesco'),
     _mk('whole milk', 2270, 'g', 'Tesco'),
     'unbranded', UNIT_TOLERANCE_UNBRANDED, False),

    ('Weight mismatch 200g vs 400g',
     _mk('sweetcorn water', 200, 'g', 'Tesco'),
     _mk('sweetcorn water', 400, 'g', 'ASDA'),
     'unbranded', UNIT_TOLERANCE_UNBRANDED, False),

    ('Unit type mismatch g vs ml',
     _mk('orange juice', 1000, 'g', 'Tesco'),
     _mk('orange juice', 1000, 'ml', 'ASDA'),
     'unbranded', UNIT_TOLERANCE_UNBRANDED, False),

    # NEW: brand stripping tests
    ('Chocolate slices vs Cherry bakewell (same brand) → must NOT match',
     _mk('mr kipling chocolate slices', None, None, 'Tesco', brand='mr kipling'),
     _mk('mr kipling cherry bakewell cakes', None, None, 'ASDA', brand='mr kipling'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Rhubarb kefir vs Honey kefir (same brand) → must NOT match',
     _mk('yeo valley kefir rhubarb fermented organic yogurt', 350, 'g', 'Tesco', brand='yeo valley'),
     _mk('yeo valley organic kefir honey yogurt', 350, 'g', 'ASDA', brand='yeo valley'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('19 Crimes Rosé vs Red wine → must NOT match',
     _mk('19 crimes revolutionary rose', 750, 'ml', 'Tesco', brand='19 crimes'),
     _mk('19 crimes red wine', 750, 'ml', 'ASDA', brand='19 crimes'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Tuna spring water vs olive oil → must NOT match',
     _mk('tuna chunks in spring water', 145, 'g', 'Tesco', brand=None, ptype='own_brand'),
     _mk('tuna chunks in olive oil', 145, 'g', 'ASDA', brand=None, ptype='own_brand'),
     'own_brand', UNIT_TOLERANCE_OWN_BRAND, False),

    ('Highland Park 12yr vs 10yr → must NOT match',
     _mk('highland park 12 year old single malt scotch whisky', 700, 'ml', 'Tesco', brand=None, ptype='unbranded'),
     _mk('highland park 10 year old single malt scotch whisky', 700, 'ml', 'ASDA', brand=None, ptype='unbranded'),
     'unbranded', UNIT_TOLERANCE_UNBRANDED, False),

    ('Verbose vs terse name (should MATCH)',
     _mk('roast chicken drumsticks', 430, 'g', 'Tesco', brand=None, ptype='own_brand'),
     _mk('roast british cooked chicken drumsticks', 430, 'g', 'Sains', brand=None, ptype='own_brand'),
     'own_brand', UNIT_TOLERANCE_OWN_BRAND, True),

    ('Pack mismatch 4x vs 10x → must NOT match',
     _mk('guinness 0 0 alcohol free draught stout', 440, 'ml', 'Tesco', brand='guinness', pq=4),
     _mk('guinness 0 0 alcohol free draught stout', 440, 'ml', 'ASDA', brand='guinness', pq=10),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Honey Cheerios 515g vs Multigrain Cheerios 540g → must NOT match',
     _mk('nestle honey cheerios cereal', 515, 'g', 'Tesco', brand='cheerios'),
     _mk('nestle cheerios multigrain cereal', 540, 'g', 'ASDA', brand='cheerios'),
     'branded', UNIT_TOLERANCE_BRANDED, False),
]

all_pass = True
for desc, ra, rb, ptype, tol, expected in tests:
    match, score = compute_similarity(ra, rb, ptype, tol)
    status = '✓' if match == expected else '✗ FAIL'
    if match != expected:
        all_pass = False
    print(f'  {status}  {desc}: match={match}, score={score:.3f}')

print(f'\nAll tests passed: {all_pass}')
if not all_pass:
    print('WARNING: some tests failed — review thresholds before proceeding')
    import sys; sys.exit(1)

# ============================================================
# BLOCKING
# ============================================================

STOPWORDS = {'the', 'a', 'an', 'of', 'and', 'with', 'in', 'for', 'to', 'by', '&', 'or', 'no'}


def _unit_bucket(uv, bucket_size=50):
    if pd.isna(uv) or uv <= 0:
        return 'no_unit'
    return str(round(float(uv) / bucket_size) * bucket_size)


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
            sub_raw = defaultdict(list)
            for idx in indices:
                sub_key = key + '||' + _unit_bucket(sub_df.loc[idx, 'unit_value'])
                sub_raw[sub_key].append(idx)
            for sub_key, sub_idx in sub_raw.items():
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
    print(f'  Blocks: {len(blocks):,}  |  Pairs: {total_pairs:,}')
    if block_sizes:
        print(f'  Size — mean: {np.mean(block_sizes):.1f}, max: {max(block_sizes)}, median: {int(np.median(block_sizes))}')
    return blocks


def run_pass(sub_df, blocks, pass_type, unit_tolerance):
    matches = []
    for _key, indices in tqdm(blocks, desc=f'  Pass={pass_type}', leave=True):
        for ia, ib in combinations(indices, 2):
            is_match, score = compute_similarity(sub_df.loc[ia], sub_df.loc[ib], pass_type, unit_tolerance)
            if is_match:
                matches.append((ia, ib, score))
    return matches

# ============================================================
# PASS 1 — BRANDED
# ============================================================

branded_df = df[df['product_type'] == 'branded'].copy()
print(f'\nPass 1 — Branded: {len(branded_df):,} products')

def branded_block_key(row):
    brand = str(row['known_brand_clean']).lower().strip()
    cat   = str(row['category']).lower().strip() if pd.notna(row['category']) else 'unknown'
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    return f'{brand}||{cat}||{utype}'

print('Building blocks...')
blocks_branded = build_blocks(branded_df, branded_block_key)
print('Running comparisons...')
matches_branded = run_pass(branded_df, blocks_branded, 'branded', UNIT_TOLERANCE_BRANDED)
print(f'Pass 1 complete: {len(matches_branded):,} matches')

# ============================================================
# PASS 2 — OWN-BRAND
# ============================================================

own_brand_df = df[df['product_type'] == 'own_brand'].copy()
print(f'\nPass 2 — Own-brand: {len(own_brand_df):,} products')

def own_brand_block_key(row):
    tier  = str(row['tier_type']).lower().strip() if pd.notna(row['tier_type']) else 'standard'
    cat   = str(row['category']).lower().strip() if pd.notna(row['category']) else 'unknown'
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    norm  = str(row['normalized_name'] or '').lower()
    tokens = [t for t in norm.split() if t not in STOPWORDS]
    first_tok = tokens[0] if tokens else 'unknown'
    return f'{tier}||{cat}||{utype}||{first_tok}'

print('Building blocks...')
blocks_own = build_blocks(own_brand_df, own_brand_block_key)
print('Running comparisons...')
matches_own = run_pass(own_brand_df, blocks_own, 'own_brand', UNIT_TOLERANCE_OWN_BRAND)
print(f'Pass 2 complete: {len(matches_own):,} matches')

# ============================================================
# PASS 3 — UNBRANDED
# ============================================================

unbranded_df = df[df['product_type'] == 'unbranded'].copy()
print(f'\nPass 3 — Unbranded: {len(unbranded_df):,} products')

def unbranded_block_key(row):
    cat   = str(row['category']).lower().strip() if pd.notna(row['category']) else 'unknown'
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    norm  = str(row['normalized_name'] or '').lower()
    tokens = [t for t in norm.split() if t not in STOPWORDS]
    sig_tokens = sorted(tokens[:2])
    tok_key = '_'.join(sig_tokens) if sig_tokens else 'unknown'
    return f'{cat}||{utype}||{tok_key}'

print('Building blocks...')
blocks_unbranded = build_blocks(unbranded_df, unbranded_block_key)
print('Running comparisons...')
matches_unbranded = run_pass(unbranded_df, blocks_unbranded, 'unbranded', UNIT_TOLERANCE_UNBRANDED)
print(f'Pass 3 complete: {len(matches_unbranded):,} matches')

# ============================================================
# UNION-FIND ASSEMBLY
# ============================================================

print('\nAssembling Union-Find...')
uf = UnionFind(len(df))

pair_scores = {}
all_matches = (
    [('branded',   m) for m in matches_branded] +
    [('own_brand', m) for m in matches_own] +
    [('unbranded', m) for m in matches_unbranded]
)

for _pass, (ia, ib, score) in all_matches:
    uf.union(ia, ib)
    pair_scores[(min(ia, ib), max(ia, ib))] = score

df['raw_cluster_id'] = df['product_idx'].apply(uf.find)
raw_cluster_sizes = df.groupby('raw_cluster_id')['product_idx'].count()
n_raw = df['raw_cluster_id'].nunique()
print(f'Raw clusters: {n_raw:,}  singletons: {(raw_cluster_sizes==1).sum():,}  multi: {(raw_cluster_sizes>1).sum():,}')

# ============================================================
# POST-CLUSTER VALIDATION — prevent transitive-link artifacts
# ============================================================
# Rebuild each Union-Find component using DIRECT match edges only.
# Products that were transitively grouped but have no direct match
# are separated into their own sub-clusters.

print('\nPost-cluster validation (direct-edge rebuild)...')

direct_pair_set = set(pair_scores.keys())  # set of (min_idx, max_idx)

validated_clusters = []   # list of lists of product indices
transitive_breaks = 0

uf_components = uf.components()  # raw_root -> [list of product indices]

for raw_root, members in tqdm(uf_components.items(), desc='  Validating', leave=True):
    if len(members) == 1:
        validated_clusters.append(members)
        continue

    # Build adjacency from DIRECT matches only
    adj = defaultdict(set)
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            ia, ib = members[i], members[j]
            key = (min(ia, ib), max(ia, ib))
            if key in direct_pair_set:
                adj[ia].add(ib)
                adj[ib].add(ia)

    # BFS to find connected components within direct-match graph
    visited = set()
    n_before = 1  # original component = 1 cluster
    n_after  = 0
    for start in members:
        if start in visited:
            continue
        comp = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            queue.extend(adj[node] - visited)
        validated_clusters.append(comp)
        n_after += 1

    if n_after > n_before:
        transitive_breaks += 1

print(f'  Transitive links broken: {transitive_breaks:,}')
print(f'  Validated clusters: {len(validated_clusters):,}')

# ============================================================
# POST-PROCESSING — same-SM and cross-tier violations
# ============================================================

def fix_same_supermarket_violation(group_df, pair_scores):
    by_sm = group_df.groupby('supermarket')
    problem_sms = [sm for sm, g in by_sm if len(g) > 1]
    if not problem_sms:
        return [group_df]
    sub_clusters = []
    remaining = group_df.copy()
    while len(remaining) > 0:
        sub = []
        seen_sms = set()
        for idx, row in remaining.iterrows():
            sm = row['supermarket']
            if sm not in seen_sms:
                sub.append(idx)
                seen_sms.add(sm)
        sub_clusters.append(remaining.loc[sub])
        remaining = remaining.drop(sub)
    return sub_clusters


def fix_cross_tier_violation(group_df):
    own = group_df[group_df['product_type'] == 'own_brand']
    if own.empty or own['tier_type'].nunique() <= 1:
        return [group_df]
    sub_clusters = []
    non_own = group_df[group_df['product_type'] != 'own_brand']
    for tier, tier_group in own.groupby('tier_type'):
        sub_clusters.append(pd.concat([tier_group, non_own]))
    return sub_clusters


print('\nRunning violation fixes...')
final_clusters = []
sm_violations_fixed   = 0
tier_violations_fixed = 0

for members in tqdm(validated_clusters, desc='  Post-processing', leave=True):
    group = df.loc[members]
    if len(group) == 1:
        final_clusters.append(group)
        continue
    sub_clusters = fix_same_supermarket_violation(group, pair_scores)
    if len(sub_clusters) > 1:
        sm_violations_fixed += 1
    for sc in sub_clusters:
        tier_subs = fix_cross_tier_violation(sc)
        if len(tier_subs) > 1:
            tier_violations_fixed += 1
        final_clusters.extend(tier_subs)

print(f'  Same-SM violations fixed:    {sm_violations_fixed:,}')
print(f'  Cross-tier violations fixed: {tier_violations_fixed:,}')
print(f'  Final cluster count:         {len(final_clusters):,}')

# Assign sequential cluster IDs (largest first)
final_clusters.sort(key=lambda g: -len(g))
cluster_id_map  = {}
match_type_map  = {}

for cid, group in enumerate(final_clusters):
    for idx in group.index:
        cluster_id_map[idx] = cid
    types = group['product_type'].value_counts()
    match_type_map[cid] = types.index[0] if len(types) else 'unknown'

df['cluster_id'] = df.index.map(cluster_id_map)

final_sizes = df.groupby('cluster_id')['product_idx'].count()
print(f'\nFinal size distribution:')
for size, count in final_sizes.value_counts().sort_index().items():
    if size <= 6:
        print(f'  size {size}: {count:,}')

# ============================================================
# OUTPUT GENERATION
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

cluster_avg_scores = defaultdict(list)
cluster_min_scores = defaultdict(list)
for (ia, ib), score in pair_scores.items():
    cid_a = cluster_id_map.get(ia)
    cid_b = cluster_id_map.get(ib)
    if cid_a == cid_b and cid_a is not None:
        cluster_avg_scores[cid_a].append(score)

avg_score_map = {cid: np.mean(scores) for cid, scores in cluster_avg_scores.items()}
min_score_map = {cid: np.min(scores)  for cid, scores in cluster_avg_scores.items()}

clusters_df = df[[
    'cluster_id', 'product_idx', 'supermarket', 'names', 'category',
    'known_brand_clean', 'own_brand', 'tier_type', 'unit_value', 'unit_type',
    'pack_quantity', 'core_product_name', 'normalized_name',
    'prices_(£)', 'prices_unit_(£)', 'product_type'
]].copy()

clusters_df['cluster_size']   = df.groupby('cluster_id')['product_idx'].transform('count')
clusters_df['n_supermarkets'] = df.groupby('cluster_id')['supermarket'].transform('nunique')
clusters_df['match_type']     = clusters_df['cluster_id'].map(match_type_map)
clusters_df['avg_pairwise_score'] = clusters_df['cluster_id'].map(avg_score_map)

clusters_df.to_csv(f'{OUTPUT_DIR}/clusters.csv', index=False)
print(f'\nSaved clusters.csv  ({len(clusters_df):,} rows)')

# Cluster summary
summary_rows = []
for cid, group in tqdm(clusters_df.groupby('cluster_id'), desc='Building summary'):
    names_avail = group['core_product_name'].dropna()
    consensus   = names_avail.loc[names_avail.str.len().idxmin()] if len(names_avail) else ''
    summary_rows.append({
        'cluster_id':                 cid,
        'cluster_size':               len(group),
        'n_supermarkets':             group['supermarket'].nunique(),
        'supermarkets_present':       '|'.join(sorted(group['supermarket'].unique())),
        'category':                   group['category'].mode()[0] if len(group) else None,
        'match_type':                 match_type_map.get(cid, 'unknown'),
        'known_brand':                group['known_brand_clean'].dropna().iloc[0] if group['known_brand_clean'].notna().any() else None,
        'tier_type':                  group['tier_type'].dropna().iloc[0] if group['tier_type'].notna().any() else None,
        'unit_value':                 group['unit_value'].dropna().mean() if group['unit_value'].notna().any() else None,
        'unit_type':                  group['unit_type'].dropna().iloc[0] if group['unit_type'].notna().any() else None,
        'pack_quantity':              group['pack_quantity'].dropna().mean() if group['pack_quantity'].notna().any() else None,
        'core_product_name_consensus': consensus,
        'avg_pairwise_score':         avg_score_map.get(cid),
        'min_pairwise_score':         min_score_map.get(cid),
    })

cluster_summary = pd.DataFrame(summary_rows)
cluster_summary.to_csv(f'{OUTPUT_DIR}/cluster_summary.csv', index=False)
print(f'Saved cluster_summary.csv  ({len(cluster_summary):,} rows)')

singletons = clusters_df[clusters_df['cluster_size'] == 1]
singletons.to_csv(f'{OUTPUT_DIR}/singletons.csv', index=False)
print(f'Saved singletons.csv  ({len(singletons):,} rows)')

# ============================================================
# STRATIFIED AUDIT SAMPLE (50 clusters)
# ============================================================

multi = cluster_summary[cluster_summary['cluster_size'] >= 2]
bp    = multi[multi['match_type'] == 'branded']
op    = multi[multi['match_type'] == 'own_brand']
up    = multi[multi['match_type'] == 'unbranded']

audit_ids = pd.concat([
    bp.sample(min(20, len(bp)), random_state=RANDOM_SEED),
    op.sample(min(20, len(op)), random_state=RANDOM_SEED),
    up.sample(min(10, len(up)), random_state=RANDOM_SEED),
])['cluster_id'].tolist()

audit_df = clusters_df[clusters_df['cluster_id'].isin(audit_ids)].copy()
audit_df = audit_df.sort_values(['cluster_id', 'supermarket'])
audit_df['AUDIT_same_core_product'] = ''
audit_df['AUDIT_weight_ok']         = ''
audit_df['AUDIT_tier_ok']           = ''
audit_df['AUDIT_notes']             = ''
audit_df.to_csv(f'{OUTPUT_DIR}/audit_sample_50.csv', index=False)
print(f'Saved audit_sample_50.csv  ({len(audit_df)} rows, {len(audit_ids)} clusters)')

# ============================================================
# DIAGNOSTIC REPORT
# ============================================================

non_singleton = cluster_summary[cluster_summary['cluster_size'] >= 2]
n_multi = len(non_singleton)
print('\n' + '=' * 60)
print('DIAGNOSTIC REPORT v2')
print('=' * 60)
print(f'Total clusters (incl singletons): {len(cluster_summary):,}')
print(f'Singletons:                       {(cluster_summary["cluster_size"]==1).sum():,}')
print(f'Multi-product clusters (≥2):      {n_multi:,}')
print(f'  4-way: {(non_singleton["n_supermarkets"]==4).sum():,}')
print(f'  3-way: {(non_singleton["n_supermarkets"]==3).sum():,}')
print(f'  2-way: {(non_singleton["n_supermarkets"]==2).sum():,}')

print(f'\nBy match type:')
for mt in ['branded', 'own_brand', 'unbranded']:
    print(f'  {mt:12s}: {(non_singleton["match_type"]==mt).sum():,}')

print(f'\nProduct coverage:')
in_cluster = clusters_df[clusters_df['cluster_size'] >= 2]
for sm in sorted(df['supermarket'].unique()):
    total   = len(df[df['supermarket'] == sm])
    matched = len(in_cluster[in_cluster['supermarket'] == sm])
    print(f'  {sm:12s}: {matched:,}/{total:,} = {matched/total*100:.1f}%')

print(f'\nQuality scores (avg_pairwise_score):')
scores_avail = non_singleton.dropna(subset=['avg_pairwise_score'])
for mt in ['branded', 'own_brand', 'unbranded']:
    sub = scores_avail[scores_avail['match_type'] == mt]['avg_pairwise_score']
    if len(sub):
        print(f'  {mt:12s}: mean={sub.mean():.3f}  p5={sub.quantile(0.05):.3f}  p25={sub.quantile(0.25):.3f}')

print(f'\nAutomated validation checks:')
violations = {'same_sm': 0, 'unit_type_mixed': 0, 'tier_mixed': 0, 'weight_high': 0, 'score_low': 0}
for cid, group in clusters_df[clusters_df['cluster_size'] >= 2].groupby('cluster_id'):
    if group['supermarket'].value_counts().max() > 1:
        violations['same_sm'] += 1
    if group['unit_type'].dropna().nunique() > 1:
        violations['unit_type_mixed'] += 1
    own = group[group['product_type'] == 'own_brand']
    if len(own) >= 2 and own['tier_type'].nunique() > 1:
        violations['tier_mixed'] += 1
    uv = group['unit_value'].dropna()
    if len(uv) >= 2 and uv.min() > 0 and uv.max() / uv.min() > 1.10:
        violations['weight_high'] += 1
    avg_s = avg_score_map.get(cid)
    if avg_s is not None and avg_s < 0.75:
        violations['score_low'] += 1

print(f'  Same-SM violations:     {violations["same_sm"]:,}  (should be 0)')
print(f'  Mixed unit type:        {violations["unit_type_mixed"]:,}  (should be 0)')
print(f'  Tier mixing:            {violations["tier_mixed"]:,}  (should be 0)')
print(f'  Weight variance >10%:   {violations["weight_high"]:,}  (flag)')
print(f'  Score below 0.75:       {violations["score_low"]:,}  (flag)')

print(f'\nTarget range: 10,000–20,000 multi-clusters')
print(f'Actual:       {n_multi:,}')
print(f'In range:     {"YES ✓" if 10_000 <= n_multi <= 20_000 else "NO — review thresholds"}')
print('=' * 60)
