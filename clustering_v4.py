"""
ShopWiser Clustering v4

Key changes vs v3:
  1. UNIT_TOLERANCE_BRANDED: 0.03 → 0.05 (wider window for branded weight rounding).
  2. SHORT_STRIPPED_THRESHOLD removed — brand is now stripped in normalise.py so
     normalized_name is already brand-free; the 0.95 guard was firing on almost all
     short branded names and blocking legitimate matches.
  3. Category normalisation (CATEGORY_ALIASES) so food_cupboard / fresh_food / bakery /
     frozen / free-from all share the same blocking bucket "grocery"; drinks and alcohol
     remain distinct.
  4. Multi-key blocking (build_multi_blocks): each product participates in multiple
     overlapping blocks derived from different key functions. Pairs are de-duplicated
     before scoring so no pair is compared twice.
     - Branded: 3 keys (+ unit_type, + category, brand-only + utype, brand + cat)
     - Own-brand: 4 keys (with/without utype × first/second token)
     - Unbranded: 4 keys (with/without utype × different 2-token windows)
  5. Truncation-aware scoring: threshold −0.05 when either product is a truncated
     Morrisons name (is_truncated = True).
  6. Updated REQUIRED_COLS to match new normalise.py output schema.
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

UNIT_TOLERANCE_BRANDED   = 0.05   # was 0.03 — wider window for weight rounding
UNIT_TOLERANCE_OWN_BRAND = 0.05
UNIT_TOLERANCE_UNBRANDED = 0.05

FUZZY_THRESHOLD          = 0.785  # was 0.82 — safe floor: "plum vs chopped tomatoes" = 0.728
FUZZY_THRESHOLD_OWNBRAND = 0.82   # was 0.884 — floor: "black beans vs black-eyed beans" = 0.804
FUZZY_THRESHOLD_NOUNIT   = 0.83   # was 0.88 — better recall for no-unit products

# SHORT_STRIPPED_THRESHOLD: lowered from v3's 0.95 → 0.92.
# Brand is now stripped at normalise.py time so more branded names have ≤3 tokens
# after normalization.  "stock cubes ham" vs "stock cubes lamb" scores ~0.91 and
# must be rejected; identical short names score 1.0.  0.92 is the right balance.
SHORT_STRIPPED_THRESHOLD = 0.92   # must be above 0.912 (ham/lamb stock cubes false positive)

PACK_QTY_MAX_RATIO   = 2.0
MAX_BLOCK_SIZE       = 200
TRUNCATION_BONUS     = 0.05        # threshold reduction for truncated names

ATTR_PENALTIES = {
    'organic':   0.20,
    'free_from': 0.20,
    'fairtrade': 0.10,
    'vegan':     0.25,
}
DIET_PENALTY = 0.10

WINE_TYPE_TOKENS = {'red', 'white', 'rose', 'rosé', 'sparkling', 'prosecco', 'champagne', 'blush'}

FISH_MEDIUM_TOKENS = {'brine', 'spring water', 'olive oil', 'sunflower oil', 'tomato', 'springwater'}

MILK_FAT_MAP = [
    ('semi-skimmed', 'semi_skimmed'),
    ('semi skimmed',  'semi_skimmed'),
    ('skimmed',       'skimmed'),
    ('full-fat',      'full_fat'),
    ('full fat',      'full_fat'),
    ('whole milk',    'whole'),
    ('whole',         'whole'),
]

NAS_PATTERNS = ['no added sugar', 'sugar free', 'sugarfree', 'zero sugar', 'no sugar added']

BRAND_EXCLUSIONS = {'extra', 'essential', 'basics', 'finest', 'select', 'special'}

# Categories that should be treated as interchangeable for blocking purposes.
# Drinks and alcohol remain distinct because a drink is NOT the same as a food item.
CATEGORY_ALIASES = {
    'food_cupboard': 'grocery',
    'fresh_food':    'grocery',
    'bakery':        'grocery',
    'frozen':        'grocery',
    'free-from':     'grocery',
    # 'drinks' → stays as 'drinks'
    # 'alcohol' → stays as 'alcohol' (if present)
}

OUTPUT_DIR  = 'data/clusters'
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print('Configuration loaded (v4).')
print(f'  Fuzzy threshold:       {FUZZY_THRESHOLD} (own-brand: {FUZZY_THRESHOLD_OWNBRAND}, no-unit: {FUZZY_THRESHOLD_NOUNIT})')
print(f'  Short stripped threshold: {SHORT_STRIPPED_THRESHOLD} (≤3 tokens, branded — lowered from v3\'s 0.95)')
print(f'  Unit tolerance:        ±{UNIT_TOLERANCE_BRANDED*100:.0f}% branded, ±{UNIT_TOLERANCE_OWN_BRAND*100:.0f}% own-brand')
print(f'  Pack ratio max:        {PACK_QTY_MAX_RATIO}')
print(f'  Truncation bonus:      -{TRUNCATION_BONUS}')
print(f'  Output directory:      {OUTPUT_DIR}')

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
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self):
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return groups

# ============================================================
# DATA LOADING
# ============================================================

df = pd.read_csv('data/normalized_products.csv', low_memory=False)
print(f'\nLoaded {len(df):,} products, {df.shape[1]} columns')

REQUIRED_COLS = [
    'supermarket', 'names', 'category', 'own_brand',
    'supermarket_brand', 'tier_type', 'known_brand',
    'pack_quantity', 'unit_value', 'unit_type',
    'attributes_keywords', 'core_product_name', 'normalized_name',
]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
assert not missing, f'Missing columns: {missing}'

df = df.reset_index(drop=True)
df['product_idx'] = df.index

df['known_brand_clean'] = df['known_brand'].where(
    ~df['known_brand'].fillna('').str.lower().isin(BRAND_EXCLUSIONS), other=None
)

df['product_type'] = np.where(
    df['known_brand_clean'].notna(), 'branded',
    np.where(df['own_brand'].astype(str).str.lower().isin(['true', '1', 'yes']), 'own_brand', 'unbranded')
)

df['has_unit']     = df['unit_value'].notna() & (df['unit_value'] > 0)
df['has_pack']     = df['pack_quantity'].notna() & (df['pack_quantity'] > 0)
df['supermarket']  = df['supermarket'].str.strip()
df['is_truncated'] = df.get('is_truncated', pd.Series(False, index=df.index)).astype(bool)

# Category normalisation for blocking
df['cat_norm'] = df['category'].str.lower().map(CATEGORY_ALIASES).fillna(df['category'].str.lower())

print(f'\nSupermarket distribution:\n{df["supermarket"].value_counts().to_string()}')
print(f'\nProduct type:\n{df["product_type"].value_counts().to_string()}')
print(f'\nCategory (normalised) distribution:\n{df["cat_norm"].value_counts().to_string()}')
print(f'\nUnit type distribution:\n{df["unit_type"].value_counts().to_string()}')
print(f'\nUnit coverage: {df["has_unit"].sum():,}/{len(df):,} ({df["has_unit"].mean()*100:.1f}%)')
print(f'Truncated names: {df["is_truncated"].sum():,}')

# ============================================================
# SIMILARITY SCORING HELPERS
# ============================================================

def _strip_brand(name: str, brand: str) -> str:
    """Remove brand prefix from normalized name (may already be stripped by normalise.py)."""
    if not brand:
        return name
    pat = re.compile(r'\b' + re.escape(brand.lower()) + r'\b', re.IGNORECASE)
    result = pat.sub('', name).strip()
    return re.sub(r'\s+', ' ', result)


def _unit_type_compatible(ut_a, ut_b) -> bool:
    if pd.isna(ut_a) or pd.isna(ut_b) or not ut_a or not ut_b:
        return True   # both missing → no constraint
    return str(ut_a).strip().lower() == str(ut_b).strip().lower()


def _unit_value_compatible(uv_a, uv_b, tolerance) -> bool:
    a_valid = pd.notna(uv_a) and float(uv_a) > 0
    b_valid = pd.notna(uv_b) and float(uv_b) > 0
    if not a_valid or not b_valid:
        return True   # missing → no constraint
    ratio = max(float(uv_a), float(uv_b)) / min(float(uv_a), float(uv_b))
    return ratio <= (1.0 + tolerance)


def _pack_compatible(pq_a, pq_b) -> bool:
    a_valid = pd.notna(pq_a) and float(pq_a) > 0
    b_valid = pd.notna(pq_b) and float(pq_b) > 0
    if not a_valid or not b_valid:
        return True
    ratio = max(float(pq_a), float(pq_b)) / min(float(pq_a), float(pq_b))
    return ratio <= PACK_QTY_MAX_RATIO


def _get_wine_type(text: str) -> str:
    t = text.lower()
    for tok in WINE_TYPE_TOKENS:
        if tok in t:
            return tok
    return ''


def _get_fish_medium(text: str) -> set:
    t = text.lower()
    return {m for m in FISH_MEDIUM_TOKENS if m in t}


def _get_milk_fat(text: str):
    t = text.lower()
    for pattern, label in MILK_FAT_MAP:
        if pattern in t:
            return label
    return None


def _has_diet_marker(text: str) -> bool:
    t = text.lower()
    if 'diet' in t:
        return True
    return any(p in t for p in NAS_PATTERNS)


_AGE_RE = re.compile(r'(\d+)\s*year', re.IGNORECASE)


def _attribute_penalty(attrs_a, attrs_b, name_a='', name_b='') -> float:
    str_a = (str(attrs_a).lower() if pd.notna(attrs_a) else '') + ' ' + str(name_a).lower()
    str_b = (str(attrs_b).lower() if pd.notna(attrs_b) else '') + ' ' + str(name_b).lower()

    penalty = 0.0

    for attr, p in ATTR_PENALTIES.items():
        if (attr in str_a) != (attr in str_b):
            penalty += p

    if _has_diet_marker(str_a) != _has_diet_marker(str_b):
        penalty += DIET_PENALTY

    wine_a = _get_wine_type(str_a)
    wine_b = _get_wine_type(str_b)
    if (wine_a or wine_b) and (wine_a != wine_b):
        penalty += 0.60

    med_a = _get_fish_medium(str_a)
    med_b = _get_fish_medium(str_b)
    if med_a and med_b and med_a != med_b:
        penalty += 0.50

    fat_a = _get_milk_fat(str_a)
    fat_b = _get_milk_fat(str_b)
    if fat_a and fat_b and fat_a != fat_b:
        penalty += 0.50

    ages_a = [int(m.group(1)) for m in _AGE_RE.finditer(str_a)]
    ages_b = [int(m.group(1)) for m in _AGE_RE.finditer(str_b)]
    if ages_a and ages_b and set(ages_a) != set(ages_b):
        penalty += 0.60

    return min(penalty, 1.0)


def compute_similarity(row_a, row_b, pass_type, unit_tolerance):
    # Hard constraint 1: same supermarket
    if row_a['supermarket'] == row_b['supermarket']:
        return False, 0.0

    # Hard constraint 2: unit type mismatch
    if not _unit_type_compatible(row_a.get('unit_type'), row_b.get('unit_type')):
        return False, 0.0

    # Hard constraint 3: unit value out of tolerance
    if not _unit_value_compatible(row_a.get('unit_value'), row_b.get('unit_value'), unit_tolerance):
        return False, 0.0

    # Hard constraint 4: pack quantity ratio
    if not _pack_compatible(row_a.get('pack_quantity'), row_b.get('pack_quantity')):
        return False, 0.0

    # Pass-specific rules
    if pass_type == 'branded':
        if str(row_a['known_brand_clean']).lower() != str(row_b['known_brand_clean']).lower():
            return False, 0.0
    elif pass_type == 'own_brand':
        tier_a = str(row_a.get('tier_type', '') or '').lower() or 'standard'
        tier_b = str(row_b.get('tier_type', '') or '').lower() or 'standard'
        if tier_a != tier_b:
            return False, 0.0
        if row_a.get('product_type') != row_b.get('product_type'):
            return False, 0.0

    raw_name_a = str(row_a.get('normalized_name', '') or '')
    raw_name_b = str(row_b.get('normalized_name', '') or '')

    # Brand stripping for branded pass.
    # normalise.py already removes the brand from normalized_name, so this is
    # typically a no-op — but kept for robustness against residual brand tokens.
    if pass_type == 'branded':
        brand = str(row_a.get('known_brand_clean', '') or '').lower()
        name_a = _strip_brand(raw_name_a, brand)
        name_b = _strip_brand(raw_name_b, brand)
    else:
        name_a, name_b = raw_name_a, raw_name_b

    # Short-name guard: both ≤2 tokens AND neither has a unit → reject
    # (prevents garbage "no-name" products from matching)
    if len(name_a.split()) <= 2 and len(name_b.split()) <= 2:
        if not (row_a['has_unit'] and row_b['has_unit']):
            return False, 0.0

    # --- FUZZY SCORING ---
    token_sort = fuzz.token_sort_ratio(name_a, name_b) / 100.0
    partial    = fuzz.partial_ratio(name_a, name_b) / 100.0

    len_a, len_b = len(name_a), len(name_b)
    len_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 1.0

    if len_ratio < 0.65:
        token_set = fuzz.token_set_ratio(name_a, name_b) / 100.0
        score = 0.45 * token_sort + 0.25 * token_set + 0.30 * partial
    else:
        score = 0.70 * token_sort + 0.30 * partial

    # Attribute penalty
    penalty = _attribute_penalty(
        row_a.get('attributes_keywords'), row_b.get('attributes_keywords'),
        name_a=raw_name_a, name_b=raw_name_b
    )
    score -= penalty

    # Threshold selection
    if not row_a['has_unit'] or not row_b['has_unit']:
        threshold = FUZZY_THRESHOLD_NOUNIT
    elif pass_type == 'own_brand':
        threshold = FUZZY_THRESHOLD_OWNBRAND
    else:
        threshold = FUZZY_THRESHOLD

    # Short stripped-name guard for branded pass: when both names are ≤3 tokens
    # (after brand stripping) require a higher similarity floor.  This prevents
    # near-synonym product types (e.g. "stock cubes ham" vs "stock cubes lamb")
    # from matching while still allowing identical short names (score = 1.0).
    if pass_type == 'branded':
        if len(name_a.split()) <= 3 and len(name_b.split()) <= 3:
            threshold = max(threshold, SHORT_STRIPPED_THRESHOLD)

    # Truncation bonus: lower effective threshold when either product name was
    # truncated by Morrisons (…), since truncated names naturally score lower.
    trunc_a = bool(row_a.get('is_truncated', False))
    trunc_b = bool(row_b.get('is_truncated', False))
    if trunc_a or trunc_b:
        threshold -= TRUNCATION_BONUS

    return score >= threshold, score

# ============================================================
# SELF-TESTS
# ============================================================

def _mk(name, uv, ut, sm, brand=None, tier='standard', ptype='branded', pq=None, attrs=None, trunc=False):
    return pd.Series({
        'normalized_name': name, 'unit_value': uv, 'unit_type': ut,
        'supermarket': sm, 'has_unit': pd.notna(uv) and uv > 0,
        'pack_quantity': pq, 'attributes_keywords': attrs,
        'known_brand_clean': brand, 'tier_type': tier, 'product_type': ptype,
        'is_truncated': trunc,
    })

tests = [
    ('Same product diff SM',
     _mk('baked beans tomato sauce', 415, 'g', 'Tesco', brand='heinz'),
     _mk('baked beans tomato sauce', 415, 'g', 'ASDA',  brand='heinz'),
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

    ('Chocolate slices vs Cherry bakewell → must NOT match',
     _mk('chocolate slices', None, None, 'Tesco', brand='mr kipling'),
     _mk('cherry bakewell cakes', None, None, 'ASDA', brand='mr kipling'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Rhubarb kefir vs Honey kefir → must NOT match',
     _mk('kefir rhubarb fermented organic yogurt', 350, 'g', 'Tesco', brand='yeo valley'),
     _mk('organic kefir honey yogurt', 350, 'g', 'ASDA', brand='yeo valley'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('19 Crimes Rosé vs Red wine → must NOT match',
     _mk('revolutionary rose', 750, 'ml', 'Tesco', brand='19 crimes'),
     _mk('red wine', 750, 'ml', 'ASDA', brand='19 crimes'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Tuna spring water vs olive oil → must NOT match',
     _mk('tuna chunks in spring water', 145, 'g', 'Tesco', brand=None, ptype='own_brand'),
     _mk('tuna chunks in olive oil', 145, 'g', 'ASDA', brand=None, ptype='own_brand'),
     'own_brand', UNIT_TOLERANCE_OWN_BRAND, False),

    ('Highland Park 12yr vs 10yr → must NOT match',
     _mk('highland park 12 year old single malt scotch whisky', 700, 'ml', 'Tesco', brand=None, ptype='unbranded'),
     _mk('highland park 10 year old single malt scotch whisky', 700, 'ml', 'ASDA', brand=None, ptype='unbranded'),
     'unbranded', UNIT_TOLERANCE_UNBRANDED, False),

    ('Pack mismatch 4x vs 10x → must NOT match',
     _mk('alcohol free draught stout', 440, 'ml', 'Tesco', brand='guinness', pq=4),
     _mk('alcohol free draught stout', 440, 'ml', 'ASDA',  brand='guinness', pq=10),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Honey Cheerios 515g vs Multigrain Cheerios 540g → must NOT match',
     _mk('honey cheerios cereal', 515, 'g', 'Tesco', brand='cheerios'),
     _mk('cheerios multigrain cereal', 540, 'g', 'ASDA', brand='cheerios'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Knorr lamb stock cubes vs ham stock cubes → must NOT match',
     _mk('stock cubes lamb', 104, 'g', 'Tesco', brand='knorr'),
     _mk('stock cubes ham', 104, 'g', 'ASDA', brand='knorr'),
     'branded', UNIT_TOLERANCE_BRANDED, False),

    ('Black beans vs Black eyed beans (own-brand) → must NOT match',
     _mk('black beans', 400, 'g', 'Tesco', brand=None, ptype='own_brand'),
     _mk('black eyed beans', 400, 'g', 'ASDA', brand=None, ptype='own_brand'),
     'own_brand', UNIT_TOLERANCE_OWN_BRAND, False),

    ('Malbec Rosé vs Malbec red wine → must NOT match',
     _mk('malbec rose wine', 750, 'ml', 'Tesco', brand=None, ptype='unbranded'),
     _mk('malbec red wine', 750, 'ml', 'ASDA', brand=None, ptype='unbranded'),
     'unbranded', UNIT_TOLERANCE_UNBRANDED, False),

    ('No Added Sugar squash vs regular squash → must NOT match',
     _mk('blackcurrant squash no added sugar', 1000, 'ml', 'Tesco', brand=None, ptype='own_brand'),
     _mk('blackcurrant squash', 1000, 'ml', 'ASDA', brand=None, ptype='own_brand'),
     'own_brand', UNIT_TOLERANCE_OWN_BRAND, False),

    ('Skimmed vs Semi-skimmed milk → must NOT match',
     _mk('skimmed milk', 2000, 'ml', 'Tesco', brand=None, ptype='own_brand'),
     _mk('semi skimmed milk', 2000, 'ml', 'ASDA', brand=None, ptype='own_brand'),
     'own_brand', UNIT_TOLERANCE_OWN_BRAND, False),
]

print('\nRunning similarity self-tests...')
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
    print(f'    Blocks: {len(blocks):,}  |  Pairs: {total_pairs:,}')
    if block_sizes:
        print(f'    Size — mean: {np.mean(block_sizes):.1f}, max: {max(block_sizes)}, median: {int(np.median(block_sizes))}')
    return blocks


def build_multi_blocks(sub_df, key_fns, max_block_size=MAX_BLOCK_SIZE):
    """
    Generate candidate pairs from multiple blocking functions, de-duplicating
    across all keys so no pair is scored more than once.
    Returns a flat list of (ia, ib) index pairs.
    """
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
    """Score a flat list of (ia, ib) index pairs."""
    matches = []
    for ia, ib in tqdm(pairs, desc=f'  Pass={pass_type}', leave=True):
        is_match, score = compute_similarity(sub_df.loc[ia], sub_df.loc[ib], pass_type, unit_tolerance)
        if is_match:
            matches.append((ia, ib, score))
    return matches

# ============================================================
# PASS 1 — BRANDED
# ============================================================

branded_df = df[df['product_type'] == 'branded'].copy()
print(f'\nPass 1 — Branded: {len(branded_df):,} products')

def _branded_key_a(row):
    """Full key: brand + cat_norm + utype"""
    brand = str(row['known_brand_clean']).lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    return f'{brand}||{cat}||{utype}'

def _branded_key_b(row):
    """Drop unit_type — catches ~44% products with inferred/missing units"""
    brand = str(row['known_brand_clean']).lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    return f'{brand}||{cat}'

def _branded_key_c(row):
    """Drop category — catches cross-category branded items"""
    brand = str(row['known_brand_clean']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    return f'{brand}||{utype}'

print('Building multi-key blocks for branded...')
pairs_branded = build_multi_blocks(branded_df, [_branded_key_a, _branded_key_b, _branded_key_c])
print('Running comparisons...')
matches_branded = run_pass(branded_df, pairs_branded, 'branded', UNIT_TOLERANCE_BRANDED)
print(f'Pass 1 complete: {len(matches_branded):,} matches')

# ============================================================
# PASS 2 — OWN-BRAND
# ============================================================

own_brand_df = df[df['product_type'] == 'own_brand'].copy()
print(f'\nPass 2 — Own-brand: {len(own_brand_df):,} products')

def _own_key_a(row):
    """Full key: tier + cat_norm + utype + first token"""
    tier  = str(row['tier_type'] or 'standard').lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    tok1  = toks[0] if toks else 'unknown'
    return f'{tier}||{cat}||{utype}||{tok1}'

def _own_key_b(row):
    """Drop unit_type: tier + cat + first token"""
    tier = str(row['tier_type'] or 'standard').lower().strip()
    cat  = str(row['cat_norm']).lower().strip()
    toks = _get_tokens(row['normalized_name'])
    tok1 = toks[0] if toks else 'unknown'
    return f'{tier}||{cat}||{tok1}'

def _own_key_c(row):
    """Second significant token: tier + cat + utype + tok2"""
    tier  = str(row['tier_type'] or 'standard').lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    tok2  = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
    return f'{tier}||{cat}||{utype}||{tok2}'

def _own_key_d(row):
    """Second token, no unit: tier + cat + tok2"""
    tier = str(row['tier_type'] or 'standard').lower().strip()
    cat  = str(row['cat_norm']).lower().strip()
    toks = _get_tokens(row['normalized_name'])
    tok2 = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
    return f'{tier}||{cat}||{tok2}'

print('Building multi-key blocks for own-brand...')
pairs_own = build_multi_blocks(own_brand_df, [_own_key_a, _own_key_b, _own_key_c, _own_key_d])
print('Running comparisons...')
matches_own = run_pass(own_brand_df, pairs_own, 'own_brand', UNIT_TOLERANCE_OWN_BRAND)
print(f'Pass 2 complete: {len(matches_own):,} matches')

# ============================================================
# PASS 3 — UNBRANDED
# ============================================================

unbranded_df = df[df['product_type'] == 'unbranded'].copy()
print(f'\nPass 3 — Unbranded: {len(unbranded_df):,} products')

def _unb_key_a(row):
    """Current: cat + utype + sorted(tok1, tok2)"""
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    sig   = sorted(toks[:2])
    tok_key = '_'.join(sig) if sig else 'unknown'
    return f'{cat}||{utype}||{tok_key}'

def _unb_key_b(row):
    """Drop unit_type: cat + sorted(tok1, tok2)"""
    cat  = str(row['cat_norm']).lower().strip()
    toks = _get_tokens(row['normalized_name'])
    sig  = sorted(toks[:2])
    tok_key = '_'.join(sig) if sig else 'unknown'
    return f'{cat}||{tok_key}'

def _unb_key_c(row):
    """Alt window: cat + utype + sorted(tok1, tok3)"""
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    picks = [toks[0]] if toks else []
    if len(toks) > 2:
        picks.append(toks[2])
    elif len(toks) > 1:
        picks.append(toks[1])
    tok_key = '_'.join(sorted(picks)) if picks else 'unknown'
    return f'{cat}||{utype}||{tok_key}'

def _unb_key_d(row):
    """Alt window: cat + utype + sorted(tok2, tok3)"""
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    picks = []
    if len(toks) > 1:
        picks.append(toks[1])
    if len(toks) > 2:
        picks.append(toks[2])
    tok_key = '_'.join(sorted(picks)) if picks else 'unknown'
    return f'{cat}||{utype}||{tok_key}'

print('Building multi-key blocks for unbranded...')
pairs_unbranded = build_multi_blocks(unbranded_df, [_unb_key_a, _unb_key_b, _unb_key_c, _unb_key_d])
print('Running comparisons...')
matches_unbranded = run_pass(unbranded_df, pairs_unbranded, 'unbranded', UNIT_TOLERANCE_UNBRANDED)
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

print('\nPost-cluster validation (direct-edge rebuild)...')

direct_pair_set = set(pair_scores.keys())
validated_clusters = []
transitive_breaks  = 0

uf_components = uf.components()

for raw_root, members in tqdm(uf_components.items(), desc='  Validating', leave=True):
    if len(members) == 1:
        validated_clusters.append(members)
        continue

    adj = defaultdict(set)
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            ia, ib = members[i], members[j]
            key = (min(ia, ib), max(ia, ib))
            if key in direct_pair_set:
                adj[ia].add(ib)
                adj[ib].add(ia)

    visited = set()
    n_before, n_after = 1, 0
    for start in members:
        if start in visited:
            continue
        comp, queue = [], [start]
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
    if not any(len(g) > 1 for _, g in by_sm):
        return [group_df]
    sub_clusters = []
    remaining = group_df.copy()
    while len(remaining) > 0:
        sub, seen_sms = [], set()
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
final_clusters        = []
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
cluster_id_map = {}
match_type_map = {}

for cid, group in enumerate(final_clusters):
    for idx in group.index:
        cluster_id_map[idx] = cid
    types = group['product_type'].value_counts()
    match_type_map[cid] = types.index[0] if len(types) else 'unknown'

df['cluster_id'] = df.index.map(cluster_id_map)

final_sizes = df.groupby('cluster_id')['product_idx'].count()
print(f'\nFinal size distribution:')
for size, count in final_sizes.value_counts().sort_index().items():
    if size <= 8:
        print(f'  size {size}: {count:,}')

# ============================================================
# OUTPUT GENERATION
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

cluster_avg_scores = defaultdict(list)
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
    'prices_(£)', 'prices_unit_(£)', 'product_type',
]].copy()

clusters_df['cluster_size']       = df.groupby('cluster_id')['product_idx'].transform('count')
clusters_df['n_supermarkets']     = df.groupby('cluster_id')['supermarket'].transform('nunique')
clusters_df['match_type']         = clusters_df['cluster_id'].map(match_type_map)
clusters_df['avg_pairwise_score'] = clusters_df['cluster_id'].map(avg_score_map)

clusters_df.to_csv(f'{OUTPUT_DIR}/clusters.csv', index=False)
print(f'\nSaved clusters.csv  ({len(clusters_df):,} rows)')

# Cluster summary
summary_rows = []
for cid, group in tqdm(clusters_df.groupby('cluster_id'), desc='Building summary'):
    names_avail = group['core_product_name'].dropna()
    consensus   = names_avail.loc[names_avail.str.len().idxmin()] if len(names_avail) else ''
    summary_rows.append({
        'cluster_id':                  cid,
        'cluster_size':                len(group),
        'n_supermarkets':              group['supermarket'].nunique(),
        'supermarkets_present':        '|'.join(sorted(group['supermarket'].unique())),
        'category':                    group['category'].mode()[0] if len(group) else None,
        'match_type':                  match_type_map.get(cid, 'unknown'),
        'known_brand':                 group['known_brand_clean'].dropna().iloc[0] if group['known_brand_clean'].notna().any() else None,
        'tier_type':                   group['tier_type'].dropna().iloc[0] if group['tier_type'].notna().any() else None,
        'unit_value':                  group['unit_value'].dropna().mean() if group['unit_value'].notna().any() else None,
        'unit_type':                   group['unit_type'].dropna().iloc[0] if group['unit_type'].notna().any() else None,
        'pack_quantity':               group['pack_quantity'].dropna().mean() if group['pack_quantity'].notna().any() else None,
        'core_product_name_consensus': consensus,
        'avg_pairwise_score':          avg_score_map.get(cid),
        'min_pairwise_score':          min_score_map.get(cid),
    })

cluster_summary = pd.DataFrame(summary_rows)
cluster_summary.to_csv(f'{OUTPUT_DIR}/cluster_summary.csv', index=False)
print(f'Saved cluster_summary.csv  ({len(cluster_summary):,} rows)')

singletons = clusters_df[clusters_df['cluster_size'] == 1]
singletons.to_csv(f'{OUTPUT_DIR}/singletons.csv', index=False)
print(f'Saved singletons.csv  ({len(singletons):,} rows)')

# Audit sample
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
print('DIAGNOSTIC REPORT v4')
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
