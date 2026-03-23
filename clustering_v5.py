"""
ShopWiser Clustering v5

Key changes vs v4:
  1. HARD CONFLICT TOKEN SETS — new precision guards for product variant confusion:
       • FLAVOR_NAMED_TOKENS   — ginger/mint/raspberry/honey etc.
         (if BOTH products have named flavor tokens that DON'T overlap → reject)
       • MILK_BASE_TOKENS      — soya/oat/almond/hazelnut etc.
         (different milk bases → always reject)
       • COOKING_STATE_TOKENS  — raw/roast/smoked/dried/cured
         (different cooking states → always reject)
       • ONE_SIDED_CONFLICT_TOKENS — 'baby'
         (presence in one product but not the other → always reject)
  2. ALCOHOL-FREE PENALTY (0.60) — wired from grocery_vocab's 'alcohol_marker'
     into _attribute_penalty; replaces the missing gap that caused cluster 793
     (Old Mout standard vs Alcohol Free).
  3. PACK_COMPATIBLE FIX — NaN pack_qty vs explicit multipack (>1) is now
     rejected (was returning True as a wildcard; caused cluster 9316
     Evian single vs 6×500ml).
  4. SCORE-BASED SM VIOLATION FIX — fix_same_supermarket_violation now
     picks the best-scoring representative per SM (highest max cross-SM pair
     score) instead of greedy iteration order.  Excluded products are recycled
     into valid sub-clusters when direct pairs exist, falling back to singletons.
     This preserves genuine 4-way clusters that the greedy approach destroyed.
  5. WIDER UNBRANDED BLOCKING — new 5th key (_unb_key_e) uses only
     cat + unit_bucket (no token dependency), catching products whose first
     tokens differ between supermarkets (e.g. "jacket potato" vs "baking
     potato").
  6. PASS 4 — CROSS-BUCKET CATCH-ALL — after passes 1–3, tentative singletons
     are re-compared across all product_type buckets at a very high threshold
     (0.90).  Handles products mis-classified branded↔unbranded that were
     stuck in different silos.  Branded vs own_brand cross-matching is
     intentionally excluded.
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

UNIT_TOLERANCE_BRANDED   = 0.05
UNIT_TOLERANCE_OWN_BRAND = 0.05
UNIT_TOLERANCE_UNBRANDED = 0.05

FUZZY_THRESHOLD          = 0.750
FUZZY_THRESHOLD_OWNBRAND = 0.805
FUZZY_THRESHOLD_NOUNIT   = 0.800

SHORT_STRIPPED_THRESHOLD = 0.87   # ≤3 equal-token branded pairs (v6: lowered from 0.92; lamb/ham
                                  # false-positive now caught by FLAVOR_NAMED_TOKENS ingredient
                                  # conflict before threshold is even evaluated)

PACK_QTY_MAX_RATIO   = 2.0
MAX_BLOCK_SIZE       = 200
TRUNCATION_BONUS     = 0.05       # threshold reduction for truncated names

ATTR_PENALTIES = {
    'organic':   0.20,
    'free_from': 0.20,
    'fairtrade': 0.10,
    'vegan':     0.25,
}
DIET_PENALTY = 0.10

# ── v5: alcohol-free penalty (wired from grocery_vocab ATTRIBUTES['alcohol_marker'])
ALCOHOL_FREE_PENALTY   = 0.60
ALCOHOL_FREE_PATTERNS  = [
    'alcohol free', 'alcohol-free', 'non-alcoholic',
    '0% alcohol', 'low alcohol', 'alcohol_marker',
]

# ── v5: Hard conflict token sets ──────────────────────────────────────────────
# FLAVOR_NAMED_TOKENS: named fruit / herb / spice flavours AND meat/seafood
# ingredient varieties that, when present in BOTH products with no overlap,
# indicate irreconcilably different variants.
#
# Meat/ingredient tokens (v6): enables lowering SHORT_STRIPPED_THRESHOLD
# from 0.92 → 0.87 because "lamb stock cubes" vs "ham stock cubes" is now
# caught here (both have named ingredient tokens that don't overlap) BEFORE
# the threshold guard is evaluated.
FLAVOR_NAMED_TOKENS = frozenset({
    # Fruit, herb, spice flavours
    'ginger', 'mint', 'raspberry', 'lemon', 'orange', 'cherry',
    'strawberry', 'blueberry', 'mango', 'blackcurrant', 'blackberry',
    'elderflower', 'rhubarb', 'lime', 'peach', 'apricot', 'vanilla',
    'caramel', 'toffee', 'honey', 'maple', 'cinnamon', 'cola',
    'lychee', 'basil', 'chilli',   # v5.1: audit fixes
    'banana', 'syrup',             # v5.2: banana vs golden-syrup transitive FP
    # Meat / poultry / seafood variety tokens (v6)
    'lamb', 'ham', 'pork', 'beef', 'chicken', 'turkey', 'duck',
    'venison', 'bacon',
    'salmon', 'tuna', 'cod', 'haddock', 'prawn', 'shrimp', 'crab',
    'mackerel', 'trout', 'sardine', 'anchovy',
})

# MILK_BASE_TOKENS: milk alternatives — mutually exclusive bases.
MILK_BASE_TOKENS = frozenset({
    'soy',          # canonical (we normalise 'soya' → 'soy' below)
    'oat', 'almond', 'hazelnut', 'cashew', 'rice', 'coconut',
    'hemp', 'pea', 'macadamia', 'pistachio',
    'ricotta', 'mascarpone',   # v5.1: pasta filling mutual exclusivity
})

# COOKING_STATE_TOKENS: mutually exclusive cooking / processing states.
COOKING_STATE_TOKENS = frozenset({
    'raw', 'roast', 'smoked', 'unsmoked', 'dried', 'cured',
})

# Normalise variant spellings to canonical form before set lookup.
_HARD_CONFLICT_NORM = {
    'soya':    'soy',
    'roasted': 'roast',
    'cans':    'can',      # v5.2: packaging normalisation (plural → singular)
    'bottles': 'bottle',   # v5.2: packaging normalisation
}

# ONE_SIDED_CONFLICT_TOKENS: presence in ONE product but not the other
# is always a hard conflict (e.g. "baby carrots" vs "carrots").
ONE_SIDED_CONFLICT_TOKENS = frozenset({
    'baby',
    'reduced',   # v5.1: "reduced fat/sugar" vs standard
    'granary',   # v5.1: Hovis Granary vs plain Wholemeal
    'cup',       # v5.1: cup noodles vs stir-fry format
    'buttons',   # v5.1: chocolate buttons vs chocolate block
    'rose',      # v5.1: rosé wine vs non-rosé (unaccented spelling)
    'light',     # v5.2: "light" variant vs full-fat/standard
    'decaf',     # v5.2: decaffeinated vs caffeinated (also catches 'decaff')
    'decaff',    # v5.2: alternative UK spelling
})

# PREPARATION_CONFLICT_PAIRS: mutually-exclusive preparation tokens —
# only fires when BOTH products contain a token from the same pair but
# they disagree (e.g. "in juice" vs "in syrup").
PREPARATION_CONFLICT_PAIRS = [frozenset({'juice', 'syrup'})]   # v5.1

# PACKAGING_FORMAT_TOKENS: mutually-exclusive physical container types.
# 'cans'/'bottles' are normalised to 'can'/'bottle' via _HARD_CONFLICT_NORM.
PACKAGING_FORMAT_TOKENS = frozenset({'can', 'bottle'})          # v5.2

# ── End v5 hard conflict sets ─────────────────────────────────────────────────

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

CATEGORY_ALIASES = {
    'food_cupboard': 'grocery',
    'fresh_food':    'grocery',
    'bakery':        'grocery',
    'frozen':        'grocery',
    'free-from':     'grocery',
}

# Pass 4 (cross-bucket catch-all) threshold
PASS4_THRESHOLD = 0.90

OUTPUT_DIR  = 'data/clusters'
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print('Configuration loaded (v5).')
print(f'  Fuzzy threshold:          {FUZZY_THRESHOLD} (own-brand: {FUZZY_THRESHOLD_OWNBRAND}, no-unit: {FUZZY_THRESHOLD_NOUNIT})')
print(f'  Short stripped threshold: {SHORT_STRIPPED_THRESHOLD}')
print(f'  Unit tolerance:           ±{UNIT_TOLERANCE_BRANDED*100:.0f}% branded, ±{UNIT_TOLERANCE_OWN_BRAND*100:.0f}% own-brand')
print(f'  Pack ratio max:           {PACK_QTY_MAX_RATIO}  (NaN vs multipack: hard reject)')
print(f'  Alcohol-free penalty:     {ALCOHOL_FREE_PENALTY}')
print(f'  Pass 4 threshold:         {PASS4_THRESHOLD}')
print(f'  Output directory:         {OUTPUT_DIR}')

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

# Guard: when imported by test_similarity.py for standalone test runs,
# all function/constant definitions above are available; raise SystemExit to
# prevent the data-loading and clustering body from executing.
# test_similarity.py catches this SystemExit via 'except SystemExit: pass'.
if globals().get('CLUSTERING_TEST_IMPORT', False):
    raise SystemExit(0)

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
    if not brand:
        return name
    pat = re.compile(r'\b' + re.escape(brand.lower()) + r'\b', re.IGNORECASE)
    result = pat.sub('', name).strip()
    return re.sub(r'\s+', ' ', result)


def _unit_type_compatible(ut_a, ut_b) -> bool:
    if pd.isna(ut_a) or pd.isna(ut_b) or not ut_a or not ut_b:
        return True
    return str(ut_a).strip().lower() == str(ut_b).strip().lower()


def _unit_value_compatible(uv_a, uv_b, tolerance, pq_a=None, pq_b=None) -> bool:
    """
    v5.2: Extended with pack-normalised comparison.
    Some retailers (e.g. ASDA) report the *total* weight of a multipack while
    others (Sains, Tesco) report the *per-unit* weight.  When both sides share
    the same pack_quantity (>1), we also try dividing each side's unit_value by
    its pack_qty before comparing — this recovers matches that would otherwise
    fail because of the 10× ratio (e.g. 360g total vs 36g per-sachet for a
    10-pack of porridge sachets).  The guard `pq_a == pq_b` prevents the
    normalisation from incorrectly linking different-count packs.
    """
    a_valid = pd.notna(uv_a) and float(uv_a) > 0
    b_valid = pd.notna(uv_b) and float(uv_b) > 0
    if not a_valid or not b_valid:
        return True
    uva, uvb = float(uv_a), float(uv_b)
    # Direct comparison
    if max(uva, uvb) / min(uva, uvb) <= (1.0 + tolerance):
        return True
    # Pack-normalised comparison — only when both pack quantities are the same
    # integer > 1, so we know neither side is a single-unit product.
    pqa_ok = pd.notna(pq_a) and float(pq_a) > 1
    pqb_ok = pd.notna(pq_b) and float(pq_b) > 1
    if pqa_ok and pqb_ok:
        pqa, pqb = float(pq_a), float(pq_b)
        if abs(pqa - pqb) / max(pqa, pqb) < 0.05:   # same pack count (within 5%)
            per_a = uva / pqa
            per_b = uvb / pqb
            # Case A: side-A reports total weight, side-B reports per-unit
            if per_a > 0 and max(per_a, uvb) / min(per_a, uvb) <= (1.0 + tolerance):
                return True
            # Case B: side-B reports total weight, side-A reports per-unit
            if per_b > 0 and max(uva, per_b) / min(uva, per_b) <= (1.0 + tolerance):
                return True
    return False


def _pack_compatible(pq_a, pq_b) -> bool:
    """
    v5 fix: if one product has NaN pack_qty and the other is an explicit
    multipack (>1 unit), reject — we cannot assume the unknown product is
    a single unit.  Both NaN → no constraint (unchanged).
    """
    a_valid = pd.notna(pq_a) and float(pq_a) > 0
    b_valid = pd.notna(pq_b) and float(pq_b) > 0
    if not a_valid and not b_valid:
        return True   # both unknown → no constraint
    if not a_valid or not b_valid:
        # One side is unknown; only compatible if the known side is a single unit
        known_qty = float(pq_b if b_valid else pq_a)
        return known_qty < 1.5   # strictly single (1) is OK; 2-packs and above are not
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


def _has_alcohol_free(text: str) -> bool:
    """v5: detect alcohol-free markers from grocery_vocab ATTRIBUTES['alcohol_marker']."""
    t = text.lower()
    return any(p in t for p in ALCOHOL_FREE_PATTERNS)


def _hard_conflict_check(name_a: str, name_b: str) -> bool:
    """
    v5 / v5.1: Returns True if the two names represent irreconcilably different
    product variants regardless of fuzzy score.

    Checks (in order):
      1. Milk base mismatch  — soya vs oat vs almond / ricotta vs mascarpone
      2. Named flavor clash  — ginger vs mint vs raspberry etc.
         (only fires when BOTH have named flavor tokens)
      3. Cooking state clash — raw vs roast vs smoked etc.
         (only fires when BOTH have cooking state tokens)
      4. One-sided always-conflict tokens — baby, reduced, granary, cup,
         buttons, rose (presence in one but not the other)
      5. Preparation type conflict — juice vs syrup etc.
    """
    toks_a_raw = set(name_a.lower().split())
    toks_b_raw = set(name_b.lower().split())

    # Normalise variant spellings (roasted→roast, soya→soy)
    toks_a = {_HARD_CONFLICT_NORM.get(t, t) for t in toks_a_raw}
    toks_b = {_HARD_CONFLICT_NORM.get(t, t) for t in toks_b_raw}

    # 1. Milk base conflict
    base_a = toks_a & MILK_BASE_TOKENS
    base_b = toks_b & MILK_BASE_TOKENS
    if base_a and base_b and not (base_a & base_b):
        return True

    # 2. Named flavor conflict (only when BOTH have named flavor tokens)
    flavor_a = toks_a & FLAVOR_NAMED_TOKENS
    flavor_b = toks_b & FLAVOR_NAMED_TOKENS
    if flavor_a and flavor_b and not (flavor_a & flavor_b):
        return True

    # 3. Cooking state conflict (only when BOTH have state tokens)
    state_a = toks_a & COOKING_STATE_TOKENS
    state_b = toks_b & COOKING_STATE_TOKENS
    if state_a and state_b and not (state_a & state_b):
        return True

    # 4. One-sided always-conflict tokens
    for tok in ONE_SIDED_CONFLICT_TOKENS:
        # Check against the raw (un-normalised) token set
        if (tok in toks_a_raw) != (tok in toks_b_raw):
            return True

    # 5. Preparation type conflict (e.g. "in juice" vs "in syrup")
    for pair in PREPARATION_CONFLICT_PAIRS:
        hits_a = toks_a_raw & pair
        hits_b = toks_b_raw & pair
        if hits_a and hits_b and hits_a != hits_b:
            return True

    # 6. Packaging format conflict (can vs bottle) — v5.2
    pkg_a = toks_a & PACKAGING_FORMAT_TOKENS
    pkg_b = toks_b & PACKAGING_FORMAT_TOKENS
    if pkg_a and pkg_b and not (pkg_a & pkg_b):
        return True

    return False


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

    # v5: alcohol-free conflict (wired from grocery_vocab ATTRIBUTES['alcohol_marker'])
    if _has_alcohol_free(str_a) != _has_alcohol_free(str_b):
        penalty += ALCOHOL_FREE_PENALTY

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
    if not _unit_value_compatible(
            row_a.get('unit_value'), row_b.get('unit_value'), unit_tolerance,
            pq_a=row_a.get('pack_quantity'), pq_b=row_b.get('pack_quantity')):
        return False, 0.0

    # Hard constraint 4: pack quantity ratio (v5: NaN vs multipack now rejected)
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

    # Hard constraint 5 (v5): irreconcilable product variant
    if _hard_conflict_check(raw_name_a, raw_name_b):
        return False, 0.0

    # Brand stripping for branded pass
    if pass_type == 'branded':
        brand = str(row_a.get('known_brand_clean', '') or '').lower()
        name_a = _strip_brand(raw_name_a, brand)
        name_b = _strip_brand(raw_name_b, brand)
    else:
        name_a, name_b = raw_name_a, raw_name_b

    # Short-name guard: both ≤2 tokens AND neither has a unit → reject
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

    # Attribute penalty (includes v5 alcohol-free)
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

    # Short stripped-name guard for branded pass
    if pass_type == 'branded':
        ta, tb = len(name_a.split()), len(name_b.split())
        if ta == tb and ta <= 3:
            threshold = max(threshold, SHORT_STRIPPED_THRESHOLD)

    # Truncation bonus
    trunc_a = bool(row_a.get('is_truncated', False))
    trunc_b = bool(row_b.get('is_truncated', False))
    if trunc_a or trunc_b:
        threshold -= TRUNCATION_BONUS

    return score >= threshold, score


def compute_similarity_pass4(row_a, row_b):
    """
    v5 Pass 4 — cross-bucket catch-all.

    High threshold (PASS4_THRESHOLD=0.90).  Allows any product_type
    combination EXCEPT branded↔own_brand (those are intentionally separate).
    All existing hard constraints + hard conflict checks apply.
    """
    if row_a['supermarket'] == row_b['supermarket']:
        return False, 0.0

    # Skip branded↔own_brand cross-matching — they're legitimately different tiers
    pt_a = str(row_a.get('product_type', '') or '').lower()
    pt_b = str(row_b.get('product_type', '') or '').lower()
    if (pt_a == 'branded') != (pt_b == 'branded'):
        if 'own_brand' in (pt_a, pt_b):
            return False, 0.0

    if not _unit_type_compatible(row_a.get('unit_type'), row_b.get('unit_type')):
        return False, 0.0
    if not _unit_value_compatible(
            row_a.get('unit_value'), row_b.get('unit_value'), UNIT_TOLERANCE_BRANDED,
            pq_a=row_a.get('pack_quantity'), pq_b=row_b.get('pack_quantity')):
        return False, 0.0
    if not _pack_compatible(row_a.get('pack_quantity'), row_b.get('pack_quantity')):
        return False, 0.0

    name_a = str(row_a.get('normalized_name', '') or '')
    name_b = str(row_b.get('normalized_name', '') or '')

    if _hard_conflict_check(name_a, name_b):
        return False, 0.0

    penalty = _attribute_penalty(
        row_a.get('attributes_keywords'), row_b.get('attributes_keywords'),
        name_a=name_a, name_b=name_b
    )
    if penalty >= 0.25:   # strong attribute mismatch → skip
        return False, 0.0

    token_sort = fuzz.token_sort_ratio(name_a, name_b) / 100.0
    token_set  = fuzz.token_set_ratio(name_a, name_b) / 100.0
    score = 0.55 * token_sort + 0.45 * token_set - penalty

    return score >= PASS4_THRESHOLD, score

# ============================================================
# SELF-TESTS  (test cases live in test_similarity.py)
# ============================================================
# _mk is kept here so the import in test_similarity.__main__ can use it.

def _mk(name, uv, ut, sm, brand=None, tier='standard', ptype='branded', pq=None, attrs=None, trunc=False):
    return pd.Series({
        'normalized_name': name, 'unit_value': uv, 'unit_type': ut,
        'supermarket': sm, 'has_unit': pd.notna(uv) and uv > 0,
        'pack_quantity': pq, 'attributes_keywords': attrs,
        'known_brand_clean': brand, 'tier_type': tier, 'product_type': ptype,
        'is_truncated': trunc,
    })

print('\nRunning similarity self-tests...')
from test_similarity import run_tests as _run_tests
_all_pass = _run_tests(
    compute_similarity,
    tol_branded=UNIT_TOLERANCE_BRANDED,
    tol_own=UNIT_TOLERANCE_OWN_BRAND,
    tol_unbranded=UNIT_TOLERANCE_UNBRANDED,
)
print(f'\nAll tests passed: {_all_pass}')
if not _all_pass:
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

# ============================================================
# PASS 1 — BRANDED
# ============================================================

branded_df = df[df['product_type'] == 'branded'].copy()
print(f'\nPass 1 — Branded: {len(branded_df):,} products')

def _branded_key_a(row):
    brand = str(row['known_brand_clean']).lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    return f'{brand}||{cat}||{utype}'

def _branded_key_b(row):
    brand = str(row['known_brand_clean']).lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    return f'{brand}||{cat}'

def _branded_key_c(row):
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
    tier  = str(row['tier_type'] or 'standard').lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    tok1  = toks[0] if toks else 'unknown'
    return f'{tier}||{cat}||{utype}||{tok1}'

def _own_key_b(row):
    tier = str(row['tier_type'] or 'standard').lower().strip()
    cat  = str(row['cat_norm']).lower().strip()
    toks = _get_tokens(row['normalized_name'])
    tok1 = toks[0] if toks else 'unknown'
    return f'{tier}||{cat}||{tok1}'

def _own_key_c(row):
    tier  = str(row['tier_type'] or 'standard').lower().strip()
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    tok2  = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
    return f'{tier}||{cat}||{utype}||{tok2}'

def _own_key_d(row):
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
    cat   = str(row['cat_norm']).lower().strip()
    utype = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    toks  = _get_tokens(row['normalized_name'])
    sig   = sorted(toks[:2])
    tok_key = '_'.join(sig) if sig else 'unknown'
    return f'{cat}||{utype}||{tok_key}'

def _unb_key_b(row):
    cat  = str(row['cat_norm']).lower().strip()
    toks = _get_tokens(row['normalized_name'])
    sig  = sorted(toks[:2])
    tok_key = '_'.join(sig) if sig else 'unknown'
    return f'{cat}||{tok_key}'

def _unb_key_c(row):
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

def _unb_key_e(row):
    """
    v5: Cross-token-window key — cat + utype + unit_bucket(25g) + tok3.
    Catches products whose tok1/tok2 differ but share a later token
    (e.g. 'jacket potato' vs 'baking potato' — both have 'potato' but at
    different positions; key_c/d don't help because tok3 is also absent).
    Using 25g buckets + tok3 keeps blocks small.
    """
    cat    = str(row['cat_norm']).lower().strip()
    utype  = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    ubucket = _unit_bucket(row['unit_value'], bucket_size=25)
    toks   = _get_tokens(row['normalized_name'])
    # Prefer tok2 (index 1) — most products have ≥2 tokens
    tok2 = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
    return f'{cat}||{utype}||{ubucket}||{tok2}'

print('Building multi-key blocks for unbranded (keys a–e)...')
pairs_unbranded_all = build_multi_blocks(
    unbranded_df, [_unb_key_a, _unb_key_b, _unb_key_c, _unb_key_d, _unb_key_e]
)

print('Running comparisons...')
matches_unbranded = run_pass(unbranded_df, pairs_unbranded_all, 'unbranded', UNIT_TOLERANCE_UNBRANDED)
print(f'Pass 3 complete: {len(matches_unbranded):,} matches')

# ============================================================
# PASS 4 — CROSS-BUCKET CATCH-ALL (v5 new)
# ============================================================

# Tentative singletons: products with no direct match in passes 1–3
matched_idx_so_far = set()
for ia, ib, _ in matches_branded + matches_own + matches_unbranded:
    matched_idx_so_far.add(ia)
    matched_idx_so_far.add(ib)

singleton_df = df[~df.index.isin(matched_idx_so_far)].copy()
print(f'\nPass 4 — Cross-bucket: {len(singleton_df):,} tentative singletons')

def _pass4_key_a(row):
    """First significant token key — catches direct brand-detection failures."""
    cat    = str(row['cat_norm']).lower().strip()
    utype  = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    ubucket = _unit_bucket(row['unit_value'], bucket_size=100)
    toks = _get_tokens(row['normalized_name'])
    tok1 = toks[0] if toks else 'unknown'
    return f'{cat}||{utype}||{ubucket}||{tok1}'

def _pass4_key_b(row):
    """Second token key — catches tok1-differs but tok2-matches cases."""
    cat    = str(row['cat_norm']).lower().strip()
    utype  = str(row['unit_type']).strip() if pd.notna(row['unit_type']) else 'none'
    ubucket = _unit_bucket(row['unit_value'], bucket_size=100)
    toks = _get_tokens(row['normalized_name'])
    tok2 = toks[1] if len(toks) > 1 else (toks[0] if toks else 'unknown')
    return f'{cat}||{utype}||{ubucket}||{tok2}'

print('Building blocks for pass 4...')
pairs_pass4 = build_multi_blocks(singleton_df, [_pass4_key_a, _pass4_key_b], max_block_size=80)

print('Running comparisons...')
matches_pass4 = []
for ia, ib in tqdm(pairs_pass4, desc='  Pass=cross_bucket', leave=True):
    is_match, score = compute_similarity_pass4(singleton_df.loc[ia], singleton_df.loc[ib])
    if is_match:
        matches_pass4.append((ia, ib, score))
print(f'Pass 4 complete: {len(matches_pass4):,} matches')

# ============================================================
# UNION-FIND ASSEMBLY
# ============================================================

print('\nAssembling Union-Find...')
uf = UnionFind(len(df))

pair_scores = {}
all_matches = (
    [('branded',      m) for m in matches_branded] +
    [('own_brand',    m) for m in matches_own] +
    [('unbranded',    m) for m in matches_unbranded] +
    [('cross_bucket', m) for m in matches_pass4]
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
    """
    v5 SCORE-BASED fix (replaces v4 greedy iteration).

    For clusters with multiple products from the same supermarket:
    - For each duplicate SM, keep the product with the highest
      max cross-SM pair score as the representative.
    - Excluded products are recycled into sub-clusters if they have
      direct pairs between them; otherwise they become singletons.

    This preserves genuine 4-way clusters that the greedy approach
    destroyed by arbitrary iteration-order selection.
    """
    by_sm = {sm: list(g.index) for sm, g in group_df.groupby('supermarket')}

    # Fast path: no duplicate SMs
    if not any(len(idxs) > 1 for idxs in by_sm.values()):
        return [group_df]

    indices = list(group_df.index)

    def best_cross_sm_score(idx, my_sm):
        """Max pair score from this product to any product from a different SM."""
        best = 0.0
        for other in indices:
            if group_df.loc[other, 'supermarket'] == my_sm:
                continue
            key = (min(idx, other), max(idx, other))
            s = pair_scores.get(key, 0.0)
            if s > best:
                best = s
        return best

    primary_indices  = []
    excluded_indices = []

    for sm, idxs in by_sm.items():
        if len(idxs) == 1:
            primary_indices.append(idxs[0])
        else:
            # Pick the representative with the highest cross-SM score
            ranked = sorted(idxs, key=lambda x: best_cross_sm_score(x, sm), reverse=True)
            primary_indices.append(ranked[0])
            excluded_indices.extend(ranked[1:])

    result = [group_df.loc[primary_indices]]

    if not excluded_indices:
        return result

    # Try to form valid sub-clusters from excluded products
    if len(excluded_indices) >= 2:
        excl_df = group_df.loc[excluded_indices]
        excl_by_sm = excl_df.groupby('supermarket').size()
        if len(excl_by_sm) >= 2:
            # Only form a sub-cluster if there are direct cross-SM pairs
            has_valid_pair = any(
                (min(ia, ib), max(ia, ib)) in pair_scores
                for ia, ib in combinations(excluded_indices, 2)
                if excl_df.loc[ia, 'supermarket'] != excl_df.loc[ib, 'supermarket']
            )
            if has_valid_pair:
                # Recurse to handle any remaining SM duplicates in the sub-group
                result.extend(fix_same_supermarket_violation(excl_df, pair_scores))
                return result

    # No valid sub-clusters — excluded become singletons
    for idx in excluded_indices:
        result.append(group_df.loc[[idx]])

    return result


def fix_cross_tier_violation(group_df):
    own = group_df[group_df['product_type'] == 'own_brand']
    if own.empty or own['tier_type'].nunique() <= 1:
        return [group_df]
    sub_clusters = []
    non_own = group_df[group_df['product_type'] != 'own_brand']
    for tier, tier_group in own.groupby('tier_type'):
        sub_clusters.append(pd.concat([tier_group, non_own]))
    return sub_clusters


def fix_unit_type_violation(group_df):
    uts = group_df['unit_type'].dropna().unique()
    if len(uts) <= 1:
        return [group_df]
    no_unit = group_df[group_df['unit_type'].isna()]
    sub_clusters = []
    for ut in uts:
        sub = group_df[group_df['unit_type'] == ut]
        if not no_unit.empty:
            sub = pd.concat([sub, no_unit])
        sub_clusters.append(sub)
    return sub_clusters


def _purge_hard_conflicts(group_df):
    """
    v5.2: Post-cluster hard-conflict purge.

    Transitive union-find chains can link products via a "bridge" product that
    has no flavor/packaging tokens, creating a cluster where two members would
    directly fail _hard_conflict_check (e.g. 'caramel' product bridged to a
    'chilli' product through a plain 'milk chocolate bar').

    Greedily removes the member with the most pairwise hard-conflicts until the
    cluster is internally consistent.  Removed products are returned as a list of
    product indices (each will become a singleton or seed a new sub-cluster).
    """
    indices = list(group_df.index)
    names   = {idx: str(group_df.loc[idx, 'normalized_name'] or '') for idx in indices}
    removed = []

    changed = True
    while changed and len(indices) >= 2:
        changed = False
        conflict_counts: dict = defaultdict(int)
        for ia, ib in combinations(indices, 2):
            if _hard_conflict_check(names[ia], names[ib]):
                conflict_counts[ia] += 1
                conflict_counts[ib] += 1
        if conflict_counts:
            # Remove the product with the most conflicts; tiebreak by higher idx
            # (later-indexed product is typically the "intruder" in most cases)
            worst = max(conflict_counts, key=lambda x: (conflict_counts[x], x))
            indices.remove(worst)
            removed.append(worst)
            changed = True

    return group_df.loc[indices], removed


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
        ut_subs = fix_unit_type_violation(sc)
        for ut_sc in ut_subs:
            tier_subs = fix_cross_tier_violation(ut_sc)
            if len(tier_subs) > 1:
                tier_violations_fixed += 1
            final_clusters.extend(tier_subs)

unit_type_violations_fixed = sum(
    1 for members in validated_clusters
    if len(members) > 1 and df.loc[members, 'unit_type'].dropna().nunique() > 1
)
print(f'  Same-SM violations fixed:      {sm_violations_fixed:,}')
print(f'  Unit-type violations fixed:    {unit_type_violations_fixed:,}')
print(f'  Cross-tier violations fixed:   {tier_violations_fixed:,}')

# v5.2: Post-cluster hard-conflict purge — split clusters where a transitive
# union-find bridge created internally inconsistent flavor/packaging pairs.
print('Running post-cluster hard-conflict purge...')
purge_products_removed = 0
purged_clusters: list = []
for cl in final_clusters:
    if len(cl) <= 1:
        purged_clusters.append(cl)
        continue
    clean_cl, conflict_idxs = _purge_hard_conflicts(cl)
    purged_clusters.append(clean_cl)
    for bad_idx in conflict_idxs:
        purged_clusters.append(cl.loc[[bad_idx]])   # singleton
    purge_products_removed += len(conflict_idxs)

final_clusters = purged_clusters
print(f'  Products purged from clusters: {purge_products_removed:,}')
print(f'  Final cluster count:           {len(final_clusters):,}')

# Assign sequential cluster IDs (largest first)
final_clusters.sort(key=lambda g: -len(g))
cluster_id_map = {}
match_type_map = {}

# Build Pass-4 pair key set for diagnostic tracking
pass4_pair_keys = {(min(ia, ib), max(ia, ib)) for ia, ib, _ in matches_pass4}

for cid, group in enumerate(final_clusters):
    for idx in group.index:
        cluster_id_map[idx] = cid
    types = group['product_type'].value_counts()
    match_type_map[cid] = types.index[0] if len(types) else 'unknown'

# Override match_type to 'cross_bucket' for clusters containing ≥1 Pass 4 pair
for cid, group in enumerate(final_clusters):
    idxs = list(group.index)
    for ia, ib in combinations(idxs, 2):
        key = (min(ia, ib), max(ia, ib))
        if key in pass4_pair_keys:
            match_type_map[cid] = 'cross_bucket'
            break

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
xp    = multi[multi['match_type'] == 'cross_bucket']

audit_ids = pd.concat([
    bp.sample(min(20, len(bp)), random_state=RANDOM_SEED),
    op.sample(min(15, len(op)), random_state=RANDOM_SEED),
    up.sample(min(10, len(up)), random_state=RANDOM_SEED),
    xp.sample(min(5,  len(xp)), random_state=RANDOM_SEED) if len(xp) else pd.DataFrame(),
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
print('DIAGNOSTIC REPORT v5')
print('=' * 60)
print(f'Total clusters (incl singletons): {len(cluster_summary):,}')
print(f'Singletons:                       {(cluster_summary["cluster_size"]==1).sum():,}')
print(f'Multi-product clusters (≥2):      {n_multi:,}')
print(f'  4-way: {(non_singleton["n_supermarkets"]==4).sum():,}')
print(f'  3-way: {(non_singleton["n_supermarkets"]==3).sum():,}')
print(f'  2-way: {(non_singleton["n_supermarkets"]==2).sum():,}')

print(f'\nBy match type:')
for mt in ['branded', 'own_brand', 'unbranded', 'cross_bucket']:
    n = (non_singleton['match_type'] == mt).sum()
    if n:
        print(f'  {mt:14s}: {n:,}')

print(f'\nPass 4 contribution:')
print(f'  Cross-bucket matches found:  {len(matches_pass4):,}')
print(f'  Cross-bucket clusters:       {(non_singleton["match_type"]=="cross_bucket").sum():,}')

print(f'\nProduct coverage:')
in_cluster = clusters_df[clusters_df['cluster_size'] >= 2]
for sm in sorted(df['supermarket'].unique()):
    total   = len(df[df['supermarket'] == sm])
    matched = len(in_cluster[in_cluster['supermarket'] == sm])
    print(f'  {sm:12s}: {matched:,}/{total:,} = {matched/total*100:.1f}%')

print(f'\nQuality scores (avg_pairwise_score):')
scores_avail = non_singleton.dropna(subset=['avg_pairwise_score'])
for mt in ['branded', 'own_brand', 'unbranded', 'cross_bucket']:
    sub = scores_avail[scores_avail['match_type'] == mt]['avg_pairwise_score']
    if len(sub):
        print(f'  {mt:14s}: mean={sub.mean():.3f}  p5={sub.quantile(0.05):.3f}  p25={sub.quantile(0.25):.3f}')

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
