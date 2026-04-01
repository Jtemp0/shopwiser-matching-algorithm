"""Clustering thresholds, token sets, and seeds (v10)."""

import random
import numpy as np

from shopwiser.paths import DATA_OUTPUTS_CLUSTERS

# ============================================================
# CONFIGURATION
# ============================================================

UNIT_TOLERANCE_BRANDED   = 0.05
UNIT_TOLERANCE_OWN_BRAND = 0.05
UNIT_TOLERANCE_UNBRANDED = 0.05

# Wider tolerances used only in completion passes (5B / 5C) where brand+category+unit_type
# are already tightly constrained.  Minor size reporting differences between retailers
# (e.g. 400 g vs 415 g) are acceptable there.
COMPLETION_UNIT_TOL = {
    'branded':   0.25,   # widened: retailers often differ slightly in stated weight
    'own_brand': 0.22,
    'unbranded': 0.22,
}

FUZZY_THRESHOLD            = 0.710   # lowered from 0.720 — meat/type ONE_SIDED guards protect precision
FUZZY_THRESHOLD_OWNBRAND   = 0.775   # lowered from 0.805
FUZZY_THRESHOLD_NOUNIT     = 0.790   # lowered from 0.800
# v8: per-type thresholds for the completion passes (5B/5C).
# Branded can go lower because brand pre-filters heavily; own_brand needs
# to stay high because generic names (e.g. "black beans" vs "black eyed
# beans") can score above 0.80 despite being different products.
COMPLETION_THRESHOLD = {
    'branded':   0.600,   # v10: completion-only threshold; guarded by hard conflicts + brand/unit gates
    'own_brand': 0.680,   # v10: completion-only threshold in tight tier/category/unit context
    'unbranded': 0.640,   # v10: completion-only threshold with unit/category guards
}
FUZZY_THRESHOLD_COMPLETION = 0.600   # kept for backward-compat references

SHORT_STRIPPED_THRESHOLD = 0.74   # ≤3 equal-token branded pairs (v7: lowered from 0.87;
                                  # ingredient/flavor conflicts caught by FLAVOR_NAMED_TOKENS and
                                  # ONE_SIDED_CONFLICT_TOKENS, making the elevated threshold
                                  # unnecessarily strict for legitimate matches with minor penalties)

PACK_QTY_MAX_RATIO   = 2.0
MAX_BLOCK_SIZE       = 200
TRUNCATION_BONUS     = 0.05       # threshold reduction for truncated names

ATTR_PENALTIES = {
    'organic':   0.20,
    'free_from': 0.20,
    'fairtrade': 0.10,
    'vegan':     0.10,   # v7: reduced from 0.25; "vegan" is often just a label difference for
                         # the same product (e.g. Cadbury Plant Bar labelled vegan in one store)
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
    # Additional fruit varieties (audit v10): kiwi/berry/melon/grape prevent
    # "Ripe & Ready Kiwi" matching "Ripe & Ready Mango" and Huel Vanilla vs Berry
    'kiwi', 'berry', 'melon', 'grape', 'pear', 'pineapple',
    'pomegranate', 'watermelon', 'passion', 'fig', 'plum',
    # Meat / poultry / seafood variety tokens (v6)
    'lamb', 'ham', 'pork', 'beef', 'chicken', 'turkey', 'duck',
    'venison', 'bacon',
    'salmon', 'tuna', 'cod', 'haddock', 'prawn', 'shrimp', 'crab',
    'mackerel', 'trout', 'sardine', 'anchovy',
    # Italian brand-variant / coffee-range tokens (v9)
    # Lavazza: Qualità Rossa (red) vs Qualità Oro (gold) — distinct product lines
    'rossa', 'oro',
    # Nescafe Azera: Intenso vs Americano — distinct coffee-style variants
    'intenso', 'americano',
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
HARD_CONFLICT_NORM = {
    'soya':          'soy',
    'roasted':       'roast',
    'cans':          'can',           # v5.2: packaging normalisation (plural → singular)
    'bottles':       'bottle',        # v5.2: packaging normalisation
    'decaffeinated': 'decaf',         # v8: treat long form as equivalent to 'decaf'
    'decaff':        'decaf',         # v8: treat UK abbreviation as equivalent to 'decaf'
}

# ONE_SIDED_CONFLICT_TOKENS: presence in ONE product but not the other
# is always a hard conflict (e.g. "baby carrots" vs "carrots").
# NOTE: the check uses _normalised_ token sets so synonym mappings in
#       HARD_CONFLICT_NORM are applied first (e.g. decaffeinated→decaf).
ONE_SIDED_CONFLICT_TOKENS = frozenset({
    'baby',
    'reduced',   # v5.1: "reduced fat/sugar" vs standard
    'granary',   # v5.1: Hovis Granary vs plain Wholemeal
    'buttons',   # v5.1: chocolate buttons vs chocolate block
    'rose',      # v5.1: rosé wine vs non-rosé (unaccented spelling)
    'light',     # v5.2: "light" variant vs full-fat/standard
    'lite',      # v9: variant spelling of 'light' (e.g. Sprite Lite) — kept in norm_name
    'decaf',     # v5.2/v8: catches decaf/decaff/decaffeinated (via HARD_CONFLICT_NORM)
    'blonde',    # v9: Starbucks Blonde Espresso vs regular Espresso — distinct roast
    'zero',      # v10: "zero sugar" / "zero calorie" vs standard — distinct product
                 #      (DIET_PENALTY alone insufficient; hard-reject is correct)
    # 'cup' removed v8: caused false rejections for pot-noodle naming variants
    #   ("cup noodles pot" vs "noodles pot" blocked incorrectly)
    # Product-category discriminators: if one product names the category type and the
    # other doesn't, they are different things (e.g. "Tomato Soup" ≠ "Tomato Sauce").
    'soup',      # "Tomato Soup" vs "Tomato Sauce" / "Tomato Paste"
    'jam',       # "Strawberry Jam" vs "Strawberry Yogurt" / "Strawberry Sauce"
    'juice',     # "Apple Juice" vs "Apple Sauce" / "Apple Cider"
    'eyed',      # "black-eyed beans" vs "black beans" — distinct legume variety
    'plum',      # "plum tomatoes" vs "chopped/tinned tomatoes" — distinct variety
    # Meat / poultry / fish directional tokens: present in one but absent in the
    # other is always a hard differentiator (e.g. "chicken pizza" ≠ "four cheese pizza").
    # These are also in FLAVOR_NAMED_TOKENS (fires when BOTH products have named
    # tokens that differ); ONE_SIDED adds the asymmetric case.
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck',
    'salmon', 'tuna', 'cod', 'bacon', 'prawn',
})

# PREPARATION_CONFLICT_PAIRS: mutually-exclusive preparation tokens —
# only fires when BOTH products contain a token from the same pair but
# they disagree (e.g. "in juice" vs "in syrup").
PREPARATION_CONFLICT_PAIRS = [frozenset({'juice', 'syrup'})]   # v5.1

# PACKAGING_FORMAT_TOKENS: mutually-exclusive physical container types.
# 'cans'/'bottles' are normalised to 'can'/'bottle' via HARD_CONFLICT_NORM.
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
    # Normalise only the hyphenated variant — all other raw categories are already
    # distinct enough to serve as fine-grained blocking keys.  Collapsing everything
    # into 'grocery' destroyed blocking precision (frozen ↔ fresh, bakery ↔ cupboard)
    # and caused the own-brand completion pool to span the entire supermarket range.
    'free-from': 'free_from',
}

# Pass 4 (cross-bucket catch-all) threshold
PASS4_THRESHOLD = 0.90

OUTPUT_DIR = DATA_OUTPUTS_CLUSTERS
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def print_config_banner() -> None:
    print('Configuration loaded (v10).')
    print(f'  Fuzzy threshold:          {FUZZY_THRESHOLD} (own-brand: {FUZZY_THRESHOLD_OWNBRAND}, no-unit: {FUZZY_THRESHOLD_NOUNIT})')
    print(f'  Short stripped threshold: {SHORT_STRIPPED_THRESHOLD}')
    print(f'  Unit tolerance:           ±{UNIT_TOLERANCE_BRANDED*100:.0f}% branded, ±{UNIT_TOLERANCE_OWN_BRAND*100:.0f}% own-brand')
    print(f'  Pack ratio max:           {PACK_QTY_MAX_RATIO}  (NaN vs multipack: hard reject)')
    print(f'  Alcohol-free penalty:     {ALCOHOL_FREE_PENALTY}')
    print(f'  Pass 4 threshold:         {PASS4_THRESHOLD}')
    print(f'  Output directory:         {OUTPUT_DIR}')
