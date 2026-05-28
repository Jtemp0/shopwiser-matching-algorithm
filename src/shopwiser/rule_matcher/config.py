"""Clustering thresholds, token sets, and seeds."""

import random
import numpy as np

from shopwiser.paths import cluster_outputs_path

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
# per-type thresholds for the completion passes (5B/5C).
# Branded can go lower because brand pre-filters heavily; own_brand needs
# to stay high because generic names (e.g. "black beans" vs "black eyed
# beans") can score above 0.80 despite being different products.
COMPLETION_THRESHOLD = {
    'branded':   0.600,   # completion-only threshold; guarded by hard conflicts + brand/unit gates
    'own_brand': 0.680,   # completion-only threshold in tight tier/category/unit context
    'unbranded': 0.640,   # completion-only threshold with unit/category guards
}
FUZZY_THRESHOLD_COMPLETION = 0.600   # kept for backward-compat references

SHORT_STRIPPED_THRESHOLD = 0.74   # ≤3 equal-token branded pairs (lowered from 0.87;
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
    'vegan':     0.10,   # reduced from 0.25; "vegan" is often just a label difference for
                         # the same product (e.g. Cadbury Plant Bar labelled vegan in one store)
}
DIET_PENALTY = 0.10

# ── alcohol-free penalty (wired from grocery_vocab ATTRIBUTES['alcohol_marker'])
ALCOHOL_FREE_PENALTY   = 0.60
ALCOHOL_FREE_PATTERNS  = [
    'alcohol free', 'alcohol-free', 'non-alcoholic',
    '0% alcohol', 'low alcohol', 'alcohol_marker',
]

# ── Hard conflict token sets ──────────────────────────────────────────────────
# Canonical home: ``shopwiser.conflict_tokens``. Re-exported here so existing
# imports (``from shopwiser.rule_matcher.config import FLAVOR_NAMED_TOKENS, …``)
# keep working without churn. Edit the vocabulary in conflict_tokens.py.
from shopwiser.conflict_tokens import (  # noqa: E402
    COOKING_STATE_TOKENS,
    FLAVOR_NAMED_TOKENS,
    HARD_CONFLICT_NORM,
    MILK_BASE_TOKENS,
    ONE_SIDED_CONFLICT_TOKENS,
    PACKAGING_FORMAT_TOKENS,
    PREPARATION_CONFLICT_PAIRS,
)

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

OUTPUT_DIR = cluster_outputs_path(sample=False)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def print_config_banner() -> None:
    print('Configuration loaded.')
    print(f'  Fuzzy threshold:          {FUZZY_THRESHOLD} (own-brand: {FUZZY_THRESHOLD_OWNBRAND}, no-unit: {FUZZY_THRESHOLD_NOUNIT})')
    print(f'  Short stripped threshold: {SHORT_STRIPPED_THRESHOLD}')
    print(f'  Unit tolerance:           ±{UNIT_TOLERANCE_BRANDED*100:.0f}% branded, ±{UNIT_TOLERANCE_OWN_BRAND*100:.0f}% own-brand')
    print(f'  Pack ratio max:           {PACK_QTY_MAX_RATIO}  (NaN vs multipack: hard reject)')
    print(f'  Alcohol-free penalty:     {ALCOHOL_FREE_PENALTY}')
    print(f'  Pass 4 threshold:         {PASS4_THRESHOLD}')
    print(f'  Output directory:         {OUTPUT_DIR}')
