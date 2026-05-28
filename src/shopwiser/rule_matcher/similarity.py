"""Pairwise product similarity (fuzzy matching, hard conflicts, penalties)."""

import re

import pandas as pd
from rapidfuzz import fuzz

from .config import *  # noqa: F403

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
    Extended with pack-normalised comparison.
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
    # One-sided pack normalisation: one SM reports total size, the other reports per-unit
    # with a pack count (e.g. ASDA "Beck's 1.65L" vs Sains "Beck's 275ml x6").
    if pqa_ok and not pqb_ok:
        # A has pack count; check if B's value ≈ A's total (uva × pq_a)
        total_a = uva * float(pq_a)
        if total_a > 0 and max(total_a, uvb) / min(total_a, uvb) <= (1.0 + tolerance):
            return True
    elif not pqa_ok and pqb_ok:
        # B has pack count; check if A's value ≈ B's total (uvb × pq_b)
        total_b = uvb * float(pq_b)
        if total_b > 0 and max(uva, total_b) / min(uva, total_b) <= (1.0 + tolerance):
            return True
    return False


def _pack_compatible(pq_a, pq_b) -> bool:
    """
    Strict version (own_brand / unbranded): NaN vs multipack → reject.
    Both NaN → no constraint.
    """
    a_valid = pd.notna(pq_a) and float(pq_a) > 0
    b_valid = pd.notna(pq_b) and float(pq_b) > 0
    if not a_valid and not b_valid:
        return True   # both unknown → no constraint
    if not a_valid or not b_valid:
        known_qty = float(pq_b if b_valid else pq_a)
        return known_qty < 1.5   # strictly single (1) is OK; 2-packs and above are not
    ratio = max(float(pq_a), float(pq_b)) / min(float(pq_a), float(pq_b))
    return ratio <= PACK_QTY_MAX_RATIO


def _pack_compatible_branded(pq_a, pq_b) -> bool:
    """
    Relaxed version for branded pass only.

    Many retailers scrape pack count inconsistently: one may record pq=5 for a
    "5 Pack" product while another leaves pack_quantity as NaN even though the
    total weight matches.  For branded products the known_brand + name similarity
    check is the primary safeguard; the unit_value check validates total weights.
    Allows NaN vs known pack_qty ≤ 8; keeps strict ratio for two known pack counts.
    """
    a_valid = pd.notna(pq_a) and float(pq_a) > 0
    b_valid = pd.notna(pq_b) and float(pq_b) > 0
    if not a_valid and not b_valid:
        return True
    if not a_valid or not b_valid:
        known_qty = float(pq_b if b_valid else pq_a)
        return known_qty <= 20   # allow common multipack sizes up to 20; unit_value is the real guard
    ratio = max(float(pq_a), float(pq_b)) / min(float(pq_a), float(pq_b))
    return ratio <= PACK_QTY_MAX_RATIO


def _get_wine_type(text: str) -> str:
    t = text.lower()
    for tok in WINE_TYPE_TOKENS:
        if re.search(r'\b' + re.escape(tok) + r'\b', t):
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
    if re.search(r'\bdiet\b', t):
        return True
    return any(p in t for p in NAS_PATTERNS)


def _has_alcohol_free(text: str) -> bool:
    """Detect alcohol-free markers from grocery_vocab ATTRIBUTES['alcohol_marker']."""
    t = text.lower()
    return any(p in t for p in ALCOHOL_FREE_PATTERNS)


def _hard_conflict_check(name_a: str, name_b: str) -> bool:
    """
    Returns True if the two names represent irreconcilably different
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
    toks_a = {HARD_CONFLICT_NORM.get(t, t) for t in toks_a_raw}
    toks_b = {HARD_CONFLICT_NORM.get(t, t) for t in toks_b_raw}

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

    # 4. One-sided always-conflict tokens (checked against NORMALISED sets so
    #    synonym mappings in HARD_CONFLICT_NORM are applied first)
    for tok in ONE_SIDED_CONFLICT_TOKENS:
        if (tok in toks_a) != (tok in toks_b):
            return True

    # 5. Preparation type conflict (e.g. "in juice" vs "in syrup")
    for pair in PREPARATION_CONFLICT_PAIRS:
        hits_a = toks_a_raw & pair
        hits_b = toks_b_raw & pair
        if hits_a and hits_b and hits_a != hits_b:
            return True

    # 6. Packaging format conflict (can vs bottle)
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

    # alcohol-free conflict (wired from grocery_vocab ATTRIBUTES['alcohol_marker'])
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

    # Hard constraint 4: pack quantity ratio
    # branded pass uses a relaxed check (unit_value is the primary safeguard for branded);
    # own_brand / unbranded keep the strict check to avoid generic-name false positives.
    _pq_fn = _pack_compatible_branded if pass_type == 'branded' else _pack_compatible
    if not _pq_fn(row_a.get('pack_quantity'), row_b.get('pack_quantity')):
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
        # frozen vs fresh_food / food_cupboard are distinct storage conditions
        # (prevents fresh whole vegetables matching frozen prepared products)
        _cat_a = str(row_a.get('category', '') or '').lower()
        _cat_b = str(row_b.get('category', '') or '').lower()
        if ('frozen' in _cat_a) != ('frozen' in _cat_b):
            return False, 0.0

    raw_name_a = str(row_a.get('normalized_name', '') or '')
    raw_name_b = str(row_b.get('normalized_name', '') or '')

    # both names empty → nothing to compare; reject
    if not raw_name_a and not raw_name_b:
        return False, 0.0

    # Hard constraint 5: irreconcilable product variant
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

    # Attribute penalty (includes alcohol-free)
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
    Pass 4 — cross-bucket catch-all.

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

    # both names empty → nothing to compare; reject (prevents empty-brand FP)
    if not name_a and not name_b:
        return False, 0.0

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
