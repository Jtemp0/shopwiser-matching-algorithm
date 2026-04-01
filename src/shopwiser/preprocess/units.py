"""Multipack patterns, unit standardisation (g/ml), price-based inference, vape guard."""

import re
from typing import Dict, Optional, Tuple

from .grocery_vocab import _VAPE_KEYWORDS

_PACK_UNIT_MULTIPLIER: Dict[str, Tuple[float, str]] = {
    'g':  (1.0,    'g'),
    'kg': (1000.0, 'g'),
    'mg': (0.001,  'g'),
    'ml': (1.0,    'ml'),
    'cl': (10.0,   'ml'),
    'l':  (1000.0, 'ml'),
}


def _is_vape_name(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in _VAPE_KEYWORDS)


def extract_multipack(name: str) -> Tuple[Optional[int], Optional[str], str, Optional[float], Optional[str]]:
    """Return (pack_quantity, pack_pattern, remaining_name, pack_unit_val, pack_unit_type).

    For the ``count_x_size`` pattern (e.g. "5 x 19.9g", "6x330ml") the per-unit
    size is converted to standardised g/ml and returned as *pack_unit_val* /
    *pack_unit_type*; the size token is NOT re-injected into the remaining string.
    The caller is responsible for computing the total unit_value as
    ``pack_unit_val * pack_quantity`` when no explicit total is found in step 4.
    All other patterns return ``None`` for the last two fields.
    """
    if not isinstance(name, str):
        return None, None, '', None, None
    name_str = name.strip()
    patterns = [
        (r'(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|cl|l\b)', 'count_x_size'),
        (r'\s[xX]\s*(\d+)\b', 'x_count'),
        (r'(\d+)\s*pack\b', 'count_pack'),
        (r'pack\s*of\s*(\d+)', 'pack_of_count'),
        (r'(\d+)\s*pk\b', 'count_pk'),
        (r'multipack\s*(\d+)', 'multipack'),
        (r'\b(\d+)\s+tea\s*bags?\b', 'bag_count'),
        (r'\b(\d+)\s+(?:portions?|sticks?|pieces?)\b', 'bag_count'),
        (r'\b(\d+)\s*(?:bags?|sachets?|tabs?|pods?|capsules?)\b', 'bag_count'),
        (r'(\d+)\s*pcs?\b', 'count_pcs'),
    ]
    for pattern, pattern_type in patterns:
        match = re.search(pattern, name_str, re.IGNORECASE)
        if match:
            quantity = int(match.group(1))
            pack_unit_val: Optional[float] = None
            pack_unit_type: Optional[str] = None
            if pattern_type == 'count_x_size':
                raw_val  = float(match.group(2))
                raw_unit = match.group(3).lower().rstrip()
                factor, base = _PACK_UNIT_MULTIPLIER.get(raw_unit, (1.0, 'g'))
                pack_unit_val  = round(raw_val * factor, 3)
                pack_unit_type = base
                remaining = name_str[:match.start()] + name_str[match.end():]
            else:
                remaining = name_str[:match.start()] + name_str[match.end():]
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return quantity, pattern_type, remaining, pack_unit_val, pack_unit_type
    bag_s = re.search(r'\b(\d{1,3})s\b', name_str, re.IGNORECASE)
    if bag_s:
        qty = int(bag_s.group(1))
        if 2 <= qty <= 500:
            remaining = name_str[:bag_s.start()] + name_str[bag_s.end():]
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return qty, 'count_s', remaining, None, None

    # Leading item count: "4 Pork Sausage Rolls" / "12 Chicken Wings"
    # Guard: quantity 2–40, followed by a capitalised word (real product noun),
    # and NOT a year (>1900), weight-like number, or pure-digit string.
    leading = re.match(r'^(\d{1,2})\s+([A-Z][a-zA-Z])', name_str)
    if leading:
        qty = int(leading.group(1))
        if 2 <= qty <= 40:
            remaining = name_str[leading.end(1):].strip()
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return qty, 'leading_count', remaining, None, None

    return None, None, name_str, None, None


def extract_and_standardize_unit(name: str) -> Tuple[Optional[float], Optional[str], str]:
    if not isinstance(name, str):
        return None, None, ''
    name_str = name.strip()

    unit_patterns = [
        (r'\((\d+(?:\.\d+)?)\s*kg\)',  1000,     'g'),
        (r'\((\d+(?:\.\d+)?)\s*g\)',   1,        'g'),
        (r'\((\d+(?:\.\d+)?)\s*l(?:tr|itre|iter)?\)',  1000, 'ml'),
        (r'\((\d+(?:\.\d+)?)\s*cl\)',  10,       'ml'),
        (r'\((\d+(?:\.\d+)?)\s*ml\)',  1,        'ml'),
        (r'(\d+(?:\.\d+)?)\s*pints?\b', 568.261, 'ml'),
        (r'(\d+(?:\.\d+)?)\s*kg\b',   1000,     'g'),
        (r'(\d+(?:\.\d+)?)\s*g\b',    1,        'g'),
        (r'(\d+(?:\.\d+)?)\s*l(?:tr|itre|iter)?\b', 1000, 'ml'),
        (r'(\d+(?:\.\d+)?)\s*cl\b',   10,       'ml'),
        (r'(\d+(?:\.\d+)?)\s*ml\b',   1,        'ml'),
    ]

    is_vape = _is_vape_name(name_str)

    for pattern, factor, base_unit in unit_patterns:
        if base_unit == 'g' and factor == 0.001 and is_vape:
            continue
        match = re.search(pattern, name_str, re.IGNORECASE)
        if match:
            value = float(match.group(1)) * factor
            remaining = re.sub(pattern, '', name_str, flags=re.IGNORECASE)
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return round(value, 3), base_unit, remaining

    if not is_vape:
        mg_match = re.search(r'(\d+(?:\.\d+)?)\s*mg\b', name_str, re.IGNORECASE)
        if mg_match:
            value = float(mg_match.group(1)) * 0.001
            remaining = re.sub(r'(\d+(?:\.\d+)?)\s*mg\b', '', name_str, flags=re.IGNORECASE)
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return round(value, 6), 'g', remaining

    return None, None, name_str


_COMMON_ML = [
    25, 35, 50, 70, 75, 100, 125, 150, 175, 187, 200, 250, 275, 284, 300,
    330, 355, 375, 400, 440, 450, 473, 500, 568, 600, 650, 700, 710, 750,
    800, 850, 900, 946, 1000, 1136, 1500, 1750, 2000, 2250, 3000, 4000, 5000,
]
_COMMON_G = [
    10, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 125, 130, 140,
    150, 160, 170, 175, 180, 190, 200, 210, 220, 225, 240, 250, 270, 280,
    300, 320, 340, 350, 360, 375, 400, 425, 450, 454, 475, 500, 510, 530,
    550, 600, 625, 650, 680, 700, 750, 800, 850, 900, 907, 950, 1000, 1100,
    1200, 1250, 1360, 1500, 1800, 2000, 2270, 2500, 3000, 4000, 5000,
]
_SNAP_THRESHOLD = 0.03  # snap if within 3% of a standard size

# Drink-bottle sizes eligible for the ×10 ppu-unit correction (see below).
# Only include sizes that are unambiguously drink containers (not small extracts).
_DRINK_BOTTLE_ML = frozenset([350, 355, 375, 440, 473, 500, 568, 700, 710, 750, 800])


def _snap_to_standard(value: float, standards: list) -> float:
    """Round *value* to the nearest standard size if within _SNAP_THRESHOLD, else return as-is."""
    for s in standards:
        if abs(value - s) / s <= _SNAP_THRESHOLD:
            return float(s)
    return value


def infer_unit_from_price(price_str, price_per_unit_str, unit_col: str) -> Tuple[Optional[float], Optional[str]]:
    """Derive product size from price ÷ price-per-unit (used as fallback when name extraction fails).

    Snaps the computed value to the nearest common package size (within 3%) to
    avoid fractional artefacts like 751.9 ml that diverge from the 750 ml
    extracted from a matching product's name, causing spurious cluster misses.
    """
    try:
        price = float(price_str)
        ppu   = float(price_per_unit_str)
    except (ValueError, TypeError):
        return None, None
    if ppu <= 0 or price <= 0:
        return None, None
    unit_col = str(unit_col).strip().lower()
    if unit_col == 'kg':
        grams = (price / ppu) * 1000
        if 5 <= grams <= 25000:
            return _snap_to_standard(round(grams, 1), _COMMON_G), 'g'
    elif unit_col in ('l', 'litre', 'liter'):
        ml = (price / ppu) * 1000
        if 10 <= ml <= 25000:
            snapped = _snap_to_standard(round(ml, 1), _COMMON_ML)
            # Some retailers (e.g. ASDA) report ppu in pence/100ml but label
            # the unit column as 'l'.  This yields an inferred value ~10× too
            # small (e.g. 75ml instead of 750ml for a standard wine bottle).
            # Heuristic: the only legitimate grocery liquid in the 50–100ml
            # range is a 75ml mini-bottle, which is vanishingly rare.  If the
            # snapped value sits in that band, try ×10 and prefer the scaled
            # result when it snaps cleanly to a known bottle size.
            if 50 <= snapped <= 100:
                ml_scaled = round(ml * 10, 1)
                snapped_scaled = _snap_to_standard(ml_scaled, _COMMON_ML)
                if snapped_scaled != ml_scaled:  # ×10 landed on a standard size
                    return snapped_scaled, 'ml'
            return snapped, 'ml'
    return None, None
