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
        (r'\b(\d+)\s+(?:portions?|sticks?|pieces?|slices?)\b', 'bag_count'),
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


def infer_unit_from_price(price_str, price_per_unit_str, unit_col: str) -> Tuple[Optional[float], Optional[str]]:
    """Derive product size from price ÷ price-per-unit (used as fallback when name extraction fails)."""
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
            return round(grams, 1), 'g'
    elif unit_col in ('l', 'litre', 'liter'):
        ml = (price / ppu) * 1000
        if 10 <= ml <= 25000:
            return round(ml, 1), 'ml'
    return None, None
