"""Price and unit-price string parsers for scraped retailer data.

Scrapers emit display strings, not numbers:

    product_price       "£2.15", "£1,099.95"
    product_unit_price  "£14.93/KG", "£1.08/litre", "£3.90 / ltr",
                        "77.0p/LT", "23.8p/EA", "87.0p/100g", "£10.67/75cl"

The pipeline contract (``data/input/raw.csv``) wants ``prices_(£)`` as a float
and ``prices_unit_(£)`` as a float per *base* unit with the base recorded in
``unit`` ('kg', 'l', 'unit', 'm') — so "87p/100g" must become 8.70 per 'kg'.
"""

import re
from typing import Optional, Tuple

_PRICE_RE = re.compile(r'^\s*£\s*([\d,]+(?:\.\d+)?)\s*$')

# numerator: "£1.08" / "£ 1,099.95" / "77.0p" / ".5p"; denominator: token after
# "/". An optional trailing annotation ("DR.WT" = drained weight, Tesco) is
# tolerated and ignored — it qualifies the weight basis, not the unit itself.
_UNIT_PRICE_RE = re.compile(
    r'^\s*(?:£\s*(?P<pounds>[\d,]*\.?\d+)|(?P<pence>[\d,]*\.?\d+)\s*p)'
    r'\s*/\s*(?P<denom>[A-Za-z0-9]+)'
    r'(?:\s+[A-Za-z.]+)?\s*$',
    re.IGNORECASE,
)

# Fixed-word denominators → (base unit, multiplier to price-per-base-unit).
_DENOM_FIXED = {
    'kg':    ('kg',   1.0),
    'g':     ('kg',   1000.0),
    'l':     ('l',    1.0),
    'litre': ('l',    1.0),
    'liter': ('l',    1.0),
    'ltr':   ('l',    1.0),
    'lt':    ('l',    1.0),
    'ml':    ('l',    1000.0),
    'cl':    ('l',    100.0),
    'ea':    ('unit', 1.0),
    'each':  ('unit', 1.0),
    'unit':  ('unit', 1.0),
    'pair':  ('unit', 1.0),
    'sqm':   ('unit', 1.0),
    'm':     ('m',    1.0),
    'mtr':   ('m',    1.0),
    'metre': ('m',    1.0),
    'meter': ('m',    1.0),
}

# Quantified denominators like "100g", "75cl", "100ml", "10g", "100gr".
_DENOM_QTY_RE = re.compile(r'^(?P<qty>\d+(?:\.\d+)?)(?P<u>gr|g|kg|ml|cl|l)$', re.IGNORECASE)
_QTY_TO_BASE = {
    'g':  ('kg', 1000.0),
    'gr': ('kg', 1000.0),
    'kg': ('kg', 1.0),
    'ml': ('l',  1000.0),
    'cl': ('l',  100.0),
    'l':  ('l',  1.0),
}


def parse_price(raw) -> Optional[float]:
    """'£2.15' / '£1,099.95' → 2.15 / 1099.95; anything else → None."""
    if not isinstance(raw, str):
        return None
    m = _PRICE_RE.match(raw)
    if not m:
        return None
    return float(m.group(1).replace(',', ''))


def parse_unit_price(raw) -> Tuple[Optional[float], Optional[str]]:
    """'87.0p/100g' → (8.70, 'kg'); unparseable → (None, None)."""
    if not isinstance(raw, str):
        return None, None
    m = _UNIT_PRICE_RE.match(raw)
    if not m:
        return None, None
    if m.group('pounds') is not None:
        value = float(m.group('pounds').replace(',', ''))
    else:
        value = float(m.group('pence').replace(',', '')) / 100.0

    denom = m.group('denom').lower()
    if denom in _DENOM_FIXED:
        base, factor = _DENOM_FIXED[denom]
        return round(value * factor, 4), base

    qm = _DENOM_QTY_RE.match(denom)
    if qm:
        qty = float(qm.group('qty'))
        if qty <= 0:
            return None, None
        base, per_base = _QTY_TO_BASE[qm.group('u').lower()]
        return round(value * per_base / qty, 4), base

    return None, None
