"""Supermarket own-label, tier, ABV, and known-brand extraction."""

import re
from typing import Optional, Tuple

from .grocery_vocab import SUPERMARKET_BRANDS, TIER_MAP, KNOWN_BRANDS

# All tier keywords sorted longest-first for trailing-tier detection.
_TIER_KWS_SORTED = sorted(TIER_MAP.keys(), key=len, reverse=True)


def _title_case(s: str) -> str:
    """Title-case that handles apostrophes correctly.

    Python's ``str.title()`` turns ``"I can't believe it"`` into
    ``"I Can'T Believe It"`` because it capitalises any letter after a
    non-letter character, including apostrophes.  This version splits on
    whitespace only.
    """
    return ' '.join(word[0].upper() + word[1:].lower() if word else word for word in s.split(' '))


def _norm_prefix(prefix: str) -> str:
    """Normalise a brand prefix so it matches names that have been pre-processed
    (``&`` → ``and``, curly-apostrophe → straight, trailing dot stripped).
    """
    p = prefix.replace('&', ' and ').replace('\u2019', "'")
    p = re.sub(r'\s+', ' ', p).strip()
    # Strip trailing dot (e.g. "Hearty Food Co." → "Hearty Food Co")
    p = p.rstrip('.')
    return p


def extract_supermarket_brand(name: str, supermarket: str) -> Tuple[Optional[str], str]:
    """Strip a supermarket own-label prefix from *name* and return
    (matched_prefix, remaining_name).

    Fixes vs prior version:
    * Normalise the prefix the same way the name is pre-processed
      (``&``→``and``, curly apostrophe, trailing dot) so that e.g.
      ``"Stockwell & Co"`` matches ``"Stockwell and Co …"``.
    * After prefix stripping, remove any *trailing* tier keyword that
      appears after a comma (Sainsbury's pattern:
      ``"Sainsbury's Camembert, Taste the Difference"`` → core =
      ``"Camembert"``, tier detected from trailing fragment).
    """
    if not isinstance(name, str):
        return None, ''
    # Collapse multi-space runs so "Stockwell  and  Co" (from "Stockwell & Co"
    # after & → ' and ' replacement with surrounding spaces) matches the
    # single-spaced _norm_prefix output.
    name_str = re.sub(r'\s+', ' ', name.strip())
    for prefix in SUPERMARKET_BRANDS.get(supermarket, []):
        norm = _norm_prefix(prefix)
        # Build pattern: use (?:\s|,) instead of [\b\s,] — \b inside a
        # character class is a literal backspace (0x08), not a word boundary.
        pattern = re.compile(r'^' + re.escape(norm) + r'(?:\s|,)?', re.IGNORECASE)
        if pattern.match(name_str):
            remaining = pattern.sub('', name_str).strip().lstrip(',').strip()
            return prefix, remaining
    return None, name_str


def strip_trailing_tier(remaining: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Remove a trailing tier keyword that follows a comma.

    Handles the Sainsbury's format where the tier appears *after* the product
    name: ``"Camembert, Taste the Difference"`` → ``("Camembert", "premium",
    "Taste the Difference")``.

    Returns ``(cleaned_name, tier_type, tier_keyword)``.  If no trailing tier
    is found all three values are unchanged / ``None``.
    """
    if not isinstance(remaining, str) or ',' not in remaining:
        return remaining, None, None

    # Work on the fragment after the last comma
    last_comma = remaining.rfind(',')
    after_comma = remaining[last_comma + 1:].strip()

    for kw in _TIER_KWS_SORTED:
        # The tier keyword must cover most of the post-comma fragment
        # (allow trailing weight tokens like "250g" that weren't stripped yet)
        pat = re.compile(r'^\s*' + re.escape(kw) + r'[\s\w]*$', re.IGNORECASE)
        if pat.match(after_comma):
            cleaned = remaining[:last_comma].strip()
            tier_type = TIER_MAP[kw]
            return cleaned, tier_type, kw.title()

    # Special case: bare "SO" after a comma is Sainsbury's SO Organic shorthand
    # (the "(Colours May Vary)" parenthetical was stripped first, leaving ", SO").
    if re.fullmatch(r'so', after_comma, re.IGNORECASE):
        cleaned = remaining[:last_comma].strip()
        return cleaned, 'dietary', 'SO Organic'

    return remaining, None, None


def extract_abv(name: str) -> Tuple[Optional[float], str]:
    """Extract alcohol-by-volume percentage.

    ``abv`` or ``vol`` marker is now **required** so that non-alcohol
    percentages (``"100% Prune Juice"``, ``"5% Fat Mince"``) are not
    incorrectly captured and stripped from the name.
    """
    if not isinstance(name, str):
        return None, ''
    pattern = r'(\d+(?:\.\d+)?)\s*%\s*(?:abv|vol)\b'
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        abv = float(match.group(1))
        remaining = re.sub(pattern, '', name, flags=re.IGNORECASE)
        remaining = re.sub(r'\s+', ' ', remaining).strip()
        return abv, remaining
    return None, name


def extract_tier(supermarket_brand: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not supermarket_brand:
        return None, None
    brand_lower = str(supermarket_brand).lower()
    for keyword, tier in TIER_MAP.items():
        if keyword in brand_lower:
            return tier, keyword.title()
    return 'standard', None


def extract_known_brand(name: str) -> Tuple[Optional[str], str]:
    """
    Detect a known commercial brand at the START of the product name only.
    Returns (brand_title, name_with_brand_stripped).
    Mid-string brand hits are intentionally ignored to prevent false
    classification of own-brand or unbranded products.
    """
    if not isinstance(name, str):
        return None, ''
    name_str = name.strip()
    sorted_brands = sorted(KNOWN_BRANDS, key=len, reverse=True)
    for brand in sorted_brands:
        pattern = re.compile(r'^' + re.escape(brand) + r'\b', re.IGNORECASE)
        if pattern.match(name_str):
            remaining = pattern.sub('', name_str).strip()
            remaining = re.sub(r'\s+', ' ', remaining)
            return _title_case(brand), remaining
    return None, name_str
