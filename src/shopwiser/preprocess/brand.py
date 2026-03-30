"""Supermarket own-label, tier, ABV, and known-brand extraction."""

import re
from typing import Optional, Tuple

from .grocery_vocab import SUPERMARKET_BRANDS, TIER_MAP, KNOWN_BRANDS


def extract_supermarket_brand(name: str, supermarket: str) -> Tuple[Optional[str], str]:
    if not isinstance(name, str):
        return None, ''
    name_str = name.strip()
    for prefix in SUPERMARKET_BRANDS.get(supermarket, []):
        pattern = re.compile(r'^' + re.escape(prefix) + r'\b', re.IGNORECASE)
        if pattern.match(name_str):
            remaining = pattern.sub('', name_str).strip()
            return prefix, remaining
    return None, name_str


def extract_abv(name: str) -> Tuple[Optional[float], str]:
    if not isinstance(name, str):
        return None, ''
    pattern = r'(\d+(?:\.\d+)?)\s*%\s*(?:abv|vol)?\b'
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
            return brand.title(), remaining
    return None, name_str
