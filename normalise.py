"""
normalise.py
============
Reads data/raw.csv → writes data/normalized_products.csv (26 columns).

Enhanced over normalisation_feature_extraction.ipynb with:
  - Expanded supermarket brand prefixes (COOK by ASDA, BAKE by ASDA,
    The BAKERY at ASDA, Market Street, JS)
  - Pint unit conversion (N pints → N × 568.261 ml)
  - Price-based unit inference (price / price_per_unit for kg/l)
  - mg guard for vape/e-liquid/nicotine products
  - Brand extraction only at string start (removes false mid-string brand hits
    that bleed brand into core_product_name)
  - is_truncated flag for Morrisons "…" names
  - unit_inferred flag for price-derived units
  - Expanded KNOWN_BRANDS list (~30 additions)
  - Tea/bags count extraction (e.g. "80s" or "x20")
"""

import pandas as pd
import numpy as np
import re
import unicodedata
import warnings
from typing import Tuple, List, Dict, Optional

# Import variables from our new config file
from grocery_vocab import (
    SUPERMARKET_BRANDS, TIER_MAP, ATTRIBUTES, DESCRIPTORS, 
    KNOWN_BRANDS, _VAPE_KEYWORDS
)

warnings.filterwarnings('ignore')

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_accents(text: str) -> str:
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def clean_parenthetical_notes(name) -> str:
    if not isinstance(name, str):
        return ''
    name = re.sub(r'\([^)]{30,}\)', '', name)
    name = re.sub(r'\(order by[^)]*\)', '', name, flags=re.IGNORECASE)
    return name.strip()


# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

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
    # Matches "4.5% ABV", "5%", "4.5 % vol"
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
    # Has a supermarket brand but no tier keyword → standard
    return 'standard', None


def extract_multipack(name: str) -> Tuple[Optional[int], Optional[str], str]:
    if not isinstance(name, str):
        return None, None, ''
    name_str = name.strip()
    patterns = [
        # "6 x 330ml" or "10x330ml" — keep unit, strip the NxUNIT count part
        (r'(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|cl|l\b)', 'count_x_size'),
        # " x6" or " x 12"
        (r'\s[xX]\s*(\d+)\b', 'x_count'),
        # "6 pack" / "12pack"
        (r'(\d+)\s*pack\b', 'count_pack'),
        # "pack of 6"
        (r'pack\s*of\s*(\d+)', 'pack_of_count'),
        # "6pk"
        (r'(\d+)\s*pk\b', 'count_pk'),
        # "multipack 6"
        (r'multipack\s*(\d+)', 'multipack'),
        # Tea/sachets count: "80s" / "20 bags" at end of name or before unit
        (r'\b(\d+)\s*(?:bags?|sachets?|tabs?|pods?|capsules?)\b', 'bag_count'),
        # "6 Pcs" / "12 pcs"
        (r'(\d+)\s*pcs?\b', 'count_pcs'),
    ]
    for pattern, pattern_type in patterns:
        match = re.search(pattern, name_str, re.IGNORECASE)
        if match:
            quantity = int(match.group(1))
            if pattern_type == 'count_x_size':
                unit_part = match.group(2) + match.group(3)
                remaining = name_str[:match.start()] + ' ' + unit_part + ' ' + name_str[match.end():]
            else:
                remaining = name_str[:match.start()] + name_str[match.end():]
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return quantity, pattern_type, remaining
    # "80s" pattern: digits followed immediately by 's' at word boundary (e.g. "Tea Bags 80s")
    bag_s = re.search(r'\b(\d{1,3})s\b', name_str, re.IGNORECASE)
    if bag_s:
        qty = int(bag_s.group(1))
        if 2 <= qty <= 500:
            remaining = name_str[:bag_s.start()] + name_str[bag_s.end():]
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return qty, 'count_s', remaining
    return None, None, name_str


def _is_vape_name(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in _VAPE_KEYWORDS)


def extract_and_standardize_unit(name: str) -> Tuple[Optional[float], Optional[str], str]:
    if not isinstance(name, str):
        return None, None, ''
    name_str = name.strip()

    unit_patterns = [
        # Parenthesised units first
        (r'\((\d+(?:\.\d+)?)\s*kg\)',  1000,     'g'),
        (r'\((\d+(?:\.\d+)?)\s*g\)',   1,        'g'),
        (r'\((\d+(?:\.\d+)?)\s*l(?:tr|itre|iter)?\)',  1000, 'ml'),
        (r'\((\d+(?:\.\d+)?)\s*cl\)',  10,       'ml'),
        (r'\((\d+(?:\.\d+)?)\s*ml\)',  1,        'ml'),
        # UK pints (BEFORE generic litre to avoid conflicts)
        (r'(\d+(?:\.\d+)?)\s*pints?\b', 568.261, 'ml'),
        # Weight
        (r'(\d+(?:\.\d+)?)\s*kg\b',   1000,     'g'),
        (r'(\d+(?:\.\d+)?)\s*g\b',    1,        'g'),
        # Volume
        (r'(\d+(?:\.\d+)?)\s*l(?:tr|itre|iter)?\b', 1000, 'ml'),
        (r'(\d+(?:\.\d+)?)\s*cl\b',   10,       'ml'),
        (r'(\d+(?:\.\d+)?)\s*ml\b',   1,        'ml'),
    ]

    is_vape = _is_vape_name(name_str)

    for pattern, factor, base_unit in unit_patterns:
        # Skip mg→g for vape/nicotine products
        if base_unit == 'g' and factor == 0.001 and is_vape:
            continue
        match = re.search(pattern, name_str, re.IGNORECASE)
        if match:
            value = float(match.group(1)) * factor
            remaining = re.sub(pattern, '', name_str, flags=re.IGNORECASE)
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return round(value, 3), base_unit, remaining

    # mg for non-vape (e.g. micronutrients in fortified food) — keep as distinct type
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


def extract_attributes(name: str) -> Tuple[List[str], List[str], str]:
    if not isinstance(name, str):
        return [], [], ''
    name_str = name.strip()
    found_types, found_keywords = [], []
    remaining = name_str
    for attr_type, keywords in ATTRIBUTES.items():
        for kw in sorted(keywords, key=len, reverse=True):
            pat = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            if pat.search(remaining):
                found_types.append(attr_type)
                found_keywords.append(kw)
                remaining = pat.sub('', remaining)
                remaining = re.sub(r'\s+', ' ', remaining).strip()
                break
    return found_types, found_keywords, remaining


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


def extract_descriptors(name: str) -> Dict[str, List[str]]:
    if not isinstance(name, str):
        return {}
    name_str = name.strip()
    found = {}
    for desc_type, keywords in DESCRIPTORS.items():
        hits = []
        for kw in sorted(keywords, key=len, reverse=True):
            if re.search(r'\b' + re.escape(kw) + r'\b', name_str, re.IGNORECASE):
                hits.append(kw)
        if hits:
            found[desc_type] = hits
    return found


# ============================================================
# FULL PIPELINE PER ROW
# ============================================================

def normalize_product_name(row: pd.Series) -> dict:
    original_name = row['names']
    supermarket   = str(row.get('supermarket', '')).strip()
    price_val     = row.get('prices_(£)')
    price_per_unit= row.get('prices_unit_(£)')
    unit_col      = str(row.get('unit', '')).strip().lower()

    result = {
        'original_name':        original_name,
        'supermarket_brand':    None,
        'tier_type':            None,
        'tier_keyword':         None,
        'known_brand':          None,
        'pack_quantity':        None,
        'pack_pattern':         None,
        'unit_value':           None,
        'unit_type':            None,
        'abv_percentage':       None,
        'unit_inferred':        False,
        'attributes_types':     [],
        'attributes_keywords':  [],
        'descriptors':          {},
        'core_product_name':    '',
        'normalized_name':      '',
    }

    # Pre-processing
    current = clean_parenthetical_notes(original_name)

    # Standardize ampersands to 'and' and normalize smart quotes
    current = current.replace('&', ' and ').replace('’', "'")

    # Step 1: Supermarket brand extraction
    sm_brand, current = extract_supermarket_brand(current, supermarket)
    result['supermarket_brand'] = sm_brand

    # Step 2: Tier from supermarket brand
    tier_type, tier_kw = extract_tier(sm_brand)
    result['tier_type']    = tier_type
    result['tier_keyword'] = tier_kw

    # Step 3: Multipack extraction
    pack_qty, pack_pat, current = extract_multipack(current)
    result['pack_quantity'] = pack_qty
    result['pack_pattern']  = pack_pat

    # Step 4: Unit extraction from name
    unit_val, unit_type, current = extract_and_standardize_unit(current)
    result['unit_value'] = unit_val
    result['unit_type']  = unit_type

    # Step 4.5: Extract Alcohol %
    abv_val, current = extract_abv(current)
    result['abv_percentage'] = abv_val

    # Step 4b: Fallback — infer unit from price if name extraction failed
    if unit_val is None:
        inferred_val, inferred_type = infer_unit_from_price(price_val, price_per_unit, unit_col)
        if inferred_val is not None:
            result['unit_value']   = inferred_val
            result['unit_type']    = inferred_type
            result['unit_inferred'] = True

    # Step 5: Attribute extraction (removes matched terms from current)
    attr_types, attr_kws, current = extract_attributes(current)
    result['attributes_types']    = attr_types
    result['attributes_keywords'] = attr_kws

    # Step 6: Known brand detection (start-of-string only; strips brand from name)
    known_brand, current = extract_known_brand(current)
    result['known_brand'] = known_brand

    # Step 7: Descriptors (detected but NOT removed)
    result['descriptors'] = extract_descriptors(current)

    # Step 8: Core product name — final cleaned string
    core = re.sub(r'\s+', ' ', current).strip()
    core = re.sub(r'^[\s\-,\.]+|[\s\-,\.]+$', '', core)
    core = re.sub(r'\(\s*\)|\[\s*\]', '', core)
    core = re.sub(r'\s*\*\s*$', '', core).strip()
    result['core_product_name'] = core

    # Step 9: Normalized name
    normalized = normalize_accents(core).lower()
    normalized = normalized.replace("'", ' ')
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    result['normalized_name'] = normalized

    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    INPUT_PATH  = Path('data/raw.csv')
    OUTPUT_PATH = Path('data/normalized_products.csv')

    print('=' * 70)
    print('ShopWiser Normalisation Pipeline')
    print('=' * 70)

    # ---- Load ----
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    initial_count = len(df)
    df = df[df['names'].notna() & (df['names'].str.strip() != '')].reset_index(drop=True)
    print(f'\nLoaded {initial_count:,} rows, removed {initial_count - len(df):,} empty names')
    print(f'Processing {len(df):,} products...')

    # ---- Truncation flag (before normalization alters names) ----
    df['is_truncated'] = df['names'].str.contains(r'[…]|\.{3}', regex=True, na=False)
    n_trunc = df['is_truncated'].sum()
    print(f'Flagged {n_trunc:,} truncated names ({n_trunc/len(df)*100:.1f}%)')

    # ---- Apply normalization ----
    try:
        from tqdm import tqdm
        tqdm.pandas(desc='Normalising')
        normalized_results = df.progress_apply(normalize_product_name, axis=1)
    except ImportError:
        print('(tqdm not available — running without progress bar)')
        normalized_results = df.apply(normalize_product_name, axis=1)

    normalized_df = pd.DataFrame(normalized_results.tolist())

    # ---- Combine ----
    df_out = pd.concat([df.reset_index(drop=True), normalized_df], axis=1)

    # ---- Stats ----
    n = len(df_out)
    print(f'\n=== Extraction Statistics ===')
    print(f'  Supermarket brand extracted : {df_out["supermarket_brand"].notna().sum():>7,}  ({df_out["supermarket_brand"].notna().mean()*100:.1f}%)')
    print(f'  Tier extracted              : {df_out["tier_keyword"].notna().sum():>7,}  ({df_out["tier_keyword"].notna().mean()*100:.1f}%)')
    print(f'  Known brand detected        : {df_out["known_brand"].notna().sum():>7,}  ({df_out["known_brand"].notna().mean()*100:.1f}%)')
    print(f'  Pack quantity extracted     : {df_out["pack_quantity"].notna().sum():>7,}  ({df_out["pack_quantity"].notna().mean()*100:.1f}%)')
    unit_any  = df_out["unit_value"].notna().sum()
    unit_name = (df_out["unit_value"].notna() & ~df_out["unit_inferred"]).sum()
    unit_inf  = df_out["unit_inferred"].sum()
    print(f'  Unit extracted (name)       : {unit_name:>7,}  ({unit_name/n*100:.1f}%)')
    print(f'  Unit inferred (price)       : {unit_inf:>7,}  ({unit_inf/n*100:.1f}%)')
    print(f'  Unit total                  : {unit_any:>7,}  ({unit_any/n*100:.1f}%)')
    print(f'  No unit at all              : {n - unit_any:>7,}  ({(n-unit_any)/n*100:.1f}%)')
    attr_n = df_out['attributes_keywords'].apply(lambda x: len(x) if isinstance(x, list) else 0).gt(0).sum()
    desc_n = df_out['descriptors'].apply(lambda x: len(x) if isinstance(x, dict) else 0).gt(0).sum()
    print(f'  Attributes extracted        : {attr_n:>7,}  ({attr_n/n*100:.1f}%)')
    print(f'  Descriptors detected        : {desc_n:>7,}  ({desc_n/n*100:.1f}%)')
    print(f'  Truncated names flagged     : {n_trunc:>7,}  ({n_trunc/n*100:.1f}%)')

    print(f'\n  Unit type distribution:')
    for ut, cnt in df_out['unit_type'].value_counts().items():
        print(f'    {str(ut):<6} {cnt:>7,}  ({cnt/n*100:.1f}%)')

    # ---- Export ----
    df_export = df_out.copy()
    df_export['attributes_types']    = df_export['attributes_types'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    df_export['attributes_keywords'] = df_export['attributes_keywords'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    df_export['descriptors']         = df_export['descriptors'].apply(lambda x: str(x) if isinstance(x, dict) and x else '')

    df_export.to_csv(OUTPUT_PATH, index=False)
    print(f'\n✓ Saved {len(df_export):,} rows × {df_export.shape[1]} columns → {OUTPUT_PATH}')
    print(f'  Columns: {list(df_export.columns)}')
