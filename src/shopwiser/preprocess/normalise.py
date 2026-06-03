"""
Per-row normalisation pipeline and CSV driver.

Reads data/input/raw.csv → writes data/input/normalized_products.csv (26 columns).

Split across ``cleaning`` (accents, parentheticals), ``brand`` (own-label, tier, ABV,
known brands), ``units`` (multipack, g/ml, price inference), and ``attributes``
(keywords + descriptors). Same behaviour as the prior monolithic module.

Enhancements (unchanged):
  Expanded supermarket brand prefixes; pint→ml; price-based unit inference;
  mg guard for vape products; brand only at string start; truncation flag;
  tea/bags count patterns; expanded KNOWN_BRANDS.
"""

import re
import warnings

import pandas as pd

from shopwiser.paths import normalized_products_path, raw_csv_path

# Placeholder written into the `category` column when the input has none.
# Keep in sync with the same constant in rule_matcher.data_prep.
DEFAULT_CATEGORY = 'uncategorised'

# ── Non-food / non-drink keyword exclusion ────────────────────────────────────
# Products whose names contain any of these tokens (whole-word, case-insensitive)
# are excluded from the output — they are non-food items miscategorised by
# retailer scrapers (tobacco, vaping hardware, kitchenware, candles, etc.).
_NON_FOOD_PATTERN = re.compile(
    r'\b(?:'
    # Tobacco & smoking
    r'cigarette|tobacco|cigars?|rolling\s+tobacco|pipe\s+tobacco'
    r'|cigarette\s+papers?|filter\s+tips?|rolling\s+paper'
    # Vaping hardware (not nicotine pouches which are borderline)
    r'|e-?liquid|vape\s+kit|vaping\s+device|starter\s+kit'
    r'|disposable\s+vape|vape\s+pen|vape\s+bar'
    # Kitchenware / housewares
    r'|whisk|spatula|colander|peeler|grater|strainer|ladle'
    r'|chopping\s+board|cutting\s+board|mixing\s+bowl|baking\s+tray'
    r'|cake\s+tin|baking\s+sheet|oven\s+glove|kitchen\s+towel'
    # Non-food containers & accessories
    r'|hot\s+water\s+bottle|reusable\s+bottle|water\s+bottle'
    r'|flask|thermos|travel\s+mug'
    # Candles & party items
    r'|birthday\s+candles?|candle\s+holder|wax\s+melts?'
    r'|party\s+bag|party\s+supplies'
    r')s?\b',
    re.IGNORECASE,
)

from .attributes import extract_attributes, extract_descriptors
from .brand import (
    extract_abv, extract_known_brand, extract_supermarket_brand,
    extract_tier, strip_trailing_tier,
)
from .cleaning import clean_parenthetical_notes, normalize_accents
from .units import extract_and_standardize_unit, extract_multipack, infer_unit_from_price

warnings.filterwarnings('ignore')


_VAPE_CATEGORY_PATTERN = re.compile(
    r'\b(?:vap(?:e|ing|our)|e-?cig|nicotine|tobacco|hookah|shisha)\b',
    re.IGNORECASE,
)


def normalize_product_name(row: pd.Series) -> dict:
    original_name = row['names']
    supermarket   = str(row.get('supermarket', '')).strip()
    price_val     = row.get('prices_(£)')
    price_per_unit= row.get('prices_unit_(£)')
    unit_col      = str(row.get('unit', '')).strip().lower()
    category_col  = str(row.get('category', '') or '').strip()

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

    current = clean_parenthetical_notes(original_name)
    current = current.replace('&', ' and ').replace('’', "'")

    sm_brand, current = extract_supermarket_brand(current, supermarket)
    result['supermarket_brand'] = sm_brand

    tier_type, tier_kw = extract_tier(sm_brand)

    # Strip trailing tier keyword (e.g. Sainsbury's "Camembert, Taste the Difference")
    # and promote it to tier if no tier was found from the prefix alone.
    current, trailing_tier_type, trailing_tier_kw = strip_trailing_tier(current)
    if trailing_tier_type and tier_type in (None, 'standard'):
        tier_type = trailing_tier_type
        tier_kw   = trailing_tier_kw

    result['tier_type']    = tier_type
    result['tier_keyword'] = tier_kw

    pack_qty, pack_pat, current, pack_unit_val, pack_unit_type = extract_multipack(current)
    result['pack_quantity'] = pack_qty
    result['pack_pattern']  = pack_pat

    unit_val, unit_type, current = extract_and_standardize_unit(current)
    result['unit_value'] = unit_val
    result['unit_type']  = unit_type

    if unit_val is None and pack_unit_val is not None and pack_qty:
        result['unit_value'] = round(pack_unit_val * pack_qty, 3)
        result['unit_type']  = pack_unit_type
        unit_val  = result['unit_value']
        unit_type = result['unit_type']

    abv_val, current = extract_abv(current)
    result['abv_percentage'] = abv_val

    # Guard: nullify mg-derived unit values for vape/tobacco products whose brand
    # keyword wasn't caught by _is_vape_name().  mg→g conversion yields tiny values
    # like 0.02 g (= 20 mg nicotine strength) which are meaningless for clustering.
    # Trigger: unit came from mg conversion (value < 0.1 g) AND the product category
    # column (or unit column) signals a non-food context.
    if (unit_type == 'g' and unit_val is not None and unit_val < 0.1
            and (_VAPE_CATEGORY_PATTERN.search(category_col)
                 or _VAPE_CATEGORY_PATTERN.search(unit_col))):
        result['unit_value'] = None
        result['unit_type']  = None
        unit_val  = None
        unit_type = None

    if unit_val is None:
        inferred_val, inferred_type = infer_unit_from_price(price_val, price_per_unit, unit_col)
        if inferred_val is not None:
            result['unit_value']   = inferred_val
            result['unit_type']    = inferred_type
            result['unit_inferred'] = True

    attr_types, attr_kws, current = extract_attributes(current)
    result['attributes_types']    = attr_types
    result['attributes_keywords'] = attr_kws

    known_brand, current = extract_known_brand(current)
    result['known_brand'] = known_brand

    result['descriptors'] = extract_descriptors(current)

    core = re.sub(r'\s+', ' ', current).strip()
    core = re.sub(r'^[\s\-,\.]+|[\s\-,\.]+$', '', core)
    core = re.sub(r'\(\s*\)|\[\s*\]', '', core)
    core = re.sub(r'\s*\*\s*$', '', core).strip()
    result['core_product_name'] = core

    # For branded products, prepend known_brand so that e.g. "Fanta Orange"
    # normalises to "fanta orange" rather than just "orange".  Without this,
    # single-flavor cores like "orange", "lemon", or "sugar" from different
    # brands appear identical and cause false-positive clustering.
    known_brand = result['known_brand']
    name_for_norm = f"{known_brand} {core}".strip() if known_brand else core
    normalized = normalize_accents(name_for_norm).lower()
    normalized = normalized.replace("'", '')
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    result['normalized_name'] = normalized

    return result


def main(*, sample: bool = False) -> None:
    """Run CSV normalisation. ``sample=True`` uses ``data/input/raw_1000.csv`` and writes ``normalized_products_sample.csv``."""
    INPUT_PATH = raw_csv_path(sample=sample)
    OUTPUT_PATH = normalized_products_path(sample=sample)

    print('=' * 70)
    print('ShopWiser Normalisation Pipeline')
    print('=' * 70)
    print(f'  Mode:   {"sample (~1000 rows)" if sample else "full dataset"}')
    print(f'  Input:  {INPUT_PATH}')
    print(f'  Output: {OUTPUT_PATH}')

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            f"Place the scraped catalogue CSV at that path, or pass --sample to use "
            f"the bundled ~1000-row sample (data/input/raw_1000.csv)."
        )

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # Validate the input schema up-front so a mis-named column from a new
    # scraper fails fast with a clear message instead of a deep KeyError.
    required = ['supermarket', 'names', 'prices_(£)', 'unit', 'own_brand']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}\n"
            f"  Expected schema : {required}\n"
            f"  Columns found   : {list(df.columns)}\n"
            f"If your scraper uses different column names, rename them to match before "
            f"running (see README → 'Adapting to new scraped data')."
        )

    # `category` is optional. It drives blocking buckets, the frozen/fresh gate,
    # and one ML ranker feature, but the pipeline runs without it — products are
    # then bucketed by brand/tier/size/name instead. When absent we materialise a
    # constant placeholder so every downstream consumer sees a uniform value
    # (no special-casing needed elsewhere). Expect a small precision reduction on
    # cross-category look-alikes (e.g. tinned soup vs fresh, frozen vs chilled).
    if 'category' not in df.columns:
        df['category'] = DEFAULT_CATEGORY
        print(
            f"\n⚠ No 'category' column in input — running without category gates "
            f"(placeholder '{DEFAULT_CATEGORY}'). Matching still works; expect a "
            f"slightly higher cross-category false-positive rate."
        )

    initial_count = len(df)
    df = df[df['names'].notna() & (df['names'].str.strip() != '')].reset_index(drop=True)
    print(f'\nLoaded {initial_count:,} rows, removed {initial_count - len(df):,} empty names')

    # Filter out non-food / non-drink items
    non_food_mask = df['names'].str.contains(_NON_FOOD_PATTERN, na=False)
    n_non_food = non_food_mask.sum()
    if n_non_food:
        print(f'Excluded {n_non_food:,} non-food/non-drink items')
        df = df[~non_food_mask].reset_index(drop=True)

    print(f'Processing {len(df):,} products...')

    df['is_truncated'] = df['names'].str.contains(r'[…]|\.{3}', regex=True, na=False)
    n_trunc = df['is_truncated'].sum()
    print(f'Flagged {n_trunc:,} truncated names ({n_trunc/len(df)*100:.1f}%)')

    try:
        from tqdm import tqdm
        tqdm.pandas(desc='Normalising')
        normalized_results = df.progress_apply(normalize_product_name, axis=1)
    except ImportError:
        print('(tqdm not available — running without progress bar)')
        normalized_results = df.apply(normalize_product_name, axis=1)

    normalized_df = pd.DataFrame(normalized_results.tolist())
    df_out = pd.concat([df.reset_index(drop=True), normalized_df], axis=1)

    n = len(df_out)
    print('\n=== Extraction Statistics ===')
    print(f'  Supermarket brand extracted : {df_out["supermarket_brand"].notna().sum():>7,}  ({df_out["supermarket_brand"].notna().mean()*100:.1f}%)')
    print(f'  Tier extracted              : {df_out["tier_keyword"].notna().sum():>7,}  ({df_out["tier_keyword"].notna().mean()*100:.1f}%)')
    print(f'  Known brand detected        : {df_out["known_brand"].notna().sum():>7,}  ({df_out["known_brand"].notna().mean()*100:.1f}%)')
    print(f'  Pack quantity extracted     : {df_out["pack_quantity"].notna().sum():>7,}  ({df_out["pack_quantity"].notna().mean()*100:.1f}%)')
    unit_any  = df_out["unit_value"].notna().sum()
    unit_name = (df_out["unit_value"].notna() & ~df_out["unit_inferred"]).sum()
    unit_inf  = df_out["unit_inferred"].sum()
    print(f'  Unit extracted (name)       : {unit_name:>7,}  ({unit_name/n*100:.1f}%)')
    print(f'  Unit inferred (price)      : {unit_inf:>7,}  ({unit_inf/n*100:.1f}%)')
    print(f'  Unit total                  : {unit_any:>7,}  ({unit_any/n*100:.1f}%)')
    print(f'  No unit at all              : {n - unit_any:>7,}  ({(n-unit_any)/n*100:.1f}%)')
    attr_n = df_out['attributes_keywords'].apply(lambda x: len(x) if isinstance(x, list) else 0).gt(0).sum()
    desc_n = df_out['descriptors'].apply(lambda x: len(x) if isinstance(x, dict) else 0).gt(0).sum()
    print(f'  Attributes extracted        : {attr_n:>7,}  ({attr_n/n*100:.1f}%)')
    print(f'  Descriptors detected        : {desc_n:>7,}  ({desc_n/n*100:.1f}%)')
    print(f'  Truncated names flagged     : {n_trunc:>7,}  ({n_trunc/n*100:.1f}%)')

    print('\n  Unit type distribution:')
    for ut, cnt in df_out['unit_type'].value_counts().items():
        print(f'    {str(ut):<6} {cnt:>7,}  ({cnt/n*100:.1f}%)')

    df_export = df_out.copy()
    df_export['attributes_types']    = df_export['attributes_types'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    df_export['attributes_keywords'] = df_export['attributes_keywords'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    df_export['descriptors']         = df_export['descriptors'].apply(lambda x: str(x) if isinstance(x, dict) and x else '')

    df_export.to_csv(OUTPUT_PATH, index=False)
    print(f'\n✓ Saved {len(df_export):,} rows × {df_export.shape[1]} columns → {OUTPUT_PATH}')
    print(f'  Columns: {list(df_export.columns)}')


if __name__ == '__main__':
    import argparse

    _p = argparse.ArgumentParser(description='Normalise raw product CSV → processed features.')
    _p.add_argument(
        '--sample',
        action='store_true',
        help='Use data/input/raw_1000.csv and write data/input/normalized_products_sample.csv',
    )
    _args = _p.parse_args()
    main(sample=_args.sample)
