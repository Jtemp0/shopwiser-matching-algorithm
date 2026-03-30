"""
Per-row normalisation pipeline and CSV driver.

Reads data/raw/raw.csv → writes data/processed/normalized_products.csv (26 columns).

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

from .attributes import extract_attributes, extract_descriptors
from .brand import extract_abv, extract_known_brand, extract_supermarket_brand, extract_tier
from .cleaning import clean_parenthetical_notes, normalize_accents
from .units import extract_and_standardize_unit, extract_multipack, infer_unit_from_price

warnings.filterwarnings('ignore')


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

    current = clean_parenthetical_notes(original_name)
    current = current.replace('&', ' and ').replace('’', "'")

    sm_brand, current = extract_supermarket_brand(current, supermarket)
    result['supermarket_brand'] = sm_brand

    tier_type, tier_kw = extract_tier(sm_brand)
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

    normalized = normalize_accents(core).lower()
    normalized = normalized.replace("'", ' ')
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    result['normalized_name'] = normalized

    return result


def main(*, sample: bool = False) -> None:
    """Run CSV normalisation. ``sample=True`` uses ``data/raw/raw_1000.csv`` and writes ``normalized_products_sample.csv``."""
    INPUT_PATH = raw_csv_path(sample=sample)
    OUTPUT_PATH = normalized_products_path(sample=sample)

    print('=' * 70)
    print('ShopWiser Normalisation Pipeline')
    print('=' * 70)
    print(f'  Mode:   {"sample (~1000 rows)" if sample else "full dataset"}')
    print(f'  Input:  {INPUT_PATH}')
    print(f'  Output: {OUTPUT_PATH}')

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    initial_count = len(df)
    df = df[df['names'].notna() & (df['names'].str.strip() != '')].reset_index(drop=True)
    print(f'\nLoaded {initial_count:,} rows, removed {initial_count - len(df):,} empty names')
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
    print(f'\n=== Extraction Statistics ===')
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

    print(f'\n  Unit type distribution:')
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
        help='Use data/raw/raw_1000.csv and write data/processed/normalized_products_sample.csv',
    )
    _args = _p.parse_args()
    main(sample=_args.sample)
