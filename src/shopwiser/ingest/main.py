"""Scraped catalogue CSV → data/input/raw.csv (the pipeline's input contract).

The live scraper emits one row per product page:

    product_url, source, product_name, product_image_url, product_review_score,
    product_review_count, product_price, product_unit_price, product_promo_price,
    product_promo_unit_price, details_category, brand_name, product_size, is_eligible

whereas the pipeline's normalise step expects

    supermarket, prices_(£), prices_unit_(£), unit, names, date, category, own_brand

This module bridges the two:

* ``source`` → the pipeline's supermarket labels (Asda→ASDA, Sainsbury's→Sains);
* display prices ("£2.15", "87.0p/100g") → floats per base unit (see parsing);
* ASDA titles carry the brand in a separate column — it is prepended to the
  title unless already there, so brand extraction downstream sees one format;
* ``own_brand`` is detected from the (rebuilt) title against the own-label
  prefixes in preprocess.grocery_vocab, plus the retailer name in brand_name;
* every row gets a coarse ``category`` (see categories module);
* rows the scraper marked ``is_eligible == False`` (pet food, homeware,
  vapes, …) are dropped by default — the matcher is a grocery pipeline.

``product_url`` is carried through as a provenance/join key back to the scrape;
normalise passes unknown columns through untouched.
"""

import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from shopwiser.paths import DATA_INPUT, RAW_CSV, RAW_1000_CSV
from shopwiser.preprocess.grocery_vocab import SUPERMARKET_BRANDS

from .categories import classify
from .parsing import parse_price, parse_unit_price

SOURCE_TO_SUPERMARKET = {
    'asda': 'ASDA',
    'tesco': 'Tesco',
    "sainsbury's": 'Sains',
    'sainsburys': 'Sains',
    'morrisons': 'Morrisons',
}

SCRAPE_REQUIRED = ['source', 'product_name', 'product_price', 'product_unit_price']

# Generic retailer tokens that mark an own-label product when they appear at
# the start of the title or anywhere in the scraper's brand_name column.
_RETAILER_TOKENS = {
    'ASDA': ['asda'],
    'Tesco': ['tesco'],
    'Sains': ["sainsbury's", 'sainsburys'],
    'Morrisons': ['morrisons'],
}


def _norm(text: str) -> str:
    """Match preprocess.brand normalisation: &→and, curly quote, single spaces."""
    t = str(text).replace('&', ' and ').replace('’', "'")
    return re.sub(r'\s+', ' ', t).strip().lower()


def _own_label_prefixes(supermarket: str) -> list:
    prefixes = [_norm(p) for p in SUPERMARKET_BRANDS.get(supermarket, [])]
    prefixes += _RETAILER_TOKENS.get(supermarket, [])
    # longest first so "asda extra special" wins over "asda"
    return sorted(set(prefixes), key=len, reverse=True)


def build_name(product_name, brand_name) -> Optional[str]:
    """Prepend the scraper's brand column unless the title already starts with it."""
    name = str(product_name).strip() if isinstance(product_name, str) else ''
    if not name:
        return None
    brand = str(brand_name).strip() if isinstance(brand_name, str) else ''
    if not brand:
        return name
    if _norm(name).startswith(_norm(brand)):
        return name
    return f'{brand} {name}'


def detect_own_brand(name: str, brand_name, supermarket: str, prefixes: list) -> bool:
    norm_name = _norm(name)
    for prefix in prefixes:
        if norm_name.startswith(prefix):
            return True
    if isinstance(brand_name, str):
        norm_brand = _norm(brand_name)
        for token in _RETAILER_TOKENS.get(supermarket, []):
            if token in norm_brand:
                return True
    return False


def ingest(
    input_path: Path,
    output_path: Path,
    *,
    scrape_date: Optional[int] = None,
    keep_ineligible: bool = False,
    write_sample: bool = False,
) -> pd.DataFrame:
    print('=' * 70)
    print('ShopWiser Ingestion: scraped catalogue → raw.csv schema')
    print('=' * 70)
    print(f'  Input:  {input_path}')
    print(f'  Output: {output_path}')

    if not input_path.exists():
        raise FileNotFoundError(f'Scraped CSV not found: {input_path}')

    df = pd.read_csv(input_path, low_memory=False)
    missing = [c for c in SCRAPE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f'Scraped CSV is missing required column(s): {missing}\n'
            f'  Columns found: {list(df.columns)}'
        )
    n_read = len(df)

    unknown = set(df['source'].dropna().unique()) - {
        s for s in df['source'].dropna().unique() if str(s).strip().lower() in SOURCE_TO_SUPERMARKET
    }
    if unknown:
        raise ValueError(
            f'Unknown retailer(s) in source column: {sorted(unknown)}\n'
            f'  Known: {sorted(SOURCE_TO_SUPERMARKET)} — extend SOURCE_TO_SUPERMARKET.'
        )

    if 'is_eligible' in df.columns and not keep_ineligible:
        n_inel = int((~df['is_eligible'].astype(bool)).sum())
        df = df[df['is_eligible'].astype(bool)].reset_index(drop=True)
        print(f'\nDropped {n_inel:,} ineligible rows (scraper flag; pet/household/non-grocery)')

    df = df[df['product_name'].notna() & (df['product_name'].astype(str).str.strip() != '')]
    df = df.reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out['supermarket'] = df['source'].astype(str).str.strip().str.lower().map(SOURCE_TO_SUPERMARKET)

    brand_col = df['brand_name'] if 'brand_name' in df.columns else pd.Series(None, index=df.index)
    out['names'] = [build_name(n, b) for n, b in zip(df['product_name'], brand_col)]

    out['prices_(£)'] = df['product_price'].map(parse_price)
    n_price_fail = int(out['prices_(£)'].isna().sum() - df['product_price'].isna().sum())

    parsed_ppu = df['product_unit_price'].map(parse_unit_price)
    out['prices_unit_(£)'] = [v for v, _ in parsed_ppu]
    out['unit'] = [u for _, u in parsed_ppu]
    n_ppu_in = int(df['product_unit_price'].notna().sum())
    n_ppu_ok = int(out['prices_unit_(£)'].notna().sum())

    out['date'] = scrape_date or int(date.today().strftime('%Y%m%d'))

    cat_col = df['details_category'] if 'details_category' in df.columns else pd.Series(None, index=df.index)
    out['category'] = [classify(c, n) for c, n in zip(cat_col, out['names'])]

    prefix_cache = {sm: _own_label_prefixes(sm) for sm in out['supermarket'].dropna().unique()}
    out['own_brand'] = [
        detect_own_brand(name, brand, sm, prefix_cache.get(sm, []))
        for name, brand, sm in zip(out['names'], brand_col, out['supermarket'])
    ]

    out['product_url'] = df.get('product_url')

    out = out[
        ['supermarket', 'prices_(£)', 'prices_unit_(£)', 'unit', 'names',
         'date', 'category', 'own_brand', 'product_url']
    ]

    n = len(out)
    print(f'\nLoaded {n_read:,} scraped rows → writing {n:,}')
    print('\n=== Ingestion statistics ===')
    print(f'  Price parsed                : {out["prices_(£)"].notna().sum():>7,} / {n:,}'
          f'  ({n_price_fail:,} parse failures)')
    print(f'  Unit price parsed           : {n_ppu_ok:>7,} / {n_ppu_in:,} present'
          f'  ({n_ppu_in - n_ppu_ok:,} unparsed)')
    print(f'  Own-brand flagged           : {out["own_brand"].sum():>7,}  ({out["own_brand"].mean()*100:.1f}%)')
    print('\n  Supermarkets:')
    for sm, cnt in out['supermarket'].value_counts().items():
        print(f'    {sm:<10} {cnt:>7,}')
    print('\n  Category distribution by supermarket (%):')
    dist = (
        out.groupby('supermarket')['category']
        .value_counts(normalize=True)
        .mul(100).round(1).unstack(fill_value=0)
    )
    print(dist.to_string())
    print('\n  Unit column:')
    for ut, cnt in out['unit'].value_counts(dropna=False).items():
        print(f'    {str(ut):<6} {cnt:>7,}  ({cnt/n*100:.1f}%)')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f'\n✓ Saved {n:,} rows × {out.shape[1]} columns → {output_path}')

    if write_sample:
        sample_path = output_path.parent / RAW_1000_CSV
        out.sample(n=min(1000, n), random_state=42).to_csv(sample_path, index=False)
        print(f'✓ Saved 1000-row development sample → {sample_path}')

    return out


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description='Transform scraped catalogue CSV into data/input/raw.csv.')
    p.add_argument('input', type=Path, help='Path to the scraped CSV (e.g. original_data.csv)')
    p.add_argument('--output', type=Path, default=DATA_INPUT / RAW_CSV,
                   help=f'Output path (default: data/input/{RAW_CSV})')
    p.add_argument('--date', type=int, default=None, metavar='YYYYMMDD',
                   help='Scrape date stamped on every row (default: today)')
    p.add_argument('--keep-ineligible', action='store_true',
                   help="Keep rows the scraper flagged is_eligible=False (pet/household/…)")
    p.add_argument('--write-sample', action='store_true',
                   help=f'Also write a seeded 1000-row {RAW_1000_CSV} for --sample runs')
    args = p.parse_args(argv)

    ingest(
        args.input,
        args.output,
        scrape_date=args.date,
        keep_ineligible=args.keep_ineligible,
        write_sample=args.write_sample,
    )


if __name__ == '__main__':
    main()
