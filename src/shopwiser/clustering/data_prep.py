"""Load normalized CSV and derive columns used by clustering passes."""

import re

import numpy as np
import pandas as pd

from shopwiser.paths import normalized_products_path

from .config import BRAND_EXCLUSIONS, CATEGORY_ALIASES

# v8: normalise synonym spellings in normalized_name so that fuzzy matching
#     can score them well (e.g. "decaffeinated" and "decaff" are the same thing).
_NAME_SYNONYM_RE = [
    # Coffee
    (re.compile(r'\bdecaffeinated\b'),          'decaff'),
    (re.compile(r'\bdecafinated\b'),            'decaff'),   # common misspelling
    # Grains / milk
    (re.compile(r'\bwhole[\s\-]+grain\b'),      'wholegrain'),
    (re.compile(r'\bsemi[\s\-]+skimmed\b'),     'semiskimmed'),
    # Yogurt spelling (UK vs US)
    (re.compile(r'\byoghurt\b'),                'yogurt'),
    (re.compile(r'\byoghurts\b'),               'yogurts'),
    # Flavour/fiber spelling (UK vs US)
    (re.compile(r'\bflavour\b'),                'flavor'),
    (re.compile(r'\bflavours\b'),               'flavors'),
    (re.compile(r'\bflavoured\b'),              'flavored'),
    (re.compile(r'\bfibre\b'),                  'fiber'),
    # Beanz vs Beans (Heinz specific, but safe broadly)
    (re.compile(r'\bbeanz\b'),                  'beans'),
    # UK/US colour spelling — "food colouring" vs "food coloring" across SMs
    (re.compile(r'\bcolouring\b'),              'coloring'),
    (re.compile(r'\bcolourings\b'),             'colorings'),
    (re.compile(r'\bcolour\b'),                 'color'),
    (re.compile(r'\bcolours\b'),                'colors'),
]


def apply_name_synonyms(s: str) -> str:
    if not isinstance(s, str):
        return s
    for pat, repl in _NAME_SYNONYM_RE:
        s = pat.sub(repl, s)
    return s


def load_prepared_dataframe(*, sample: bool = False) -> pd.DataFrame:
    path = normalized_products_path(sample=sample)
    df = pd.read_csv(path, low_memory=False)
    print(f'\nLoaded {len(df):,} products from {path.name}, {df.shape[1]} columns')

    REQUIRED_COLS = [
        'supermarket', 'names', 'category', 'own_brand',
        'supermarket_brand', 'tier_type', 'known_brand',
        'pack_quantity', 'unit_value', 'unit_type',
        'attributes_keywords', 'core_product_name', 'normalized_name',
    ]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, f'Missing columns: {missing}'

    df = df.reset_index(drop=True)
    df['product_idx'] = df.index

    df['known_brand_clean'] = df['known_brand'].where(
        ~df['known_brand'].fillna('').str.lower().isin(BRAND_EXCLUSIONS), other=None
    )

    df['product_type'] = np.where(
        df['known_brand_clean'].notna(), 'branded',
        np.where(df['own_brand'].astype(str).str.lower().isin(['true', '1', 'yes']), 'own_brand', 'unbranded')
    )

    df['has_unit']     = df['unit_value'].notna() & (df['unit_value'] > 0)
    df['has_pack']     = df['pack_quantity'].notna() & (df['pack_quantity'] > 0)
    df['supermarket']  = df['supermarket'].str.strip()
    df['is_truncated'] = df.get('is_truncated', pd.Series(False, index=df.index)).astype(bool)

    df['cat_norm'] = df['category'].str.lower().map(CATEGORY_ALIASES).fillna(df['category'].str.lower())

    df['normalized_name'] = df['normalized_name'].apply(apply_name_synonyms)

    print(f'\nSupermarket distribution:\n{df["supermarket"].value_counts().to_string()}')
    print(f'\nProduct type:\n{df["product_type"].value_counts().to_string()}')
    print(f'\nCategory (normalised) distribution:\n{df["cat_norm"].value_counts().to_string()}')
    print(f'\nUnit type distribution:\n{df["unit_type"].value_counts().to_string()}')
    print(f'\nUnit coverage: {df["has_unit"].sum():,}/{len(df):,} ({df["has_unit"].mean()*100:.1f}%)')
    print(f'Truncated names: {df["is_truncated"].sum():,}')
    return df
