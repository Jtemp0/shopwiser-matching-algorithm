"""
Cleaning, normalisation, and feature extraction for raw product rows.

- ``grocery_vocab`` — lists and maps shared by extractors
- ``cleaning`` — accents, parenthetical cleanup
- ``brand`` — supermarket own-label, tier, ABV, known brands
- ``units`` — multipack and g/ml extraction, price fallback
- ``attributes`` — attribute keywords and descriptors
- ``normalise`` — ``normalize_product_name`` (per row) and ``main`` (CSV CLI)
"""

from .normalise import main, normalize_product_name

__all__ = ['main', 'normalize_product_name']
