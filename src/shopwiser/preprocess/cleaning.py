"""Lightweight string cleanup before extraction."""

import re
import unicodedata


def normalize_accents(text: str) -> str:
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def clean_parenthetical_notes(name) -> str:
    if not isinstance(name, str):
        return ''
    # Specific short patterns known to add noise to core_product_name
    name = re.sub(r'\(\s*serves?\s*\d+\s*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(\s*colou?rs?\s+(?:may\s+)?vary[^)]*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(\s*min\s+\d+\s*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(\s*sugar\s+levy[^)]*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(\s*food\s+pack\s*\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(\s*\d+\s*g\s*\*\s*\)', '', name, flags=re.IGNORECASE)
    # Generic: any parenthetical ≥30 chars or containing 'order by'
    name = re.sub(r'\([^)]{30,}\)', '', name)
    name = re.sub(r'\(order by[^)]*\)', '', name, flags=re.IGNORECASE)
    return name.strip()
