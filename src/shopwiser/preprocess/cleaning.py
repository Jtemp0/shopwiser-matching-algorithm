"""Lightweight string cleanup before extraction."""

import re
import unicodedata


def normalize_accents(text: str) -> str:
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def clean_parenthetical_notes(name) -> str:
    if not isinstance(name, str):
        return ''
    name = re.sub(r'\([^)]{30,}\)', '', name)
    name = re.sub(r'\(order by[^)]*\)', '', name, flags=re.IGNORECASE)
    return name.strip()
