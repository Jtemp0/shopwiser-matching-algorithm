"""Product attributes and non-removing descriptor tags from vocabulary."""

import re
from typing import Dict, List, Tuple

from .grocery_vocab import ATTRIBUTES, DESCRIPTORS


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
