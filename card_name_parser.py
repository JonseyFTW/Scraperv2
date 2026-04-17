"""Parse structured fields out of SportsCardPro `product_name` strings.

SCP encodes everything in a single product_name. Examples:

    "Joe Burrow #30"
    "Joe Burrow #30 [Yellow Pyramids]"
    "Joe Burrow [Silver] #30 /185"
    "Joe Burrow #RPA-25 /99"
    "Topps Chrome Refractor Auto"              (no #, no brackets)
    "Upper Deck #1/1000"                       (# first, / next -> print_run=1000, card_number=1)

Rules:
  - player_name   : text before whichever of '[', '#', or '/<digit>' appears first
  - variant_label : contents of the first [...] block
  - card_number   : token after first '#', accepting [A-Za-z0-9-]
  - print_run     : first integer after a '/' anywhere in the string

Absent fields return None. Empty/None input -> all None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_CARD_NUMBER_RE = re.compile(r"#\s*([A-Za-z0-9\-]+)")
_PRINT_RUN_RE   = re.compile(r"/\s*(\d{1,6})\b")
_BRACKET_RE     = re.compile(r"\[([^\]]+)\]")
_PRINT_RUN_SPLIT_RE = re.compile(r"/\s*\d")


@dataclass
class ParsedName:
    player_name:   Optional[str]
    card_number:   Optional[str]
    print_run:     Optional[int]
    variant_label: Optional[str]


def parse_product_name(product_name: Optional[str]) -> ParsedName:
    if not product_name:
        return ParsedName(None, None, None, None)

    m = _BRACKET_RE.search(product_name)
    variant_label = m.group(1).strip() if m else None

    m = _CARD_NUMBER_RE.search(product_name)
    card_number = m.group(1) if m else None

    m = _PRINT_RUN_RE.search(product_name)
    print_run = int(m.group(1)) if m else None

    positions = [i for i in (product_name.find("["), product_name.find("#")) if i >= 0]
    m_slash = _PRINT_RUN_SPLIT_RE.search(product_name)
    if m_slash:
        positions.append(m_slash.start())
    split_at = min(positions) if positions else len(product_name)
    player_name = product_name[:split_at].strip() or None

    return ParsedName(player_name, card_number, print_run, variant_label)
