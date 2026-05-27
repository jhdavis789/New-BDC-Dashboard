"""Adapter registry — maps a BDC ticker to its structural SOI parser.

This public sample ships two reference adapters (ARCC, OBDC). The full
internal build registers ~40 BDCs; the architecture is identical.
"""
from __future__ import annotations

from typing import Type

from .base import V2SOIParser
from .arcc import ARCCParser
from .obdc import OBDCParser

REGISTRY: dict[str, Type[V2SOIParser]] = {
    "ARCC": ARCCParser,
    "OBDC": OBDCParser,
}


def get_adapter(ticker: str) -> Type[V2SOIParser]:
    if ticker not in REGISTRY:
        raise KeyError(f"no V2 adapter for ticker {ticker}")
    return REGISTRY[ticker]
