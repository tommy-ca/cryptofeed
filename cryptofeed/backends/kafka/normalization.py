"""
Normalization utilities for Kafka topic/partition/header usage.

This module provides centralized string normalization functions for exchange and symbol
names used across the Kafka backend. Consolidates duplicate implementations from
topic_manager.py, partitioner.py, and headers.py to comply with DRY principle.

Functions:
    normalize_symbol: Normalize trading symbol for Kafka usage
    normalize_exchange: Normalize exchange name for Kafka usage

Design Rationale:
    - Single source of truth for normalization rules
    - Consistent behavior across topic naming, partition routing, and header encoding
    - 'unknown' fallback prevents empty strings in Kafka metadata
    - Idempotent operations (normalize(normalize(x)) == normalize(x))

Usage:
    >>> from cryptofeed.backends.kafka.normalization import normalize_symbol, normalize_exchange
    >>> normalize_symbol("BTC/USD")
    'btc-usd'
    >>> normalize_exchange("Binance")
    'binance'
"""

from __future__ import annotations


def normalize_symbol(symbol: str | None) -> str:
    """
    Normalize trading symbol for Kafka topic/partition/header usage.

    Normalization Rules:
        1. Convert to lowercase
        2. Replace '/' with '-' (BTC/USD → btc-usd)
        3. Replace '_' with '-' (BTC_USD → btc-usd)
        4. Strip leading/trailing whitespace
        5. Return 'unknown' for None or empty string

    Args:
        symbol: Trading symbol string (e.g., 'BTC/USD', 'ETH_USDT', None)

    Returns:
        Normalized symbol string (lowercase, hyphens only, no whitespace)
        Returns 'unknown' if input is None or empty/whitespace-only

    Examples:
        >>> normalize_symbol('BTC/USD')
        'btc-usd'
        >>> normalize_symbol('BTC_USD')
        'btc-usd'
        >>> normalize_symbol(' ETH-BTC ')
        'eth-btc'
        >>> normalize_symbol(None)
        'unknown'
        >>> normalize_symbol('')
        'unknown'
        >>> normalize_symbol('  ')
        'unknown'

    Rationale:
        - Kafka topic names: Lowercase, hyphens preferred over slashes/underscores
        - Partition keys: Consistent format ensures same routing for equivalent symbols
        - Headers: UTF-8 safe encoding (hyphens safer than slashes)

    Properties:
        - Idempotent: normalize_symbol(normalize_symbol(x)) == normalize_symbol(x)
        - Never returns empty string (uses 'unknown' fallback)
        - Type-safe: Always returns str, never None
    """
    if symbol is None:
        return "unknown"
    s = str(symbol)
    if not s.strip():
        return "unknown"
    return s.strip().replace("/", "-").replace("_", "-").lower()


def normalize_exchange(exchange: str | None) -> str:
    """
    Normalize exchange name for Kafka topic/partition/header usage.

    Normalization Rules:
        1. Convert to lowercase
        2. Strip leading/trailing whitespace
        3. Return 'unknown' for None or empty string

    Args:
        exchange: Exchange name string (e.g., 'Binance', 'OKX', None)

    Returns:
        Normalized exchange string (lowercase, no whitespace)
        Returns 'unknown' if input is None or empty/whitespace-only

    Examples:
        >>> normalize_exchange('Binance')
        'binance'
        >>> normalize_exchange(' OKX ')
        'okx'
        >>> normalize_exchange('COINBASE')
        'coinbase'
        >>> normalize_exchange(None)
        'unknown'
        >>> normalize_exchange('')
        'unknown'

    Rationale:
        - Consistent casing across topic names, partition keys, headers
        - 'unknown' fallback prevents empty string routing issues

    Properties:
        - Idempotent: normalize_exchange(normalize_exchange(x)) == normalize_exchange(x)
        - Never returns empty string (uses 'unknown' fallback)
        - Type-safe: Always returns str, never None
    """
    if exchange is None:
        return "unknown"
    s = str(exchange)
    if not s.strip():
        return "unknown"
    return s.strip().lower()


__all__ = ["normalize_symbol", "normalize_exchange"]
