"""Temporary shim for the legacy protobuf helper path.

All logic now lives under `cryptofeed.backends.protobuf`."""

from __future__ import annotations

import warnings

from cryptofeed.backends.protobuf.helpers import serialize_to_protobuf, get_converter
from cryptofeed.backends.protobuf import converters
from cryptofeed.backends.protobuf.serialization import (  # noqa: F401
    _CONVERTER_MAP,
    _SCHEMA_CLASS_MAP,
)
from cryptofeed.backends.protobuf.converters import (  # noqa: F401
    trade_to_proto,
    ticker_to_proto,
    candle_to_proto,
    funding_to_proto,
    orderbook_to_proto,
    liquidation_to_proto,
    open_interest_to_proto,
    index_to_proto,
    balance_to_proto,
    position_to_proto,
    fill_to_proto,
    order_info_to_proto,
    order_to_proto,
    transaction_to_proto,
)

warnings.warn(
    "cryptofeed.backends.protobuf_helpers is deprecated; "
    "import from cryptofeed.backends.protobuf instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    'serialize_to_protobuf',
    'get_converter',
    '_CONVERTER_MAP',
    '_SCHEMA_CLASS_MAP',
    'trade_to_proto',
    'ticker_to_proto',
    'candle_to_proto',
    'funding_to_proto',
    'orderbook_to_proto',
    'liquidation_to_proto',
    'open_interest_to_proto',
    'index_to_proto',
    'balance_to_proto',
    'position_to_proto',
    'fill_to_proto',
    'order_info_to_proto',
    'order_to_proto',
    'transaction_to_proto',
]
