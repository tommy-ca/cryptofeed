''' 
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf bindings import wrapper.

Provides convenient imports for generated protobuf message types
from the normalized-data-schema-crypto specification.
'''

# ruff: noqa: F401

SCHEMA_VERSION = "v2beta1"  # Updated to v2beta1 for new optional fields

# Import all generated protobuf modules from v2beta1
# v2beta1 includes optional fields: Trade (maker, event_time, match_id, liquidity_flag),
# Level2Book (event_time, last_update_id)
try:
    from gen.python.cryptofeed.normalized.v2beta1 import trade_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import order_book_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import ticker_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import candle_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import funding_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import liquidation_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import open_interest_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import index_price_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import balance_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import position_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import fill_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import order_info_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import transaction_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import order_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import trade_side_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import price_level_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import level2_delta_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import nbbo_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import top_of_book_pb2
    from gen.python.cryptofeed.normalized.v2beta1 import events_pb2
    REQUIRED_MODULES = {
        'trade_pb2': 'Trade',
        'order_book_pb2': 'Level2Book',
        'ticker_pb2': 'Ticker',
        'candle_pb2': 'Candle',
        'funding_pb2': 'FundingRate',
        'liquidation_pb2': 'Liquidation',
        'open_interest_pb2': 'OpenInterest',
        'index_price_pb2': 'IndexPrice',
        'balance_pb2': 'Balance',
        'position_pb2': 'Position',
        'fill_pb2': 'Fill',
        'order_info_pb2': 'OrderInfo',
        'transaction_pb2': 'Transaction',
        'order_pb2': 'Order',
        'trade_side_pb2': 'TradeSide',
        'price_level_pb2': 'PriceLevel',
        'level2_delta_pb2': 'Level2Delta',
        'nbbo_pb2': 'Nbbo',
        'top_of_book_pb2': 'TopOfBook',
        'events_pb2': 'Events',
    }

except ImportError as e:
    raise ImportError(
        f"Failed to import protobuf bindings: {e}\n"
        f"Ensure protobuf schemas have been generated with 'buf generate proto/'"
    ) from e

def validate_bindings(required=None) -> None:
    """Validate that generated protobuf modules are present."""

    required = required or REQUIRED_MODULES.keys()
    missing = [name for name in required if name not in globals()]
    if missing:
        raise ImportError(f"Missing protobuf modules: {', '.join(missing)}")


validate_bindings()

__all__ = [
    'SCHEMA_VERSION',
    'REQUIRED_MODULES',
    'validate_bindings',
] + list(REQUIRED_MODULES.keys())
