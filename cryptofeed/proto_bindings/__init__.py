'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf bindings import wrapper.

Provides convenient imports for generated protobuf message types
from the normalized-data-schema-crypto specification.
'''

# Import all generated protobuf modules
try:
    from gen.python.cryptofeed.normalized.v1 import trade_pb2
    from gen.python.cryptofeed.normalized.v1 import order_book_pb2
    from gen.python.cryptofeed.normalized.v1 import ticker_pb2
    from gen.python.cryptofeed.normalized.v1 import candle_pb2
    from gen.python.cryptofeed.normalized.v1 import funding_pb2
    from gen.python.cryptofeed.normalized.v1 import liquidation_pb2
    from gen.python.cryptofeed.normalized.v1 import open_interest_pb2
    from gen.python.cryptofeed.normalized.v1 import index_price_pb2
    from gen.python.cryptofeed.normalized.v1 import balance_pb2
    from gen.python.cryptofeed.normalized.v1 import position_pb2
    from gen.python.cryptofeed.normalized.v1 import fill_pb2
    from gen.python.cryptofeed.normalized.v1 import order_info_pb2
    from gen.python.cryptofeed.normalized.v1 import transaction_pb2
    from gen.python.cryptofeed.normalized.v1 import order_pb2
    from gen.python.cryptofeed.normalized.v1 import trade_side_pb2
    from gen.python.cryptofeed.normalized.v1 import price_level_pb2
    from gen.python.cryptofeed.normalized.v1 import level2_delta_pb2
    from gen.python.cryptofeed.normalized.v1 import nbbo_pb2
    from gen.python.cryptofeed.normalized.v1 import top_of_book_pb2
    from gen.python.cryptofeed.normalized.v1 import events_pb2
except ImportError as e:
    raise ImportError(
        f"Failed to import protobuf bindings: {e}\n"
        f"Ensure protobuf schemas have been generated with 'buf generate proto/'"
    ) from e

__all__ = [
    'trade_pb2',
    'order_book_pb2',
    'ticker_pb2',
    'candle_pb2',
    'funding_pb2',
    'liquidation_pb2',
    'open_interest_pb2',
    'index_price_pb2',
    'balance_pb2',
    'position_pb2',
    'fill_pb2',
    'order_info_pb2',
    'transaction_pb2',
    'order_pb2',
    'trade_side_pb2',
    'price_level_pb2',
    'level2_delta_pb2',
    'nbbo_pb2',
    'top_of_book_pb2',
    'events_pb2',
]
