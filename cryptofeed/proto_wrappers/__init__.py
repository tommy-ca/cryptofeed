'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrappers for C extension data types.

This module adds to_proto() methods to cryptofeed C extension types
via monkey-patching, enabling protobuf serialization.

Import this module to enable protobuf serialization for all data types:
    import cryptofeed.proto_wrappers

Individual type wrappers can be imported separately:
    import cryptofeed.proto_wrappers.trade
    import cryptofeed.proto_wrappers.orderbook
'''

# Import all wrappers to register to_proto() methods
# Note: Import using 'from . import' to avoid circular imports
from . import trade
# from . import orderbook  # TODO: Implement orderbook wrapper

__all__ = ['trade']
