"""
Adapters for converting between legacy cryptofeed types and protobuf messages.

This module provides backward compatibility during the migration to protobuf-based
data types while maintaining the existing API surface.
"""

from .trade_adapter import TradeAdapter
from .ticker_adapter import TickerAdapter
from .book_adapter import BookAdapter
from .funding_adapter import FundingAdapter
from .candle_adapter import CandleAdapter
from .order_info_adapter import OrderInfoAdapter
from .balance_adapter import BalanceAdapter
from .fill_adapter import FillAdapter
from .position_adapter import PositionAdapter

__all__ = [
    'TradeAdapter',
    'TickerAdapter', 
    'BookAdapter',
    'FundingAdapter',
    'CandleAdapter',
    'OrderInfoAdapter',
    'BalanceAdapter',
    'FillAdapter',
    'PositionAdapter'
]
