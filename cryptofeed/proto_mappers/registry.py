from typing import Any, Callable, Dict, Tuple


def _msg_key(obj: Any) -> str:
    # Prefer stable protobuf full_name over Python module/class name
    desc = getattr(obj, 'DESCRIPTOR', None)
    if desc is not None and getattr(desc, 'full_name', None):
        return desc.full_name  # e.g., 'cryptofeed.exchanges.binance.v1.Trade'
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


class MapperRegistry:
    def __init__(self):
        self._map: Dict[str, Tuple[Callable, str]] = {}

    def register(self, name: str, fn: Callable, target: str):
        # Name should be a protobuf message full_name where possible
        self._map[name] = (fn, target)
        return self

    def register_by_full_name(self, full_name: str, fn: Callable, target: str):
        return self.register(full_name, fn, target)

    def map(self, obj: Any, md):
        name = _msg_key(obj)
        if name not in self._map:
            raise KeyError(f"No mapper for {name}")
        fn, _ = self._map[name]
        return fn(obj, md)


def default_registry():
    # Import mapper modules only; avoid importing generated pb2 modules by path
    from importlib import import_module
    reg = MapperRegistry()

    # Binance
    b_map = import_module('cryptofeed.proto_mappers.binance')
    reg.register_by_full_name('cryptofeed.exchanges.binance.v1.Trade', b_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register_by_full_name('cryptofeed.exchanges.binance.v1.BookTicker', b_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register_by_full_name('cryptofeed.exchanges.binance.v1.DepthUpdate', b_map.to_common_l2_from_depth, 'cryptofeed.v1.L2Book')
    reg.register_by_full_name('cryptofeed.exchanges.binance.v1.Funding', b_map.to_common_funding, 'cryptofeed.v1.Funding')

    # OKX
    o_map = import_module('cryptofeed.proto_mappers.okx')
    reg.register_by_full_name('cryptofeed.exchanges.okx.v1.Trade', o_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register_by_full_name('cryptofeed.exchanges.okx.v1.Ticker', o_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register_by_full_name('cryptofeed.exchanges.okx.v1.OrderBook', o_map.to_common_l2_from_orderbook, 'cryptofeed.v1.L2Book')
    reg.register_by_full_name('cryptofeed.exchanges.okx.v1.Funding', o_map.to_common_funding, 'cryptofeed.v1.Funding')

    # Bybit
    y_map = import_module('cryptofeed.proto_mappers.bybit')
    reg.register_by_full_name('cryptofeed.exchanges.bybit.v1.Trade', y_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register_by_full_name('cryptofeed.exchanges.bybit.v1.Ticker', y_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register_by_full_name('cryptofeed.exchanges.bybit.v1.OrderBook', y_map.to_common_l2_from_orderbook, 'cryptofeed.v1.L2Book')
    reg.register_by_full_name('cryptofeed.exchanges.bybit.v1.Funding', y_map.to_common_funding, 'cryptofeed.v1.Funding')

    # Bitget
    g_map = import_module('cryptofeed.proto_mappers.bitget')
    reg.register_by_full_name('cryptofeed.exchanges.bitget.v1.Trade', g_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register_by_full_name('cryptofeed.exchanges.bitget.v1.Ticker', g_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register_by_full_name('cryptofeed.exchanges.bitget.v1.OrderBook', g_map.to_common_l2_from_orderbook, 'cryptofeed.v1.L2Book')
    reg.register_by_full_name('cryptofeed.exchanges.bitget.v1.Funding', g_map.to_common_funding, 'cryptofeed.v1.Funding')

    return reg
