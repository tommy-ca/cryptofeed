from typing import Any, Callable, Dict, Tuple


def _fqname(obj: Any) -> str:
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


class MapperRegistry:
    def __init__(self):
        self._map: Dict[str, Tuple[Callable, str]] = {}

    def register(self, fqname: str, fn: Callable, target: str):
        self._map[fqname] = (fn, target)
        return self

    def map(self, obj: Any, md):
        name = _fqname(obj)
        if name not in self._map:
            raise KeyError(f"No mapper for {name}")
        fn, _ = self._map[name]
        return fn(obj, md)


def default_registry():
    # Lazy imports to avoid runtime conflicts
    from importlib import import_module
    reg = MapperRegistry()
    # Binance
    b_mod = import_module('cryptofeed.exchanges.binance.v1.binance_pb2')
    b_map = import_module('cryptofeed.proto_mappers.binance')
    reg.register(f"{b_mod.__name__}.Trade", b_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register(f"{b_mod.__name__}.BookTicker", b_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register(f"{b_mod.__name__}.DepthUpdate", b_map.to_common_l2_from_depth, 'cryptofeed.v1.L2Book')
    reg.register(f"{b_mod.__name__}.Funding", b_map.to_common_funding, 'cryptofeed.v1.Funding')
    
    # OKX
    o_mod = import_module('cryptofeed.exchanges.okx.v1.okx_pb2')
    o_map = import_module('cryptofeed.proto_mappers.okx')
    reg.register(f"{o_mod.__name__}.Trade", o_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register(f"{o_mod.__name__}.Ticker", o_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register(f"{o_mod.__name__}.OrderBook", o_map.to_common_l2_from_orderbook, 'cryptofeed.v1.L2Book')
    reg.register(f"{o_mod.__name__}.Funding", o_map.to_common_funding, 'cryptofeed.v1.Funding')
    # Bybit
    y_mod = import_module('cryptofeed.exchanges.bybit.v1.bybit_pb2')
    y_map = import_module('cryptofeed.proto_mappers.bybit')
    reg.register(f"{y_mod.__name__}.Trade", y_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register(f"{y_mod.__name__}.Ticker", y_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register(f"{y_mod.__name__}.OrderBook", y_map.to_common_l2_from_orderbook, 'cryptofeed.v1.L2Book')
    reg.register(f"{y_mod.__name__}.Funding", y_map.to_common_funding, 'cryptofeed.v1.Funding')
    # Bitget
    g_mod = import_module('cryptofeed.exchanges.bitget.v1.bitget_pb2')
    g_map = import_module('cryptofeed.proto_mappers.bitget')
    reg.register(f"{g_mod.__name__}.Trade", g_map.to_common_trade, 'cryptofeed.v1.Trade')
    reg.register(f"{g_mod.__name__}.Ticker", g_map.to_common_ticker, 'cryptofeed.v1.Ticker')
    reg.register(f"{g_mod.__name__}.OrderBook", g_map.to_common_l2_from_orderbook, 'cryptofeed.v1.L2Book')
    reg.register(f"{g_mod.__name__}.Funding", g_map.to_common_funding, 'cryptofeed.v1.Funding')
return reg
