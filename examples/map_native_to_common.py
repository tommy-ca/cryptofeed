#!/usr/bin/env python3
import os
import sys
import json
from importlib import import_module

# Usage: python examples/map_native_to_common.py <exchange> <message> <json_file>
# exchange: binance|okx|bybit|bitget
# message: Trade|Ticker|OrderBook|DepthUpdate|BookTicker|Funding

def project_root(start: str) -> str:
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, 'gen', 'python')) and os.path.isfile(os.path.join(cur, 'buf.yaml')):
            return cur
        nxt = os.path.abspath(os.path.join(cur, os.pardir))
        if nxt == cur:
            return cur
        cur = nxt

ROOT = project_root(os.path.dirname(__file__))
GEN = os.path.join(ROOT, 'gen', 'python')
if GEN not in sys.path:
    sys.path.insert(0, GEN)

if len(sys.argv) < 4:
    print('Usage: map_native_to_common.py <exchange> <message> <json_file>')
    sys.exit(1)

ex, msg, jf = sys.argv[1:4]
mod_map = {
  'binance': 'cryptofeed.exchanges.binance.v1.binance_pb2',
  'okx': 'cryptofeed.exchanges.okx.v1.okx_pb2',
  'bybit': 'cryptofeed.exchanges.bybit.v1.bybit_pb2',
  'bitget': 'cryptofeed.exchanges.bitget.v1.bitget_pb2',
}
map_map = {
  'binance': 'cryptofeed.proto_mappers.binance',
  'okx': 'cryptofeed.proto_mappers.okx',
  'bybit': 'cryptofeed.proto_mappers.bybit',
  'bitget': 'cryptofeed.proto_mappers.bitget',
}

pb2 = import_module(mod_map[ex])
mapper = import_module(map_map[ex])
md = import_module('cryptofeed.v1.market_data_pb2')

payload = json.load(open(jf))

# Build a minimal native instance based on message
native = getattr(pb2, msg)()
if ex in ('binance','bybit') and hasattr(native, 'symbol') and 's' in payload:
    native.symbol = payload.get('s')
elif 'instId' in payload and hasattr(native, 'inst_id'):
    native.inst_id = payload['instId']

# Set some common fields heuristically
if hasattr(native, 'price') and 'p' in payload:
    native.price.value = payload['p']
if hasattr(native, 'quantity') and 'q' in payload:
    native.quantity.value = payload['q']
if hasattr(native, 'best_bid') and 'bestBid' in payload:
    native.best_bid.value = payload['bestBid']
if hasattr(native, 'best_ask') and 'bestAsk' in payload:
    native.best_ask.value = payload['bestAsk']
if hasattr(native, 'bid_price') and 'b' in payload:
    native.bid_price.value = payload['b']
if hasattr(native, 'ask_price') and 'a' in payload:
    native.ask_price.value = payload['a']

# Map
fn_name = {
  'Trade': 'to_common_trade',
  'Ticker': 'to_common_ticker',
  'BookTicker': 'to_common_ticker',
  'OrderBook': 'to_common_l2_from_orderbook',
  'DepthUpdate': 'to_common_l2_from_depth',
  'Funding': 'to_common_funding',
}[msg]
fn = getattr(mapper, fn_name)
common_msg = fn(native, md)
print(common_msg)
