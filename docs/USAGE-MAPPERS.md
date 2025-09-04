# Using Exchange-Native Mappers

This project ships exchange-native protobuf schemas and small mapper functions to convert native messages to normalized `cryptofeed.v1` types.

## Basic Mapping

```bash
# Ensure buf-generated Python is on PYTHONPATH
export PYTHONPATH=$(pwd)/gen/python:$PYTHONPATH
python examples/map_native_to_common.py binance BookTicker tests/fixtures/exchange-native/binance/bookTicker_spot.json
```

## Programmatic Usage (Registry)

```python
from cryptofeed.proto_mappers.registry import default_registry
from cryptofeed.v1 import market_data_pb2 as md
from cryptofeed.exchanges.binance.v1 import binance_pb2 as b

reg = default_registry()
native = b.Trade(symbol='BTCUSDT', trade_id='1', is_buyer_maker=False)
common_trade = reg.map(native, md)
```

- Book/L2/BookDelta/Funding are supported via per-exchange mapping functions.
- `raw_data` on native messages is preserved in the normalized message where applicable.

## Notes
- Options instruments are mapped to `InstrumentType.OPTION`. Strike/expiry parsing is intentionally out of scope for now.
- Symbol parsing uses simple heuristics; consider metadata-driven parsing in upstream feeds for edge cases.
