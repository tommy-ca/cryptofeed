# Level2Delta Mapping Notes

**Goal**: Define a deterministic transformation from Python `OrderBook.delta` payloads to the `cryptofeed.normalized.v1.Level2Delta` protobuf message.

> Note: The legacy `cryptofeed.proto_mappers` helpers were removed; any Level2Delta mapper should live alongside the consolidated protobuf helpers under `cryptofeed/backends/protobuf/`.

## Current Python Shape

```python
delta = {
    BID: [(price, size, order_id_optional)],
    ASK: [(price, size, order_id_optional)],
}
timestamp = orderbook.timestamp  # float seconds or None
sequence = orderbook.sequence_number  # int or None
checksum = orderbook.checksum  # str/int or None
```

## Proposed Mapping

| Python Source | Proto Field | Notes |
|---------------|-------------|-------|
| `orderbook.exchange` | `Level2Delta.exchange` | Normalized exchange identifier |
| `orderbook.symbol` | `Level2Delta.symbol` | Normalized instrument |
| `delta[BID]` tuples | `Level2Delta.bids` | Transform each tuple into `PriceLevel(price, size)` |
| `delta[ASK]` tuples | `Level2Delta.asks` | Same as bids, respecting ascending order |
| `orderbook.timestamp` | `Level2Delta.timestamp` | Convert float seconds → µs; omit field when `None` |
| `orderbook.sequence_number` | `Level2Delta.sequence` | Emit when present |
| `orderbook.checksum` | `Level2Delta.checksum` | Cast to string |

### Open Questions

1. **Zero/Negative Sizes**: Some venues send delete instructions (size = 0). Confirm downstream expectations.
2. **Order IDs**: Python tuples may include order IDs in certain venues. Do we surface them or drop?
3. **Empty Deltas**: Decide whether to emit messages when bids/asks arrays are empty (heartbeat) or skip entirely.

## Next Steps

1. Confirm tuple structure across supported venues (spot vs derivatives).
2. Prototype transformation helper in Python (`proto_mappers/order_book.py`).
3. Backfill unit tests covering add/update/delete scenarios per venue.
4. Update documentation (`PYTHON_PROTO_ALIGNMENT.md`) once decisions are finalized.
