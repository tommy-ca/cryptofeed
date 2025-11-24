# Cryptofeed Normalized v2 Protobuf Schemas

Authoritative field mapping for v2 message types used by `shift-left-streaming-lakehouse`.

## Decimal Fidelity Rule (REQ-011)
- Default numeric type: `double` (lossy, acceptable for most exchanges).
- If an exchange needs precision beyond ~1e-9, switch the affected numeric fields to `bytes` and **add** a message-level `int32 scale` describing the quantization exponent. Keep original field numbers; place `scale` in a high, currently unused slot (e.g., 15).

## Timestamp Rule (REQ-007)
- All timestamps use `google.protobuf.Timestamp`.

## Field Matrix (v1 field numbers reused where possible)
| Message | Field | No. | Type | Notes |
|---|---|---|---|---|
| Trade | exchange | 1 | string | unchanged |
| Trade | symbol | 2 | string | unchanged |
| Trade | side | 3 | enum | unchanged |
| Trade | trade_id | 4 | string | unchanged |
| Trade | price | 5 | double | switch to bytes+scale if >1e-9 precision needed |
| Trade | amount | 6 | double | switch to bytes+scale if >1e-9 precision needed |
| Trade | timestamp | 7 | google.protobuf.Timestamp | standardized |
| Trade | sequence_number | 8 | uint64 | gap detection |
| Trade | (reserved) | 9 | — | reserved from v1 trade_type to prevent reuse |
| Ticker | exchange | 1 | string | unchanged |
| Ticker | symbol | 2 | string | unchanged |
| Ticker | best_bid_price | 3 | double | reuses v1 bid slot |
| Ticker | best_ask_price | 4 | double | reuses v1 ask slot |
| Ticker | best_bid_size | 5 | double | |
| Ticker | best_ask_size | 6 | double | |
| Ticker | timestamp | 7 | google.protobuf.Timestamp | |
| Ticker | sequence_number | 8 | uint64 | |
| Book | exchange | 1 | string | unchanged |
| Book | symbol | 2 | string | unchanged |
| Book | bids | 3 | repeated PriceLevelV2 | price/size double |
| Book | asks | 4 | repeated PriceLevelV2 | price/size double |
| Book | timestamp | 5 | google.protobuf.Timestamp | snapshot/delta aligned |
| Book | sequence_number | 6 | uint64 | |
| Book | checksum | 7 | string | retained |
| Candle | exchange | 1 | string | unchanged |
| Candle | symbol | 2 | string | unchanged |
| Candle | start | 3 | google.protobuf.Timestamp | was int64 µs |
| Candle | end | 4 | google.protobuf.Timestamp | was int64 µs |
| Candle | interval | 5 | string | unchanged |
| Candle | trades | 6 | uint64 | was optional int64 |
| Candle | open | 7 | double | switch to bytes+scale if precision-critical |
| Candle | close | 8 | double | |
| Candle | high | 9 | double | |
| Candle | low | 10 | double | |
| Candle | volume | 11 | double | |
| Candle | closed | 12 | bool | |
| Candle | timestamp (close/end) | 13 | google.protobuf.Timestamp | |
| Candle | sequence_number | 14 | uint64 | |

### Launch Decision Table (Day 1 defaults)
| Field group | Default Type | Bytes+Scale? | Scale field number |
|---|---|---|---|
| Trade.price / Trade.amount | double | No | 15 (reserved if ever enabled) |
| Ticker bid/ask prices & sizes | double | No | 15 (reserved if ever enabled) |
| OrderBook price/quantity | double | No | 15 (reserved if ever enabled) |
| Candle OHLCV | double | No | 15 (reserved if ever enabled) |


## Schema Hygiene
- Syntax: `proto3`
- Package: `cryptofeed.normalized.v2`
- Run `buf lint proto/cryptofeed/normalized/v2` to validate style and reserved fields.
