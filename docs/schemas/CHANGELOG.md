# Schema Changelog

## v3.2.0 (unpublished)
- Added Binance/Tardis parity fields to `cryptofeed.normalized.v2beta1.Trade`:
  - `maker` (optional bool)
  - `event_time` (optional int64 µs)
  - `match_id` (optional string)
  - `liquidity_flag` (optional string)
- Added parity fields to `cryptofeed.normalized.v2beta1.Level2Book` (order book):
  - `event_time` (optional int64 µs)
  - `last_update_id` (optional string)
- All new fields are optional to remain wire-compatible; decimal scale remains 1e-8 and timestamp `timestamp` continues to represent trade/book match time. Use `event_time` when the venue provides an explicit event timestamp.

## v3.1.0
- Current published baseline (no schema changes documented in this file previously).
