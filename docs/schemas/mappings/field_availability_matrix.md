# Field Availability Matrix

This document provides a comprehensive overview of which protobuf v2beta1 schema fields are supported by each exchange integration in Cryptofeed. This matrix helps developers understand field coverage and guides future implementation work.

**Last Updated:** 2025-12-14
**Schema Version:** v2beta1
**Document Status:** Living document - updated as exchange implementations evolve

---

## Trade Fields Availability

The `Trade` message in protobuf v2beta1 includes four optional metadata fields beyond the core price/volume data:

| Exchange | maker | event_time | match_id | liquidity_flag | Implementation Status | Notes |
|----------|-------|------------|----------|----------------|----------------------|-------|
| **Binance** | ✅ SUPPORTED | ✅ SUPPORTED | ✅ SUPPORTED | ⏸️ NOT_AVAILABLE | **PRODUCTION** | Extracts 'm', 'E', 'a' fields from aggTrade WebSocket messages |
| Binance Delivery | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | Futures markets follow same schema as spot |
| Binance US | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | US-regulated variant uses identical API structure |
| **OKX** | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | ⏸️ PLANNED | PLANNED | API provides maker/taker in 'side' field, event time in 'ts' |
| **Coinbase** | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | WebSocket provides maker/taker via 'maker_order_id'/'taker_order_id' |
| **Kraken** | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | ⏸️ NOT_AVAILABLE | ⏸️ NOT_AVAILABLE | PLANNED | Limited metadata in public trade feed |
| Bybit | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | Similar structure to Binance for USDT perpetuals |
| Bitfinex | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | Trade ID available, event time not provided |
| Deribit | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Derivatives-focused exchange with rich metadata |
| Huobi | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | API structure similar to Binance |
| Gate.io | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | Trade feed includes maker/taker flag |
| KuCoin | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | WebSocket provides trade details |
| Phemex | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | ⏸️ NOT_AVAILABLE | PLANNED | Contract trading focus |
| Bitmex | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | Derivatives platform with detailed trade data |
| FTX (archived) | ❌ NOT_AVAILABLE | ❌ NOT_AVAILABLE | ❌ NOT_AVAILABLE | ❌ NOT_AVAILABLE | DEPRECATED | Exchange no longer operational |
| Other Exchanges | ⏸️ VARIES | ⏸️ VARIES | ⏸️ VARIES | ⏸️ VARIES | VARIES | See individual exchange documentation |

---

## OrderBook Fields Availability

The `OrderBook` message in protobuf v2beta1 includes two optional metadata fields:

| Exchange | event_time | last_update_id | Implementation Status | Notes |
|----------|------------|----------------|----------------------|-------|
| **Binance** | ✅ SUPPORTED | ✅ SUPPORTED | **PRODUCTION** | Extracts 'E' (event time) and 'u' (final update ID) from depthUpdate messages |
| Binance Delivery | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Futures order book updates follow same schema |
| Binance US | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | US variant uses identical API structure |
| **OKX** | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | WebSocket provides 'ts' (timestamp) and 'seqId' (sequence) |
| **Coinbase** | ⏸️ NOT_AVAILABLE | ⏸️ PLANNED | PLANNED | Sequence number available, event time not provided by API |
| **Kraken** | ⏸️ NOT_AVAILABLE | ⏸️ NOT_AVAILABLE | PLANNED | Limited metadata in order book snapshots |
| Bybit | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Order book updates include timestamp and sequence |
| Bitfinex | ⏸️ PLANNED | ⏸️ NOT_AVAILABLE | PLANNED | Event time available, sequence tracking varies |
| Deribit | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Derivatives order books with full metadata |
| Huobi | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | WebSocket includes event timestamps |
| Gate.io | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Order book feed includes event metadata |
| KuCoin | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Sequence numbers for gap detection |
| Phemex | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Contract order book updates |
| Bitmex | ⏸️ PLANNED | ⏸️ PLANNED | PLANNED | Derivatives order book with sequences |
| Other Exchanges | ⏸️ VARIES | ⏸️ VARIES | VARIES | See individual exchange documentation |

---

## Field Definitions

### Trade Fields

| Field Name | Type | Description | Exchange API Examples |
|------------|------|-------------|----------------------|
| **maker** | `bool` | True if buyer is maker side, False if buyer is taker side | Binance: `m` field, OKX: derived from `side` |
| **event_time** | `float` → `int64 µs` | Exchange event timestamp (when trade occurred on exchange matching engine) | Binance: `E` field (milliseconds), OKX: `ts` field |
| **match_id** | `string` | Exchange-specific match identifier (may differ from trade_id) | Binance: `a` field (aggregate trade ID), Coinbase: match ID in full feed |
| **liquidity_flag** | `string` | Exchange-specific liquidity role indicator (e.g., "M" for maker, "T" for taker) | Deribit: `liquidation` field, OKX: custom flags |

### OrderBook Fields

| Field Name | Type | Description | Exchange API Examples |
|------------|------|-------------|----------------------|
| **event_time** | `float` → `int64 µs` | Exchange event timestamp (when order book update occurred on exchange) | Binance: `E` field (milliseconds), OKX: `ts` field |
| **last_update_id** | `int64` | Final update ID in this event (for gap detection and sequence validation) | Binance: `u` field, Coinbase: `sequence` field |

---

## Implementation Status Legend

- ✅ **SUPPORTED** - Field extraction implemented and production-ready
- ⏸️ **PLANNED** - Exchange API provides this field, implementation pending
- ⏸️ **NOT_AVAILABLE** - Exchange API does not provide this field
- ❌ **NOT_AVAILABLE** - Exchange no longer operational or deprecated
- **VARIES** - Field availability depends on specific market/channel

---

## Data Quality Metrics

Current field population rates (as of 2025-12-14):

| Field | Overall Population Rate | Binance Population Rate | Notes |
|-------|------------------------|------------------------|-------|
| Trade.maker | ~8% (1/12 exchanges) | 100% | Only Binance implemented |
| Trade.event_time | ~8% (1/12 exchanges) | 100% | Only Binance implemented |
| Trade.match_id | ~8% (1/12 exchanges) | 100% | Only Binance implemented |
| Trade.liquidity_flag | 0% (0/12 exchanges) | 0% | No exchanges implemented yet |
| OrderBook.event_time | ~8% (1/12 exchanges) | 100% | Only Binance implemented |
| OrderBook.last_update_id | ~8% (1/12 exchanges) | 100% | Only Binance implemented |

**Target:** 75%+ population rate across top 12 exchanges by volume

---

## Exchange API Documentation References

Official API documentation for field extraction:

- **Binance Spot:** [WebSocket Streams](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
  - Trade: `<symbol>@aggTrade` stream
  - OrderBook: `<symbol>@depth` stream
- **OKX:** [WebSocket Public Channels](https://www.okx.com/docs-v5/en/#websocket-api-public-channel)
  - Trade: `trades` channel
  - OrderBook: `books` channel
- **Coinbase:** [WebSocket Feed](https://docs.cloud.coinbase.com/exchange/docs/websocket-channels)
  - Trade: `matches` channel
  - OrderBook: `level2` channel with sequences
- **Kraken:** [WebSocket API](https://docs.kraken.com/websockets/)
  - Trade: `trade` subscription
  - OrderBook: `book` subscription

---

## Migration Guidance

When implementing field extraction for a new exchange:

1. **Review Exchange API Documentation:** Identify which v2beta1 fields the exchange provides in raw WebSocket/REST responses
2. **Update This Matrix:** Mark fields as SUPPORTED or NOT_AVAILABLE based on API capabilities
3. **Implement Extraction Logic:** See `docs/schemas/migration/adding_exchange_fields.md` for step-by-step guide
4. **Reference Binance Implementation:** Use `cryptofeed/exchanges/binance.py` as canonical example
5. **Update Field Population Metrics:** After implementation, measure population rates in production

---

## Related Documentation

- **Binance Field Mapping:** `docs/schemas/mappings/binance_field_mapping.md` - Detailed field extraction specification
- **Migration Guide:** `docs/schemas/migration/adding_exchange_fields.md` - Step-by-step implementation template
- **Protobuf Schema:** `cryptofeed/backends/protobuf/bindings/` - v2beta1 schema definitions
- **Trade Mapping:** `docs/schemas/mappings/trade_mapping.md` - Cross-platform trade field reconciliation
- **OrderBook Mapping:** `docs/schemas/mappings/order_book_mapping.md` - Cross-platform order book field reconciliation

---

## Feedback & Contributions

This matrix is maintained by the Cryptofeed development team. To contribute:

- **Report Missing Fields:** Open GitHub issue with exchange name and API field details
- **Implement New Exchange:** Follow migration guide and submit PR with test coverage
- **Update Status:** PRs welcome to reflect implementation progress

**Contact:** Cryptofeed maintainers via GitHub issues
