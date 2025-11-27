# Data Types and Product Types Exploration - Complete Analysis

## Overview

This document provides a comprehensive analysis of Cryptofeed's 17 normalized protobuf data types across exchange integrations, categorized by product types and mapped to exchange coverage patterns.

## Data Types Overview (17 Total)

### Universal Data Types (All Product Types)
- **Trade** - Individual trade executions with price, amount, side, timestamp
- **Ticker** - Best bid/ask quotes with microsecond timestamps
- **Level2Book** - Full order book snapshots with bids/asks ladders
- **Level2Delta** - Incremental order book updates
- **TopOfBook** - L1 best bid/ask ladders
- **Candle** - OHLCV bars with configurable intervals
- **Balance** - Wallet balances with available/reserved amounts
- **Fill** - Trade fills linked to orders with fee information
- **Order** - Order submissions with price/amount details
- **OrderInfo** - Venue-reported order state and status
- **Transaction** - Wallet transactions (deposits/withdrawals)

### Derivatives-Only Data Types
- **Funding** - Perpetual swap funding rates and mark prices
- **Liquidation** - Forced position closures with liquidation details
- **OpenInterest** - Outstanding contract quantities
- **IndexPrice** - Index calculations and reference prices
- **Position** - Open derivatives positions with P&L tracking

### Cross-Exchange Data Types
- **Nbbo** - National Best Bid and Offer across multiple exchanges

## Product Type Categories

### Core Product Types
- **SPOT** - Traditional spot trading pairs (BTC-USD)
- **PERPETUAL** - Perpetual futures contracts (BTC-PERP)
- **FUTURES** - Dated futures contracts (BTC-20251227)
- **OPTION** - Options contracts (BTC-20251227-CALL-50000)
- **CURRENCY** - Currency pairs for FX
- **FX** - Foreign exchange pairs

## Data Type ↔ Product Type Matrix

| Data Type | SPOT | PERPETUAL | FUTURES | OPTION | Notes |
|-----------|------|-----------|---------|--------|-------|
| **Trade** | ✅ | ✅ | ✅ | ✅ | Universal market data |
| **Ticker** | ✅ | ✅ | ✅ | ✅ | Universal market data |
| **Level2Book** | ✅ | ✅ | ✅ | ✅ | Universal market data |
| **Level2Delta** | ✅ | ✅ | ✅ | ✅ | Universal market data |
| **TopOfBook** | ✅ | ✅ | ✅ | ✅ | Universal market data |
| **Candle** | ✅ | ✅ | ✅ | ✅ | Universal market data |
| **Funding** | ❌ | ✅ | ❌ | ❌ | Perpetual-specific |
| **Liquidation** | ❌ | ✅ | ✅ | ❌ | Derivatives liquidations |
| **OpenInterest** | ❌ | ✅ | ✅ | ❌ | Derivatives OI |
| **IndexPrice** | ❌ | ✅ | ✅ | ❌ | Derivatives reference |
| **Balance** | ✅ | ✅ | ✅ | ✅ | Account-specific |
| **Position** | ❌ | ✅ | ✅ | ❌ | Derivatives positions |
| **Fill** | ✅ | ✅ | ✅ | ✅ | Account-specific |
| **Order** | ✅ | ✅ | ✅ | ✅ | Account-specific |
| **OrderInfo** | ✅ | ✅ | ✅ | ✅ | Account-specific |
| **Transaction** | ✅ | ✅ | ✅ | ✅ | Account-specific |
| **Nbbo** | ✅ | ✅ | ✅ | ✅ | Cross-exchange aggregation |

## Exchange Coverage Patterns

### Major Exchange Examples

- **Binance (SPOT)**: Trade, Ticker, Level2Book, Level2Delta, TopOfBook, Candle, Balance, Fill, Order, OrderInfo, Transaction
- **Binance Futures (PERPETUAL/FUTURES)**: All SPOT types + Funding, Liquidation, OpenInterest, Position
- **Bybit (SPOT)**: Trade, Ticker, Level2Book, Level2Delta, TopOfBook, Candle
- **Bybit (PERPETUAL/FUTURES)**: All SPOT types + Funding, Liquidation, OpenInterest, IndexPrice, Position

### Coverage Observations
- **Spot exchanges**: 11 core data types (market + account)
- **Derivatives exchanges**: 16 data types (all except Nbbo)
- **Account data**: Universally supported across authenticated feeds
- **Market data**: Comprehensive coverage with some variation in depth

## Key Findings & Gaps

### Strengths
- **Comprehensive Schema**: 17 data types with consistent protobuf serialization
- **Product Type Flexibility**: Clear separation between spot and derivatives data
- **Precision Standards**: Well-defined decimal scaling (1e-8 for prices/volumes)
- **Universal Fields**: exchange, symbol, timestamp present in all types

### Identified Gaps
- **Nbbo Implementation**: Limited cross-exchange aggregation support
- **Options Coverage**: Few exchanges support options data types
- **Index Price**: Not universally available on all derivatives exchanges
- **Real-time Position Updates**: Some exchanges lack streaming position data

## Field Precision Standards
- **Prices**: 1e-8 scale (8 decimal places)
- **Volumes/Amounts**: 1e-8 scale
- **Funding Rates**: 1e-4 scale (4 decimal places)
- **Timestamps**: Microseconds since Unix epoch
- **Balances**: Native precision (varies by currency)

## Conclusion

This analysis provides a solid foundation for understanding Cryptofeed's data type architecture and can guide future development of exchange integrations and consumer applications.</content>
<parameter name="filePath">docs/analysis/data-types-exploration.md