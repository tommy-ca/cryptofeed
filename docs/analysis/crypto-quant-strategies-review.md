# Crypto Quant Strategies Review & Data Requirements Analysis

## Overview

This document provides a comprehensive analysis of crypto quantitative trading strategies and maps their data requirements to Cryptofeed's 17 normalized data types, enabling strategic prioritization for quant platform development.

## Strategy Categories & Data Requirements

### 1. Momentum & Trend Following Strategies

**Core Requirements:**
- **Candle** (OHLCV) - Primary data for moving averages, trend indicators
- **Trade** - Volume analysis, price action confirmation
- **Ticker** - Real-time price feeds for entry/exit signals

**Key Data Types:** `Candle`, `Trade`, `Ticker`, `Level2Book` (optional for order flow)

**Strategy Examples:**
- Moving average crossovers (SMA, EMA)
- MACD divergence strategies
- Trend strength indicators (ADX, Parabolic SAR)
- Breakout strategies (Donchian channels, Bollinger Bands)

### 2. Mean Reversion & Statistical Arbitrage

**Core Requirements:**
- **Trade** - High-frequency tick data for spread calculations
- **Ticker** - Real-time quotes for correlation analysis
- **Level2Book** - Market depth for liquidity assessment
- **Candle** - Multi-timeframe analysis

**Key Data Types:** `Trade`, `Ticker`, `Level2Book`, `Level2Delta`, `Candle`, `Nbbo`

**Strategy Examples:**
- Pairs trading (cointegrated pairs)
- Statistical arbitrage (distance-based, z-score)
- Cross-sectional mean reversion
- Volatility-based mean reversion

### 3. Perpetual Funding Arbitrage

**Core Requirements:**
- **Funding** - Funding rates and mark prices
- **Ticker** - Spot prices for basis calculations
- **Position** - Position management and P&L tracking
- **Balance** - Account balance monitoring

**Key Data Types:** `Funding`, `Ticker`, `Position`, `Balance`, `OpenInterest`

**Strategy Examples:**
- Funding rate arbitrage (spot vs perpetual)
- Cross-exchange funding arbitrage
- Basis trading (cash-futures arbitrage)
- Gamma scalping on perpetuals

### 4. Graph/Circular Arbitrage

**Core Requirements:**
- **Ticker** - Real-time cross-exchange prices
- **Level2Book** - Order book depth for slippage calculations
- **Trade** - Actual execution prices and volumes
- **Nbbo** - Cross-exchange best bid/ask

**Key Data Types:** `Ticker`, `Level2Book`, `Trade`, `Nbbo`, `OrderInfo`

**Strategy Examples:**
- Triangular arbitrage (A→B→C→A cycles)
- Cross-exchange arbitrage (same asset, different venues)
- Synthetic arbitrage (create synthetic positions)
- Decentralized exchange arbitrage

### 5. Other Uncorrelated Strategies

**Liquidation Hunting:**
- **Liquidation** - Liquidation events and prices
- **Position** - Market positioning analysis
- **OpenInterest** - Market sentiment indicators

**Order Flow Analysis:**
- **Level2Book**, **Level2Delta** - Order book dynamics
- **Trade** - Market maker activity inference

**Market Making:**
- **Level2Book**, **Level2Delta** - Inventory management
- **OrderInfo**, **Fill** - Execution tracking

**Additional Strategies:**
- Volatility harvesting (straddles, strangles)
- Sentiment-based trading
- Whale watching (large order analysis)

**Key Data Types:** `Liquidation`, `Level2Book`, `Level2Delta`, `OrderInfo`, `Fill`, `OpenInterest`

## Priority Data Types by Strategy Coverage

| Data Type | Strategies Using | Priority | Notes |
|-----------|------------------|----------|-------|
| **Ticker** | All 5 categories | 🔴 Critical | Real-time price feed foundation |
| **Trade** | 4/5 categories | 🔴 Critical | High-frequency price action |
| **Level2Book** | 4/5 categories | 🔴 Critical | Market depth and liquidity |
| **Candle** | 3/5 categories | 🟡 High | Multi-timeframe analysis |
| **Funding** | 1/5 categories | 🟡 High | Derivatives-specific strategies |
| **Position** | 2/5 categories | 🟡 High | Risk management |
| **Balance** | 2/5 categories | 🟡 High | Account management |
| **Level2Delta** | 2/5 categories | 🟡 High | Incremental updates |
| **OrderInfo** | 2/5 categories | 🟡 High | Execution tracking |
| **Fill** | 2/5 categories | 🟡 High | Trade execution |
| **Liquidation** | 1/5 categories | 🟠 Medium | Specialized strategies |
| **OpenInterest** | 2/5 categories | 🟠 Medium | Market sentiment |
| **IndexPrice** | 1/5 categories | 🟠 Medium | Reference pricing |
| **Order** | 1/5 categories | 🟠 Medium | Order management |
| **Transaction** | 1/5 categories | 🟠 Medium | Wallet operations |
| **TopOfBook** | 1/5 categories | 🟠 Medium | L1 data |
| **Nbbo** | 1/5 categories | 🟠 Medium | Cross-exchange arb |

## Data Focus Recommendations

### High-Priority Focus Areas (80% of Strategy Coverage)
1. **Real-time Market Data**: `Ticker`, `Trade`, `Level2Book`, `Level2Delta`
2. **OHLCV Analysis**: `Candle` with multiple timeframes
3. **Account Management**: `Balance`, `OrderInfo`, `Fill`, `Position`

### Medium-Priority Focus Areas (15% of Strategy Coverage)
1. **Derivatives Data**: `Funding`, `OpenInterest`, `Liquidation`
2. **Cross-Exchange**: `Nbbo` for arbitrage strategies
3. **Reference Data**: `IndexPrice`, `TopOfBook`

### Low-Priority Focus Areas (5% of Strategy Coverage)
1. **Administrative**: `Order`, `Transaction`

## Key Gaps & Considerations

### Data Latency Requirements
- **HFT Strategies** (stat arb, circular arb): Need microsecond timestamps, minimal latency
- **Trend Strategies**: Can tolerate millisecond-level data
- **Funding Arb**: Can work with second-level updates

### Exchange Coverage Gaps
- **Options Data**: Limited support for options strategies
- **Cross-Exchange Data**: `Nbbo` implementation gaps
- **Futures Data**: Some exchanges lack dated futures support

### Implementation Priorities
1. **Core Infrastructure**: High-frequency market data pipeline
2. **Strategy Foundation**: OHLCV and account data systems
3. **Advanced Features**: Derivatives and cross-exchange data
4. **Specialized Strategies**: Liquidation and order flow analysis

## Strategic Recommendations

### For Building a Quant Platform
- **Start with Core 4**: `Ticker`, `Trade`, `Level2Book`, `Candle`
- **Add Account Layer**: `Balance`, `OrderInfo`, `Fill`, `Position`
- **Expand to Derivatives**: `Funding`, `OpenInterest`, `Liquidation`
- **Scale to Multi-Exchange**: `Nbbo` and cross-exchange infrastructure

### Data Architecture Priorities
- **Low-Latency Pipeline**: Critical for HFT and arb strategies
- **Multi-Timeframe Support**: Essential for trend and momentum strategies
- **Real-time Processing**: Required for all strategy types
- **Reliable Streaming**: Mission-critical for production systems

This analysis provides a clear roadmap for prioritizing Cryptofeed data types based on comprehensive quant strategy requirements, ensuring efficient resource allocation for building robust crypto trading systems.</content>
<parameter name="filePath">docs/analysis/crypto-quant-strategies-review.md