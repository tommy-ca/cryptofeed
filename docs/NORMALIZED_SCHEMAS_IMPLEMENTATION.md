# Normalized Data Types Implementation with Buf Schema Registry

## Overview

This document outlines the implementation of normalized data types and channels for the cryptofeed project using Protocol Buffers (protobuf) and Buf Schema Registry for cross-language schema management in data lakehouse and quantitative trading engines.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Schema Design](#schema-design)
3. [Buf Schema Registry Setup](#buf-schema-registry-setup)
4. [Data Type Mappings](#data-type-mappings)
5. [Integration Strategy](#integration-strategy)
6. [Migration Plan](#migration-plan)
7. [Language-Specific Code Generation](#language-specific-code-generation)
8. [Deployment and Versioning](#deployment-and-versioning)
9. [Performance Considerations](#performance-considerations)
10. [Best Practices](#best-practices)

## Architecture Overview

### Current State
The cryptofeed library currently uses Cython-based data types (`types.pyx`) with the following key characteristics:
- Python-centric implementation
- Direct dictionary serialization
- Exchange-specific handling
- Limited cross-language compatibility

### Proposed Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Buf Schema Registry                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │   Common v1     │  │ Market Data v1  │  │ Account Data   │
│  │                 │  │                 │  │     v1         │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              Generated Code (Multi-Language)                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────── │
│  │ Python  │ │   Go    │ │  Rust   │ │  Java   │ │   C++    │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────── │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Consumers                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │  Data Lakehouse │  │ Quant Engines   │  │  Analytics     │
│  │                 │  │                 │  │   Platform     │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
└─────────────────────────────────────────────────────────────┘
```

## Schema Design

### Schema Structure

The protobuf schemas are organized into three main modules:

#### 1. Common Types (`cryptofeed/v1/common.proto`)
- **Purpose**: Shared enums, types, and constants
- **Key Components**:
  - Exchange identifiers enum
  - Trading sides, order types, statuses
  - Instrument types
  - Decimal representation for high precision
  - Symbol structure
  - Data channel types

#### 2. Market Data (`cryptofeed/v1/market_data.proto`)
- **Purpose**: Public market data structures
- **Key Components**:
  - Trade executions
  - Ticker/BBO data
  - Order book levels (L1, L2, L3)
  - Book deltas
  - Funding rates
  - Open interest
  - Liquidations
  - Index prices
  - Candlestick/OHLCV data

#### 3. Account Data (`cryptofeed/v1/account_data.proto`)
- **Purpose**: Private account and trading data
- **Key Components**:
  - Order information and status
  - Account balances
  - Transactions (deposits/withdrawals)
  - Trade fills/executions
  - Position information

#### 4. Event Streaming (`cryptofeed/v1/events.proto`)
- **Purpose**: Event envelope and streaming infrastructure
- **Key Components**:
  - Generic event wrapper
  - Batch processing support
  - Subscription management
  - Heartbeat and error handling

### Design Principles

1. **Backward Compatibility**: Using protobuf field numbers and optional fields
2. **Precision**: String-based decimal representation for financial data
3. **Extensibility**: Oneof fields and Any types for future extensions
4. **Efficiency**: Optimized for both storage and transmission
5. **Type Safety**: Strong typing with comprehensive enums

## Buf Schema Registry Setup

### Installation

```bash
# Install Buf CLI
curl -sSL "https://github.com/bufbuild/buf/releases/latest/download/buf-$(uname -s)-$(uname -m)" -o "/usr/local/bin/buf"
chmod +x "/usr/local/bin/buf"
```

### Repository Structure

```
crypto-data/
├── cryptofeed/
│   ├── proto/
│   │   └── cryptofeed/
│   │       └── v1/
│   │           ├── common.proto
│   │           ├── market_data.proto
│   │           ├── account_data.proto
│   │           └── events.proto
│   ├── buf.yaml
│   ├── buf.gen.yaml
│   └── gen/
│       ├── python/
│       ├── go/
│       ├── rust/
│       ├── java/
│       ├── cpp/
│       └── typescript/
```

### Configuration Files

#### buf.yaml
```yaml
version: v2
modules:
  - path: proto
deps:
  - buf.build/googleapis/googleapis
breaking:
  use:
    - FILE
lint:
  use:
    - BASIC
    - COMMENTS
    - FILE_LOWER_SNAKE_CASE
```

#### buf.gen.yaml
```yaml
version: v2
managed:
  enabled: true
plugins:
  - remote: buf.build/protocolbuffers/plugins/python
    out: gen/python
  - remote: buf.build/protocolbuffers/plugins/go
    out: gen/go
  # Additional language plugins...
```

### Schema Registry Operations

```bash
# Push schemas to Buf Schema Registry
buf push --tag v1.0.0

# Generate code
buf generate

# Lint schemas
buf lint

# Check breaking changes
buf breaking --against '.git#branch=main'
```

## Data Type Mappings

### Current Types → Protobuf Mapping

| Current Type | Protobuf Message | Key Changes |
|--------------|------------------|-------------|
| `Trade` | `cryptofeed.v1.Trade` | Added receipt_timestamp, standardized enums |
| `Ticker` | `cryptofeed.v1.Ticker` | Added receipt_timestamp, symbol structure |
| `L1Book` | `cryptofeed.v1.L1Book` | Renamed fields for clarity |
| `OrderBook` | `cryptofeed.v1.L2Book` | Separated L2/L3, added delta support |
| `Funding` | `cryptofeed.v1.Funding` | Added optional fields, better timestamps |
| `Candle` | `cryptofeed.v1.Candle` | Enhanced with start/end times, interval |
| `OrderInfo` | `cryptofeed.v1.OrderInfo` | Added receipt_timestamp, enum status |
| `Balance` | `cryptofeed.v1.Balance` | Added receipt_timestamp |
| `Fill` | `cryptofeed.v1.Fill` | Enhanced with fee currency, liquidity |
| `Position` | `cryptofeed.v1.Position` | Added margin, leverage fields |

### Decimal Precision Handling

```python
# Current: Python Decimal
amount = Decimal("123.456789")

# Protobuf: String-based Decimal
decimal_pb = Decimal()
decimal_pb.value = "123.456789"
```

### Exchange Mapping

```python
# Current: String constants
BINANCE = 'BINANCE'

# Protobuf: Enum
exchange = Exchange.EXCHANGE_BINANCE
```

## Integration Strategy

### Phase 1: Parallel Implementation

1. **Generate Protobuf Classes**: Create Python bindings
2. **Adapter Layer**: Convert between current types and protobuf
3. **Backend Updates**: Add protobuf serialization support
4. **Testing**: Validate data integrity and performance

### Phase 2: Backend Integration

1. **Storage Backends**: Update all backends to support protobuf
2. **Streaming**: Implement protobuf-based event streaming
3. **Compression**: Add compression support for efficient transport
4. **Monitoring**: Add metrics for protobuf usage

### Phase 3: API Evolution

1. **New APIs**: Expose protobuf-based APIs
2. **Client Libraries**: Generate client libraries for multiple languages
3. **Documentation**: Update API documentation
4. **Migration Tools**: Provide conversion utilities

### Adapter Pattern Implementation

```python
from cryptofeed.types import Trade as LegacyTrade
from cryptofeed.proto.v1.market_data_pb2 import Trade as ProtoTrade
from cryptofeed.proto.v1.common_pb2 import Exchange, Side, Decimal

class TradeAdapter:
    @staticmethod
    def to_protobuf(legacy_trade: LegacyTrade) -> ProtoTrade:
        proto_trade = ProtoTrade()
        proto_trade.exchange = Exchange.Value(f"EXCHANGE_{legacy_trade.exchange}")
        proto_trade.symbol.symbol = legacy_trade.symbol
        proto_trade.side = Side.Value(f"SIDE_{legacy_trade.side.upper()}")
        proto_trade.amount.value = str(legacy_trade.amount)
        proto_trade.price.value = str(legacy_trade.price)
        proto_trade.id = legacy_trade.id or ""
        proto_trade.type = legacy_trade.type or ""
        proto_trade.timestamp.FromSeconds(int(legacy_trade.timestamp))
        return proto_trade
    
    @staticmethod
    def from_protobuf(proto_trade: ProtoTrade) -> LegacyTrade:
        return LegacyTrade(
            exchange=Exchange.Name(proto_trade.exchange).replace("EXCHANGE_", ""),
            symbol=proto_trade.symbol.symbol,
            side=Side.Name(proto_trade.side).replace("SIDE_", "").lower(),
            amount=Decimal(proto_trade.amount.value),
            price=Decimal(proto_trade.price.value),
            timestamp=proto_trade.timestamp.ToSeconds(),
            id=proto_trade.id,
            type=proto_trade.type
        )
```

## Migration Plan

### Timeline: 6-Month Implementation

#### Month 1-2: Foundation
- [ ] Set up Buf Schema Registry
- [ ] Define and validate protobuf schemas
- [ ] Generate initial language bindings
- [ ] Create adapter layer
- [ ] Unit tests for adapters

#### Month 3-4: Integration
- [ ] Update backend storage systems
- [ ] Implement protobuf serialization
- [ ] Add compression support
- [ ] Performance benchmarking
- [ ] Integration tests

#### Month 5-6: Rollout
- [ ] Deploy to staging environment
- [ ] A/B testing with production traffic
- [ ] Client library distribution
- [ ] Documentation and examples
- [ ] Production deployment

### Risk Mitigation

1. **Data Integrity**: Comprehensive validation and testing
2. **Performance**: Benchmarking and optimization
3. **Backward Compatibility**: Maintain dual support during transition
4. **Rollback Strategy**: Feature flags and gradual migration

## Language-Specific Code Generation

### Python Integration

```python
# Installation
pip install protobuf grpcio-tools

# Generated usage
from cryptofeed.proto.v1 import market_data_pb2
from cryptofeed.proto.v1 import common_pb2

# Create a trade
trade = market_data_pb2.Trade()
trade.exchange = common_pb2.Exchange.EXCHANGE_BINANCE
trade.symbol.symbol = "BTC-USD"
```

### Go Integration

```go
// go.mod
module github.com/your-org/quant-engine

require (
    github.com/your-org/cryptofeed-schemas/gen/go v1.0.0
    google.golang.org/protobuf v1.28.0
)

// Usage
import pb "github.com/your-org/cryptofeed-schemas/gen/go/cryptofeed/v1"

trade := &pb.Trade{
    Exchange: pb.Exchange_EXCHANGE_BINANCE,
    Symbol: &pb.Symbol{Symbol: "BTC-USD"},
}
```

### Rust Integration

```rust
// Cargo.toml
[dependencies]
prost = "0.11"
cryptofeed-schemas = "1.0"

// Usage
use cryptofeed_schemas::cryptofeed::v1::{Trade, Exchange};

let trade = Trade {
    exchange: Exchange::ExchangeBinance as i32,
    symbol: Some(Symbol { symbol: "BTC-USD".to_string(), ..Default::default() }),
    ..Default::default()
};
```

## Deployment and Versioning

### Schema Versioning Strategy

1. **Semantic Versioning**: Major.Minor.Patch (e.g., v1.2.3)
2. **Breaking Changes**: Increment major version
3. **Backward Compatible**: Increment minor version
4. **Bug Fixes**: Increment patch version

### Deployment Pipeline

```yaml
# .github/workflows/schema-deploy.yml
name: Schema Deployment
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    steps:
      - uses: actions/checkout@v3
      - uses: bufbuild/buf-setup-action@v1
      - name: Push to Registry
        run: buf push --tag ${{ github.ref_name }}
      - name: Generate Code
        run: buf generate
      - name: Publish Packages
        run: |
          # Publish to PyPI, npm, crates.io, etc.
```

### Schema Evolution Examples

```protobuf
// Adding optional field (backward compatible)
message Trade {
  // existing fields...
  optional string venue = 11;  // New optional field
}

// Adding enum value (backward compatible)
enum Exchange {
  // existing values...
  EXCHANGE_NEW_EXCHANGE = 42;  // New exchange
}
```

## Performance Considerations

### Benchmarking Results (Projected)

| Operation | Current (Cython) | Protobuf | Improvement |
|-----------|------------------|----------|-------------|
| Serialization | 100μs | 45μs | 55% faster |
| Deserialization | 120μs | 50μs | 58% faster |
| Memory Usage | 100MB | 65MB | 35% reduction |
| Wire Size | 1.0KB | 0.7KB | 30% smaller |

### Optimization Strategies

1. **Field Ordering**: Place frequently used fields first
2. **Compression**: Use gzip/snappy for large payloads
3. **Batch Processing**: Group events for efficient transmission
4. **Connection Pooling**: Reuse connections for schema registry
5. **Lazy Loading**: Load schemas on demand

### Memory Management

```python
# Efficient protobuf usage
import cryptofeed.proto.v1.market_data_pb2 as md

# Object pooling
trade_pool = []

def get_trade():
    if trade_pool:
        trade = trade_pool.pop()
        trade.Clear()
        return trade
    return md.Trade()

def return_trade(trade):
    trade_pool.append(trade)
```

## Best Practices

### Schema Design

1. **Field Numbering**: Reserve ranges for different purposes
   - 1-15: Core fields (single-byte encoding)
   - 16-100: Extended fields
   - 101+: Optional/experimental fields

2. **Naming Conventions**:
   - Use snake_case for field names
   - Use UPPER_SNAKE_CASE for enum values
   - Prefix enum values with enum name

3. **Documentation**: Comprehensive field documentation

### Code Generation

1. **Automation**: Automate code generation in CI/CD
2. **Validation**: Validate generated code with tests
3. **Distribution**: Use package managers for distribution

### Error Handling

```python
from google.protobuf.message import DecodeError

def safe_parse_trade(data: bytes):
    try:
        trade = md.Trade()
        trade.ParseFromString(data)
        return trade
    except DecodeError as e:
        logger.error(f"Failed to parse trade: {e}")
        return None
```

### Testing Strategy

```python
import pytest
from cryptofeed.adapters import TradeAdapter
from cryptofeed.types import Trade as LegacyTrade

def test_trade_roundtrip():
    # Create legacy trade
    legacy_trade = LegacyTrade(
        exchange='BINANCE',
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.5'),
        price=Decimal('50000.0'),
        timestamp=1234567890.0
    )
    
    # Convert to protobuf
    proto_trade = TradeAdapter.to_protobuf(legacy_trade)
    
    # Convert back to legacy
    converted_trade = TradeAdapter.from_protobuf(proto_trade)
    
    # Verify equality
    assert legacy_trade == converted_trade
```

### Monitoring and Observability

```python
import prometheus_client as prom

# Metrics
schema_registry_hits = prom.Counter('schema_registry_hits_total')
protobuf_serialization_time = prom.Histogram('protobuf_serialization_seconds')
protobuf_size_bytes = prom.Histogram('protobuf_message_size_bytes')

def serialize_with_metrics(message):
    with protobuf_serialization_time.time():
        data = message.SerializeToString()
    protobuf_size_bytes.observe(len(data))
    return data
```

## Implementation Checklist

### Pre-Implementation
- [ ] Buf CLI installed and configured
- [ ] Schema registry access configured
- [ ] Development environment setup
- [ ] Team training on protobuf concepts

### Schema Development
- [ ] Common types defined
- [ ] Market data schemas implemented
- [ ] Account data schemas implemented
- [ ] Event streaming schemas implemented
- [ ] Schema validation and linting passing

### Integration Development
- [ ] Python adapter layer implemented
- [ ] Backend storage updates completed
- [ ] Code generation pipeline established
- [ ] Unit tests written and passing
- [ ] Integration tests implemented

### Deployment Preparation
- [ ] Performance benchmarks completed
- [ ] Documentation updated
- [ ] Migration strategy finalized
- [ ] Rollback procedures documented
- [ ] Monitoring and alerting configured

### Go-Live
- [ ] Staging environment deployment
- [ ] A/B testing results validated
- [ ] Production deployment completed
- [ ] Post-deployment monitoring active
- [ ] Team training completed

## Conclusion

The implementation of normalized data types using Protocol Buffers and Buf Schema Registry will provide:

1. **Cross-Language Compatibility**: Seamless integration across different technology stacks
2. **Schema Evolution**: Safe and backward-compatible schema updates
3. **Performance**: Improved serialization and reduced bandwidth usage
4. **Type Safety**: Strong typing and validation across all systems
5. **Maintainability**: Centralized schema management and automated code generation

This foundation will enable efficient data sharing between cryptofeed, data lakehouse systems, and quantitative trading engines while maintaining high performance and reliability standards.

## Next Steps

1. **Approve Schema Design**: Review and approve the protobuf schemas
2. **Set Up Infrastructure**: Configure Buf Schema Registry and CI/CD pipelines
3. **Begin Implementation**: Start with the adapter layer development
4. **Stakeholder Alignment**: Ensure all teams understand the migration plan
5. **Resource Allocation**: Assign development resources and timeline

For questions or clarifications, please refer to the [Buf documentation](https://docs.buf.build/) or contact the development team.
