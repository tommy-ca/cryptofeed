# Technical Design Document

## Project Description
Universe screener pipeline extension for Cryptofeed - a standalone package enabling real-time cryptocurrency symbol screening through composable linear filter chains, publishing filtered universes to Kafka topics.

## Design Overview

### Architecture Principles
- **Extension, Not Modification**: Zero changes to `cryptofeed/` directory
- **Simplicity Over Complexity**: Linear chains, stdlib usage, 500-800 LOC target
- **Filters Own State**: No central MetricsAggregator
- **JSON Over Protobuf**: Human-readable, debuggable output
- **NO MOCKS Testing**: Real fixtures, 85%+ coverage

### Simplified Architecture (Post-Technical Review)

```
┌────────────────────────────────────────────────────────────┐
│ cryptofeed_screener/ (Extension Package - 500-800 LOC)    │
│                                                             │
│  Trade/Ticker/OrderBook (from cryptofeed)                  │
│         ↓                                                   │
│  ScreenerPipeline (extends AggregateCallback)              │
│         ↓                                                   │
│  Broadcast to Filters: filter.update(data)                 │
│  [VolumeFilter, MomentumFilter, RSIFilter]                 │
│         ↓                                                   │
│  Periodic Evaluation (every 60s):                          │
│    passed = [s for s in symbols                            │
│              if all(f.should_include(s) for f in filters)] │
│         ↓                                                   │
│  KafkaCallback.write(topic, json.dumps(result))            │
│         ↓                                                   │
│  Kafka: cryptofeed.screener.momentum                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Core cryptofeed: NO MODIFICATIONS (REQ-1, REQ-12)
```

## Component Specifications

### 1. Filter Base Class (REQ-3)

**File:** `cryptofeed_screener/filters/base.py` (~20 LOC)

```python
from abc import ABC, abstractmethod

class Filter(ABC):
    """Base filter for symbol screening.

    Filters maintain their own state and answer: should this symbol be included?
    """

    @abstractmethod
    def update(self, data) -> None:
        """Process incoming market data (Trade, Ticker, OrderBook).

        Args:
            data: Market data object from cryptofeed

        Raises:
            Exception: Implementation-specific errors (caught by pipeline)
        """
        pass

    @abstractmethod
    def should_include(self, symbol: str) -> bool:
        """Return True if symbol passes this filter's criteria.

        Args:
            symbol: Symbol to evaluate (e.g., "BTC-USDT")

        Returns:
            True if symbol passes, False otherwise (including insufficient data)
        """
        pass
```

**Responsibilities:**
- Define contract for all filters
- Enforce two-method interface (update, should_include)
- Document state ownership (filters maintain their own state)

**Design Rationale:**
- No async overhead (filters are CPU-bound, not I/O)
- No separate filter() method (YAGNI - pipeline handles iteration)
- No SymbolMetrics dataclass (filters query their own state)

### 2. VolumeFilter (Stateless) (REQ-4)

**File:** `cryptofeed_screener/filters/volume.py` (~30 LOC)

```python
from dataclasses import dataclass
from decimal import Decimal
from collections import defaultdict
from cryptofeed_screener.filters.base import Filter

@dataclass
class VolumeFilter(Filter):
    """Filter symbols by minimum 24-hour trading volume."""
    min_volume: Decimal
    volumes: dict = None  # symbol -> Decimal

    def __post_init__(self):
        if self.volumes is None:
            self.volumes = defaultdict(Decimal)

    def update(self, data) -> None:
        """Accumulate volume from trades."""
        if hasattr(data, 'symbol') and hasattr(data, 'amount'):
            self.volumes[data.symbol] += Decimal(str(data.amount))

    def should_include(self, symbol: str) -> bool:
        """Return True if volume >= threshold."""
        return self.volumes.get(symbol, Decimal(0)) >= self.min_volume
```

**State Management:**
- `volumes: defaultdict(Decimal)` - cumulative volume per symbol
- Bounded by universe size (O(symbols), typically 10K)

**Performance:**
- `update()`: O(1) - dict lookup + addition
- `should_include()`: O(1) - dict lookup + comparison
- Memory: ~800 bytes per symbol (symbol string + Decimal)

### 3. RSIFilter (Stateful with pandas) (REQ-5)

**File:** `cryptofeed_screener/filters/rsi.py` (~50 LOC)

```python
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
from cryptofeed_screener.filters.base import Filter

@dataclass
class RSIFilter(Filter):
    """Filter symbols by RSI (Relative Strength Index) range."""
    min_rsi: int = 30
    max_rsi: int = 70
    period: int = 14
    price_history: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))

    def update(self, data) -> None:
        """Accumulate price history from trades."""
        if hasattr(data, 'symbol') and hasattr(data, 'price'):
            self.price_history[data.symbol].append(float(data.price))

    def should_include(self, symbol: str) -> bool:
        """Compute RSI and check if within range."""
        prices = self.price_history.get(symbol)
        if not prices or len(prices) < self.period:
            return False  # Insufficient data

        rsi = self._compute_rsi(list(prices))
        return rsi is not None and self.min_rsi <= rsi <= self.max_rsi

    def _compute_rsi(self, prices: list[float]) -> float | None:
        """Compute RSI using pandas (TradingView-compatible)."""
        series = pd.Series(prices)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=self.period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.period, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
```

**State Management:**
- `price_history: defaultdict(lambda: deque(maxlen=100))` - bounded price windows
- deque auto-evicts oldest when maxlen reached (REQ-8)

**Performance:**
- `update()`: O(1) - deque append (constant time eviction)
- `should_include()`: O(period) - pandas .ewm() computation
- Memory: ~800 bytes per symbol (deque of 100 floats)

**TradingView Validation (REQ-5):**
- Uses Wilder's smoothing (ewm with adjust=False)
- Matches TradingView RSI calculation (±0.5% tolerance)
- Test with fixture data from TradingView export

### 4. ScreenerPipeline (REQ-2, REQ-9)

**File:** `cryptofeed_screener/pipeline.py` (~150 LOC)

```python
from cryptofeed.backends.aggregate import AggregateCallback
from cryptofeed.backends.kafka import KafkaCallback
from cryptofeed_screener.filters.base import Filter
import time
import json
import structlog

LOG = structlog.get_logger()

class ScreenerPipeline(AggregateCallback):
    """Linear filter pipeline - processes symbols through sequential filters."""

    def __init__(self, filters: list[Filter], kafka: KafkaCallback, topic: str, emit_interval: int = 60):
        super().__init__(kafka)
        self.filters = filters
        self.kafka = kafka
        self.topic = topic
        self.emit_interval = emit_interval
        self.symbols = set()  # Track all symbols seen
        self._last_emit = 0

    async def __call__(self, data, receipt_timestamp: float):
        """Process incoming market data.

        1. Update all filters with new data (with error boundaries)
        2. Track symbols encountered
        3. Periodically evaluate and emit filtered universes
        """
        # Update all filters (REQ-9: exception boundaries)
        for filter in self.filters:
            try:
                filter.update(data)
            except Exception as e:
                LOG.error("filter.update_failed",
                         filter=filter.__class__.__name__,
                         symbol=getattr(data, 'symbol', None),
                         error=str(e))
                # Continue processing other filters (fault isolation)

        # Track symbols
        if hasattr(data, 'symbol'):
            self.symbols.add(data.symbol)

        # Periodic emit (REQ-2)
        if time.time() - self._last_emit >= self.emit_interval:
            passed = [s for s in self.symbols if all(f.should_include(s) for f in self.filters)]

            result = {
                "timestamp": time.time(),
                "symbols": passed,
                "total": len(passed)
            }

            try:
                await self.kafka.write(self.topic, json.dumps(result).encode())
                LOG.info("screener.emit",
                        topic=self.topic,
                        symbols_in=len(self.symbols),
                        symbols_out=len(passed),
                        filters=len(self.filters))
            except Exception as e:
                LOG.error("kafka.publish_failed",
                         topic=self.topic,
                         error=str(e))
                # Don't raise - continue pipeline operation

            self._last_emit = time.time()
```

**Responsibilities:**
- Extend AggregateCallback (reuse Cryptofeed lifecycle)
- Broadcast data to filters via update()
- Periodic evaluation with all() combination
- Kafka publishing with error handling
- Structured logging (REQ-10)

**Error Handling Strategy (REQ-9):**
- filter.update() failures: Log and continue (isolated failure)
- kafka.write() failures: Log and continue (don't halt pipeline)
- Use structlog for queryable logs

**Performance (REQ-15):**
- `__call__()`: O(filters) - linear broadcast to filters
- Periodic evaluation: O(symbols × filters) - list comprehension with all()
- Target: <10ms for 10K symbols × 10 filters

### 5. ScreenerConfig (Pydantic) (REQ-6)

**File:** `cryptofeed_screener/config.py` (~50 LOC)

```python
from pydantic import BaseModel, Field, field_validator
from cryptofeed_screener.filters import VolumeFilter, RSIFilter, MomentumFilter

class ScreenerConfig(BaseModel):
    """Configuration for a screener pipeline."""
    name: str = Field(..., pattern=r'^[a-z0-9_]+$')
    topic: str = Field(..., pattern=r'^[a-z0-9._-]+$')
    emit_interval: int = Field(default=60, ge=1, le=3600)
    max_symbols: int = Field(default=10000, ge=1, le=100000)
    filters: list[dict]  # Validated by build_filters()

    @field_validator('filters')
    def validate_filters(cls, v):
        if not v or len(v) > 100:
            raise ValueError("filters: 1-100 filters required")
        return v

    def build_filters(self) -> list[Filter]:
        """Build filter instances from config (REQ-6: allowlist registry)."""
        FILTER_REGISTRY = {
            "VolumeFilter": VolumeFilter,
            "RSIFilter": RSIFilter,
            "MomentumFilter": MomentumFilter,
        }

        filters = []
        for cfg in self.filters:
            filter_cls = FILTER_REGISTRY.get(cfg["type"])
            if not filter_cls:
                raise ValueError(f"Unknown filter: {cfg['type']}")
            filters.append(filter_cls(**{k: v for k, v in cfg.items() if k != "type"}))
        return filters

# Load from YAML (REQ-6: yaml.safe_load only)
import yaml
with open("config/screener.yaml") as f:
    config = ScreenerConfig(**yaml.safe_load(f))
```

**Security Hardening (REQ-6, REQ-16):**
- Pydantic validation (strict mode, regex patterns)
- yaml.safe_load() exclusively (prevents code execution)
- Allowlist filter registry (prevents injection)
- Resource limits (max_filters=100, max_symbols=10000)

### 6. CI Boundary Enforcement (REQ-12)

**File:** `.github/workflows/extension-boundary.yml`

```yaml
name: Extension Boundary Enforcement

on: [pull_request]

jobs:
  verify-boundary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Verify no core files modified
        run: |
          # Get changed files
          CHANGED=$(git diff --name-only origin/${{ github.base_ref }}...HEAD)

          # Check for cryptofeed/ modifications
          if echo "$CHANGED" | grep '^cryptofeed/'; then
            echo "ERROR: Extension boundary violated!"
            echo "Modified core files:"
            echo "$CHANGED" | grep '^cryptofeed/'
            exit 1
          fi

          echo "✓ Extension boundary respected - no core files modified"

      - name: Verify cryptofeed dependency
        run: |
          if ! grep -q 'cryptofeed' cryptofeed_screener/setup.py; then
            echo "ERROR: Missing cryptofeed dependency"
            exit 1
          fi
```

## Data Models

### Filter State Structures

**VolumeFilter State:**
```python
{
    "BTC-USDT": Decimal("125000000.50"),
    "ETH-USDT": Decimal("75000000.00"),
    # ... up to 10K symbols
}
```

**RSIFilter State:**
```python
{
    "BTC-USDT": deque([45000.0, 45100.0, ..., 45500.0], maxlen=100),
    "ETH-USDT": deque([3000.0, 3010.0, ..., 3050.0], maxlen=100),
    # ... up to 10K symbols
}
```

### Kafka Message Format (REQ-7: JSON)

```json
{
  "timestamp": 1707566400.5,
  "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
  "total": 3
}
```

## Technology Stack

**Core Dependencies (REQ-Dependencies):**
- Python 3.10+ (modern type hints)
- cryptofeed ≥2.5.0 (AggregateCallback interface)
- pandas ≥1.5.0 (indicator calculations)
- Pydantic ≥2.0.0 (config validation)
- PyYAML ≥6.0 (YAML parsing)
- confluent-kafka ≥2.0.0 (Kafka client, via cryptofeed)
- structlog ≥23.1.0 (structured logging)

**Development Dependencies:**
- pytest ≥7.0.0
- pytest-asyncio ≥0.21.0
- pytest-cov ≥4.0.0
- ruff ≥0.1.0 (linting)
- mypy ≥1.0.0 (type checking)

## Sequence Diagrams

### Pipeline Execution Flow

```
User               FeedHandler         ScreenerPipeline      Filters           KafkaCallback
 |                      |                     |                  |                    |
 |--config.yaml-------->|                     |                  |                    |
 |                      |                     |                  |                    |
 |                      |--new Pipeline------>|                  |                    |
 |                      |<--------------------|                  |                    |
 |                      |                     |                  |                    |
 |                      |--run()------------->|                  |                    |
 |                      |                     |                  |                    |
 |                      |                     |<--Trade data-----|                    |
 |                      |                     |                  |                    |
 |                      |                     |--update(trade)-->|                    |
 |                      |                     |--update(trade)-->|                    |
 |                      |                     |--update(trade)-->|                    |
 |                      |                     |                  |                    |
 |                      |                     |<--60s elapsed----|                    |
 |                      |                     |                  |                    |
 |                      |                     |--should_include(sym)-->|              |
 |                      |                     |<--True/False-----------|              |
 |                      |                     |                  |                    |
 |                      |                     |--write(topic, json)--->|              |
 |                      |                     |                  |    |--Kafka publish|
 |                      |                     |<--------------------|  |              |
```

### Filter Update Flow (Error Boundary)

```
ScreenerPipeline     VolumeFilter       RSIFilter         Log
      |                   |                  |             |
      |--update(trade)--->|                  |             |
      |<------------------|                  |             |
      |                   |                  |             |
      |--update(trade)---------------------->|             |
      |                   |         (exception raised)     |
      |                   |                  |             |
      |--log error---------------------------------------->|
      |<---------------------------------------------------|
      |                   |                  |             |
      |--continue to next filter                           |
```

## Performance Targets (REQ-15)

### Component-Level Benchmarks

| Component | Operation | Target | Measurement |
|-----------|-----------|--------|-------------|
| VolumeFilter | update() | <0.1µs | dict lookup + addition |
| RSIFilter | update() | <1µs | deque append |
| RSIFilter | should_include() | <500µs | pandas .ewm() |
| ScreenerPipeline | broadcast update | <10µs | 10 filters × 1µs |
| ScreenerPipeline | periodic eval | <10ms | 10K symbols × 10 filters |
| JSON serialization | encode | <50µs | 100 symbols |
| Kafka publish | write | <10ms p50 | confluent-kafka |

### Memory Targets (REQ-8)

| Component | Memory per Symbol | 10K Symbols |
|-----------|-------------------|-------------|
| VolumeFilter | ~800 bytes | ~8MB |
| RSIFilter | ~800 bytes | ~8MB |
| Pipeline symbol set | ~100 bytes | ~1MB |
| **Total** | **~1.7KB** | **~17MB** |

Target: <100MB total for 10K symbols (includes Python overhead)

## Testing Strategy (REQ-11)

### Test Coverage Requirements
- Overall: 85%+ (higher than standard 80%)
- Critical paths: 95%+ (filter logic, error boundaries, Kafka publish)

### NO MOCKS Principle

**What to Test With Real Implementations:**
- ✅ Trade/Ticker/OrderBook objects (from cryptofeed fixtures)
- ✅ Kafka integration (docker-compose with Redpanda)
- ✅ Pandas indicator calculations (against TradingView fixtures)
- ✅ Pydantic validation (real YAML configs)
- ✅ deque bounded memory (real data accumulation)

**What NOT to Mock:**
- ❌ Filter.update() or Filter.should_include()
- ❌ KafkaCallback or KafkaProducer
- ❌ Pandas .ewm() computation
- ❌ JSON serialization

### Test Organization

```
tests/
├── unit/
│   ├── filters/
│   │   ├── test_volume_filter.py (~50 LOC)
│   │   ├── test_rsi_filter.py (~80 LOC)
│   │   └── test_macd_filter.py (~80 LOC)
│   ├── test_pipeline.py (~100 LOC)
│   └── test_config.py (~50 LOC)
├── integration/
│   ├── test_kafka_screener.py (~100 LOC)
│   └── conftest.py (docker-compose fixtures)
└── fixtures/
    ├── trades_btc_2024_01.json (Trade fixtures)
    ├── tradingview_rsi_expected.json (validation data)
    └── screener_config.yaml (config examples)
```

### TradingView Validation (REQ-5)

```python
def test_rsi_matches_tradingview():
    """Validate RSI calculation against TradingView export."""
    # Load TradingView fixture (BTC-USD, 2024-01-01 to 2024-01-31)
    with open("fixtures/tradingview_rsi_expected.json") as f:
        expected = json.load(f)

    # Compute RSI using our implementation
    filter = RSIFilter(period=14)
    for trade in load_trades("fixtures/trades_btc_2024_01.json"):
        filter.update(trade)

    rsi = filter._compute_rsi(list(filter.price_history["BTC-USD"]))

    # Validate within tolerance
    assert abs(rsi - expected["rsi_14"]) < 0.005 * expected["rsi_14"]  # ±0.5%
```

## Security Considerations (REQ-16)

### YAML Deserialization (CRITICAL)
```python
# ✅ ALWAYS use safe_load
config = yaml.safe_load(f)

# ❌ NEVER use load or unsafe_load (code execution vulnerability)
# config = yaml.load(f, Loader=yaml.Loader)
```

### Dynamic Filter Loading
```python
# ✅ Allowlist registry pattern
FILTER_REGISTRY = {
    "VolumeFilter": VolumeFilter,
    "RSIFilter": RSIFilter,
}

# ❌ NEVER use eval() or __import__()
# filter_cls = eval(cfg["type"])
```

### Kafka Authentication (REQ-7, REQ-16)
```python
kafka = KafkaCallback(
    bootstrap_servers=["kafka:9092"],
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-512",
    sasl_username=os.getenv("KAFKA_USER"),
    sasl_password=os.getenv("KAFKA_PASSWORD"),
)
```

### Input Validation
```python
# Symbol name validation (prevent path traversal)
SYMBOL_PATTERN = r'^[A-Z0-9/_-]+$'

# Resource limits (prevent DoS)
MAX_FILTERS = 100
MAX_SYMBOLS = 10000
```

## Implementation Roadmap

### Phase 0: Extension Package Setup (Day 1, 4 hours)
- Create package structure
- Setup CI boundary check
- Verify imports from cryptofeed

### Phase 1: Core Pipeline + Basic Filters (Day 2-3, 2 days)
- Implement Filter base class
- Implement VolumeFilter, MomentumFilter
- Implement ScreenerPipeline
- Implement Pydantic config
- Unit tests (~200 LOC)

### Phase 2: Stateful Filters + Kafka (Day 4-5, 2 days)
- Implement RSIFilter with pandas
- Implement MACDFilter with pandas
- Kafka integration (JSON serialization)
- Integration tests (~150 LOC)
- TradingView validation

### Phase 3: Documentation + Observability (Day 6, 1 day)
- User guide + examples
- Structured logging (structlog)
- API reference

**Total: 6 days, 500-800 LOC**

## Traceability Matrix

| Design Component | Requirements Covered |
|------------------|---------------------|
| Filter base class | REQ-3 (Filter Base Contract) |
| VolumeFilter | REQ-4 (Stateless Volume Filter) |
| RSIFilter | REQ-5 (Stateful RSI Filter) |
| ScreenerPipeline | REQ-2 (Linear Pipeline), REQ-9 (Error Handling), REQ-15 (Performance) |
| ScreenerConfig | REQ-6 (Configuration), REQ-16 (Security) |
| CI workflow | REQ-12 (Extension Boundary) |
| Structured logging | REQ-10 (Observability) |
| Test strategy | REQ-11 (Test Coverage) |
| deque usage | REQ-8 (Bounded Memory) |
| Kafka integration | REQ-7 (Kafka Output) |

## Design Decisions

### 1. Linear Chains (No DAG)
**Decision:** Use simple list of filters, no branching
**Rationale:** 95% of use cases don't need branching, YAGNI principle
**Trade-off:** Users run multiple pipelines for multiple strategies

### 2. Filters Own State (No MetricsAggregator)
**Decision:** Each filter maintains only the state it needs
**Rationale:** Single responsibility, no state duplication
**Trade-off:** Slight memory overhead if multiple filters need same data

### 3. JSON Serialization (Not Protobuf)
**Decision:** Use json.dumps() for Kafka messages
**Rationale:** Human-readable debugging, no schema evolution complexity
**Trade-off:** ~30% larger messages (acceptable for low-volume screener output)

### 4. pandas for Indicators (Not Custom Classes)
**Decision:** Use pandas .ewm() for RSI/MACD
**Rationale:** Battle-tested, TradingView-compatible, 5 lines per indicator
**Trade-off:** pandas dependency (acceptable - widely used)

### 5. stdlib deque (Not Custom RollingWindow)
**Decision:** Use collections.deque(maxlen=N)
**Rationale:** Stdlib, bounded memory, zero bugs
**Trade-off:** None - deque is optimal for this use case

## Next Steps

After design approval:
1. Run `/kiro:spec-tasks universe-screener-pipeline -y` to generate implementation tasks
2. Begin Phase 0 (extension package setup)
3. Follow TDD workflow with NO MOCKS principle
