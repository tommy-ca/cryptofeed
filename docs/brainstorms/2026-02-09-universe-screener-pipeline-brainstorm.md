# Universe Screener Pipeline - Brainstorm

**Date:** 2026-02-09
**Status:** Brainstorming Complete
**Related Plan:** `/home/tommyk/.claude/plans/starry-wishing-abelson.md`

---

## What We're Building

A **pluggable filter pipeline architecture** for screening cryptocurrency symbols based on real-time market data. Users compose linear or branching filter chains to build custom screening strategies (e.g., high volume → momentum → low RSI). Filtered universes are published to Kafka topics for downstream strategy engines and dashboards to consume.

### Core Value Proposition

- **Composability:** Build complex screening logic from simple, reusable filter components
- **Flexibility:** Support diverse strategies without hardcoding specific screener logic
- **Efficiency:** Pre-filter reduces bandwidth; downstream filters only process qualifying symbols
- **Real-time:** Continuous metric computation and universe updates from streaming data

### Target Use Cases

1. **Pre-trade universe construction:** Filter 10K symbols → 500 candidates based on volume/liquidity
2. **Multi-strategy screening:** Branch pipeline into "momentum", "mean-reversion", "volatility-breakout" universes
3. **Dynamic watchlists:** RSI oversold → recent volume spike → publish to alert topic
4. **Quant research:** Test different filter combinations by swapping pipeline stages

---

## Engineering Principles

### SOLID Principles
- **Single Responsibility**: Each filter has one clear purpose (volume filtering, RSI calculation, etc.)
- **Open/Closed**: Pipeline is open for extension (new filter types) but closed for modification (core pipeline engine remains stable)
- **Liskov Substitution**: All filters implement the same `Filter` interface and are interchangeable
- **Interface Segregation**: Filters only depend on the interfaces they use (stateless filters don't need `update()`)
- **Dependency Inversion**: Pipeline depends on `Filter` abstraction, not concrete filter implementations

### KISS (Keep It Simple, Stupid)
- Simple linear chains before complex branching
- Each filter does one thing well
- YAML configuration over complex programmatic APIs
- Avoid premature optimization - profile before optimizing

### DRY (Don't Repeat Yourself)
- Reusable filter components across pipelines
- Shared `RollingWindow` and `MetricsAggregator` utilities
- Common protobuf schemas for all screener outputs
- Template-based pipeline configurations

### YAGNI (You Aren't Gonna Need It)
- No DAG merging (just branching) - defer until needed
- No filter state persistence - rebuild on restart is simpler
- No auto-optimization of filter ordering - user controls
- No built-in backtesting - that's consumer responsibility

### Extension, Not Modification (CRITICAL)
**Core Constraint:** Build the universe screener **on top of cryptofeed**, not inside it. Do not modify core cryptofeed code.

**Implementation Strategy:**
- **Separate package**: `cryptofeed_screener/` as an extension package (or `cryptofeed/extensions/screener/`)
- **Import, don't modify**: Use existing types (`Trade`, `Ticker`, `OrderBook`) without changing them
- **Extend patterns**: Implement screener callbacks extending `AggregateCallback` (existing pattern)
- **Reuse infrastructure**: Use existing Kafka backend, protobuf serialization, feed subscription
- **Plugin architecture**: Register filters dynamically without touching core feed handler

**Benefits:**
- No risk of breaking existing cryptofeed functionality
- Independent versioning and release cycle
- Can be developed/tested in isolation
- Easier to maintain and review
- Follows Open/Closed Principle at the package level

**Integration Points:**
```python
# User code - no core changes required
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Binance
from cryptofeed.defines import TRADES, TICKER
from cryptofeed_screener import ScreenerPipeline, VolumeFilter, RSIFilter

pipeline = ScreenerPipeline.from_yaml("config/screener.yaml")
fh = FeedHandler()
fh.add_feed(
    Binance,
    channels=[TRADES, TICKER],
    symbols=pipeline.get_input_symbols(),
    callbacks={TRADES: pipeline, TICKER: pipeline}
)
fh.run()
```

**Enforcement:**
- **Code Review**: All PRs must demonstrate zero changes to `cryptofeed/` directory
- **CI Check**: Automated test fails if any core file is modified
- **Testing**: Screener tests run against installed cryptofeed package, not source
- **Documentation**: Clearly mark public extension points in cryptofeed docs

---

## Why This Approach

### Selected Architecture: Filter Pipeline (Approach 3)

**Rationale:**
- User needs **hybrid approach** (pre-filter + real-time) with flexibility for multiple strategies
- Pipeline naturally supports this: early filters = pre-filter stage, later filters = real-time refinement
- **Incremental phasing** maps well: Phase 1 = basic filters, Phase 2 = indicator filters, Phase 3 = branching
- Enables A/B testing of screening strategies without code changes

**Trade-offs Accepted:**
- More complex than single-purpose screener (violates YAGNI slightly)
- Steeper learning curve for simple use cases
- Requires more testing (filter combinations)

**Mitigations:**
- Provide pre-built pipeline templates for common patterns (high-volume-momentum, oversold-bounce)
- Keep individual filters simple and well-documented
- YAML configuration hides complexity for most users

---

## Key Decisions

### 1. Pipeline Structure: Linear + Branching

```
UniverseInput (10K symbols)
    ↓
VolumeFilter (min=1M) → 500 symbols
    ↓
    ├─→ Branch A: MomentumFilter → RSIFilter → Kafka: screener.momentum
    │
    └─→ Branch B: VolatilityFilter → MACDFilter → Kafka: screener.volatility
```

- **Linear chains** for simple flows
- **Branching** enables parallel strategies from same pre-filtered set
- **No DAG complexity** (no merging, no cycles)

### 2. Filter Types

**Stateless Filters** (operate on current data only):
- `VolumeFilter` - 24h volume threshold
- `PriceChangeFilter` - % change over time window
- `SpreadFilter` - bid-ask spread threshold

**Stateful Filters** (maintain rolling windows/indicators):
- `RSIFilter` - Relative Strength Index bounds
- `MACDFilter` - MACD signal crossovers
- `EMAFilter` - EMA crossover detection
- `ATRFilter` - Average True Range volatility

**Base Contract:**
```python
class Filter(ABC):
    @abstractmethod
    async def filter(self, symbols: List[SymbolMetrics]) -> List[SymbolMetrics]:
        """Return symbols passing this filter's criteria."""
        pass

    async def update(self, data) -> None:
        """Update internal state with new market data (trades, tickers)."""
        pass
```

### 3. Input Universe: Static Symbol List

- Pipeline starts with a **pre-defined symbol list** (from config or UniverseBuilder)
- No dynamic subscription management (scope creep)
- UniverseBuilder can run separately (cron job) to refresh the input list periodically
- Keeps pipeline focused on filtering, not symbol discovery

### 4. Output: Kafka Topics Per Branch

- Each pipeline **output node** publishes to a dedicated Kafka topic
- Topic naming: `cryptofeed.screener.{pipeline_name}.{branch_id}`
- Message format: Protobuf `ScreenerResult` with filtered symbol list + metrics
- Emit interval: Configurable per output (default 60s)

### 5. Configuration: YAML-Driven

```yaml
screener:
  pipelines:
    - name: "multi_strategy"
      input_symbols: "config/universe.yaml"  # or UniverseBuilder output
      emit_interval: 60

      filters:
        - id: "volume_prefilter"
          type: VolumeFilter
          min_volume_24h: 1000000

        - id: "branch_momentum"
          type: MomentumFilter
          parent: "volume_prefilter"
          threshold: 0.05
          output_topic: "cryptofeed.screener.momentum"

        - id: "branch_volatility"
          type: VolatilityFilter
          parent: "volume_prefilter"
          min_atr: 2.0
          output_topic: "cryptofeed.screener.volatility"
```

### 6. Package Structure: Extension Architecture

**Critical Decision:** Build as an **extension package**, not core modification.

```
cryptofeed_screener/              # Separate package (or cryptofeed/extensions/screener/)
├── __init__.py                   # Public API
├── pipeline.py                   # ScreenerPipeline (extends AggregateCallback)
├── filters/
│   ├── __init__.py
│   ├── base.py                   # Filter abstract base class
│   ├── stateless.py              # Volume, PriceChange, Spread filters
│   └── stateful.py               # RSI, MACD, EMA, ATR filters
├── aggregators/
│   ├── rolling_window.py         # Time-series data structure
│   └── metrics.py                # MetricsAggregator
├── backends/
│   └── kafka.py                  # KafkaScreenerBackend (uses core KafkaCallback)
├── config/
│   └── parser.py                 # YAML config parsing
└── tests/
    ├── unit/
    └── integration/

# Core cryptofeed - NO MODIFICATIONS
cryptofeed/
├── types.pyx                     # UNCHANGED - reused by screener
├── backends/aggregate.py         # UNCHANGED - screener extends AggregateCallback
├── backends/kafka/               # UNCHANGED - screener reuses infrastructure
└── ...
```

**Integration Pattern:**
- Screener imports from cryptofeed: `from cryptofeed.backends.aggregate import AggregateCallback`
- No modifications to cryptofeed source code
- Screener installed as separate package: `pip install cryptofeed cryptofeed-screener`
- Or as an optional extension: `pip install cryptofeed[screener]`

---

## Implementation Phases

### Phase 0: Extension Package Setup (Week 1)
- [ ] Create `cryptofeed_screener/` package structure (separate from core)
- [ ] Define extension's `setup.py` with cryptofeed as dependency
- [ ] Verify integration: import cryptofeed types without modifications
- [ ] Setup testing infrastructure (pytest, fixtures)
- [ ] Document extension philosophy in README

**Deliverable:** Standalone extension package that imports cryptofeed

### Phase 1: Core Pipeline Engine (Week 1-2)
- [ ] `Filter` base class with `filter()` and `update()` contracts
- [ ] `FilterPipeline` orchestrator (builds DAG, routes data)
- [ ] `ScreenerPipeline` callback implementing `AggregateCallback` interface (extends, doesn't modify)
- [ ] YAML configuration parser
- [ ] Basic stateless filters: `VolumeFilter`, `PriceChangeFilter`, `SpreadFilter`
- [ ] Unit tests for pipeline routing and filter chaining

**Deliverable:** Can build simple linear pipelines with basic filters (zero core modifications)

### Phase 2: Stateful Filters (Week 2-3)
- [ ] `RollingWindow` helper for time-series data
- [ ] `MetricsAggregator` to feed filters with computed metrics
- [ ] Indicator filters: `RSIFilter`, `EMAFilter`, `MACDFilter`, `ATRFilter`
- [ ] Unit tests for indicators against known values

**Deliverable:** Can build pipelines with technical indicator filters

### Phase 3: Branching + Kafka Output (Week 3-4)
- [ ] Branching logic in `FilterPipeline`
- [ ] `KafkaScreenerBackend` for publishing filtered universes
- [ ] Protobuf schema: `ScreenerResult` message
- [ ] Integration tests with docker-compose Kafka

**Deliverable:** Multi-branch pipelines publishing to multiple Kafka topics

### Phase 4: Documentation + Templates (Week 4-5)
- [ ] User guide with pipeline configuration examples
- [ ] Pre-built templates: `high-volume-momentum.yaml`, `oversold-bounce.yaml`
- [ ] Example consumer (Python script reading screener topics)
- [ ] Architecture diagrams

---

## Open Questions

### 1. Filter Ordering Optimization
**Question:** Should the pipeline automatically reorder filters for efficiency (e.g., move cheap filters first)?
**Decision:** Not in Phase 1. User controls order explicitly via config. Can add auto-optimization in Phase 4 if needed.

### 2. Filter State Persistence
**Question:** Should filter state (RSI history, EMA values) persist across restarts?
**Decision:** No. Filters rebuild state from scratch on restart. This is simpler and aligns with stream processing philosophy (stateless recovery). If needed, can add optional state backends later.

### 3. UniverseBuilder Integration
**Question:** Should UniverseBuilder be part of this spec or a separate spec?
**Decision:** Separate spec. UniverseBuilder outputs a static symbol list file. This spec consumes that file as input. Keeps concerns separated.

### 4. Multi-Exchange Pipelines
**Question:** Can a pipeline filter symbols across multiple exchanges?
**Decision:** Yes. Input universe can include `{exchange, symbol}` tuples. Filters operate on `SymbolMetrics` which includes `exchange` field. No special handling needed.

### 5. Filter Parameter Tuning
**Question:** How do users discover optimal filter parameters (e.g., RSI threshold)?
**Decision:** Out of scope for Phase 1. Document in user guide that parameter tuning requires backtesting (downstream consumer responsibility). Could add a "pipeline simulator" tool in Phase 5.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking extension boundary** | **Critical** | **Strict code review: all PRs must show zero core file modifications. CI check: fail if cryptofeed/ files touched.** |
| Pipeline complexity overwhelming simple users | High | Provide 5-10 pre-built templates for common patterns |
| Memory growth with many stateful filters | Medium | Bounded rolling windows, document memory requirements |
| Configuration errors hard to debug | Medium | Strict YAML validation, clear error messages with line numbers |
| Filter combination explosion in testing | Low | Focus tests on filter contracts, not all combinations |
| Performance bottleneck in pipeline routing | Low | Profile early, optimize hot paths (likely just dict lookups) |
| Extension package versioning conflicts with core | Medium | Pin cryptofeed version compatibility in setup.py, semantic versioning |

---

## Success Criteria

### Architectural
- [ ] **Zero modifications to core cryptofeed code** (CRITICAL - must pass)
- [ ] Extension package installs independently: `pip install cryptofeed-screener`
- [ ] All screener functionality imports from cryptofeed, doesn't modify it
- [ ] Can upgrade cryptofeed version without breaking screener (semver compatibility)

### Functional
- [ ] Users can build linear filter chains from YAML config
- [ ] Users can build branching pipelines (1 input → N outputs)
- [ ] Stateful filters (RSI, MACD) compute correctly from streaming data
- [ ] Filtered universes publish to Kafka with <1s latency

### Non-Functional
- [ ] Pipeline processes ≥10K symbols with <1GB memory
- [ ] Individual filter overhead <1ms per symbol
- [ ] Config validation catches 95%+ of user errors
- [ ] Documentation enables users to build custom filters

### Acceptance
- [ ] Can replicate TradingView's "Top Volume Gainers" screener
- [ ] Can build a "Oversold Bounce" strategy pipeline
- [ ] Downstream consumer can read Kafka topics and execute trades

---

## Next Steps

1. **Review and refine** this brainstorm with `/compound-engineering:document-review`
2. **Create specification** with `/kiro:spec-init "universe-screener-pipeline"`
3. **Generate requirements** with `/kiro:spec-requirements universe-screener-pipeline`
4. **Design architecture** with `/kiro:spec-design universe-screener-pipeline`
5. **Implement TDD** with `/kiro:spec-impl universe-screener-pipeline`

---

## Related Documents

- Initial Plan: `/home/tommyk/.claude/plans/starry-wishing-abelson.md`
- Spec Location: `.kiro/specs/universe-screener-pipeline/` (to be created)
- CLAUDE.md Guidance: Ingestion layer boundary, spec-driven development, TDD
- Related Specs: `market-data-kafka-producer` (Kafka infrastructure), `protobuf-callback-serialization` (message format)
