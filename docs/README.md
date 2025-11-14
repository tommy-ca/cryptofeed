# Cryptofeed Documentation

Welcome to the Cryptofeed documentation. This guide will help you navigate all available documentation.

## Quick Navigation

### For Users Getting Started
- **[Core Documentation](core/)** - Essential guides for using Cryptofeed
  - Configuration, data types, callbacks, authentication channels, performance tuning

### For Developers Integrating
- **[Architecture](architecture/)** - System design and integration patterns
- **[Exchange Integration](exchanges/)** - Adding new exchanges and exchange-specific guides
- **[Consumer Integration](consumers/)** - Building downstream consumers and storage solutions

### For Operations & Deployment
- **[Kafka Setup & Operations](kafka/)** - Kafka producer configuration, tuning, and troubleshooting
- **[Proxy Configuration](proxy/)** - HTTP/SOCKS proxy setup and usage
- **[Operations Runbooks](operations/)** - Migration procedures and operational guides
- **[Monitoring & Alerts](monitoring/)** - Prometheus, Grafana dashboards, and alerting

### For Data & Schema
- **[Schemas & Data Types](schemas/)** - Protobuf schema documentation and examples
- **[Specifications](specs/)** - Feature specifications and implementation details

### For Analysis & Research
- **[Technical Analysis](analysis/)** - Architectural analysis, exploration reports, and research findings
- **[Archive](archive/)** - Completed execution reports and historical documentation

### For Testing
- **[Testing & Verification](testing/)** - E2E testing procedures and test planning

### For Benchmarks
- **[Performance Benchmarks](benchmarks/)** - Performance metrics and optimization results

---

## Documentation Organization

```
docs/
├── core/                 # User-facing documentation
├── architecture/         # System design and patterns
├── exchanges/           # Exchange integration guides
├── kafka/               # Kafka operations
├── proxy/               # Proxy system documentation
├── schemas/             # Data schema and protobuf docs
├── consumers/           # Consumer integration guides
├── monitoring/          # Monitoring and alerting
├── operations/          # Operational runbooks
├── testing/             # Testing documentation
├── benchmarks/          # Performance benchmarks
├── investigations/      # Technical investigations
├── analysis/            # Research and analysis
├── specs/               # Specifications
└── archive/             # Historical reports
```

---

## Finding What You Need

### "How do I..."

| Task | Location |
|------|----------|
| Configure Cryptofeed? | [Configuration Guide](core/configuration.md) |
| Connect to an exchange? | [Exchange Integration](exchanges/) |
| Set up Kafka? | [Kafka Documentation](kafka/) |
| Use HTTP proxies? | [Proxy Guide](proxy/user-guide.md) |
| Build a consumer? | [Consumer Integration](consumers/integration-guide.md) |
| Understand the data flow? | [Architecture: Data Flow](architecture/data-flow.md) |
| Migrate from CCXT? | [Consumer Migration Guide](consumers/migration-guide.md) |
| View performance metrics? | [Kafka Performance Tuning](kafka/producer-tuning.md) |
| Set up monitoring? | [Monitoring Guide](monitoring/README.md) |
| Add a new exchange? | [Exchange Integration](exchanges/adding-new-exchange.md) |

---

## Core User Documentation

* **[High Level Overview](core/high-level.md)** - System overview and core concepts
* **[Data Types](core/data-types.md)** - Cryptofeed data structure and types
* **[Callbacks](core/callbacks.md)** - Implementing data callbacks
* **[Configuration](core/configuration.md)** - Configuration and setup
* **[Authenticated Channels](core/authenticated-channels.md)** - Private data channels
* **[Performance Considerations](core/performance.md)** - Performance tuning and optimization
* **[REST Endpoints](core/rest-endpoints.md)** - REST API endpoints
* **[Data Integrity for Orderbooks](core/orderbook-validation.md)** - Orderbook validation

---

## Enterprise Features

* **[Transparent Proxy System](proxy/)** - HTTP/SOCKS proxy support for all exchanges
* **[Proxy Testing Guide](proxy/testing.md)** - Test suites and integration verification
* **[Kafka Producer](kafka/)** - High-performance Kafka integration for market data

---

## Specifications & Development

### Specification Status
* **[Specification Dashboard](specs/SPEC_STATUS.md)** - Overview of all specifications
* **[Completed Specifications](specs/completed/)** - Finished and released features
* **[Active Development](../.kiro/specs/)** - Current work-in-progress specs

### Key Specifications
* **[Normalized Data Schema](schemas/)** - Baseline schemas, governance, and monitoring
* **[Protobuf Serialization](architecture/protobuf-serialization.md)** - Binary message format
* **[Kafka Producer Integration](architecture/kafka-producer.md)** - Event streaming setup

---

## Exchange Integration

* **[Adding a New Exchange](exchanges/adding-new-exchange.md)** - Implementation guide
* **[Native Exchange Blueprint](exchanges/native-exchange-blueprint.md)** - Pattern for native integrations
* **[CCXT Generic Integration](exchanges/ccxt-generic.md)** - Using CCXT for exchanges
* **[Backpack Exchange](exchanges/backpack.md)** - Native Backpack integration

---

## Operations & Deployment

### Kafka Operations
* **[Kafka Configuration](kafka/config-translation-examples.md)** - Config setup
* **[Producer Tuning](kafka/producer-tuning.md)** - Performance optimization
* **[Troubleshooting](kafka/troubleshooting.md)** - Common issues and solutions
* **[Schema Management](kafka/schema-registry-setup.md)** - Schema registry setup
* **[Rollback Procedures](kafka/rollback-procedures.md)** - Emergency procedures

### Operations & Migrations
* **[Migration Runbooks](operations/runbooks/)** - Migration procedures
* **[Operational Guides](operations/)** - Deployment and management

### Monitoring
* **[Monitoring Setup](monitoring/README.md)** - Prometheus and Grafana configuration
* **[Alert Rules](monitoring/README.md)** - Alerting configuration

---

## Consumer Development

* **[Consumer Integration Guide](consumers/integration-guide.md)** - Building consumers
* **[Consumer Templates](consumers/templates/)** - Code templates for common patterns
* **[Migration Guide](consumers/migration-guide.md)** - Moving to Cryptofeed

---

## Testing & Quality

* **[E2E Testing](testing/e2e-testing/)** - End-to-end test procedures
* **[Testing Planning](testing/e2e-testing/planning/)** - Test plans and coordination
* **[Test Results](testing/e2e-testing/results/)** - Execution results and analysis

---

## Analysis & Research

### Technical Analysis
* **[Codebase Exploration](analysis/codebase-exploration/)** - Codebase structure and patterns
* **[Architecture Analysis](analysis/architecture-analysis/)** - System design findings
* **[Market Data Analysis](analysis/market-data-analysis/)** - Data schema and mapping analysis
* **[Protobuf Analysis](analysis/protobuf-analysis/)** - Serialization research

### Historical Reports
* **[Exploration Reports](archive/explorations/)** - Research and discovery documentation
* **[Execution Reports](archive/execution-reports/)** - Completion reports and deliverables

---

## Technical Investigations

* **[Issues & Investigations](investigations/)** - Technical deep-dives and problem analysis

---

## Performance Benchmarks

* **[Performance Metrics](benchmarks/)** - Throughput, latency, and resource usage benchmarks

---

## Contributing to Documentation

Documentation should follow these guidelines:

1. **Core Documentation** (`core/`) - User-facing guides and how-tos
2. **Architecture** (`architecture/`) - System design and technical decisions
3. **Analysis** (`analysis/`) - Research, exploration, and validation reports
4. **Archive** (`archive/`) - Completed work and historical reports

For more information, see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## External Resources

* [Repository](https://github.com/cryptofeed-project/cryptofeed)
* [Issue Tracker](https://github.com/cryptofeed-project/cryptofeed/issues)
* [Changelog](../CHANGES.md)
* [License](../LICENSE)

---

*Last updated: November 14, 2025*
