# ADR-001: Deprecate Legacy Kafka Backend

**Date:** 2025-11-26 (updated 2025-11-29)

**Status:** accepted

## Context

The original Kafka backend implementation (`cryptofeed.backends.kafka`) was built using the BackendQueue pattern with JSON serialization. While functional, it has several limitations:

- **No Protobuf Support:** Cannot leverage binary serialization for efficiency
- **Limited Partition Strategies:** Only supports basic partitioning
- **Monolithic Design:** All Kafka-related code in a single module
- **No Advanced Features:** Missing exactly-once semantics, message headers, health checks
- **Maintenance Burden:** Updates require touching large, complex files

The market-data-kafka-producer specification delivered a modern, modular implementation with:

- Protobuf serialization support
- Multiple partition strategies (Composite, Symbol, Exchange, RoundRobin)
- Message headers with routing metadata
- Exactly-once semantics
- Comprehensive health checks and monitoring
- Clean separation of concerns

## Decision

We will deprecate the legacy Kafka backend (`cryptofeed.backends.kafka`) in favor of the modern implementation (`cryptofeed.backends.kafka.*`).

**Deprecation Plan:**

1. Emit deprecation warnings for all legacy class usage
2. Provide migration tools and comprehensive documentation
3. Maintain legacy classes for critical bug fixes only (no new features)
4. Monitor usage statistics to determine removal timeline
5. Remove legacy classes after 90 days of zero observed usage

**Timeline:** Q2 2026 (subject to usage statistics) with an interim cutoff: protobuf mode on `KafkaCallback` will be removed after **January 31, 2026**; `KafkaProtobufCallback` is the supported path for protobuf payloads.

## Consequences

### Positive

- **Reduced Maintenance Burden:** Focus on single implementation
- **Modern Features Available:** All users can leverage protobuf, partition strategies, headers
- **Cleaner Codebase:** Remove duplicate functionality and complexity
- **Better Performance:** Protobuf serialization is faster and more compact
- **Enhanced Monitoring:** Modern backend has comprehensive health checks

### Negative

- **Breaking Change:** Users must migrate existing code
- **Migration Effort Required:** Configuration and code updates needed
- **Potential Disruption:** Production systems require careful migration planning
- **Documentation Overhead:** Must maintain both implementations during transition

### Mitigations

- **Automated Migration Tools:** CLI tool to translate configurations
- **Comprehensive Documentation:** Step-by-step migration guide
- **Extended Support Period:** Minimum 6 months before removal
- **Clear Communication:** Multi-channel updates on timeline
- **Rollback Procedures:** Emergency restoration plan if needed

## Alternatives Considered

### Alternative 1: Maintain Both Implementations

**Rationale:** Avoid breaking changes, support both indefinitely

**Rejected Because:**
- Double maintenance burden
- Confusion over which implementation to use
- Legacy code prevents architectural improvements
- Technical debt accumulates

### Alternative 2: Hard Cutover (No Deprecation Period)

**Rationale:** Clean break, remove immediately

**Rejected Because:**
- Too disruptive for production users
- No time for migration planning
- High risk of breaking critical systems
- Community backlash likely

### Alternative 3: Feature Parity First

**Rationale:** Add all modern features to legacy backend before deprecation

**Rejected Because:**
- Defeats purpose of modular redesign
- Extends timeline significantly
- Increases complexity of legacy code
- No clear benefit over direct migration

## References

- Market-Data-Kafka-Producer Spec: `.kiro/specs/market-data-kafka-producer/`
- Migration Guide: `docs/kafka/migration-guide-phase2-maintenance.md`
- Deprecation Timeline: `docs/kafka/deprecation-timeline.md`
- Modern Backend Implementation: `cryptofeed/backends/kafka/`

## Notes

This decision was made after successful completion of market-data-kafka-producer specification and validation that the modern implementation meets all production requirements with superior architecture and features.

Usage statistics will be monitored monthly to ensure migration timeline aligns with community adoption rates. Timeline may be extended if adoption is slower than expected.
