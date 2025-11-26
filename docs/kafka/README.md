# Kafka Backend Documentation

This directory contains comprehensive documentation for Cryptofeed's Kafka backend migration and maintenance.

## Quick Links

### For Users

- **[Migration Guide](migration-guide-phase2-maintenance.md)** - Step-by-step guide to migrate from legacy to modern backend
- **[Deprecation Timeline](deprecation-timeline.md)** - Official timeline for legacy backend deprecation
- **[Communication Plan](communication-plan.md)** - How timeline updates are communicated

### For Operators

- **[Progress Reports](progress-reports/)** - Monthly migration progress tracking
- **[Decision Log](decisions/)** - Architectural Decision Records (ADRs) for Kafka backend evolution

## Migration Timeline

**Current Phase:** Documentation (90% complete)
**Target Completion:** Q2 2026
**Status:** On track

See [deprecation-timeline.md](deprecation-timeline.md) for complete timeline.

## Getting Started

### New Users - Use Modern Backend

```python
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig

config = KafkaConfig(bootstrap_servers="localhost:9092")
callback = KafkaCallback(config)
```

### Existing Users - Migrate from Legacy

1. Read [migration-guide-phase2-maintenance.md](migration-guide-phase2-maintenance.md)
2. Run: `python -m cryptofeed.tools.kafka_config_migrate --input legacy.yaml --output modern.yaml`
3. Update imports to `cryptofeed.backends.kafka.callback`

## References

- Decision Log: [decisions/](decisions/)
- Progress Reports: [progress-reports/](progress-reports/)
- Specification: `.kiro/specs/kafka-backend-maintenance/`

---

**Last Updated:** 2025-11-26
