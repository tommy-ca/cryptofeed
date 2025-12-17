# Kafka Backend Migration (Phase 2 → Maintenance)

This guide helps operators and developers migrate from the legacy `cryptofeed.backends.kafka` implementation and the `cryptofeed.kafka_callback` shim to the modern Phase 2 Kafka backend with maintenance tooling.

## What changed
- Centralized deprecation warnings with actionable guidance.
- Legacy → modern config translator (`cryptofeed.backends.kafka.migration`).
- CLI migrator: `python -m cryptofeed.tools.kafka_config_migrate`.
- Health checks for legacy and modern backends (`cryptofeed.backends.kafka.health`).

## Quick migration steps
1. Translate config:
   ```bash
   python -m cryptofeed.tools.kafka_config_migrate --input legacy_kafka.yaml --output modern_kafka.yaml --pretty
   ```
2. Update imports:
   - Replace `from cryptofeed.kafka_callback import ...` with `from cryptofeed.backends.kafka.callback import ...`.
3. Swap callbacks:
   - Use `KafkaCallback` or `KafkaProtobufCallback`.
4. Run health check (optional):
   ```python
   from cryptofeed.backends.kafka.health import KafkaHealthCheck
   from cryptofeed.backends.kafka.callback import KafkaConfig
   cfg = KafkaConfig.from_yaml("modern_kafka.yaml")
   print(KafkaHealthCheck.check_modern(cfg).as_dict())
   ```

## Legacy deprecation timeline
- Import and class usage now emit `DeprecationWarning` with migration links.
- Shim `cryptofeed.kafka_callback` remains functional but marked for removal in the next major release.
- Legacy classes (`TradeKafka`, `BookKafka`, etc.) are frozen; critical fixes only.
- Planned removal window: **Q2 2026** (after 90 days of zero observed legacy usage per usage tracking).

## Configuration translation details
- Legacy keys mapped automatically (e.g., `topic_prefix` → `topic.prefix`, `partition_strategy` → `partition.strategy`).
- Defaults mirror legacy behavior: `topic.strategy=per_symbol`, `partition.strategy=composite`.
- Unmapped legacy keys are reported via warnings in the CLI and translator result.

## Validation & regression checks
- Migration validation API: `validate_migration(legacy_dict, expected_modern)` returns equivalence report.
- Regression suites:
  - `tests/unit/kafka/test_legacy_equivalence.py`
  - `tests/integration/kafka/test_config_migration_workflow.py`
  - `tests/unit/kafka/test_regression_stability.py`

## Health monitoring
- Connectivity checks for both legacy and modern configs.
- Periodic health loop with optional alert hook:
  ```python
  from cryptofeed.backends.kafka.health import start_periodic_health_checks, KafkaHealthCheck
  cfg = KafkaConfig.from_yaml("modern_kafka.yaml")
  async def main():
      await start_periodic_health_checks(
          10,
          lambda: KafkaHealthCheck.check_modern(cfg),
          alert_fn=lambda status: print("ALERT", status.as_dict()),
      )
  ```

## Troubleshooting
- Warnings about unmapped options: review output from the CLI and add modern equivalents.
- Legacy classes still work but log migration guidance—keep them only during transition.
- Health check failures: verify `bootstrap_servers` reachability and auth settings.

## References
- Modern callback API: `cryptofeed.backends.kafka.callback`
- Protobuf callback: `cryptofeed.backends.kafka.protobuf_callback`
- Config models: `cryptofeed.backends.kafka.config` / `cryptofeed.kafka_config` (shim, deprecated)
- CLI: `cryptofeed.tools.kafka_config_migrate`
