"""
Example demonstrating integrated maintenance components for Kafka backend.

This example shows how to use the MaintenanceCoordinator for unified
management of all Kafka backend maintenance operations:
- Tracking legacy usage with integrated monitoring
- Performing configuration migration with documentation updates
- Executing health checks with alerting

Implements task 6.1 integration capabilities.

Note: This example requires a working Kafka cluster. For testing without Kafka,
      run the unit tests in tests/unit/kafka/test_maintenance_integration.py
"""

import logging
import sys

# Add parent directory to path to allow direct execution
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptofeed.backends.kafka.maintenance.integration import (
    MaintenanceCoordinator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

LOG = logging.getLogger(__name__)


def main():
    """Demonstrate integrated maintenance operations."""

    # Initialize the unified maintenance coordinator
    coordinator = MaintenanceCoordinator()

    LOG.info("=" * 80)
    LOG.info("Kafka Backend Maintenance Integration Demo")
    LOG.info("=" * 80)

    # 1. Track legacy usage across integrated systems
    LOG.info("\n1. Tracking legacy component usage...")
    legacy_result = coordinator.handle_legacy_usage(
        component="TradeKafka",
        context={
            "exchange": "binance",
            "symbol": "BTC-USD",
            "reason": "migration_demo"
        }
    )
    LOG.info(f"   - Success: {legacy_result.success}")
    LOG.info(f"   - Events captured: {len(legacy_result.events)}")
    for event in legacy_result.events:
        LOG.info(f"     * {event.event_type}: {event.message}")

    # 2. Perform configuration migration with documentation
    LOG.info("\n2. Performing configuration migration...")
    legacy_config = {
        "bootstrap_servers": ["localhost:9092"],
        "topic_prefix": "crypto",
        "partition_strategy": "symbol",
        "compression_type": "gzip",
    }

    migration_result = coordinator.handle_configuration_migration(legacy_config)
    LOG.info(f"   - Success: {migration_result.success}")
    LOG.info(f"   - Events captured: {len(migration_result.events)}")
    if migration_result.warnings:
        LOG.info(f"   - Warnings: {migration_result.warnings}")
    for event in migration_result.events:
        LOG.info(f"     * {event.event_type}: {event.message}")

    # 3. Execute health check with alerting
    LOG.info("\n3. Executing health check with alerting...")
    health_result = coordinator.handle_health_check(
        bootstrap_servers=["localhost:9092"],
        implementation="modern"
    )
    LOG.info(f"   - Success: {health_result.success}")
    LOG.info(f"   - Events captured: {len(health_result.events)}")
    for event in health_result.events:
        LOG.info(f"     * {event.severity.upper()}: {event.message}")

    # 4. Get comprehensive system status
    LOG.info("\n4. Retrieving comprehensive system status...")
    status = coordinator.get_system_status()

    LOG.info("\n   Migration Analytics:")
    analytics = status["migration_analytics"]
    LOG.info(f"   - Migration percentage: {analytics['migration_percentage']:.1f}%")
    LOG.info(f"   - Timeline recommendation: {analytics['timeline_recommendation'].reason}")

    LOG.info("\n   Timeline Status:")
    timeline = status["timeline_status"]
    for name, milestone in timeline["milestones"].items():
        LOG.info(f"   - {name}: {milestone['status']} ({milestone['completion']}%)")

    LOG.info(f"\n   Communication history: {status['communication_history']} events")

    # 5. Demonstrate component bridge usage
    LOG.info("\n5. Using individual component bridges...")

    # Deprecation-Monitoring Bridge
    LOG.info("\n   a) Deprecation-Monitoring Bridge:")
    deprecation_analytics = coordinator.deprecation_monitoring.get_migration_analytics()
    LOG.info(f"      - Legacy usage count: {deprecation_analytics['legacy_usage']['total_legacy_usage']}")
    LOG.info(f"      - Modern usage count: {deprecation_analytics['modern_usage']['total_modern_usage']}")

    # Health-Alerting Bridge
    LOG.info("\n   b) Health-Alerting Bridge:")
    LOG.info(f"      - Alert threshold: {coordinator.health_alerting.alert_threshold_ms}ms")
    LOG.info(f"      - Communication channels: {coordinator.health_alerting.comm_system.get_registered_channels()}")

    LOG.info("\n" + "=" * 80)
    LOG.info("Demo completed successfully!")
    LOG.info("=" * 80)


if __name__ == "__main__":
    main()
