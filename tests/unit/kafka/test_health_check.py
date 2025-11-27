"""Health check tests for Kafka backends."""

from __future__ import annotations

from typing import Dict

import pytest

from cryptofeed.backends.kafka.health import (
    KafkaHealthCheck,
    start_periodic_health_checks,
)
from cryptofeed.backends.kafka.callback import KafkaConfig


class DummyProducer:
    def __init__(self, config: Dict):
        self.config = config
        self._listed = False

    def list_topics(self, timeout=None):
        self._listed = True
        return {}


class FailingProducer:
    def __init__(self, config: Dict):
        raise RuntimeError("boom")


def test_health_check_connectivity_success():
    status = KafkaHealthCheck.check_connectivity(
        ["kafka:9092"], implementation="modern", producer_factory=DummyProducer
    )
    assert status.ok is True
    assert status.error is None
    assert status.details["bootstrap"] == ["kafka:9092"]


def test_health_check_connectivity_failure():
    status = KafkaHealthCheck.check_connectivity(
        ["kafka:9092"], implementation="modern", producer_factory=FailingProducer
    )
    assert status.ok is False
    assert status.error
    assert status.details["bootstrap"] == ["kafka:9092"]


def test_health_check_modern_uses_kafka_config():
    config = KafkaConfig(bootstrap_servers=["k1:9092"], acks="all")
    status = KafkaHealthCheck.check_modern(
        config, producer_factory=DummyProducer, timeout_ms=1000
    )
    assert status.ok is True


@pytest.mark.asyncio
async def test_periodic_health_check_runs_max_times():
    calls = []

    def _check():
        calls.append("ran")
        return KafkaHealthCheck.check_connectivity(
            ["k:9092"], implementation="modern", producer_factory=DummyProducer
        )

    task = await start_periodic_health_checks(
        interval_sec=0.01, check_fn=_check, max_runs=3
    )
    await task
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_periodic_health_check_triggers_alert():
    alerts = []

    def _check():
        return KafkaHealthCheck.check_connectivity(
            ["k:9092"], implementation="modern", producer_factory=FailingProducer
        )

    def _alert(status):
        alerts.append(status)

    task = await start_periodic_health_checks(
        interval_sec=0.01, check_fn=_check, max_runs=1, alert_fn=_alert
    )
    await task
    assert len(alerts) == 1
    assert alerts[0].ok is False
