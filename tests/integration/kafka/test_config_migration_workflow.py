"""Integration test for config migration workflow (legacy -> modern -> KafkaCallback)."""

from __future__ import annotations


import pytest

from cryptofeed.backends.kafka.migration import translate_legacy_config, validate_migration
from cryptofeed.kafka_callback import KafkaCallback
from cryptofeed.backends.kafka.callback import KafkaConfig


class DummyKafkaProducer:
    """Lightweight dummy to avoid real Kafka."""

    def __init__(self, bootstrap_servers=None, **kwargs):
        self.bootstrap_servers = bootstrap_servers
        self.started = False

    def connect(self):
        self.started = True

    def list_topics(self, timeout=None):
        return {}

    def produce(self, *args, **kwargs):
        return None

    def flush(self, timeout=None):
        return 0


@pytest.mark.integration
def test_legacy_to_modern_to_callback_workflow(monkeypatch):
    legacy = {
        "bootstrap_servers": ["kafka:9092"],
        "topic_prefix": "staging",
        "acks": "1",
        "compression_type": "snappy",
        "partition_strategy": "round_robin",
    }

    # Translate legacy to modern config
    translated = translate_legacy_config(legacy)
    modern: KafkaConfig = translated.modern_config

    # Validate equivalence
    report = validate_migration(legacy, modern)
    assert report.is_equivalent
    assert report.differences == []

    # Instantiate KafkaCallback with translated config using dummy producer
    monkeypatch.setenv("KAFKA_BOOTSTRAP", "true")
    callback = KafkaCallback(
        kafka_config=modern,
        producer_factory=DummyKafkaProducer,
    )
    assert callback.bootstrap_servers == ["kafka:9092"]
    assert callback.topic_config.prefix == "staging"
    assert callback.partition_config.strategy == "round_robin"

    # Ensure dummy producer connected
    assert callback._producer is not None
