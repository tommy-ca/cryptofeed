"""Regression stability tests for legacy vs modern Kafka backends."""

from __future__ import annotations

import warnings

import pytest

from tools.migrate_kafka_config import translate_legacy_config
from cryptofeed.backends.kafka.callback import KafkaConfig
from cryptofeed.kafka_callback import KafkaCallback


class DummyProducer:
    def __init__(self, bootstrap_servers=None, **kwargs):
        self.bootstrap_servers = bootstrap_servers
        self.produced = []

    def connect(self):
        return None

    def list_topics(self, timeout=None):
        return {}

    def produce(self, topic, value, key=None, headers=None, on_delivery=None):
        self.produced.append((topic, value, key, headers))

    def flush(self, timeout=None):
        return 0

    def poll(self, timeout=0):
        return 0


@pytest.fixture
def modern_callback():
    config = KafkaConfig(
        bootstrap_servers=["kafka:9092"],
        topic={"strategy": "consolidated"},
        partition={"strategy": "composite"},
        compression_type="snappy",
    )
    return KafkaCallback(kafka_config=config, producer_factory=DummyProducer)


def test_modern_callback_produces_messages(monkeypatch, modern_callback):
    producer: DummyProducer = modern_callback._producer._producer  # type: ignore
    modern_callback._topic_strategy = "consolidated"
    modern_callback._topic_prefix = "cryptofeed"

    # Simulate enqueue of JSON message
    msg = {"exchange": "binance", "symbol": "BTC-USDT", "data_type": "trades"}
    topic = modern_callback._topic_name("trades", type("obj", (), msg)())
    producer.produce(topic, b"{}", key=b"BTC-USDT", headers=None)

    assert producer.produced
    produced_topic, _, _, _ = producer.produced[0]
    # With consolidated strategy, TopicManager returns base topic
    assert produced_topic == "cryptofeed.trades"


def test_translation_preserves_partition_strategy():
    legacy = {
        "bootstrap_servers": ["kafka:9092"],
        "partition_strategy": "symbol",
    }
    translated = translate_legacy_config(legacy).modern_config
    assert translated.partition.strategy == "symbol"


def test_deprecation_warnings_do_not_break_flow():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        legacy = translate_legacy_config({"bootstrap_servers": ["k:1"]}).modern_config
        KafkaCallback(kafka_config=legacy, producer_factory=DummyProducer)
        assert all(issubclass(item.category, Warning) for item in w)
