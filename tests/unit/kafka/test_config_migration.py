"""Tests for legacy->modern Kafka configuration translation."""

from __future__ import annotations

import pytest

from cryptofeed.backends.kafka.migration import (
    translate_legacy_config,
    detect_legacy_config,
    diff_configs,
)
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig, KafkaPartitionConfig


def test_detect_legacy_config_true():
    legacy = {"topic_prefix": "legacy_feeds", "bootstrap_servers": ["k:9092"]}
    assert detect_legacy_config(legacy) is True


def test_detect_legacy_config_false():
    modern = {"bootstrap_servers": ["k:9092"], "topic": {"strategy": "consolidated"}}
    assert detect_legacy_config(modern) is False


def test_translate_minimal_legacy_config():
    legacy = {"bootstrap_servers": ["kafka:9092"]}
    result = translate_legacy_config(legacy)

    modern = result.modern_config
    assert modern.bootstrap_servers == ["kafka:9092"]
    # Legacy defaults favor per_symbol / composite to preserve old topic layout
    assert modern.topic.strategy == "per_symbol"
    assert modern.partition.strategy == "composite"
    assert result.unmapped_options == {}
    assert result.warnings == []


def test_translate_with_topic_prefix_and_compression():
    legacy = {
        "bootstrap_servers": ["k1:9092", "k2:9092"],
        "topic_prefix": "staging",
        "acks": "1",
        "compression_type": "snappy",
    }
    result = translate_legacy_config(legacy)
    modern = result.modern_config

    assert modern.bootstrap_servers == ["k1:9092", "k2:9092"]
    assert modern.acks == "1"
    assert modern.compression_type == "snappy"
    assert modern.topic.prefix == "staging"
    assert modern.topic.strategy == "per_symbol"
    assert result.unmapped_options == {}


def test_translate_maps_partition_strategy():
    legacy = {
        "bootstrap_servers": ["k:9092"],
        "partition_strategy": "round_robin",
    }
    modern = translate_legacy_config(legacy).modern_config
    assert modern.partition.strategy == "round_robin"


def test_translate_unmapped_keys_are_reported():
    legacy = {"bootstrap_servers": ["k:9092"], "unknown_option": True}
    result = translate_legacy_config(legacy)
    assert result.unmapped_options == {"unknown_option": True}
    assert result.warnings  # should include notice


def test_translate_requires_bootstrap_servers():
    with pytest.raises(ValueError):
        translate_legacy_config({})


def test_diff_configs_reports_changes():
    modern_a = KafkaConfig(
        bootstrap_servers=["k:9092"],
        topic=KafkaTopicConfig(strategy="per_symbol"),
        partition=KafkaPartitionConfig(strategy="composite"),
    )
    modern_b = KafkaConfig(
        bootstrap_servers=["k:9092"],
        topic=KafkaTopicConfig(strategy="consolidated"),
        partition=KafkaPartitionConfig(strategy="composite"),
    )
    diffs = diff_configs(modern_a, modern_b)
    assert ("topic", modern_a.topic, modern_b.topic) in diffs

