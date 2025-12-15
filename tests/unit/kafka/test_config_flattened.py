"""Unit tests for flattened KafkaConfig dataclass (Task 15.2).

Tests verify that the simplified dataclass provides identical behavior
to the previous 4-class Pydantic implementation while reducing complexity.

Target: Single KafkaConfig dataclass (60 LOC) replacing 4 Pydantic classes (328 LOC).
"""

import pytest
import tempfile
import os
from pathlib import Path
from dataclasses import is_dataclass


def test_kafka_config_is_dataclass():
    """Verify KafkaConfig uses standard library dataclass, not Pydantic."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    assert is_dataclass(KafkaConfig)
    # Should not have Pydantic model attributes
    assert not hasattr(KafkaConfig, 'model_validate')
    assert not hasattr(KafkaConfig, 'model_config')


def test_kafka_config_minimal():
    """Test minimal configuration with only bootstrap_servers."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    config = KafkaConfig(bootstrap_servers="localhost:9092")
    assert config.bootstrap_servers == "localhost:9092"
    assert config.topic_prefix == "cryptofeed"
    assert config.partition_strategy == "composite"
    assert config.compression_type == "gzip"
    assert config.acks == "all"
    assert config.enable_idempotence is True


def test_kafka_config_all_fields():
    """Test that all essential fields are available in flat structure."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    config = KafkaConfig(
        bootstrap_servers="kafka:9092",
        topic_prefix="production",
        topic_strategy="consolidated",
        partition_strategy="symbol",
        partitions_per_topic=12,
        replication_factor=2,
        compression_type="snappy",
        acks="1",
        enable_idempotence=False,
        retries=5,
        retry_backoff_ms=200,
        batch_size=32768,
        linger_ms=20,
        prometheus_port=9090,
    )

    assert config.bootstrap_servers == "kafka:9092"
    assert config.topic_prefix == "production"
    assert config.topic_strategy == "consolidated"
    assert config.partition_strategy == "symbol"
    assert config.partitions_per_topic == 12
    assert config.replication_factor == 2
    assert config.compression_type == "snappy"
    assert config.acks == "1"
    assert config.enable_idempotence is False
    assert config.retries == 5
    assert config.retry_backoff_ms == 200
    assert config.batch_size == 32768
    assert config.linger_ms == 20
    assert config.prometheus_port == 9090


def test_kafka_config_from_yaml_minimal():
    """Test from_yaml() loads minimal configuration."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    yaml_content = """
bootstrap_servers: localhost:9092
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = f.name

    try:
        config = KafkaConfig.from_yaml(temp_path)
        assert config.bootstrap_servers == "localhost:9092"
        assert config.topic_prefix == "cryptofeed"
    finally:
        os.unlink(temp_path)


def test_kafka_config_from_yaml_complete():
    """Test from_yaml() loads complete configuration with all fields."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    yaml_content = """
bootstrap_servers: kafka1:9092,kafka2:9092
topic_prefix: production
topic_strategy: consolidated
partition_strategy: composite
partitions_per_topic: 12
replication_factor: 3
compression_type: snappy
acks: all
enable_idempotence: true
retries: 3
retry_backoff_ms: 100
batch_size: 16384
linger_ms: 10
prometheus_port: 9090
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = f.name

    try:
        config = KafkaConfig.from_yaml(temp_path)
        assert config.bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert config.topic_prefix == "production"
        assert config.topic_strategy == "consolidated"
        assert config.partition_strategy == "composite"
        assert config.partitions_per_topic == 12
        assert config.replication_factor == 3
        assert config.compression_type == "snappy"
        assert config.acks == "all"
        assert config.enable_idempotence is True
        assert config.retries == 3
        assert config.retry_backoff_ms == 100
        assert config.batch_size == 16384
        assert config.linger_ms == 10
        assert config.prometheus_port == 9090
    finally:
        os.unlink(temp_path)


def test_kafka_config_from_yaml_with_path_object():
    """Test from_yaml() accepts Path objects."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    yaml_content = """
bootstrap_servers: localhost:9092
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = f.name

    try:
        config = KafkaConfig.from_yaml(Path(temp_path))
        assert config.bootstrap_servers == "localhost:9092"
    finally:
        os.unlink(temp_path)


def test_kafka_config_from_yaml_missing_file():
    """Test from_yaml() raises FileNotFoundError for missing file."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    with pytest.raises(FileNotFoundError):
        KafkaConfig.from_yaml("/nonexistent/config.yaml")


def test_kafka_config_backward_compatibility_nested_topic():
    """Test backward compatibility with nested topic config format."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    # Old format had nested topic dict
    yaml_content = """
bootstrap_servers: localhost:9092
topic:
  strategy: consolidated
  prefix: production
  partitions_per_topic: 12
  replication_factor: 3
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = f.name

    try:
        config = KafkaConfig.from_yaml(temp_path)
        # Should flatten nested structure
        assert config.topic_prefix == "production"
        assert config.topic_strategy == "consolidated"
        assert config.partitions_per_topic == 12
        assert config.replication_factor == 3
    finally:
        os.unlink(temp_path)


def test_kafka_config_backward_compatibility_nested_partition():
    """Test backward compatibility with nested partition config format."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    yaml_content = """
bootstrap_servers: localhost:9092
partition:
  strategy: symbol
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = f.name

    try:
        config = KafkaConfig.from_yaml(temp_path)
        assert config.partition_strategy == "symbol"
    finally:
        os.unlink(temp_path)


def test_kafka_config_defaults_match_original():
    """Test that default values match original Pydantic implementation."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    config = KafkaConfig(bootstrap_servers="localhost:9092")

    # Topic defaults
    assert config.topic_prefix == "cryptofeed"
    assert config.topic_strategy == "consolidated"
    assert config.partitions_per_topic == 3
    assert config.replication_factor == 3

    # Partition defaults
    assert config.partition_strategy == "composite"

    # Producer defaults
    assert config.compression_type == "gzip"
    assert config.acks == "all"
    assert config.enable_idempotence is True
    assert config.retries == 3
    assert config.retry_backoff_ms == 100
    assert config.batch_size == 16384
    assert config.linger_ms == 10


def test_kafka_config_prometheus_port_optional():
    """Test that prometheus_port is optional (defaults to None)."""
    from cryptofeed.backends.kafka.config import KafkaConfig

    config = KafkaConfig(bootstrap_servers="localhost:9092")
    assert config.prometheus_port is None

    config_with_port = KafkaConfig(
        bootstrap_servers="localhost:9092",
        prometheus_port=9090
    )
    assert config_with_port.prometheus_port == 9090


def test_kafka_config_no_pydantic_dependency():
    """Verify config module does not import Pydantic."""
    import sys
    from cryptofeed.backends.kafka import config as config_module

    # Check module source for Pydantic imports
    import inspect
    source = inspect.getsource(config_module)
    assert 'from pydantic' not in source
    assert 'import pydantic' not in source
