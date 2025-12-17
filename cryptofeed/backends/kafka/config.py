"""Kafka backend configuration using simplified dataclass.

Replaces 4 Pydantic classes (328 LOC) with single dataclass (60 LOC).
Maintains backward compatibility with nested YAML format while flattening structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class KafkaConfig:
    """Simplified Kafka backend configuration (flattened from 4 Pydantic models).

    Single-level configuration replacing nested KafkaTopicConfig, KafkaPartitionConfig,
    and KafkaProducerConfig. All fields are flattened for simplicity.

    Attributes:
        bootstrap_servers: Kafka broker address(es) (required).
                          Can be single string "kafka:9092" or comma-separated "kafka1:9092,kafka2:9092"

        # Topic configuration (formerly KafkaTopicConfig)
        topic_prefix: Topic name prefix. Default: 'cryptofeed'
        topic_strategy: Topic naming strategy ('consolidated' or 'per_symbol'). Default: 'consolidated'
        partitions_per_topic: Number of partitions per topic. Default: 3
        replication_factor: Replication factor for topics. Default: 3

        # Partition configuration (formerly KafkaPartitionConfig)
        partition_strategy: Partition key strategy. Default: 'composite'
                           Options: 'composite', 'symbol', 'exchange', 'round_robin'

        # Producer configuration (formerly KafkaProducerConfig)
        compression_type: Compression algorithm. Default: 'gzip'
                         Options: 'none', 'gzip', 'snappy', 'lz4', 'zstd'
        acks: Delivery guarantee ('0', '1', 'all'). Default: 'all'
        enable_idempotence: Enable idempotent producer. Default: True
        retries: Number of retries on failure. Default: 3
        retry_backoff_ms: Backoff time between retries (ms). Default: 100
        batch_size: Maximum batch size in bytes. Default: 16384
        linger_ms: Time to wait before sending batch (ms). Default: 10

        # Monitoring configuration
        prometheus_port: Optional Prometheus metrics port. Default: None

    Example:
        >>> # Minimal config
        >>> config = KafkaConfig(bootstrap_servers="localhost:9092")
        >>>
        >>> # From YAML
        >>> config = KafkaConfig.from_yaml("config/kafka.yaml")
        >>>
        >>> # Complete config
        >>> config = KafkaConfig(
        ...     bootstrap_servers="kafka:9092",
        ...     topic_prefix="production",
        ...     partition_strategy="symbol",
        ...     compression_type="snappy"
        ... )
    """

    # Required field
    bootstrap_servers: str

    # Topic configuration
    topic_prefix: str = "cryptofeed"
    topic_strategy: str = "consolidated"
    partitions_per_topic: int = 3
    replication_factor: int = 3

    # Partition configuration
    partition_strategy: str = "composite"

    # Producer configuration
    compression_type: str = "gzip"
    acks: str = "all"
    enable_idempotence: bool = True
    retries: int = 3
    retry_backoff_ms: int = 100
    batch_size: int = 16384
    linger_ms: int = 10

    # Monitoring (optional)
    prometheus_port: Optional[int] = None

    def __init__(self, **kwargs):
        """Initialize with backward compatibility for nested configs.

        Supports both flat and nested (deprecated) formats:
        - Flat: KafkaConfig(bootstrap_servers="kafka:9092", topic_prefix="prod")
        - Nested: KafkaConfig(bootstrap_servers="kafka:9092", topic={...}, partition={...}, producer={...})
        """
        # Flatten nested configs if present (backward compatibility)
        if 'topic' in kwargs or 'partition' in kwargs or 'producer' in kwargs:
            kwargs = self._flatten_nested_config(kwargs)

        # Set all fields
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> KafkaConfig:
        """Load configuration from YAML file.

        Supports both flat and nested (backward compatible) YAML formats:

        Flat format:
            bootstrap_servers: kafka:9092
            topic_prefix: production
            partition_strategy: symbol

        Nested format (backward compatible):
            bootstrap_servers: kafka:9092
            topic:
              prefix: production
              strategy: consolidated
            partition:
              strategy: symbol

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            KafkaConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Configuration file is empty: {yaml_path}")

        # Flatten nested structure for backward compatibility
        config_dict = cls._flatten_nested_config(config_dict)

        return cls(**config_dict)

    @classmethod
    def _flatten_nested_config(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested YAML structure for backward compatibility.

        Converts old nested format:
            topic: {strategy: consolidated, prefix: prod}
            partition: {strategy: symbol}

        To flat format:
            topic_strategy: consolidated
            topic_prefix: prod
            partition_strategy: symbol

        Args:
            config_dict: Configuration dictionary (may contain nested dicts or deprecated config objects)

        Returns:
            Flattened configuration dictionary
        """
        flattened = dict(config_dict)

        # Flatten topic config (handle both dict and KafkaTopicConfig object)
        if 'topic' in flattened:
            topic_config = flattened.pop('topic')
            # Convert object to dict if needed
            if hasattr(topic_config, '_kwargs'):
                topic_config = topic_config._kwargs
            if isinstance(topic_config, dict):
                if 'strategy' in topic_config:
                    flattened['topic_strategy'] = topic_config['strategy']
                if 'prefix' in topic_config:
                    flattened['topic_prefix'] = topic_config['prefix']
                # Handle both 'partitions' and 'partitions_per_topic' field names
                if 'partitions_per_topic' in topic_config:
                    flattened['partitions_per_topic'] = topic_config['partitions_per_topic']
                elif 'partitions' in topic_config:
                    flattened['partitions_per_topic'] = topic_config['partitions']
                if 'replication_factor' in topic_config:
                    flattened['replication_factor'] = topic_config['replication_factor']

        # Flatten partition config (handle both dict and KafkaPartitionConfig object)
        if 'partition' in flattened:
            partition_config = flattened.pop('partition')
            # Convert object to dict if needed
            if hasattr(partition_config, '_kwargs'):
                partition_config = partition_config._kwargs
            if isinstance(partition_config, dict):
                if 'strategy' in partition_config:
                    flattened['partition_strategy'] = partition_config['strategy']

        # Flatten producer config (handle both dict and KafkaProducerConfig object)
        if 'producer' in flattened:
            producer_config = flattened.pop('producer')
            # Convert object to dict if needed
            if hasattr(producer_config, '_kwargs'):
                producer_config = producer_config._kwargs
            if isinstance(producer_config, dict):
                # Map all producer-specific fields
                producer_field_mappings = {
                    'compression_type': 'compression_type',
                    'acks': 'acks',
                    'enable_idempotence': 'enable_idempotence',
                    'retries': 'retries',
                    'retry_backoff_ms': 'retry_backoff_ms',
                    'batch_size': 'batch_size',
                    'linger_ms': 'linger_ms',
                }
                for old_key, new_key in producer_field_mappings.items():
                    if old_key in producer_config:
                        flattened[new_key] = producer_config[old_key]

        # Handle legacy 'partitions' field name (convert to partitions_per_topic)
        if 'partitions' in flattened:
            if 'partitions_per_topic' not in flattened:
                flattened['partitions_per_topic'] = flattened['partitions']
            flattened.pop('partitions')

        return flattened


# Backward compatibility: Keep old class names as aliases with deprecation warnings
class KafkaTopicConfig:
    """Deprecated: Use KafkaConfig with flat structure instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "KafkaTopicConfig is deprecated. Use KafkaConfig with flat structure instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Store args as both _kwargs and attributes for compatibility
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class KafkaPartitionConfig:
    """Deprecated: Use KafkaConfig with flat structure instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "KafkaPartitionConfig is deprecated. Use KafkaConfig with flat structure instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class KafkaProducerConfig:
    """Deprecated: Use KafkaConfig with flat structure instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "KafkaProducerConfig is deprecated. Use KafkaConfig with flat structure instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
