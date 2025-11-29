"""Kafka backend configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class KafkaTopicConfig(BaseModel):
    """Configuration for Kafka topic management.

    Attributes:
        strategy: Topic naming strategy ('consolidated' or 'per_symbol').
                 Default: 'consolidated'
        prefix: Topic name prefix (e.g., 'cryptofeed', 'production').
               Whitespace-only prefixes default to 'cryptofeed'.
               Default: 'cryptofeed'
        partitions_per_topic: Number of partitions per topic.
                             Must be > 0. Default: 3
        replication_factor: Replication factor for topic.
                           Must be > 0. Default: 3

    Example:
        >>> config = KafkaTopicConfig(
        ...     strategy='consolidated',
        ...     prefix='production',
        ...     partitions_per_topic=12,
        ...     replication_factor=3
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default="consolidated", description="Topic naming strategy")
    prefix: str = Field(default="cryptofeed", description="Topic name prefix")
    partitions_per_topic: int = Field(default=3, description="Partitions per topic")
    replication_factor: int = Field(default=3, description="Replication factor")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate topic strategy is supported."""
        if v not in {"consolidated", "per_symbol"}:
            raise ValueError(
                f"Invalid topic strategy: {v}. "
                f"Must be 'consolidated' or 'per_symbol'"
            )
        return v

    @field_validator("prefix", mode="before")
    @classmethod
    def normalize_prefix(cls, v: Optional[str]) -> str:
        """Normalize prefix: whitespace-only becomes 'cryptofeed'."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return "cryptofeed"
        return v

    @field_validator("partitions_per_topic")
    @classmethod
    def validate_partitions(cls, v: int) -> int:
        """Validate partitions_per_topic is positive."""
        if v <= 0:
            raise ValueError("partitions_per_topic must be > 0")
        return v

    @field_validator("replication_factor")
    @classmethod
    def validate_replication(cls, v: int) -> int:
        """Validate replication_factor is positive."""
        if v <= 0:
            raise ValueError("replication_factor must be > 0")
        return v


class KafkaPartitionConfig(BaseModel):
    """Configuration for partition key strategies.

    Attributes:
        strategy: Partitioner strategy to use.
                 Options: 'composite' (default), 'symbol', 'exchange', 'round_robin'
                 - composite: Route by exchange-symbol (recommended)
                 - symbol: Route by symbol only (cross-exchange analysis)
                 - exchange: Route by exchange only (exchange-specific processing)
                 - round_robin: No ordering (maximum parallelism)

    Example:
        >>> config = KafkaPartitionConfig(strategy='composite')
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default="composite", description="Partition key strategy")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate partition strategy is supported."""
        valid_strategies = {"composite", "symbol", "exchange", "round_robin"}
        if v.lower() not in valid_strategies:
            raise ValueError(
                f"Invalid partition strategy: {v}. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}"
            )
        return v.lower()


class KafkaProducerConfig(BaseModel):
    """Configuration for Kafka producer client.

    Attributes:
        bootstrap_servers: List of Kafka broker addresses (required).
                          Example: ['kafka1:9092', 'kafka2:9092']
        acks: Delivery guarantee ('0', '1', 'all'). Default: 'all'
        idempotence: Enable idempotent producer (prevent duplicates). Default: True
        retries: Number of retries on failure. Default: 3
        retry_backoff_ms: Backoff time between retries in milliseconds. Default: 100
        batch_size: Maximum batch size in bytes. Default: 16384
        linger_ms: Time to wait before sending batch (ms). Default: 10
        compression_type: Compression algorithm. Default: 'snappy'
                         Options: 'none', 'gzip', 'snappy', 'lz4', 'zstd'

    Example:
        >>> config = KafkaProducerConfig(
        ...     bootstrap_servers=['kafka:9092'],
        ...     acks='all',
        ...     compression_type='snappy'
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    bootstrap_servers: list[str] = Field(
        description="Kafka broker addresses (required)"
    )
    acks: str = Field(default="all", description="Delivery guarantee")
    idempotence: bool = Field(default=True, description="Enable idempotence")
    retries: int = Field(default=3, description="Number of retries")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff (ms)")
    batch_size: int = Field(default=16384, description="Batch size (bytes)")
    linger_ms: int = Field(default=10, description="Linger time (ms)")
    compression_type: str = Field(default="snappy", description="Compression type")
    protobuf_cutoff: Optional[str] = Field(
        default=None,
        description="Cutoff date (YYYY-MM-DD) after which KafkaCallback protobuf mode is disabled; overrides env CF_KAFKA_PROTOBUF_CUTOFF",
    )

    @field_validator("bootstrap_servers")
    @classmethod
    def validate_bootstrap_servers(cls, v: list[str]) -> list[str]:
        """Validate bootstrap_servers is not empty."""
        if not v:
            raise ValueError("bootstrap_servers cannot be empty")
        return v

    @field_validator("acks")
    @classmethod
    def validate_acks(cls, v: str) -> str:
        """Validate acks value."""
        if v not in {"0", "1", "all"}:
            raise ValueError(f"acks must be '0', '1', or 'all', got {v}")
        return v

    @field_validator("retries")
    @classmethod
    def validate_retries(cls, v: int) -> int:
        """Validate retries is non-negative."""
        if v < 0:
            raise ValueError("retries must be >= 0")
        return v

    @field_validator("retry_backoff_ms")
    @classmethod
    def validate_retry_backoff(cls, v: int) -> int:
        """Validate retry_backoff_ms is non-negative."""
        if v < 0:
            raise ValueError("retry_backoff_ms must be >= 0")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch_size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v

    @field_validator("linger_ms")
    @classmethod
    def validate_linger(cls, v: int) -> int:
        """Validate linger_ms is non-negative."""
        if v < 0:
            raise ValueError("linger_ms must be >= 0")
        return v

    @field_validator("compression_type")
    @classmethod
    def validate_compression(cls, v: str) -> str:
        """Validate compression_type is supported."""
        valid = {"none", "gzip", "snappy", "lz4", "zstd"}
        if v not in valid:
            raise ValueError(
                f"compression_type must be one of {valid}, got {v}"
            )
        return v


class KafkaConfig(BaseModel):
    """Top-level Kafka configuration combining all settings.

    Combines producer, topic, and partition configurations into a single,
    loadable configuration object. Supports loading from YAML files and
    Python dictionaries.

    Attributes:
        bootstrap_servers: Kafka broker addresses (required).
        topic: Topic configuration (nested KafkaTopicConfig).
        partition: Partition configuration (nested KafkaPartitionConfig).
        acks: Producer acks setting. Default: 'all'
        idempotence: Enable idempotence. Default: True
        retries: Retry count. Default: 3
        retry_backoff_ms: Retry backoff. Default: 100
        batch_size: Batch size. Default: 16384
        linger_ms: Linger time. Default: 10
        compression_type: Compression type. Default: 'snappy'

    Example:
        >>> # From dictionary
        >>> config = KafkaConfig.from_dict({
        ...     'bootstrap_servers': ['kafka:9092'],
        ...     'acks': 'all',
        ...     'topic': {'strategy': 'consolidated'},
        ...     'partition': {'strategy': 'composite'}
        ... })
        >>>
        >>> # From YAML file
        >>> config = KafkaConfig.from_yaml('config/kafka.yaml')
    """

    model_config = ConfigDict(extra="forbid")

    bootstrap_servers: list[str] = Field(description="Kafka broker addresses")
    topic: KafkaTopicConfig = Field(
        default_factory=KafkaTopicConfig,
        description="Topic configuration"
    )
    partition: KafkaPartitionConfig = Field(
        default_factory=KafkaPartitionConfig,
        description="Partition configuration"
    )
    acks: str = Field(default="all", description="Delivery guarantee")
    idempotence: bool = Field(default=True, description="Enable idempotence")
    retries: int = Field(default=3, description="Number of retries")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff (ms)")
    batch_size: int = Field(default=16384, description="Batch size (bytes)")
    linger_ms: int = Field(default=10, description="Linger time (ms)")
    compression_type: str = Field(default="snappy", description="Compression type")

    @field_validator("bootstrap_servers")
    @classmethod
    def validate_bootstrap_servers(cls, v: list[str]) -> list[str]:
        """Validate bootstrap_servers is not empty."""
        if not v:
            raise ValueError("bootstrap_servers cannot be empty")
        return v

    @field_validator("acks")
    @classmethod
    def validate_acks(cls, v: str) -> str:
        """Validate acks value."""
        if v not in {"0", "1", "all"}:
            raise ValueError(f"acks must be '0', '1', or 'all', got {v}")
        return v

    @field_validator("compression_type")
    @classmethod
    def validate_compression(cls, v: str) -> str:
        """Validate compression_type."""
        valid = {"none", "gzip", "snappy", "lz4", "zstd"}
        if v not in valid:
            raise ValueError(f"compression_type must be one of {valid}")
        return v

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> KafkaConfig:
        """Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary with bootstrap_servers,
                        optional topic, partition, and producer settings.

        Returns:
            KafkaConfig instance

        Raises:
            ValueError: If configuration is invalid
            KeyError: If required fields are missing
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> KafkaConfig:
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            KafkaConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or configuration is incomplete
        """
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Configuration file is empty: {yaml_path}")

        return cls.from_dict(config_dict)
