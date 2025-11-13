"""Config translator for migrating legacy Kafka configurations to Phase 2 format.

This module handles conversion of legacy kafka.py configurations to the new
KafkaCallback format, including YAML file translation and dry-run preview.

Key Features:
- Parse legacy configuration dictionaries and YAML files
- Translate to Phase 2 KafkaCallback format
- Preserve all producer settings
- Dry-run mode for preview before applying
- Clear error messages with migration guidance
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class LegacyKafkaConfig(BaseModel):
    """Legacy Kafka configuration schema.

    Represents the structure of legacy cryptofeed.backends.kafka configurations.
    Supports all historical settings that need to be migrated to Phase 2.
    """

    model_config = ConfigDict(extra="forbid")

    bootstrap_servers: list[str] = Field(description="Kafka broker addresses")
    topic_prefix: str = Field(default="cryptofeed", description="Topic name prefix")
    acks: str = Field(default="1", description="Delivery guarantee (legacy default: 1)")
    retries: int = Field(default=3, description="Number of retries")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff (ms)")
    batch_size: int = Field(default=16384, description="Batch size (bytes)")
    linger_ms: int = Field(default=10, description="Linger time (ms)")
    compression_type: str = Field(default="snappy", description="Compression type")
    per_symbol_topics: Optional[bool] = Field(default=None, description="Use per-symbol topic naming")

    @field_validator("bootstrap_servers")
    @classmethod
    def validate_bootstrap_servers(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("bootstrap_servers cannot be empty")
        return v

    @field_validator("acks")
    @classmethod
    def validate_acks(cls, v: str) -> str:
        if v not in {"0", "1", "all"}:
            raise ValueError(f"acks must be '0', '1', or 'all', got {v}")
        return v

    @field_validator("compression_type")
    @classmethod
    def validate_compression(cls, v: str) -> str:
        valid = {"none", "gzip", "snappy", "lz4", "zstd"}
        if v not in valid:
            raise ValueError(f"compression_type must be one of {valid}, got {v}")
        return v


class Phase2KafkaConfig(BaseModel):
    """Phase 2 KafkaCallback configuration schema.

    This is the target schema that legacy configurations are translated to.
    Importing this from kafka_callback would create circular imports, so we
    define the structure here.
    """

    model_config = ConfigDict(extra="forbid")

    bootstrap_servers: list[str] = Field(description="Kafka broker addresses")
    topic: Dict[str, Any] = Field(
        default_factory=lambda: {"strategy": "per_symbol", "prefix": "cryptofeed"},
        description="Topic configuration"
    )
    partition: Dict[str, Any] = Field(
        default_factory=lambda: {"strategy": "composite"},
        description="Partition configuration"
    )
    acks: str = Field(default="all", description="Delivery guarantee")
    idempotence: bool = Field(default=True, description="Enable idempotence")
    retries: int = Field(default=3, description="Number of retries")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff (ms)")
    batch_size: int = Field(default=16384, description="Batch size (bytes)")
    linger_ms: int = Field(default=10, description="Linger time (ms)")
    compression_type: str = Field(default="snappy", description="Compression type")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Phase2KafkaConfig":
        """Load configuration from dictionary."""
        return cls(**config_dict)


class ConfigTranslator:
    """Translate legacy Kafka configurations to Phase 2 format.

    This class provides methods to convert legacy kafka.py configurations
    to the new KafkaCallback format, handling both dictionaries and YAML files.

    Example:
        >>> translator = ConfigTranslator()
        >>> legacy = {'bootstrap_servers': ['kafka:9092']}
        >>> phase2 = translator.translate(legacy)
        >>> translator.save_yaml(phase2, 'phase2_config.yaml')
    """

    def translate(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Translate legacy config dictionary to Phase 2 format.

        Args:
            legacy_config: Legacy configuration dictionary

        Returns:
            Phase 2 configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate legacy config
        try:
            legacy = LegacyKafkaConfig(**legacy_config)
        except Exception as e:
            raise ValueError(f"Invalid legacy configuration: {e}")

        # Determine topic strategy
        topic_strategy = "per_symbol"  # Default for backward compatibility
        if legacy.per_symbol_topics is False:
            topic_strategy = "consolidated"
        elif legacy.per_symbol_topics is True:
            topic_strategy = "per_symbol"

        # Build Phase 2 config
        phase2_dict = {
            "bootstrap_servers": legacy.bootstrap_servers,
            "topic": {
                "strategy": topic_strategy,
                "prefix": legacy.topic_prefix,
                "partitions_per_topic": 3,  # Phase 2 default
                "replication_factor": 3,  # Phase 2 default
            },
            "partition": {
                "strategy": "composite",  # Phase 2 default
            },
            "acks": legacy.acks,
            "idempotence": True if legacy.acks == "all" else False,
            "retries": legacy.retries,
            "retry_backoff_ms": legacy.retry_backoff_ms,
            "batch_size": legacy.batch_size,
            "linger_ms": legacy.linger_ms,
            "compression_type": legacy.compression_type,
        }

        return phase2_dict

    def translate_yaml_file(
        self,
        input_file: str | Path,
        output_file: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Translate legacy YAML configuration file to Phase 2 format.

        Args:
            input_file: Path to legacy YAML configuration
            output_file: Optional path to save Phase 2 configuration

        Returns:
            Phase 2 configuration dictionary

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If YAML is invalid or configuration is incomplete
        """
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {input_path}")

        try:
            with open(input_path, 'r') as f:
                legacy_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {input_path}: {e}")

        if legacy_dict is None:
            raise ValueError(f"Configuration file is empty: {input_path}")

        # Translate
        phase2_dict = self.translate(legacy_dict)

        # Save if output path provided
        if output_file:
            self.save_yaml(phase2_dict, output_file)

        return phase2_dict

    def save_yaml(self, config_dict: Dict[str, Any], output_file: str | Path) -> None:
        """Save Phase 2 configuration to YAML file.

        Args:
            config_dict: Phase 2 configuration dictionary
            output_file: Path to save YAML file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def dry_run(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Preview what would be translated without saving.

        Args:
            legacy_config: Legacy configuration dictionary

        Returns:
            Dictionary with 'original', 'translated', 'changes', and 'summary'
        """
        phase2_dict = self.translate(legacy_config)

        # Identify changes
        changes = {}

        # Check topic strategy
        if legacy_config.get("per_symbol_topics") is not None:
            changes["topic_strategy"] = {
                "before": "per_symbol" if legacy_config.get("per_symbol_topics") else "consolidated",
                "after": phase2_dict["topic"]["strategy"],
            }

        # Check if new fields added
        new_fields = set(phase2_dict.keys()) - set(legacy_config.keys())
        if new_fields:
            changes["added_fields"] = list(new_fields)

        # Build summary
        summary_lines = []
        summary_lines.append("Translation Preview:")
        summary_lines.append(f"- Bootstrap servers: {len(phase2_dict['bootstrap_servers'])} broker(s)")
        summary_lines.append(f"- Topic strategy: {phase2_dict['topic']['strategy']}")
        summary_lines.append(f"- Topic prefix: {phase2_dict['topic']['prefix']}")
        summary_lines.append(f"- Partition strategy: {phase2_dict['partition']['strategy']}")
        summary_lines.append(f"- Producer acks: {phase2_dict['acks']}")

        if changes:
            summary_lines.append("\nKey changes:")
            if "topic_strategy" in changes:
                before = changes["topic_strategy"]["before"]
                after = changes["topic_strategy"]["after"]
                summary_lines.append(f"- Topic strategy: {before} -> {after}")
            if "added_fields" in changes:
                summary_lines.append(f"- New fields: {', '.join(changes['added_fields'])}")

        return {
            "original": legacy_config,
            "translated": phase2_dict,
            "changes": changes,
            "summary": "\n".join(summary_lines),
        }
