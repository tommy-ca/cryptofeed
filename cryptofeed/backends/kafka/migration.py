"""
Configuration migration utilities for Kafka backend.

Provides translation from legacy (Phase 0/1) Kafka configuration
structures to the modern Phase 2 `KafkaConfig` model used by
`cryptofeed.backends.kafka.callback`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .callback import KafkaConfig, KafkaTopicConfig, KafkaPartitionConfig


# ---- Data classes ---------------------------------------------------------


@dataclass
class MigrationResult:
    """Result of a legacy-to-modern configuration translation."""

    modern_config: KafkaConfig
    unmapped_options: Dict[str, Any]
    warnings: List[str]


@dataclass
class MigrationValidationReport:
    """Validation outcome for a legacy→modern migration comparison."""

    is_equivalent: bool
    differences: List[Tuple[str, Any, Any]]
    unmapped_options: Dict[str, Any]
    warnings: List[str]


# ---- Core translation helpers --------------------------------------------


LEGACY_KEY_MAP = {
    "bootstrap_servers": "bootstrap_servers",
    "acks": "acks",
    "idempotence": "idempotence",
    "retries": "retries",
    "retry_backoff_ms": "retry_backoff_ms",
    "batch_size": "batch_size",
    "linger_ms": "linger_ms",
    "compression_type": "compression_type",
    # Topic / partition specific legacy keys
    "topic_prefix": ("topic", "prefix"),
    "topic_strategy": ("topic", "strategy"),
    "partitions_per_topic": ("topic", "partitions_per_topic"),
    "replication_factor": ("topic", "replication_factor"),
    "partition_strategy": ("partition", "strategy"),
}

# Legacy defaults chosen to mirror historic behavior (per-symbol topics).
LEGACY_DEFAULT_TOPIC = KafkaTopicConfig(strategy="per_symbol")
LEGACY_DEFAULT_PARTITION = KafkaPartitionConfig(strategy="composite")


def translate_legacy_config(legacy_config: Dict[str, Any]) -> MigrationResult:
    """
    Translate a legacy Kafka configuration dictionary into a modern KafkaConfig.

    Args:
        legacy_config: Dictionary containing legacy configuration keys.

    Returns:
        MigrationResult with modern KafkaConfig and any unmapped legacy options.

    Raises:
        ValueError: If required fields (bootstrap_servers) are missing or invalid.
    """
    if not isinstance(legacy_config, dict):
        raise ValueError("legacy_config must be a dictionary")

    if "bootstrap_servers" not in legacy_config:
        raise ValueError("legacy_config missing required key 'bootstrap_servers'")

    # Initialize target sections with legacy-friendly defaults
    topic_kwargs = LEGACY_DEFAULT_TOPIC.model_dump()
    partition_kwargs = LEGACY_DEFAULT_PARTITION.model_dump()
    producer_kwargs: Dict[str, Any] = {}
    unmapped: Dict[str, Any] = {}
    warnings: List[str] = []

    for key, value in legacy_config.items():
        if key not in LEGACY_KEY_MAP:
            unmapped[key] = value
            continue

        target = LEGACY_KEY_MAP[key]
        # bootstrap_servers handled explicitly when constructing KafkaConfig
        if target == "bootstrap_servers":
            continue
        if isinstance(target, tuple):
            section, section_key = target
            if section == "topic":
                topic_kwargs[section_key] = value
            elif section == "partition":
                # Normalize strategy to lower-case for compatibility
                if isinstance(value, str):
                    value = value.lower()
                partition_kwargs[section_key] = value
        else:
            producer_kwargs[target] = value

    # Build modern KafkaConfig
    modern = KafkaConfig(
        bootstrap_servers=legacy_config["bootstrap_servers"],
        topic=KafkaTopicConfig(**topic_kwargs),
        partition=KafkaPartitionConfig(**partition_kwargs),
        **producer_kwargs,
    )

    # Emit warning when unmapped options exist
    if unmapped:
        warnings.append(
            f"Unmapped legacy options: {', '.join(sorted(unmapped.keys()))}"
        )

    return MigrationResult(
        modern_config=modern,
        unmapped_options=unmapped,
        warnings=warnings,
    )


def detect_legacy_config(config: Dict[str, Any]) -> bool:
    """
    Heuristically detect whether a config dictionary is using the legacy format.

    Returns True when it contains legacy-only keys like topic_prefix/partition_strategy
    or when it omits modern nested 'topic'/'partition' sections (minimal legacy configs).
    """
    legacy_keys = {"topic_prefix", "partition_strategy", "topic_strategy"}
    if any(key in config for key in legacy_keys):
        return True
    # Minimal legacy configs lacked nested topic/partition sections.
    if "topic" not in config and "partition" not in config:
        return True
    return False


def diff_configs(modern_a: KafkaConfig, modern_b: KafkaConfig) -> List[Tuple[str, Any, Any]]:
    """
    Compare two KafkaConfig instances and list differences.

    Useful for functional equivalence checks between translated and expected configs.
    """
    diffs: List[Tuple[str, Any, Any]] = []

    def _simple(attr: str):
        a_val = getattr(modern_a, attr)
        b_val = getattr(modern_b, attr)
        if a_val != b_val:
            diffs.append((attr, a_val, b_val))

    for attr in [
        "bootstrap_servers",
        "acks",
        "idempotence",
        "retries",
        "retry_backoff_ms",
        "batch_size",
        "linger_ms",
        "compression_type",
    ]:
        _simple(attr)

    # Topic and partition comparisons
    if modern_a.topic != modern_b.topic:
        diffs.append(("topic", modern_a.topic, modern_b.topic))
    if modern_a.partition != modern_b.partition:
        diffs.append(("partition", modern_a.partition, modern_b.partition))

    return diffs


def validate_migration(
    legacy_config: Dict[str, Any],
    expected_modern: KafkaConfig | None = None,
) -> MigrationValidationReport:
    """
    Validate a migration by translating legacy config and comparing to target modern config.

    Args:
        legacy_config: Legacy configuration dictionary.
        expected_modern: Optional expected KafkaConfig to compare against. If not
            provided, the translation result is compared to itself (always equivalent).

    Returns:
        MigrationValidationReport capturing equivalence, differences, and unmapped options.
    """
    translation = translate_legacy_config(legacy_config)
    translated_modern = translation.modern_config

    if expected_modern is None:
        differences: List[Tuple[str, Any, Any]] = []
        is_equivalent = True
    else:
        differences = diff_configs(translated_modern, expected_modern)
        is_equivalent = len(differences) == 0

    warnings = list(translation.warnings)
    if translation.unmapped_options:
        warnings.append(
            f"Unmapped legacy options detected: {', '.join(sorted(translation.unmapped_options))}"
        )

    return MigrationValidationReport(
        is_equivalent=is_equivalent,
        differences=differences,
        unmapped_options=translation.unmapped_options,
        warnings=warnings,
    )


__all__ = [
    "MigrationResult",
    "MigrationValidationReport",
    "translate_legacy_config",
    "detect_legacy_config",
    "diff_configs",
    "validate_migration",
]
