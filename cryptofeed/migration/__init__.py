"""Kafka configuration migration utilities for Phase 2 upgrade.

This package provides tools for migrating legacy kafka.py configurations
to the new KafkaCallback format, including:

- Config translation: Convert legacy configs to Phase 2 format
- Config validation: Validate Phase 2 configurations
- CLI tool: Command-line interface for migration workflows

Key modules:
- config_translator: Legacy to Phase 2 conversion
- config_validator: Configuration validation and testing
- cli: Command-line interface

Example:
    >>> from cryptofeed.migration.config_translator import ConfigTranslator
    >>> translator = ConfigTranslator()
    >>> legacy = {'bootstrap_servers': ['kafka:9092']}
    >>> phase2 = translator.translate(legacy)
"""

from cryptofeed.migration.config_translator import ConfigTranslator
from cryptofeed.migration.config_validator import ConfigValidator
from cryptofeed.migration.cli import MigrationCLI

__all__ = [
    "ConfigTranslator",
    "ConfigValidator",
    "MigrationCLI",
]
