"""Tests for Kafka config migration translation from legacy to Phase 2.

This module tests the config_translator module which handles conversion
of legacy kafka.py configurations to the new KafkaCallback format.

Test organization follows TDD approach:
1. RED: Tests fail with no implementation
2. GREEN: Minimal implementation passes tests
3. REFACTOR: Improve code quality
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from cryptofeed.migration.config_translator import (
    ConfigTranslator,
    LegacyKafkaConfig,
    Phase2KafkaConfig,
)


class TestLegacyConfigParsing:
    """Test parsing legacy kafka.py configuration formats."""

    def test_parse_simple_legacy_config(self):
        """Parse minimal legacy kafka configuration."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
        }
        config = LegacyKafkaConfig(**legacy_dict)
        assert config.bootstrap_servers == ['kafka:9092']

    def test_parse_legacy_config_with_topics(self):
        """Parse legacy config with explicit topic names."""
        legacy_dict = {
            'bootstrap_servers': ['kafka1:9092', 'kafka2:9092'],
            'topic_prefix': 'old_feed',
        }
        config = LegacyKafkaConfig(**legacy_dict)
        assert config.bootstrap_servers == ['kafka1:9092', 'kafka2:9092']
        assert config.topic_prefix == 'old_feed'

    def test_parse_legacy_config_with_producer_settings(self):
        """Parse legacy config with producer settings."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'acks': '1',
            'retries': 5,
            'compression_type': 'gzip',
        }
        config = LegacyKafkaConfig(**legacy_dict)
        assert config.acks == '1'
        assert config.retries == 5
        assert config.compression_type == 'gzip'

    def test_legacy_config_defaults(self):
        """Verify legacy config has sensible defaults."""
        legacy_dict = {'bootstrap_servers': ['kafka:9092']}
        config = LegacyKafkaConfig(**legacy_dict)
        assert config.topic_prefix == 'cryptofeed'
        assert config.acks == '1'  # Legacy default
        assert config.retries == 3


class TestConfigTranslation:
    """Test translating legacy configs to Phase 2 format."""

    def test_translate_minimal_legacy_config(self):
        """Translate minimal legacy config to Phase 2."""
        legacy_dict = {'bootstrap_servers': ['kafka:9092']}
        translator = ConfigTranslator()
        phase2_dict = translator.translate(legacy_dict)

        assert phase2_dict['bootstrap_servers'] == ['kafka:9092']
        assert phase2_dict['topic']['strategy'] == 'per_symbol'  # Backward compat
        assert phase2_dict['partition']['strategy'] == 'composite'
        assert phase2_dict['acks'] == '1'  # Preserves legacy acks

    def test_translate_legacy_with_custom_topic_prefix(self):
        """Translate legacy config preserving custom topic prefix."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'topic_prefix': 'production',
        }
        translator = ConfigTranslator()
        phase2_dict = translator.translate(legacy_dict)

        assert phase2_dict['topic']['prefix'] == 'production'

    def test_translate_legacy_acks_settings(self):
        """Translate various acks settings correctly."""
        translator = ConfigTranslator()

        # Legacy 0 -> Phase 2 0
        result = translator.translate({'bootstrap_servers': ['kafka:9092'], 'acks': '0'})
        assert result['acks'] == '0'

        # Legacy 1 -> Phase 2 1
        result = translator.translate({'bootstrap_servers': ['kafka:9092'], 'acks': '1'})
        assert result['acks'] == '1'

    def test_translate_legacy_compression_settings(self):
        """Translate compression settings."""
        translator = ConfigTranslator()

        for compression in ['none', 'gzip', 'snappy', 'lz4']:
            result = translator.translate({
                'bootstrap_servers': ['kafka:9092'],
                'compression_type': compression,
            })
            assert result['compression_type'] == compression

    def test_translate_legacy_producer_settings(self):
        """Translate all producer settings."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'acks': '1',
            'retries': 5,
            'retry_backoff_ms': 200,
            'batch_size': 32768,
            'linger_ms': 20,
            'compression_type': 'snappy',
        }
        translator = ConfigTranslator()
        result = translator.translate(legacy_dict)

        assert result['acks'] == '1'
        assert result['retries'] == 5
        assert result['retry_backoff_ms'] == 200
        assert result['batch_size'] == 32768
        assert result['linger_ms'] == 20
        assert result['compression_type'] == 'snappy'

    def test_translate_legacy_to_phase2_config_object(self):
        """Translate and instantiate Phase 2 config object."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'acks': '1',
            'topic_prefix': 'test',
        }
        translator = ConfigTranslator()
        phase2_dict = translator.translate(legacy_dict)

        # Should be valid Phase 2 config
        phase2_config = Phase2KafkaConfig.from_dict(phase2_dict)
        assert phase2_config.bootstrap_servers == ['kafka:9092']
        assert phase2_config.topic['prefix'] == 'test'
        assert phase2_config.acks == '1'

    def test_translate_per_symbol_explicit(self):
        """When legacy config indicates per-symbol, preserve that."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'per_symbol_topics': True,
        }
        translator = ConfigTranslator()
        result = translator.translate(legacy_dict)

        assert result['topic']['strategy'] == 'per_symbol'

    def test_translate_consolidated_option(self):
        """Support explicit consolidated topic naming in legacy format."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'per_symbol_topics': False,
        }
        translator = ConfigTranslator()
        result = translator.translate(legacy_dict)

        assert result['topic']['strategy'] == 'consolidated'


class TestYAMLRoundTrip:
    """Test translating YAML files and round-tripping."""

    def test_translate_legacy_yaml_file(self):
        """Load legacy YAML, translate, and validate."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
  - kafka:9093
topic_prefix: myapp
acks: '1'
retries: 5
compression_type: snappy
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(legacy_yaml)
            f.flush()

            try:
                translator = ConfigTranslator()
                phase2_dict = translator.translate_yaml_file(f.name)

                assert phase2_dict['bootstrap_servers'] == ['kafka:9092', 'kafka:9093']
                assert phase2_dict['topic']['prefix'] == 'myapp'
                assert phase2_dict['acks'] == '1'
                assert phase2_dict['compression_type'] == 'snappy'
            finally:
                Path(f.name).unlink()

    def test_save_translated_yaml(self):
        """Save translated config to YAML file."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'topic_prefix': 'test',
            'acks': '1',
        }
        translator = ConfigTranslator()
        phase2_dict = translator.translate(legacy_dict)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name

        try:
            translator.save_yaml(phase2_dict, output_path)

            # Verify saved file can be read back
            with open(output_path, 'r') as f:
                loaded = yaml.safe_load(f)

            assert loaded['bootstrap_servers'] == ['kafka:9092']
            assert loaded['topic']['prefix'] == 'test'
        finally:
            Path(output_path).unlink()

    def test_translate_yaml_to_yaml(self):
        """Full workflow: legacy YAML -> translate -> Phase 2 YAML."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
topic_prefix: legacy
acks: '1'
compression_type: gzip
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as src:
            src.write(legacy_yaml)
            src.flush()
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as dst:
            dst_path = dst.name

        try:
            translator = ConfigTranslator()
            translator.translate_yaml_file(src_path, output_file=dst_path)

            # Verify output
            with open(dst_path, 'r') as f:
                phase2_config = yaml.safe_load(f)

            assert phase2_config['bootstrap_servers'] == ['kafka:9092']
            assert phase2_config['topic']['prefix'] == 'legacy'
            assert phase2_config['acks'] == '1'
        finally:
            Path(src_path).unlink()
            Path(dst_path).unlink()


class TestDryRunMode:
    """Test dry-run preview of translations."""

    def test_dry_run_shows_changes(self):
        """Dry-run mode shows what would be translated."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'topic_prefix': 'old',
            'acks': '1',
        }
        translator = ConfigTranslator()
        diff = translator.dry_run(legacy_dict)

        # Should return dict with before/after
        assert 'original' in diff
        assert 'translated' in diff
        assert diff['original']['bootstrap_servers'] == ['kafka:9092']
        assert diff['translated']['bootstrap_servers'] == ['kafka:9092']

    def test_dry_run_highlights_changes(self):
        """Dry-run marks which fields changed."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'acks': '1',  # Will stay '1'
        }
        translator = ConfigTranslator()
        diff = translator.dry_run(legacy_dict)

        # Should indicate which fields changed
        assert 'changes' in diff
        # New topic/partition config added as new fields
        assert 'added_fields' in diff['changes'] or 'topic' in diff['translated']

    def test_dry_run_summary_message(self):
        """Dry-run provides human-readable summary."""
        legacy_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'topic_prefix': 'legacy',
        }
        translator = ConfigTranslator()
        diff = translator.dry_run(legacy_dict)

        # Should have summary message
        assert 'summary' in diff
        assert isinstance(diff['summary'], str)
        assert len(diff['summary']) > 0


class TestErrorHandling:
    """Test error handling for invalid configurations."""

    def test_missing_bootstrap_servers_raises_error(self):
        """Missing bootstrap_servers raises clear error."""
        with pytest.raises(ValueError, match="bootstrap_servers"):
            translator = ConfigTranslator()
            translator.translate({})

    def test_invalid_acks_value_raises_error(self):
        """Invalid acks value raises clear error."""
        with pytest.raises(ValueError, match="acks"):
            translator = ConfigTranslator()
            translator.translate({
                'bootstrap_servers': ['kafka:9092'],
                'acks': 'invalid',
            })

    def test_invalid_compression_raises_error(self):
        """Invalid compression type raises clear error."""
        with pytest.raises(ValueError, match="compression"):
            translator = ConfigTranslator()
            translator.translate({
                'bootstrap_servers': ['kafka:9092'],
                'compression_type': 'invalid',
            })

    def test_file_not_found_raises_error(self):
        """Non-existent file raises clear error."""
        translator = ConfigTranslator()
        with pytest.raises(FileNotFoundError):
            translator.translate_yaml_file('/nonexistent/path/config.yaml')

    def test_invalid_yaml_raises_error(self):
        """Invalid YAML raises clear error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [content")
            f.flush()

            try:
                translator = ConfigTranslator()
                with pytest.raises(ValueError, match="YAML"):
                    translator.translate_yaml_file(f.name)
            finally:
                Path(f.name).unlink()

    def test_suggestion_for_unknown_field(self):
        """Unknown fields in legacy config raise helpful error."""
        with pytest.raises(Exception) as exc_info:
            translator = ConfigTranslator()
            translator.translate({
                'bootstrap_servers': ['kafka:9092'],
                'unknown_field': 'value',
            })
        # Should have suggestion in error
        assert 'unknown' in str(exc_info.value).lower() or 'field' in str(exc_info.value).lower()


class TestRealWorldExamples:
    """Test against realistic legacy configuration examples."""

    def test_example_simple_single_exchange(self):
        """Single exchange setup (legacy)."""
        legacy = {
            'bootstrap_servers': ['localhost:9092'],
        }
        translator = ConfigTranslator()
        result = translator.translate(legacy)

        assert result['bootstrap_servers'] == ['localhost:9092']
        assert result['acks'] == '1'  # Legacy default preserved

    def test_example_production_setup(self):
        """Production setup with multiple brokers."""
        legacy = {
            'bootstrap_servers': [
                'kafka-1.prod.example.com:9092',
                'kafka-2.prod.example.com:9092',
                'kafka-3.prod.example.com:9092',
            ],
            'topic_prefix': 'production',
            'acks': '1',
            'retries': 10,
            'compression_type': 'snappy',
        }
        translator = ConfigTranslator()
        result = translator.translate(legacy)

        assert len(result['bootstrap_servers']) == 3
        assert result['topic']['prefix'] == 'production'
        assert result['acks'] == '1'
        assert result['retries'] == 10

    def test_example_high_throughput_setup(self):
        """High-throughput configuration."""
        legacy = {
            'bootstrap_servers': ['kafka:9092'],
            'batch_size': 65536,
            'linger_ms': 100,
            'compression_type': 'lz4',
            'acks': '1',
        }
        translator = ConfigTranslator()
        result = translator.translate(legacy)

        assert result['batch_size'] == 65536
        assert result['linger_ms'] == 100
        assert result['compression_type'] == 'lz4'

    def test_example_strict_delivery_setup(self):
        """Strict delivery guarantee setup."""
        legacy = {
            'bootstrap_servers': ['kafka:9092'],
            'acks': 'all',
            'retries': 999,
            'compression_type': 'none',
        }
        translator = ConfigTranslator()
        result = translator.translate(legacy)

        assert result['acks'] == 'all'
        assert result['retries'] == 999
        assert result['compression_type'] == 'none'
