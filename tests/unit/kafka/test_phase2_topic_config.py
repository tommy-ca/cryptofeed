"""Phase 2: Task 6 - Unit tests for topic naming and configuration.

This module tests:
- Task 6.1: Topic naming logic (consolidated vs per-symbol)
- Task 6.2: Topic strategy configuration
- Task 6.3: Configuration parsing and validation

All tests are written FIRST (TDD: RED phase) before implementation.
"""

import pytest
from pathlib import Path
import tempfile

from cryptofeed.kafka_callback import (
    TopicManager,
    KafkaTopicConfig,
    KafkaConfig,
)


# ============================================================================
# Task 6.1: Topic Naming Logic Tests
# ============================================================================


class TestTaskSixOneTopicNamingLogic:
    """Test topic naming logic for consolidated vs per-symbol strategies."""

    def test_consolidated_topic_single_data_type(self):
        """Consolidated strategy should generate cryptofeed.trade for trade."""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='BTC-USD',
            exchange='coinbase',
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.trade', \
            "Consolidated strategy should return cryptofeed.<data_type>"

    def test_consolidated_topic_multiple_data_types(self):
        """Consolidated strategy should handle multiple data types."""
        test_cases = [
            ('trade', 'cryptofeed.trade'),
            ('orderbook', 'cryptofeed.orderbook'),
            ('ticker', 'cryptofeed.ticker'),
            ('candle', 'cryptofeed.candle'),
            ('funding', 'cryptofeed.funding'),
            ('liquidation', 'cryptofeed.liquidation'),
        ]
        for data_type, expected_topic in test_cases:
            topic = TopicManager.get_topic(
                data_type=data_type,
                symbol='BTC-USD',
                exchange='coinbase',
                strategy='consolidated'
            )
            assert topic == expected_topic, \
                f"Failed for data_type={data_type}"

    def test_per_symbol_topic_naming(self):
        """Per-symbol strategy should generate cryptofeed.{type}.{exchange}.{symbol}."""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='BTC-USD',
            exchange='coinbase',
            strategy='per_symbol'
        )
        assert topic == 'cryptofeed.trade.coinbase.btc-usd', \
            "Per-symbol strategy should include exchange and symbol"

    def test_per_symbol_topic_case_normalization(self):
        """Per-symbol topics should normalize symbol and exchange to lowercase."""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='BTC-USD',
            exchange='COINBASE',
            strategy='per_symbol'
        )
        assert topic == 'cryptofeed.trade.coinbase.btc-usd', \
            "Exchange and symbol should be lowercase in per-symbol topics"

    def test_symbol_normalization_underscore_to_hyphen(self):
        """Symbol normalization should convert underscores to hyphens."""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='BTC_USD',
            exchange='binance',
            strategy='per_symbol'
        )
        assert 'btc-usd' in topic, \
            "Underscores should be converted to hyphens in topic name"

    def test_symbol_normalization_various_formats(self):
        """Verify symbol normalization works with various formats."""
        test_cases = [
            ('BTC-USD', 'btc-usd'),
            ('btc-usd', 'btc-usd'),
            ('BTC_USD', 'btc-usd'),
            ('btc_USD', 'btc-usd'),
        ]
        for symbol_input, expected_normalized in test_cases:
            topic = TopicManager.get_topic(
                data_type='trade',
                symbol=symbol_input,
                exchange='test',
                strategy='per_symbol'
            )
            assert expected_normalized in topic, \
                f"Symbol {symbol_input} should normalize to {expected_normalized}"

    def test_topic_name_length_validation(self):
        """Topic name should not exceed Kafka limit of 249 characters."""
        # Create a very long symbol and exchange
        long_symbol = 'A' * 100
        long_exchange = 'B' * 100

        topic = TopicManager.get_topic(
            data_type='trade',
            symbol=long_symbol,
            exchange=long_exchange,
            strategy='per_symbol'
        )

        assert len(topic) <= 249, \
            f"Topic name {len(topic)} chars exceeds Kafka limit of 249"

    def test_consolidated_ignores_symbol_exchange(self):
        """Consolidated strategy should ignore symbol and exchange parameters."""
        topic1 = TopicManager.get_topic(
            data_type='trade',
            symbol='BTC-USD',
            exchange='coinbase',
            strategy='consolidated'
        )
        topic2 = TopicManager.get_topic(
            data_type='trade',
            symbol='ETH-USDT',
            exchange='binance',
            strategy='consolidated'
        )
        assert topic1 == topic2 == 'cryptofeed.trade', \
            "Consolidated strategy should produce identical topics for same data_type"


# ============================================================================
# Task 6.2: Topic Strategy Configuration Tests
# ============================================================================


class TestTaskSixTwoTopicStrategyConfig:
    """Test topic strategy configuration loading and validation."""

    def test_load_consolidated_strategy_from_config(self):
        """KafkaTopicConfig should load consolidated strategy."""
        config = KafkaTopicConfig(strategy='consolidated')
        assert config.strategy == 'consolidated', \
            "Config should store consolidated strategy"

    def test_load_per_symbol_strategy_from_config(self):
        """KafkaTopicConfig should load per_symbol strategy."""
        config = KafkaTopicConfig(strategy='per_symbol')
        assert config.strategy == 'per_symbol', \
            "Config should store per_symbol strategy"

    def test_custom_prefix_configuration(self):
        """KafkaTopicConfig should support custom prefix."""
        config = KafkaTopicConfig(prefix='production')
        assert config.prefix == 'production', \
            "Config should store custom prefix"

    def test_default_prefix_is_cryptofeed(self):
        """Default prefix should be 'cryptofeed'."""
        config = KafkaTopicConfig()
        assert config.prefix == 'cryptofeed', \
            "Default prefix should be cryptofeed"

    def test_whitespace_prefix_defaults_to_cryptofeed(self):
        """Whitespace-only prefix should default to cryptofeed."""
        config = KafkaTopicConfig(prefix='   ')
        assert config.prefix == 'cryptofeed', \
            "Whitespace-only prefix should default to cryptofeed"

    def test_invalid_strategy_raises_error(self):
        """Invalid strategy value should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            KafkaTopicConfig(strategy='invalid_strategy')
        assert 'consolidated' in str(exc_info.value).lower() or \
               'per_symbol' in str(exc_info.value).lower(), \
            "Error message should mention valid strategies"

    def test_strategy_values_case_sensitive(self):
        """Strategy values should be case-sensitive (lowercase required)."""
        # uppercase should fail
        with pytest.raises(ValueError):
            KafkaTopicConfig(strategy='CONSOLIDATED')

    def test_partitions_per_topic_configuration(self):
        """KafkaTopicConfig should accept partitions_per_topic parameter."""
        config = KafkaTopicConfig(partitions_per_topic=12)
        assert config.partitions_per_topic == 12, \
            "Config should store partitions_per_topic value"

    def test_replication_factor_configuration(self):
        """KafkaTopicConfig should accept replication_factor parameter."""
        config = KafkaTopicConfig(replication_factor=2)
        assert config.replication_factor == 2, \
            "Config should store replication_factor value"

    def test_partitions_must_be_positive(self):
        """partitions_per_topic must be > 0."""
        with pytest.raises(ValueError):
            KafkaTopicConfig(partitions_per_topic=0)

        with pytest.raises(ValueError):
            KafkaTopicConfig(partitions_per_topic=-1)

    def test_replication_factor_must_be_positive(self):
        """replication_factor must be > 0."""
        with pytest.raises(ValueError):
            KafkaTopicConfig(replication_factor=0)

        with pytest.raises(ValueError):
            KafkaTopicConfig(replication_factor=-1)


# ============================================================================
# Task 6.3: Configuration Parsing and Validation Tests
# ============================================================================


class TestTaskSixThreeConfigParsing:
    """Test configuration parsing from YAML and dictionaries."""

    def test_yaml_parsing_valid_configuration(self):
        """KafkaConfig should load valid YAML configuration."""
        yaml_content = """
bootstrap_servers:
  - localhost:9092
  - localhost:9093
topic:
  strategy: consolidated
  prefix: cryptofeed
partition:
  strategy: composite
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = KafkaConfig.from_yaml(f.name)
            assert config.bootstrap_servers == ['localhost:9092', 'localhost:9093'], \
                "Should parse bootstrap_servers from YAML"
            assert config.topic.strategy == 'consolidated', \
                "Should parse topic strategy from YAML"
            assert config.partition.strategy == 'composite', \
                "Should parse partition strategy from YAML"

            Path(f.name).unlink()

    def test_yaml_parsing_invalid_syntax_raises_error(self):
        """Invalid YAML syntax should raise error."""
        invalid_yaml = """
bootstrap_servers:
  - localhost:9092
invalid: [unclosed bracket
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()

            with pytest.raises(Exception):  # YAML parsing error
                KafkaConfig.from_yaml(f.name)

            Path(f.name).unlink()

    def test_pydantic_validation_field_types(self):
        """Pydantic should validate field types."""
        # bootstrap_servers should be a list
        with pytest.raises(ValueError):
            KafkaConfig(bootstrap_servers='localhost:9092')  # string instead of list

    def test_pydantic_validation_constraints(self):
        """Pydantic should validate field constraints."""
        # bootstrap_servers cannot be empty
        with pytest.raises(ValueError):
            KafkaConfig(bootstrap_servers=[])

    def test_dict_parsing_minimal_config(self):
        """KafkaConfig should load from minimal dictionary."""
        config_dict = {
            'bootstrap_servers': ['kafka:9092']
        }
        config = KafkaConfig.from_dict(config_dict)
        assert config.bootstrap_servers == ['kafka:9092'], \
            "Should load bootstrap_servers from dict"
        assert config.topic.strategy == 'consolidated', \
            "Should use default topic strategy"
        assert config.partition.strategy == 'composite', \
            "Should use default partition strategy"

    def test_dict_parsing_complete_config(self):
        """KafkaConfig should load complete configuration from dictionary."""
        config_dict = {
            'bootstrap_servers': ['kafka1:9092', 'kafka2:9092'],
            'topic': {
                'strategy': 'per_symbol',
                'prefix': 'production'
            },
            'partition': {
                'strategy': 'symbol'
            },
            'acks': 'all',
            'compression_type': 'snappy'
        }
        config = KafkaConfig.from_dict(config_dict)
        assert config.topic.strategy == 'per_symbol', \
            "Should parse nested topic config from dict"
        assert config.partition.strategy == 'symbol', \
            "Should parse nested partition config from dict"

    def test_yaml_file_not_found_raises_error(self):
        """Loading non-existent YAML file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            KafkaConfig.from_yaml('/nonexistent/file.yaml')

    def test_yaml_empty_file_raises_error(self):
        """Empty YAML file should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            f.flush()

            with pytest.raises(ValueError):
                KafkaConfig.from_yaml(f.name)

            Path(f.name).unlink()

    def test_missing_bootstrap_servers_raises_error(self):
        """Configuration without bootstrap_servers should raise error."""
        config_dict = {
            'topic': {'strategy': 'consolidated'}
        }
        with pytest.raises(ValueError):
            KafkaConfig.from_dict(config_dict)

    def test_invalid_acks_value_raises_error(self):
        """Invalid acks value should raise error."""
        config_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'acks': 'invalid'
        }
        with pytest.raises(ValueError):
            KafkaConfig.from_dict(config_dict)

    def test_invalid_compression_type_raises_error(self):
        """Invalid compression_type should raise error."""
        config_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'compression_type': 'invalid'
        }
        with pytest.raises(ValueError):
            KafkaConfig.from_dict(config_dict)

    def test_valid_acks_values(self):
        """Valid acks values should be accepted."""
        for acks_value in ['0', '1', 'all']:
            config = KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                acks=acks_value
            )
            assert config.acks == acks_value, \
                f"Should accept acks value {acks_value}"

    def test_valid_compression_types(self):
        """Valid compression types should be accepted."""
        valid_types = ['none', 'gzip', 'snappy', 'lz4', 'zstd']
        for comp_type in valid_types:
            config = KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                compression_type=comp_type
            )
            assert config.compression_type == comp_type, \
                f"Should accept compression_type {comp_type}"

    def test_extra_fields_rejected(self):
        """Extra fields not in config schema should be rejected."""
        config_dict = {
            'bootstrap_servers': ['kafka:9092'],
            'unknown_field': 'value'
        }
        with pytest.raises(ValueError):
            KafkaConfig.from_dict(config_dict)

    def test_environment_variable_override_not_implemented(self):
        """Note: Environment variable overrides would be implemented as separate feature."""
        # This test documents that env var override is a future feature
        # Currently, configuration is loaded from YAML/dict only
        pass


# ============================================================================
# Integration Tests: Configuration Round-Trip
# ============================================================================


class TestConfigurationRoundTrip:
    """Test that configuration can be saved and restored accurately."""

    def test_config_round_trip_to_dict(self):
        """KafkaConfig should round-trip through to_dict()."""
        original_dict = {
            'bootstrap_servers': ['kafka:9092', 'kafka:9093'],
            'topic': {'strategy': 'consolidated', 'prefix': 'prod'},
            'partition': {'strategy': 'symbol'}
        }
        config = KafkaConfig.from_dict(original_dict)

        # Convert back to dict via model_dump()
        restored_dict = config.model_dump()

        assert restored_dict['bootstrap_servers'] == original_dict['bootstrap_servers']
        assert restored_dict['topic']['strategy'] == original_dict['topic']['strategy']
        assert restored_dict['partition']['strategy'] == original_dict['partition']['strategy']

    def test_config_round_trip_yaml(self):
        """KafkaConfig should round-trip through YAML."""
        yaml_content = """
bootstrap_servers:
  - kafka:9092
topic:
  strategy: per_symbol
  prefix: staging
partition:
  strategy: exchange
acks: '1'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config1 = KafkaConfig.from_yaml(f.name)

            # Re-save and reload
            yaml_path = Path(f.name)
            with open(yaml_path, 'w') as fw:
                import yaml
                yaml.dump(config1.model_dump(), fw)

            config2 = KafkaConfig.from_yaml(yaml_path)

            assert config1.topic.strategy == config2.topic.strategy
            assert config1.bootstrap_servers == config2.bootstrap_servers

            yaml_path.unlink()
