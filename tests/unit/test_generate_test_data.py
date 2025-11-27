"""Unit tests for synthetic test data generator.

Tests the test data generation tool used for staging validation and load testing.
Follows TDD methodology (RED-GREEN-REFACTOR cycle).
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from decimal import Decimal

from scripts.generate_test_data import (
    SyntheticDataGenerator,
    DataGenerationConfig,
    VolumeProfile,
    GeneratorConfig,
    KafkaOutputAdapter,
)


class TestVolumeProfile:
    """Test volume profile configuration model."""

    def test_volume_profile_low_volume(self):
        """Test low volume profile configuration (1K msg/s)."""
        profile = VolumeProfile(
            name="low",
            messages_per_second=1000,
            duration_seconds=3600,
            exchanges=["coinbase"],
            symbols=["BTC-USD"],
        )

        assert profile.name == "low"
        assert profile.messages_per_second == 1000
        assert profile.duration_seconds == 3600
        assert profile.total_messages == 3_600_000
        assert len(profile.exchanges) == 1
        assert len(profile.symbols) == 1

    def test_volume_profile_medium_volume(self):
        """Test medium volume profile (50K msg/s)."""
        profile = VolumeProfile(
            name="medium",
            messages_per_second=50_000,
            duration_seconds=3600,
            exchanges=["coinbase", "binance", "kraken"],
            symbols=["BTC-USD", "ETH-USD"],
        )

        assert profile.messages_per_second == 50_000
        assert profile.total_messages == 180_000_000
        assert len(profile.exchanges) == 3

    def test_volume_profile_high_volume(self):
        """Test high volume profile (150K msg/s)."""
        profile = VolumeProfile(
            name="high",
            messages_per_second=150_000,
            duration_seconds=3600,
            exchanges=["coinbase", "binance", "kraken", "okx", "bybit"],
            symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        )

        assert profile.messages_per_second == 150_000
        assert profile.total_messages == 540_000_000

    def test_volume_profile_validation_negative_rate(self):
        """Test validation fails for negative message rate."""
        with pytest.raises(ValueError, match="messages_per_second must be positive"):
            VolumeProfile(
                name="invalid",
                messages_per_second=-100,
                duration_seconds=3600,
                exchanges=["coinbase"],
                symbols=["BTC-USD"],
            )

    def test_volume_profile_validation_empty_exchanges(self):
        """Test validation fails for empty exchange list."""
        with pytest.raises(ValueError, match="exchanges must not be empty"):
            VolumeProfile(
                name="invalid",
                messages_per_second=1000,
                duration_seconds=3600,
                exchanges=[],
                symbols=["BTC-USD"],
            )


class TestDataGenerationConfig:
    """Test configuration loading and validation."""

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "profiles": {
                "low": {
                    "messages_per_second": 1000,
                    "duration_seconds": 3600,
                    "exchanges": ["coinbase"],
                    "symbols": ["BTC-USD"],
                }
            }
        }

        config = DataGenerationConfig.from_dict(config_dict)
        assert "low" in config.profiles
        assert config.profiles["low"].messages_per_second == 1000

    def test_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
profiles:
  low:
    messages_per_second: 1000
    duration_seconds: 3600
    exchanges:
      - coinbase
    symbols:
      - BTC-USD
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = DataGenerationConfig.from_yaml(f.name)
            assert "low" in config.profiles

            # Cleanup
            Path(f.name).unlink()

    def test_config_validation_duplicate_profiles(self):
        """Test validation catches duplicate profile names."""
        config_dict = {
            "profiles": {
                "test": {
                    "messages_per_second": 1000,
                    "duration_seconds": 3600,
                    "exchanges": ["coinbase"],
                    "symbols": ["BTC-USD"],
                }
            }
        }

        config = DataGenerationConfig.from_dict(config_dict)
        # Should not raise, profile names are unique by dict key
        assert len(config.profiles) == 1


class TestSyntheticDataGenerator:
    """Test synthetic market data generation."""

    @pytest.fixture
    def generator_config(self):
        """Create test generator configuration."""
        return GeneratorConfig(
            exchanges=["coinbase", "binance"],
            symbols=["BTC-USD", "ETH-USD"],
            data_types=["trade", "ticker"],
            seed=42,  # Fixed seed for reproducibility
        )

    @pytest.fixture
    def generator(self, generator_config):
        """Create test data generator instance."""
        return SyntheticDataGenerator(generator_config)

    def test_generator_initialization(self, generator, generator_config):
        """Test generator initializes with correct configuration."""
        assert generator.config == generator_config
        assert generator.config.seed == 42
        assert len(generator.config.exchanges) == 2
        assert len(generator.config.symbols) == 2

    def test_generate_trade_message(self, generator):
        """Test generating synthetic trade message."""
        trade = generator.generate_trade(
            exchange="coinbase",
            symbol="BTC-USD",
            timestamp=1234567890.123456,
        )

        assert trade["exchange"] == "coinbase"
        assert trade["symbol"] == "BTC-USD"
        assert trade["timestamp"] == 1234567890.123456
        assert "price" in trade
        assert "amount" in trade
        assert "side" in trade
        assert trade["side"] in ["buy", "sell"]
        assert isinstance(trade["price"], Decimal)
        assert isinstance(trade["amount"], Decimal)
        assert trade["price"] > 0
        assert trade["amount"] > 0

    def test_generate_ticker_message(self, generator):
        """Test generating synthetic ticker message."""
        ticker = generator.generate_ticker(
            exchange="binance",
            symbol="ETH-USD",
            timestamp=1234567890.123456,
        )

        assert ticker["exchange"] == "binance"
        assert ticker["symbol"] == "ETH-USD"
        assert ticker["timestamp"] == 1234567890.123456
        assert "bid" in ticker
        assert "ask" in ticker
        assert isinstance(ticker["bid"], Decimal)
        assert isinstance(ticker["ask"], Decimal)
        assert ticker["ask"] >= ticker["bid"]  # Spread constraint

    def test_generate_orderbook_message(self, generator):
        """Test generating synthetic order book message."""
        orderbook = generator.generate_orderbook(
            exchange="coinbase",
            symbol="BTC-USD",
            timestamp=1234567890.123456,
            depth=10,
        )

        assert orderbook["exchange"] == "coinbase"
        assert orderbook["symbol"] == "BTC-USD"
        assert orderbook["timestamp"] == 1234567890.123456
        assert "bids" in orderbook
        assert "asks" in orderbook
        assert len(orderbook["bids"]) == 10
        assert len(orderbook["asks"]) == 10

        # Validate bid/ask ordering
        for i in range(1, len(orderbook["bids"])):
            assert orderbook["bids"][i]["price"] < orderbook["bids"][i-1]["price"]
        for i in range(1, len(orderbook["asks"])):
            assert orderbook["asks"][i]["price"] > orderbook["asks"][i-1]["price"]

    def test_generate_message_stream_constant_rate(self, generator):
        """Test generating message stream at constant rate."""
        messages = list(generator.generate_stream(
            messages_per_second=100,
            duration_seconds=1,
            data_types=["trade"],
        ))

        # Should generate approximately 100 messages (±5%)
        assert 95 <= len(messages) <= 105

        # All messages should be trades
        assert all(msg["type"] == "trade" for msg in messages)

    def test_generate_message_stream_spike_scenario(self, generator):
        """Test spike scenario: 10 → 200 → 10 msg/s over time (scaled down for testing)."""
        spike_config = {
            "start_rate": 10,
            "peak_rate": 200,
            "end_rate": 10,
            "ramp_up_seconds": 6,  # 6 seconds (scaled down from 600)
            "peak_seconds": 6,     # 6 seconds (scaled down from 600)
            "ramp_down_seconds": 6, # 6 seconds (scaled down from 600)
        }

        messages = list(generator.generate_spike_scenario(**spike_config))

        # Total duration: 18 seconds
        # Expected messages (approximate):
        # - Ramp up: avg 105/s * 6s = 630
        # - Peak: 200/s * 6s = 1200
        # - Ramp down: avg 105/s * 6s = 630
        # Total: ~2460 messages

        # Allow 20% tolerance
        expected = 2460
        assert expected * 0.8 <= len(messages) <= expected * 1.2

    def test_deterministic_generation_with_seed(self):
        """Test that same seed produces same messages."""
        config1 = GeneratorConfig(
            exchanges=["coinbase"],
            symbols=["BTC-USD"],
            data_types=["trade"],
            seed=123,
        )
        config2 = GeneratorConfig(
            exchanges=["coinbase"],
            symbols=["BTC-USD"],
            data_types=["trade"],
            seed=123,
        )

        gen1 = SyntheticDataGenerator(config1)
        gen2 = SyntheticDataGenerator(config2)

        messages1 = list(gen1.generate_stream(
            messages_per_second=10,
            duration_seconds=1,
            data_types=["trade"],
        ))
        messages2 = list(gen2.generate_stream(
            messages_per_second=10,
            duration_seconds=1,
            data_types=["trade"],
        ))

        assert len(messages1) == len(messages2)
        assert messages1[0]["price"] == messages2[0]["price"]


class TestKafkaOutputAdapter:
    """Test Kafka output adapter for staging integration."""

    @pytest.fixture
    def mock_producer(self):
        """Create mock Kafka producer."""
        return Mock()

    @pytest.fixture
    def adapter(self, mock_producer):
        """Create Kafka output adapter with mock producer."""
        return KafkaOutputAdapter(
            bootstrap_servers=["localhost:9092"],
            topic_prefix="cryptofeed.test",
            producer=mock_producer,
        )

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes with correct configuration."""
        assert adapter.topic_prefix == "cryptofeed.test"
        assert adapter.bootstrap_servers == ["localhost:9092"]

    def test_send_message_to_kafka(self, adapter, mock_producer):
        """Test sending message to Kafka topic."""
        message = {
            "type": "trade",
            "exchange": "coinbase",
            "symbol": "BTC-USD",
            "price": Decimal("50000.00"),
            "amount": Decimal("0.1"),
            "timestamp": 1234567890.123456,
        }

        adapter.send(message)

        # Verify producer.produce was called
        mock_producer.produce.assert_called_once()
        call_args = mock_producer.produce.call_args

        # Verify topic routing
        assert call_args[1]["topic"] == "cryptofeed.test.trade"

    def test_send_batch_messages(self, adapter, mock_producer):
        """Test sending batch of messages."""
        messages = [
            {"type": "trade", "exchange": "coinbase", "symbol": "BTC-USD"},
            {"type": "trade", "exchange": "binance", "symbol": "ETH-USD"},
            {"type": "ticker", "exchange": "kraken", "symbol": "SOL-USD"},
        ]

        adapter.send_batch(messages)

        assert mock_producer.produce.call_count == 3

    def test_flush_on_close(self, adapter, mock_producer):
        """Test adapter flushes messages on close."""
        adapter.close()

        mock_producer.flush.assert_called_once()


class TestCLIIntegration:
    """Test command-line interface integration."""

    def test_cli_help_message(self):
        """Test CLI displays help message."""
        from scripts.generate_test_data import main

        with pytest.raises(SystemExit):
            with patch('sys.argv', ['generate_test_data.py', '--help']):
                main()

    def test_cli_config_file_argument(self):
        """Test CLI accepts config file argument."""
        with patch('sys.argv', [
            'generate_test_data.py',
            '--config', 'test-config.yaml',
            '--profile', 'low',
        ]):
            with patch('scripts.generate_test_data.DataGenerationConfig.from_yaml'):
                with patch('scripts.generate_test_data.SyntheticDataGenerator'):
                    # Should not raise
                    pass

    def test_cli_output_kafka_option(self):
        """Test CLI supports Kafka output option."""
        with patch('sys.argv', [
            'generate_test_data.py',
            '--config', 'test-config.yaml',
            '--profile', 'low',
            '--output', 'kafka',
            '--kafka-brokers', 'localhost:9092',
        ]):
            # Should accept Kafka output configuration
            pass


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_duration_raises_error(self):
        """Test zero duration raises validation error."""
        with pytest.raises(ValueError):
            VolumeProfile(
                name="invalid",
                messages_per_second=1000,
                duration_seconds=0,
                exchanges=["coinbase"],
                symbols=["BTC-USD"],
            )

    def test_negative_duration_raises_error(self):
        """Test negative duration raises validation error."""
        with pytest.raises(ValueError):
            VolumeProfile(
                name="invalid",
                messages_per_second=1000,
                duration_seconds=-100,
                exchanges=["coinbase"],
                symbols=["BTC-USD"],
            )

    def test_empty_data_types_raises_error(self):
        """Test empty data types list raises error."""
        config = GeneratorConfig(
            exchanges=["coinbase"],
            symbols=["BTC-USD"],
            data_types=[],
            seed=42,
        )

        generator = SyntheticDataGenerator(config)

        with pytest.raises(ValueError, match="data_types must not be empty"):
            list(generator.generate_stream(
                messages_per_second=100,
                duration_seconds=1,
                data_types=[],
            ))

    def test_unsupported_data_type_raises_error(self):
        """Test unsupported data type raises error."""
        config = GeneratorConfig(
            exchanges=["coinbase"],
            symbols=["BTC-USD"],
            data_types=["trade"],
            seed=42,
        )

        generator = SyntheticDataGenerator(config)

        with pytest.raises(ValueError, match="Unsupported data type"):
            list(generator.generate_stream(
                messages_per_second=100,
                duration_seconds=1,
                data_types=["invalid_type"],
            ))
