#!/usr/bin/env python3
"""
Synthetic Market Data Generator for Staging Validation

Generates realistic test data for load testing the Kafka producer backend.
Supports configurable volume profiles (low/medium/high) and spike scenarios.

Usage:
    # Generate low volume test data (1K msg/s for 1 hour)
    python scripts/generate-test-data.py --config scripts/test-data-config.yaml --profile low

    # Generate spike scenario (10K → 200K → 10K msg/s)
    python scripts/generate-test-data.py --config scripts/test-data-config.yaml --profile spike

    # Output to Kafka topics for staging validation
    python scripts/generate-test-data.py \
        --config scripts/test-data-config.yaml \
        --profile medium \
        --output kafka \
        --kafka-brokers localhost:9092

Features:
- Configurable volume profiles (1K, 50K, 150K msg/s)
- Spike scenario support (ramp up, peak, ramp down)
- Multiple data types (trade, ticker, orderbook, funding)
- Kafka output for staging integration
- File output (JSON lines or Parquet)
- Deterministic generation with seed support
"""
import sys
import random
import time
import argparse
import yaml
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field


@dataclass
class VolumeProfile:
    """Volume profile configuration for test data generation."""

    name: str
    exchanges: List[str]
    symbols: List[str]
    messages_per_second: int = 0
    duration_seconds: int = 0
    data_types: List[str] = field(default_factory=lambda: ["trade"])
    description: str = ""
    spike_scenario: bool = False
    start_rate: Optional[int] = None
    peak_rate: Optional[int] = None
    end_rate: Optional[int] = None
    ramp_up_seconds: Optional[int] = None
    peak_seconds: Optional[int] = None
    ramp_down_seconds: Optional[int] = None

    @property
    def total_messages(self) -> int:
        """Calculate total messages to generate."""
        if self.spike_scenario:
            # Calculate average rate for spike scenario
            if not all([self.start_rate, self.peak_rate, self.end_rate,
                       self.ramp_up_seconds, self.peak_seconds, self.ramp_down_seconds]):
                raise ValueError("Spike scenario requires all rate parameters")

            # Approximate total (trapezoidal integration)
            ramp_up_avg = (self.start_rate + self.peak_rate) / 2
            ramp_down_avg = (self.peak_rate + self.end_rate) / 2

            total = (
                ramp_up_avg * self.ramp_up_seconds +
                self.peak_rate * self.peak_seconds +
                ramp_down_avg * self.ramp_down_seconds
            )
            return int(total)
        else:
            return self.messages_per_second * self.duration_seconds

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate based on scenario type
        if self.spike_scenario:
            # Spike scenario validation
            if not all([self.start_rate, self.peak_rate, self.end_rate,
                       self.ramp_up_seconds, self.peak_seconds, self.ramp_down_seconds]):
                raise ValueError("Spike scenario requires all rate and duration parameters")
        else:
            # Constant rate validation
            if self.messages_per_second <= 0:
                raise ValueError("messages_per_second must be positive")

            if self.duration_seconds <= 0:
                raise ValueError("duration_seconds must be positive")

        # Common validation
        if not self.exchanges:
            raise ValueError("exchanges must not be empty")

        if not self.symbols:
            raise ValueError("symbols must not be empty")


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generator."""

    exchanges: List[str]
    symbols: List[str]
    data_types: List[str]
    seed: Optional[int] = None


@dataclass
class DataGenerationConfig:
    """Test data configuration loaded from YAML file."""

    profiles: Dict[str, VolumeProfile]
    kafka_config: Optional[Dict[str, Any]] = None
    file_output_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataGenerationConfig':
        """Load configuration from dictionary."""
        profiles = {}
        for name, profile_dict in config_dict.get("profiles", {}).items():
            # Create a copy and remove description if present
            profile_data = dict(profile_dict)
            description = profile_data.pop("description", "")

            # Create profile with name, description, and remaining fields
            profiles[name] = VolumeProfile(
                name=name,
                description=description,
                **profile_data
            )

        return cls(
            profiles=profiles,
            kafka_config=config_dict.get("kafka"),
            file_output_config=config_dict.get("file_output"),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DataGenerationConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class SyntheticDataGenerator:
    """Generate synthetic market data for testing."""

    # Base prices for common symbols (starting points)
    BASE_PRICES = {
        "BTC-USD": Decimal("50000.00"),
        "ETH-USD": Decimal("3000.00"),
        "SOL-USD": Decimal("100.00"),
        "AVAX-USD": Decimal("30.00"),
        "MATIC-USD": Decimal("0.80"),
        "LINK-USD": Decimal("15.00"),
    }

    def __init__(self, config: GeneratorConfig):
        """Initialize generator with configuration."""
        self.config = config
        self.rng = random.Random(config.seed)

        # Initialize price state for each symbol
        self.current_prices = {}
        for symbol in config.symbols:
            base_price = self.BASE_PRICES.get(symbol, Decimal("100.00"))
            self.current_prices[symbol] = base_price

    def generate_trade(
        self,
        exchange: str,
        symbol: str,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Generate synthetic trade message."""
        # Get current price and add random walk
        current_price = self.current_prices[symbol]
        price_change_pct = Decimal(str(self.rng.uniform(-0.001, 0.001)))  # ±0.1%
        new_price = current_price * (1 + price_change_pct)

        # Update price state
        self.current_prices[symbol] = new_price

        # Quantize to 2 decimals
        quantize_pattern = Decimal("0.01")
        price = new_price.quantize(quantize_pattern, rounding=ROUND_HALF_UP)

        # Generate random amount (0.001 to 10.0)
        amount = Decimal(str(self.rng.uniform(0.001, 10.0)))
        amount = amount.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        return {
            "type": "trade",
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": timestamp,
            "price": price,
            "amount": amount,
            "side": self.rng.choice(["buy", "sell"]),
        }

    def generate_ticker(
        self,
        exchange: str,
        symbol: str,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Generate synthetic ticker message."""
        current_price = self.current_prices[symbol]

        # Bid slightly below current, ask slightly above (spread)
        spread_pct = Decimal(str(self.rng.uniform(0.0001, 0.001)))  # 0.01-0.1% spread
        bid = current_price * (1 - spread_pct)
        ask = current_price * (1 + spread_pct)

        # Quantize to 2 decimals
        quantize_pattern = Decimal("0.01")
        bid = bid.quantize(quantize_pattern, rounding=ROUND_HALF_UP)
        ask = ask.quantize(quantize_pattern, rounding=ROUND_HALF_UP)

        return {
            "type": "ticker",
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": timestamp,
            "bid": bid,
            "ask": ask,
        }

    def generate_orderbook(
        self,
        exchange: str,
        symbol: str,
        timestamp: float,
        depth: int = 10,
    ) -> Dict[str, Any]:
        """Generate synthetic order book message."""
        current_price = self.current_prices[symbol]

        # Generate bids (descending prices from current)
        bids = []
        for i in range(depth):
            price_offset = Decimal(str(0.001 * (i + 1)))  # 0.1% per level
            price = current_price * (1 - price_offset)
            amount = Decimal(str(self.rng.uniform(0.1, 5.0)))

            quantize_pattern = Decimal("0.01")
            price = price.quantize(quantize_pattern, rounding=ROUND_HALF_UP)
            amount = amount.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            bids.append({"price": price, "amount": amount})

        # Generate asks (ascending prices from current)
        asks = []
        for i in range(depth):
            price_offset = Decimal(str(0.001 * (i + 1)))  # 0.1% per level
            price = current_price * (1 + price_offset)
            amount = Decimal(str(self.rng.uniform(0.1, 5.0)))

            quantize_pattern = Decimal("0.01")
            price = price.quantize(quantize_pattern, rounding=ROUND_HALF_UP)
            amount = amount.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            asks.append({"price": price, "amount": amount})

        return {
            "type": "orderbook",
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": timestamp,
            "bids": bids,
            "asks": asks,
        }

    def generate_funding(
        self,
        exchange: str,
        symbol: str,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Generate synthetic funding rate message."""
        # Funding rate typically -0.1% to +0.1%
        rate = Decimal(str(self.rng.uniform(-0.001, 0.001)))
        rate = rate.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

        return {
            "type": "funding",
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": timestamp,
            "rate": rate,
        }

    def generate_stream(
        self,
        messages_per_second: int,
        duration_seconds: int,
        data_types: List[str],
    ) -> Iterator[Dict[str, Any]]:
        """Generate message stream at constant rate."""
        if not data_types:
            raise ValueError("data_types must not be empty")

        total_messages = messages_per_second * duration_seconds
        interval_seconds = 1.0 / messages_per_second

        start_time = time.time()
        base_timestamp = datetime.now(datetime.UTC).timestamp() if hasattr(datetime, 'UTC') else datetime.utcnow().timestamp()

        for i in range(total_messages):
            # Calculate timestamp for this message
            elapsed = i * interval_seconds
            timestamp = base_timestamp + elapsed

            # Randomly select exchange, symbol, and data type
            exchange = self.rng.choice(self.config.exchanges)
            symbol = self.rng.choice(self.config.symbols)
            data_type = self.rng.choice(data_types)

            # Generate message based on type
            if data_type == "trade":
                message = self.generate_trade(exchange, symbol, timestamp)
            elif data_type == "ticker":
                message = self.generate_ticker(exchange, symbol, timestamp)
            elif data_type == "orderbook":
                message = self.generate_orderbook(exchange, symbol, timestamp)
            elif data_type == "funding":
                message = self.generate_funding(exchange, symbol, timestamp)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")

            yield message

            # Rate limiting (optional, for real-time simulation)
            # elapsed_real = time.time() - start_time
            # expected_elapsed = (i + 1) * interval_seconds
            # sleep_time = expected_elapsed - elapsed_real
            # if sleep_time > 0:
            #     time.sleep(sleep_time)

    def generate_spike_scenario(
        self,
        start_rate: int,
        peak_rate: int,
        end_rate: int,
        ramp_up_seconds: int,
        peak_seconds: int,
        ramp_down_seconds: int,
    ) -> Iterator[Dict[str, Any]]:
        """Generate spike scenario: ramp up → peak → ramp down."""
        base_timestamp = datetime.now(datetime.UTC).timestamp() if hasattr(datetime, 'UTC') else datetime.utcnow().timestamp()
        message_count = 0

        # Phase 1: Ramp up (linear increase from start_rate to peak_rate)
        for elapsed in range(ramp_up_seconds):
            progress = elapsed / ramp_up_seconds
            current_rate = int(start_rate + (peak_rate - start_rate) * progress)

            for _ in range(current_rate):
                timestamp = base_timestamp + elapsed + (message_count / current_rate)
                exchange = self.rng.choice(self.config.exchanges)
                symbol = self.rng.choice(self.config.symbols)
                data_type = self.rng.choice(self.config.data_types)

                if data_type == "trade":
                    message = self.generate_trade(exchange, symbol, timestamp)
                elif data_type == "ticker":
                    message = self.generate_ticker(exchange, symbol, timestamp)
                elif data_type == "orderbook":
                    message = self.generate_orderbook(exchange, symbol, timestamp)
                else:
                    message = self.generate_funding(exchange, symbol, timestamp)

                yield message
                message_count += 1

        # Phase 2: Peak (constant peak_rate)
        peak_start = ramp_up_seconds
        for elapsed in range(peak_seconds):
            for _ in range(peak_rate):
                timestamp = base_timestamp + peak_start + elapsed + (message_count / peak_rate)
                exchange = self.rng.choice(self.config.exchanges)
                symbol = self.rng.choice(self.config.symbols)
                data_type = self.rng.choice(self.config.data_types)

                if data_type == "trade":
                    message = self.generate_trade(exchange, symbol, timestamp)
                elif data_type == "ticker":
                    message = self.generate_ticker(exchange, symbol, timestamp)
                elif data_type == "orderbook":
                    message = self.generate_orderbook(exchange, symbol, timestamp)
                else:
                    message = self.generate_funding(exchange, symbol, timestamp)

                yield message
                message_count += 1

        # Phase 3: Ramp down (linear decrease from peak_rate to end_rate)
        ramp_down_start = ramp_up_seconds + peak_seconds
        for elapsed in range(ramp_down_seconds):
            progress = elapsed / ramp_down_seconds
            current_rate = int(peak_rate - (peak_rate - end_rate) * progress)

            for _ in range(current_rate):
                timestamp = base_timestamp + ramp_down_start + elapsed + (message_count / current_rate)
                exchange = self.rng.choice(self.config.exchanges)
                symbol = self.rng.choice(self.config.symbols)
                data_type = self.rng.choice(self.config.data_types)

                if data_type == "trade":
                    message = self.generate_trade(exchange, symbol, timestamp)
                elif data_type == "ticker":
                    message = self.generate_ticker(exchange, symbol, timestamp)
                elif data_type == "orderbook":
                    message = self.generate_orderbook(exchange, symbol, timestamp)
                else:
                    message = self.generate_funding(exchange, symbol, timestamp)

                yield message
                message_count += 1


class KafkaOutputAdapter:
    """Output adapter for Kafka topics."""

    def __init__(
        self,
        bootstrap_servers: List[str],
        topic_prefix: str = "cryptofeed.test",
        producer=None,
    ):
        """Initialize Kafka output adapter."""
        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self._producer = producer

        if self._producer is None:
            # Import here to avoid dependency if not using Kafka output
            from confluent_kafka import Producer

            config = {
                "bootstrap.servers": ",".join(bootstrap_servers),
                "acks": "all",
                "enable.idempotence": True,
                "compression.type": "snappy",
            }
            self._producer = Producer(config)

    def send(self, message: Dict[str, Any]) -> None:
        """Send single message to Kafka topic."""
        data_type = message["type"]
        topic = f"{self.topic_prefix}.{data_type}"

        # Serialize message (convert Decimal to string for JSON)
        serialized = self._serialize_message(message)

        # Produce to Kafka
        self._producer.produce(
            topic=topic,
            value=serialized.encode("utf-8"),
            key=message["symbol"].encode("utf-8"),
        )

    def send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send batch of messages to Kafka."""
        for message in messages:
            self.send(message)

        # Poll to trigger delivery callbacks
        self._producer.poll(0)

    def close(self) -> None:
        """Flush and close producer."""
        if self._producer is not None:
            self._producer.flush()

    def _serialize_message(self, message: Dict[str, Any]) -> str:
        """Serialize message to JSON (converting Decimal to string)."""
        def decimal_encoder(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(message, default=decimal_encoder)


class FileOutputAdapter:
    """Output adapter for file storage (JSON lines or Parquet)."""

    def __init__(
        self,
        output_dir: str,
        format_type: str = "jsonlines",
        compression: str = "gzip",
    ):
        """Initialize file output adapter."""
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        self.compression = compression

        # Create output directory if not exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize output files
        self.output_files = {}

    def send(self, message: Dict[str, Any]) -> None:
        """Write single message to file."""
        data_type = message["type"]

        # Get or create output file for this data type
        if data_type not in self.output_files:
            filename = f"{data_type}.jsonl"
            if self.compression == "gzip":
                filename += ".gz"

            filepath = self.output_dir / filename
            self.output_files[data_type] = filepath.open("a")

        # Serialize and write
        serialized = self._serialize_message(message)
        self.output_files[data_type].write(serialized + "\n")

    def send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Write batch of messages to file."""
        for message in messages:
            self.send(message)

    def close(self) -> None:
        """Close all output files."""
        for file_handle in self.output_files.values():
            file_handle.close()

    def _serialize_message(self, message: Dict[str, Any]) -> str:
        """Serialize message to JSON."""
        def decimal_encoder(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(message, default=decimal_encoder)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for Kafka producer staging validation"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to test data configuration YAML file",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Volume profile name (low, medium, high, spike, smoke)",
    )
    parser.add_argument(
        "--output",
        default="stdout",
        choices=["stdout", "kafka", "file"],
        help="Output destination (default: stdout)",
    )
    parser.add_argument(
        "--kafka-brokers",
        default="localhost:9092",
        help="Kafka broker addresses (comma-separated)",
    )
    parser.add_argument(
        "--file-output-dir",
        default="./test-data-output",
        help="Output directory for file output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without generating data",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = DataGenerationConfig.from_yaml(args.config)

    # Get requested profile
    if args.profile not in config.profiles:
        print(f"Error: Profile '{args.profile}' not found in configuration")
        print(f"Available profiles: {', '.join(config.profiles.keys())}")
        return 1

    profile = config.profiles[args.profile]

    print(f"\nProfile: {profile.name}")
    print(f"Description: {profile.description}")
    print(f"Message rate: {profile.messages_per_second:,} msg/s")
    print(f"Duration: {profile.duration_seconds} seconds")
    print(f"Total messages: {profile.total_messages:,}")
    print(f"Exchanges: {', '.join(profile.exchanges)}")
    print(f"Symbols: {', '.join(profile.symbols)}")
    print(f"Data types: {', '.join(profile.data_types)}")
    print()

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return 0

    # Initialize generator
    generator_config = GeneratorConfig(
        exchanges=profile.exchanges,
        symbols=profile.symbols,
        data_types=profile.data_types,
        seed=args.seed,
    )
    generator = SyntheticDataGenerator(generator_config)

    # Initialize output adapter
    if args.output == "kafka":
        print(f"Output: Kafka ({args.kafka_brokers})")
        output_adapter = KafkaOutputAdapter(
            bootstrap_servers=args.kafka_brokers.split(","),
        )
    elif args.output == "file":
        print(f"Output: File ({args.file_output_dir})")
        output_adapter = FileOutputAdapter(
            output_dir=args.file_output_dir,
        )
    else:
        print("Output: stdout")
        output_adapter = None

    # Generate messages
    print("\nGenerating test data...")
    start_time = time.time()
    message_count = 0

    try:
        if profile.spike_scenario:
            # Spike scenario
            messages = generator.generate_spike_scenario(
                start_rate=profile.start_rate,
                peak_rate=profile.peak_rate,
                end_rate=profile.end_rate,
                ramp_up_seconds=profile.ramp_up_seconds,
                peak_seconds=profile.peak_seconds,
                ramp_down_seconds=profile.ramp_down_seconds,
            )
        else:
            # Constant rate scenario
            messages = generator.generate_stream(
                messages_per_second=profile.messages_per_second,
                duration_seconds=profile.duration_seconds,
                data_types=profile.data_types,
            )

        # Process messages
        for message in messages:
            if output_adapter is not None:
                output_adapter.send(message)
            else:
                # Print to stdout (first 10 messages only)
                if message_count < 10:
                    print(json.dumps(message, default=str))

            message_count += 1

            # Progress update every 10K messages
            if message_count % 10_000 == 0:
                elapsed = time.time() - start_time
                rate = message_count / elapsed if elapsed > 0 else 0
                print(f"Generated {message_count:,} messages ({rate:.0f} msg/s)")

    finally:
        # Close output adapter
        if output_adapter is not None:
            output_adapter.close()

    elapsed = time.time() - start_time
    final_rate = message_count / elapsed if elapsed > 0 else 0

    print(f"\nGeneration complete!")
    print(f"Total messages: {message_count:,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average rate: {final_rate:.0f} msg/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
