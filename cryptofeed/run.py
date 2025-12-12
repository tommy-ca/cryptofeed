"""
Container entrypoint for cryptofeed.

This module provides a command-line interface for running cryptofeed
in containerized environments (Docker, Kubernetes).

Usage:
    python -m cryptofeed.run --config /config/config.yaml
    python -m cryptofeed.run --help
"""
import argparse
import asyncio
import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from cryptofeed.config import Config
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.health_server import HealthServer
from cryptofeed.exchanges import EXCHANGE_MAP
from cryptofeed.backends.kafka.callback import (
    TradeKafka, BookKafka, TickerKafka, FundingKafka,
    OpenInterestKafka, LiquidationsKafka, CandlesKafka
)
from cryptofeed.settings import Settings


LOG = logging.getLogger('feedhandler')


# Mapping of data channel names to Kafka callback classes
KAFKA_CALLBACK_MAP = {
    'trades': TradeKafka,
    'l2_book': BookKafka,
    'ticker': TickerKafka,
    'funding': FundingKafka,
    'open_interest': OpenInterestKafka,
    'liquidations': LiquidationsKafka,
    'candles': CandlesKafka,
}


def setup_logging(level: str = 'INFO'):
    """Configure logging for container environment."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cryptofeed - Cryptocurrency market data ingestion platform',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default=os.environ.get('CRYPTOFEED_CONFIG', '/config/config.yaml'),
        help='Path to configuration file (YAML)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default=os.environ.get('LOG_LEVEL', 'INFO'),
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )

    parser.add_argument(
        '--proxy-config',
        type=str,
        default=os.environ.get('PROXY_CONFIG_PATH', '/config/proxy.yaml'),
        help='Path to proxy configuration file (YAML)'
    )

    return parser.parse_args()


def interpolate_env_vars(value: Any) -> Any:
    """
    Recursively interpolate environment variables in config values.

    Replaces ${VAR_NAME} with the value from os.environ.get('VAR_NAME').
    If the environment variable is not set, returns None for that value.

    Args:
        value: Configuration value (can be str, dict, list, or other)

    Returns:
        Value with environment variables interpolated
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        if matches:
            result = value
            for var_name in matches:
                env_value = os.environ.get(var_name, '')
                result = result.replace(f'${{{var_name}}}', env_value)
            # Return None if the result is empty (env var not set)
            return result if result else None
        return value
    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [interpolate_env_vars(v) for v in value]
    else:
        return value


def setup_kafka_callbacks(kafka_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Kafka backend callbacks from configuration.

    Args:
        kafka_config: Kafka configuration dictionary from YAML

    Returns:
        Dictionary mapping channel names to Kafka callback instances
    """
    if not kafka_config:
        return {}

    # Extract Kafka producer configuration
    bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])

    # Allow override from environment variable
    env_bootstrap = os.environ.get('KAFKA_BOOTSTRAP_SERVERS')
    if env_bootstrap:
        bootstrap_servers = env_bootstrap.split(',')

    kafka_kwargs = {
        'bootstrap_servers': ','.join(bootstrap_servers) if isinstance(bootstrap_servers, list) else bootstrap_servers,
    }

    # Add optional producer settings
    if 'acks' in kafka_config:
        kafka_kwargs['acks'] = kafka_config['acks']
    if 'compression_type' in kafka_config:
        kafka_kwargs['compression_type'] = kafka_config['compression_type']
    if 'enable_idempotence' in kafka_config:
        kafka_kwargs['enable_idempotence'] = kafka_config['enable_idempotence']
    if 'request_timeout_ms' in kafka_config:
        kafka_kwargs['request_timeout_ms'] = kafka_config['request_timeout_ms']
    if 'retries' in kafka_config:
        kafka_kwargs['retries'] = kafka_config['retries']
    if 'retry_backoff_ms' in kafka_config:
        kafka_kwargs['retry_backoff_ms'] = kafka_config['retry_backoff_ms']
    if 'linger_ms' in kafka_config:
        kafka_kwargs['linger_ms'] = kafka_config['linger_ms']
    if 'batch_size' in kafka_config:
        kafka_kwargs['batch_size'] = kafka_config['batch_size']

    # Create callback instances for each channel type
    callbacks = {}
    for channel, callback_cls in KAFKA_CALLBACK_MAP.items():
        callbacks[channel] = callback_cls(**kafka_kwargs)

    LOG.info(f"Configured Kafka callbacks with bootstrap servers: {kafka_kwargs['bootstrap_servers']}")
    return callbacks


def configure_exchange(fh: FeedHandler, exchange_name: str, exchange_config: Dict[str, Any],
                      kafka_callbacks: Optional[Dict[str, Any]] = None):
    """
    Configure and add an exchange to the FeedHandler.

    Args:
        fh: FeedHandler instance
        exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        exchange_config: Exchange configuration from YAML
        kafka_callbacks: Optional Kafka callback instances
    """
    # Normalize exchange name to uppercase for lookup
    exchange_key = exchange_name.upper()

    # Skip if this is a special config section (not an exchange)
    if exchange_key in ('LOG', 'UVLOOP', 'KAFKA', 'IGNORE_INVALID_INSTRUMENTS',
                        'BACKEND_MULTIPROCESSING', 'EXCHANGE_CREDENTIALS'):
        return

    # Check if exchange is supported
    if exchange_key not in EXCHANGE_MAP:
        LOG.warning(f"Exchange '{exchange_name}' not found in EXCHANGE_MAP, skipping")
        return

    # Get channels and symbols from config
    channels = exchange_config.get('channels', [])
    symbols = exchange_config.get('symbols', [])

    if not channels or not symbols:
        LOG.warning(f"Exchange '{exchange_name}' has no channels or symbols configured, skipping")
        return

    # Build callbacks dict for this exchange (filter to only requested channels)
    callbacks = {}
    if kafka_callbacks:
        for channel in channels:
            if channel in kafka_callbacks:
                callbacks[channel] = kafka_callbacks[channel]

    # Build feed kwargs
    feed_kwargs = {
        'channels': channels,
        'symbols': symbols,
    }

    if callbacks:
        feed_kwargs['callbacks'] = callbacks

    # Add optional exchange-specific parameters
    if 'max_depth' in exchange_config:
        feed_kwargs['max_depth'] = exchange_config['max_depth']

    # Add the feed to FeedHandler
    try:
        fh.add_feed(exchange_key, **feed_kwargs)
        LOG.info(f"Added {exchange_name} feed: {len(symbols)} symbols, {len(channels)} channels")
    except Exception as e:
        LOG.error(f"Failed to add {exchange_name} feed: {e}", exc_info=True)


def load_proxy_mapping(path: str) -> Optional[Dict[str, Any]]:
    """Load proxy YAML as a plain mapping (no templating)."""
    proxy_path = Path(path)
    if not proxy_path.exists():
        return None
    data = yaml.safe_load(proxy_path.read_text()) or {}
    return data or None


async def run_feedhandler(config_path: str, proxy_config_path: Optional[str] = None):
    """
    Run feedhandler with configuration from file.

    Args:
        config_path: Path to YAML configuration file

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config file is invalid
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    LOG.info(f"Loading configuration from: {config_path}")

    # Load configuration via pydantic Settings (YAML + env with env highest)
    settings = Settings(config_path=config_path)
    config_dict = settings.to_feed_config()
    LOG.info("Config: loaded via Settings (YAML path=%s, env overrides applied)", config_path)

    # Extract Kafka configuration
    kafka_config = config_dict.get('kafka', {})
    kafka_callbacks = setup_kafka_callbacks(kafka_config) if kafka_config else None
    proxy_mapping = load_proxy_mapping(proxy_config_path or '/config/proxy.yaml')

    # Create FeedHandler instance
    config_obj = Config(config=config_dict)
    fh = FeedHandler(config=config_obj, proxy_settings=proxy_mapping)

    # Configure exchanges from YAML
    exchanges_configured = 0
    for key, value in config_dict.items():
        if isinstance(value, dict) and ('channels' in value or 'symbols' in value):
            configure_exchange(fh, key, value, kafka_callbacks)
            exchanges_configured += 1

    if exchanges_configured == 0:
        LOG.warning("No exchanges configured in YAML file")
        LOG.warning("Add exchange configurations to start receiving data")
        LOG.warning("Example: Add 'binance:', 'coinbase:', etc. sections to your config.yaml")

    # Initialize health server
    health_server = HealthServer(
        port=int(os.environ.get('HEALTH_PORT', '8080')),
        shutdown_timeout=30.0
    )

    # Start health server
    await health_server.start()
    LOG.info("Health server started")

    # Setup graceful shutdown handler
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        LOG.info(f"Received signal {sig}, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start FeedHandler in the background
    LOG.info("Starting FeedHandler...")

    # Run FeedHandler without blocking (start_loop=False)
    fh.run(start_loop=False, install_signal_handlers=False)

    # Wait for shutdown signal
    LOG.info("Cryptofeed is running. Press Ctrl+C or send SIGTERM to stop.")
    await shutdown_event.wait()

    # Graceful shutdown
    LOG.info("Shutting down gracefully...")

    # Stop health server
    await health_server.shutdown()
    await health_server.stop()

    # Stop FeedHandler
    loop = asyncio.get_event_loop()
    await fh.stop_async(loop=loop)

    LOG.info("Shutdown complete")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    LOG.info("=" * 60)
    LOG.info("Cryptofeed Container Starting")
    LOG.info("=" * 60)
    LOG.info(f"Config file: {args.config}")
    LOG.info(f"Log level: {args.log_level}")

    try:
        # Run asyncio event loop
        asyncio.run(run_feedhandler(args.config, args.proxy_config))
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    except Exception as e:
        LOG.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        LOG.info("Cryptofeed Container Stopped")


if __name__ == '__main__':
    main()
