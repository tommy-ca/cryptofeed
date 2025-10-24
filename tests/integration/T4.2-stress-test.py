"""
Stress test for concurrent proxy operations.

Run 20+ feeds simultaneously through Mullvad proxies for configurable duration.
Monitor memory, connections, and error rates.

Usage:
    python tests/integration/T4.2-stress-test.py --duration=3600 --feeds=20
    
Environment Variables:
    CRYPTOFEED_TEST_SOCKS_PROXY: Proxy URL for all feeds
    CRYPTOFEED_TEST_STRESS_DURATION: Duration in seconds (default: 300)
    CRYPTOFEED_TEST_STRESS_FEEDS: Number of concurrent feeds (default: 20)
"""
import argparse
import asyncio
import logging
import os
import psutil
import time
from collections import defaultdict
from decimal import Decimal
from typing import Dict, List

from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, L2_BOOK, TICKER
from cryptofeed.proxy import ProxySettings, ProxyConfig, ConnectionProxies, init_proxy_system


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


class StressTestMetrics:
    """Track metrics during stress test execution."""
    
    def __init__(self):
        self.start_time = time.time()
        self.message_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.feed_start_times = {}
        self.feed_first_message = {}
        self.memory_snapshots = []
        self.process = psutil.Process()
        
    def record_message(self, feed_name: str, channel: str):
        """Record successful message receipt."""
        key = f"{feed_name}:{channel}"
        self.message_counts[key] += 1
        
        if key not in self.feed_first_message:
            elapsed = time.time() - self.feed_start_times.get(feed_name, self.start_time)
            self.feed_first_message[key] = elapsed
            LOG.info(f"First message from {key} after {elapsed:.2f}s")
    
    def record_error(self, feed_name: str, error: Exception):
        """Record error occurrence."""
        error_type = type(error).__name__
        key = f"{feed_name}:{error_type}"
        self.error_counts[key] += 1
        LOG.error(f"Error in {feed_name}: {error_type} - {str(error)}")
    
    def snapshot_memory(self):
        """Take memory usage snapshot."""
        mem_info = self.process.memory_info()
        snapshot = {
            'timestamp': time.time() - self.start_time,
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
        }
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def print_summary(self):
        """Print test summary."""
        duration = time.time() - self.start_time
        total_messages = sum(self.message_counts.values())
        total_errors = sum(self.error_counts.values())
        
        print("\n" + "="*80)
        print("STRESS TEST SUMMARY")
        print("="*80)
        print(f"Duration: {duration:.2f}s")
        print(f"Total Messages: {total_messages:,}")
        print(f"Total Errors: {total_errors}")
        print(f"Message Rate: {total_messages / duration:.2f} msg/s")
        
        if self.memory_snapshots:
            first_mem = self.memory_snapshots[0]['rss_mb']
            last_mem = self.memory_snapshots[-1]['rss_mb']
            mem_growth = ((last_mem - first_mem) / first_mem) * 100
            print(f"\nMemory Usage:")
            print(f"  Initial: {first_mem:.2f} MB")
            print(f"  Final: {last_mem:.2f} MB")
            print(f"  Growth: {mem_growth:+.2f}%")
        
        print(f"\nMessages by Feed:")
        for key, count in sorted(self.message_counts.items()):
            print(f"  {key}: {count:,}")
        
        if self.error_counts:
            print(f"\nErrors by Type:")
            for key, count in sorted(self.error_counts.items()):
                print(f"  {key}: {count}")
        
        print("="*80)
        
        # Success criteria check
        success = True
        if mem_growth > 5.0:
            print("⚠️  FAIL: Memory growth exceeds 5%")
            success = False
        if total_errors > total_messages * 0.05:  # >5% error rate
            print("⚠️  FAIL: Error rate exceeds 5%")
            success = False
        if total_messages == 0:
            print("⚠️  FAIL: No messages received")
            success = False
        
        if success:
            print("✅ PASS: All success criteria met")
        
        return success


async def run_stress_test(proxy_url: str, duration: int, num_feeds: int):
    """
    Run stress test with multiple concurrent feeds.
    
    Args:
        proxy_url: SOCKS5 proxy URL
        duration: Test duration in seconds
        num_feeds: Number of concurrent feeds to run
    """
    # Initialize proxy system
    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(
            http=ProxyConfig(url=proxy_url),
            websocket=ProxyConfig(url=proxy_url)
        )
    )
    init_proxy_system(settings)
    
    # Initialize metrics
    metrics = StressTestMetrics()
    
    # Define feed configurations
    feed_configs = [
        # High-volume pairs
        {'exchange': 'Binance', 'symbols': ['BTC-USDT', 'ETH-USDT'], 'channels': [TRADES]},
        {'exchange': 'Binance', 'symbols': ['BTC-USDT'], 'channels': [L2_BOOK], 'max_depth': 10},
        {'exchange': 'Coinbase', 'symbols': ['BTC-USD', 'ETH-USD'], 'channels': [TRADES]},
        {'exchange': 'Coinbase', 'symbols': ['BTC-USD'], 'channels': [L2_BOOK]},
        
        # Additional exchanges
        {'exchange': 'Bybit', 'symbols': ['BTC-USDT-PERP', 'ETH-USDT-PERP'], 'channels': [TRADES]},
        {'exchange': 'Bybit', 'symbols': ['BTC-USDT-PERP'], 'channels': [TICKER]},
        {'exchange': 'Kraken', 'symbols': ['BTC-USD', 'ETH-USD'], 'channels': [TRADES]},
        {'exchange': 'Kraken', 'symbols': ['BTC-USD'], 'channels': [L2_BOOK], 'max_depth': 10},
    ]
    
    # Replicate configs to reach target feed count
    while len(feed_configs) < num_feeds:
        feed_configs.extend(feed_configs[:num_feeds - len(feed_configs)])
    
    feed_configs = feed_configs[:num_feeds]
    
    # Create callbacks
    def make_callback(feed_name: str, channel: str):
        async def callback(**kwargs):
            try:
                metrics.record_message(feed_name, channel)
            except Exception as e:
                metrics.record_error(feed_name, e)
        return callback
    
    # Build FeedHandler
    fh = FeedHandler()
    
    for i, config in enumerate(feed_configs):
        feed_name = f"{config['exchange']}_{i}"
        metrics.feed_start_times[feed_name] = time.time()
        
        # Import exchange class dynamically
        exchange_module = __import__(f"cryptofeed.exchanges.{config['exchange'].lower()}", 
                                     fromlist=[config['exchange']])
        exchange_class = getattr(exchange_module, config['exchange'])
        
        # Build callbacks dict
        callbacks = {}
        for channel in config['channels']:
            callbacks[channel] = make_callback(feed_name, channel)
        
        # Create feed instance
        feed_kwargs = {
            'symbols': config['symbols'],
            'channels': config['channels'],
            'callbacks': callbacks,
        }
        if 'max_depth' in config:
            feed_kwargs['max_depth'] = config['max_depth']
        
        feed = exchange_class(**feed_kwargs)
        fh.add_feed(feed)
        LOG.info(f"Added {feed_name}: {config['symbols']} {config['channels']}")
    
    # Start memory monitoring task
    async def memory_monitor():
        while True:
            metrics.snapshot_memory()
            await asyncio.sleep(30)  # Snapshot every 30 seconds
    
    monitor_task = asyncio.create_task(memory_monitor())
    
    # Run feeds for specified duration
    LOG.info(f"Starting {num_feeds} feeds for {duration} seconds...")
    try:
        await asyncio.wait_for(
            fh.run_async(),
            timeout=duration
        )
    except asyncio.TimeoutError:
        LOG.info("Test duration reached, shutting down...")
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    # Print summary
    success = metrics.print_summary()
    return success


def main():
    parser = argparse.ArgumentParser(description='Stress test for concurrent proxy operations')
    parser.add_argument('--duration', type=int, 
                       default=int(os.getenv('CRYPTOFEED_TEST_STRESS_DURATION', '300')),
                       help='Test duration in seconds (default: 300)')
    parser.add_argument('--feeds', type=int,
                       default=int(os.getenv('CRYPTOFEED_TEST_STRESS_FEEDS', '20')),
                       help='Number of concurrent feeds (default: 20)')
    parser.add_argument('--proxy', type=str,
                       default=os.getenv('CRYPTOFEED_TEST_SOCKS_PROXY'),
                       help='SOCKS5 proxy URL (default: from env)')
    
    args = parser.parse_args()
    
    if not args.proxy:
        print("Error: Proxy URL required (--proxy or CRYPTOFEED_TEST_SOCKS_PROXY)")
        return 1
    
    LOG.info(f"Configuration:")
    LOG.info(f"  Proxy: {args.proxy}")
    LOG.info(f"  Duration: {args.duration}s")
    LOG.info(f"  Feeds: {args.feeds}")
    
    try:
        success = asyncio.run(run_stress_test(args.proxy, args.duration, args.feeds))
        return 0 if success else 1
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
        return 1
    except Exception as e:
        LOG.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
