"""Regional proxy validation matrix for Binance E2E tests.

Tests Binance accessibility through different Mullvad relay regions:
- US East (New York): Expected HTTP 451 geofencing
- EU (Frankfurt, Germany): Full access expected
- Asia (Singapore): Full access expected

Usage:
    export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true
    export KAFKA_BOOTSTRAP_SERVERS=localhost:19092
    python -m pytest tests/integration/kafka/test_regional_proxy_matrix.py -v -s
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from cryptofeed.defines import TRADES
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.proxy import ProxySettings, init_proxy_system
from tests.integration.kafka.helpers import ConsumedRecord, consume_one
from tests.integration.kafka.topic_provision import ensure_topics_exist
from uuid import uuid4


BINANCE_E2E_ENV = "CRYPTODATA_RUN_BINANCE_KAFKA_E2E"

# Mullvad relay endpoints by region
REGIONAL_RELAYS = {
    "US_EAST": "socks5://us-nyc-wg-socks5-301.relays.mullvad.net:1080",
    "EU_CENTRAL": "socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080",
    "ASIA_PACIFIC": "socks5://sg-sin-wg-socks5-001.relays.mullvad.net:1080",
}


def _env_enabled() -> bool:
    value = os.getenv(BINANCE_E2E_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


class _TestKafkaProtobufCallback(KafkaProtobufCallback):
    """Test shim that accepts the multiprocess kwarg used by FeedHandler.start."""

    def start(self, loop, multiprocess: bool | None = None):  # type: ignore[override]
        return super().start(loop)


async def _test_binance_via_proxy(
    redpanda: str,
    proxy_url: str,
    region_name: str,
    expected_accessible: bool,
) -> tuple[bool, str]:
    """Test Binance REST/WS access via specific proxy.

    Returns:
        (success, message) tuple
    """
    # Configure proxy
    os.environ["CRYPTOFEED_PROXY_ENABLED"] = "true"
    os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL"] = proxy_url
    os.environ["CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL"] = proxy_url

    # Initialize proxy system
    from cryptofeed.proxy import load_proxy_settings
    settings = load_proxy_settings()
    init_proxy_system(settings)

    # Test REST access first (exchangeInfo)
    try:
        import aiohttp
        from aiohttp_socks import ProxyConnector

        connector = ProxyConnector.from_url(proxy_url)
        timeout = aiohttp.ClientTimeout(total=20)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get("https://api.binance.com/api/v3/exchangeInfo") as resp:
                if resp.status == 451:
                    if expected_accessible:
                        return False, f"❌ {region_name}: HTTP 451 (geofenced, unexpected)"
                    else:
                        return True, f"✅ {region_name}: HTTP 451 (geofenced, expected)"
                elif resp.status != 200:
                    return False, f"⚠️  {region_name}: HTTP {resp.status} (unexpected error)"

    except Exception as exc:
        return False, f"⚠️  {region_name}: REST failed - {exc}"

    # If REST succeeds, test WS via E2E flow
    try:
        topic = "cryptofeed.trade"
        await ensure_topics_exist(redpanda, [topic])

        fh = FeedHandler()
        loop = asyncio.get_running_loop()

        kafka_cb = _TestKafkaProtobufCallback(
            bootstrap_servers=[redpanda],
            producer_factory=None,
            metrics_exporter=None,
            metrics_enabled=False,
        )
        kafka_cb._topic_strategy = "consolidated"
        kafka_cb._enable_partition_key_cache = False
        kafka_cb.start(loop)

        if not kafka_cb.is_connected():
            return False, f"⚠️  {region_name}: Kafka connection failed"

        async def _handler(obj, receipt_timestamp):
            await kafka_cb._handle_message("trade", obj, receipt_timestamp)

        fh.add_feed(
            "BINANCE",
            symbols=["BTC-USDT"],
            channels=[TRADES],
            callbacks={TRADES: [_handler]},
        )

        for feed in fh.feeds:
            feed.start(loop)

        # Try to consume one message
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=30.0,
                group_id=f"cf-regional-{uuid4().hex}",
                offset_reset="latest",
            )

            # Success!
            if expected_accessible:
                result = True, f"✅ {region_name}: Full access (REST + WS)"
            else:
                result = False, f"⚠️  {region_name}: Unexpected success (expected geofencing)"

        except AssertionError:
            # Timeout waiting for message
            result = False, f"⚠️  {region_name}: WS timeout (no messages received)"

        finally:
            # Cleanup
            for feed in fh.feeds:
                feed.stop()
                await feed.shutdown()

            if hasattr(kafka_cb, "stop"):
                await kafka_cb.stop()

        # Reset proxy system
        init_proxy_system(ProxySettings(enabled=False))

        return result

    except Exception as exc:
        return False, f"⚠️  {region_name}: E2E test failed - {exc}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.regional
async def test_binance_regional_access_matrix(redpanda):
    """Test Binance access through different regional proxies.

    Expected behavior:
    - US East: HTTP 451 geofencing (Binance blocked in US)
    - EU Central: Full access
    - Asia Pacific: Full access
    """
    if not _env_enabled():
        pytest.skip(
            f"Regional validation disabled. Set {BINANCE_E2E_ENV}=true to enable. "
            f"See docs/e2e/TEST_PLAN.md for details."
        )

    print("\n" + "="*70)
    print("Regional Proxy Validation Matrix: Binance")
    print("="*70 + "\n")

    results = {}

    # Test US East (expected geofencing)
    print(f"Testing US East (New York)...")
    success, msg = await _test_binance_via_proxy(
        redpanda,
        REGIONAL_RELAYS["US_EAST"],
        "US East (NYC)",
        expected_accessible=False,  # Expect geofencing
    )
    results["US_EAST"] = (success, msg)
    print(f"  {msg}\n")

    # Test EU Central (expected accessible)
    print(f"Testing EU Central (Frankfurt)...")
    success, msg = await _test_binance_via_proxy(
        redpanda,
        REGIONAL_RELAYS["EU_CENTRAL"],
        "EU Central (FRA)",
        expected_accessible=True,  # Expect full access
    )
    results["EU_CENTRAL"] = (success, msg)
    print(f"  {msg}\n")

    # Test Asia Pacific (expected accessible)
    print(f"Testing Asia Pacific (Singapore)...")
    success, msg = await _test_binance_via_proxy(
        redpanda,
        REGIONAL_RELAYS["ASIA_PACIFIC"],
        "Asia Pacific (SIN)",
        expected_accessible=True,  # Expect full access
    )
    results["ASIA_PACIFIC"] = (success, msg)
    print(f"  {msg}\n")

    # Summary
    print("="*70)
    print("Regional Validation Summary")
    print("="*70)

    for region, (success, msg) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {region:15} | {msg}")

    print("="*70 + "\n")

    # Assert all regions behaved as expected
    failures = [msg for success, msg in results.values() if not success]
    assert not failures, f"Regional validation failures: {failures}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.regional
async def test_binance_eu_proxy_quick_validation(redpanda):
    """Quick validation: EU proxy provides full Binance access.

    This is a faster version of the regional matrix test that only
    tests the EU relay (most commonly used).
    """
    if not _env_enabled():
        pytest.skip(f"Regional validation disabled. Set {BINANCE_E2E_ENV}=true to enable.")

    print("\n🔧 Quick regional validation: EU Central (Frankfurt)\n")

    success, msg = await _test_binance_via_proxy(
        redpanda,
        REGIONAL_RELAYS["EU_CENTRAL"],
        "EU Central (FRA)",
        expected_accessible=True,
    )

    print(f"  {msg}\n")

    assert success, f"EU proxy validation failed: {msg}"
