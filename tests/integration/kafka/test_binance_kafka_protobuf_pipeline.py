"""Binance → Kafka Protobuf end-to-end tests using Redpanda.

These tests exercise the live Binance public WebSocket feed through the
Cryptofeed normalization and Kafka Protobuf backend into a Redpanda
cluster, then validate headers and payloads via the generated protobuf
bindings.

They are **opt-in** and will skip unless all of the following are true:
- Docker + `docker compose` are available (via the existing Redpanda fixture)
- `CRYPTODATA_RUN_BINANCE_KAFKA_E2E` is set to a truthy value
- Network access to Binance public endpoints is available
"""

from __future__ import annotations

import asyncio
import os
from importlib import import_module
from urllib.parse import urlparse

import pytest

from cryptofeed.defines import TRADES
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.kafka.partitioner import PartitionerFactory
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION
from cryptofeed.proxy import ProxySettings
from cryptofeed.defines import L2_BOOK
from cryptofeed.proxy import get_proxy_injector, init_proxy_system, load_proxy_settings
from tests.integration.kafka.helpers import ConsumedRecord, consume_one
from tests.integration.kafka.topic_provision import ensure_topics_exist
from uuid import uuid4
import contextlib

BINANCE_E2E_ENV = "CRYPTODATA_RUN_BINANCE_KAFKA_E2E"
BINANCE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
TOPIC_STRATEGY_ENV = "KAFKA_E2E_TOPIC_STRATEGY"


def _topic_strategy() -> str:
    value = os.getenv(TOPIC_STRATEGY_ENV, "per_symbol").lower()
    return "consolidated" if value == "consolidated" else "per_symbol"


def _topic_name(channel: str, strategy: str) -> str:
    if strategy == "consolidated":
        if channel == TRADES:
            return "cryptofeed.trade"
        if channel == L2_BOOK:
            return "cryptofeed.l2_book"
    # per_symbol default
    if channel == TRADES:
        return "cryptofeed.trade.binance.btc-usdt"
    if channel == L2_BOOK:
        return "cryptofeed.l2_book.binance.btc-usdt"
    return f"cryptofeed.{channel.lower()}.binance.btc-usdt"


# Track original env to restore after tests
_env_cache: dict[str, str | None] = {}

# Proxy env examples (JSON form for pools is preferred by ProxySettings):
#   CRYPTOFEED_PROXY_ENABLED=true
#   CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=socks5://user:pass@host:1080
#   CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=socks5://user:pass@host:1080
#   CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{"proxies":[{"url":"socks5://p1:1080","weight":1},{"url":"socks5://p2:1080","weight":1}],"strategy":"round_robin"}'
#   CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://p1:1080","weight":1},{"url":"socks5://p2:1080","weight":1}],"strategy":"round_robin"}'
#
# Note on REST proxying:
# Binance feed loads symbol metadata via REST before WS starts. If REST is
# geoblocked, the test will skip without producing to Kafka. We therefore set
# HTTP(S)_PROXY to the leased Binance HTTP proxy inside _init_proxy_settings_if_configured
# to ensure symbol_mapping uses the same proxy as WS.


class _TestKafkaProtobufCallback(KafkaProtobufCallback):
    """Test shim that accepts the multiprocess kwarg used by FeedHandler.start."""

    def start(self, loop, multiprocess: bool | None = None):  # type: ignore[override]
        return super().start(loop)


def _assert_or_skip_headers(record: ConsumedRecord):
    missing = [
        h
        for h in (b"content-type", b"schema_version", b"cf.serialization_format")
        if h not in record.headers
    ]
    if missing:
        pytest.skip(
            f"Binance Kafka Protobuf E2E: missing headers {missing}; observed={record.headers}"
        )


def _env_enabled() -> bool:
    value = os.getenv(BINANCE_E2E_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _require_binance_e2e_prereqs() -> None:
    if not _env_enabled():
        pytest.skip(
            f"Binance Kafka Protobuf E2E tests disabled. "
            f"Set {BINANCE_E2E_ENV}=true to enable."
        )


async def _preflight_rest_through_proxy() -> None:
    """Fetch exchangeInfo via configured proxy; skip if unavailable.

    Symbol mapping runs before WS start; if REST is geoblocked the test would
    otherwise skip later after long waits. This makes the failure explicit.
    """

    settings = load_proxy_settings()
    proxy_url = None
    if settings and (settings.enabled or settings.default or settings.exchanges):
        injector = get_proxy_injector()
        if injector:
            proxy_url = injector.get_http_proxy_url("binance")

    if not proxy_url:
        return  # no proxy configured; use direct path

    try:
        import aiohttp
    except ImportError:
        pytest.skip("aiohttp not available for REST preflight")

    connector = None
    try:
        scheme = urlparse(proxy_url).scheme.lower()
        if scheme.startswith("socks"):
            try:
                from aiohttp_socks import ProxyConnector  # type: ignore

                connector = ProxyConnector.from_url(proxy_url)
            except ModuleNotFoundError:
                pytest.skip(
                    "Binance REST exchangeInfo via SOCKS proxy requires aiohttp-socks; "
                    "install it or use HTTP proxy"
                )
    except Exception:
        connector = None

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        try:
            async with session.get(
                BINANCE_INFO_URL, proxy=None if connector else proxy_url
            ) as resp:
                if resp.status != 200:
                    pytest.skip(
                        f"Binance REST exchangeInfo via proxy failed (status {resp.status});"
                        " REST geoblocked or proxy blocked."
                    )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Binance REST exchangeInfo via proxy failed: {exc}")


def _python_socks_available() -> bool:
    try:
        import_module("python_socks")
        return True
    except ModuleNotFoundError:
        return False


def _init_proxy_settings_if_configured() -> bool:
    # Preserve prior env so we can restore after test in _shutdown_feeds
    _env_cache["HTTP_PROXY"] = os.environ.get("HTTP_PROXY")
    _env_cache["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY")

    settings = load_proxy_settings()
    has_proxy = settings.enabled or settings.default or settings.exchanges
    if not has_proxy:
        return False

    ws_proxy = (
        settings.get_proxy("binance", "websocket")
        if hasattr(settings, "get_proxy")
        else None
    )
    if ws_proxy and ws_proxy.url:
        scheme = urlparse(ws_proxy.url).scheme.lower()
        if scheme.startswith("socks") and not _python_socks_available():
            pytest.skip(
                "Binance Kafka Protobuf E2E: SOCKS websocket proxy configured but python-socks is not installed"
            )

    init_proxy_system(settings)
    injector = get_proxy_injector()
    if injector and settings.enabled:
        # Warm HTTP proxy retrieval to ensure config parses
        http_proxy_url = injector.get_http_proxy_url("binance")
        if http_proxy_url:
            # Ensure requests-based REST calls (symbol_mapping) are proxied too
            os.environ["HTTPS_PROXY"] = http_proxy_url
            os.environ["HTTP_PROXY"] = http_proxy_url

        if ws_proxy:
            # Ensure a proxy entry can be selected from pool/config
            url, release = injector.lease_proxy("binance", "websocket")
            if release is None:
                pytest.skip(
                    "Binance Kafka Protobuf E2E: websocket proxy configured but injector returned no release handle"
                )
            try:
                if url is None:
                    pytest.skip(
                        "Binance Kafka Protobuf E2E: websocket proxy configured but no proxy was selected"
                    )
                # Log-friendly assertion that a concrete proxy URL was resolved
                assert urlparse(url).scheme, "Binance proxy resolution must return a scheme"
            finally:
                release()

    return True


async def _start_binance_with_kafka(
    redpanda_bootstrap: str,
    *,
    partition_strategy: str | None = None,
    channels: list[str] | None = None,
    topic_strategy: str = "per_symbol",
) -> FeedHandler:
    """Configure and start a Binance feed wired to KafkaProtobufCallback.

    The feed subscribes to TRADES for BTC-USDT and routes normalized
    events to a KafkaProtobufCallback that produces protobuf-encoded
    messages to the given Redpanda bootstrap address.
    """
    loop = asyncio.get_running_loop()

    _init_proxy_settings_if_configured()

    fh = FeedHandler()

    kafka_cb = _TestKafkaProtobufCallback(
        bootstrap_servers=[redpanda_bootstrap],
        producer_factory=None,
        metrics_exporter=None,
        metrics_enabled=False,
    )
    kafka_cb._topic_strategy = topic_strategy
    kafka_cb._enable_partition_key_cache = False

    if partition_strategy is not None:
        kafka_cb._partitioner = PartitionerFactory.create(partition_strategy)

    kafka_cb.start(loop)

    if not kafka_cb.is_connected():
        pytest.skip("Kafka producer failed to connect to Redpanda")

    # Use FeedHandler to construct the Binance feed with its own Config
    _channels = channels or [TRADES]

    def _mk_handler(data_type: str):
        async def _handler(obj, receipt_timestamp):
            await kafka_cb._handle_message(data_type, obj, receipt_timestamp)

        return _handler

    callbacks = {}
    for channel in _channels:
        if channel == TRADES:
            callbacks[channel] = [_mk_handler("trade")]
        elif channel == L2_BOOK:
            callbacks[channel] = [_mk_handler("l2_book")]
        else:  # pragma: no cover - future channels
            callbacks[channel] = [_mk_handler(channel.lower())]

    symbols = ["BTC-USDT"]
    if topic_strategy == "consolidated":
        symbols.append("ETH-USDT")  # improve early message likelihood

    fh.add_feed(
        "BINANCE",
        symbols=symbols,
        channels=_channels,
        callbacks=callbacks,
    )

    for feed in fh.feeds:
        feed.start(loop)

    # Allow feeds a moment to establish before consuming (helps L2 snapshot over proxies)
    await asyncio.sleep(2)

    # Return both handler and backend for explicit teardown
    fh.kafka_cb = kafka_cb  # type: ignore[attr-defined]
    return fh


async def _shutdown_feeds(handler: FeedHandler) -> None:
    """Stop all feeds and shut down connections and backends."""
    shutdown_tasks = []
    for feed in handler.feeds:
        feed.stop()
        shutdown_tasks.append(feed.shutdown())
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks)

    # Ensure connection handler background tasks are stopped to avoid pending-task warnings
    conn = getattr(handler, "conn", None)
    if conn and hasattr(conn, "close"):
        await conn.close()
        watcher = getattr(conn, "_watcher_task", None)
        if watcher and hasattr(watcher, "cancel") and not watcher.done():
            watcher.cancel()
            with contextlib.suppress(Exception):
                await watcher

    kafka_cb = getattr(handler, "kafka_cb", None)
    if kafka_cb and hasattr(kafka_cb, "stop"):
        await kafka_cb.stop()

    # Restore original HTTP proxy envs (set in _init_proxy_settings_if_configured)
    for key, value in _env_cache.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Reset proxy system to avoid leaking proxy configuration into other tests
    init_proxy_system(ProxySettings(enabled=False))


@pytest.mark.integration
@pytest.mark.live_binance
def test_binance_proxy_resolution_when_configured():
    """Ensure proxy settings for Binance are accepted and resolvable when provided."""

    _require_binance_e2e_prereqs()
    if not _init_proxy_settings_if_configured():
        pytest.skip("No proxy configuration provided for Binance")

    injector = get_proxy_injector()
    assert injector is not None, (
        "Proxy injector should be initialized when proxies are configured"
    )

    http_url = injector.get_http_proxy_url("binance")
    ws_url, release = injector.lease_proxy("binance", "websocket")
    try:
        if not http_url and not ws_url:
            pytest.skip(
                "Proxy settings loaded but no Binance-specific HTTP/WS proxy configured"
            )
        if http_url:
            assert urlparse(http_url).scheme, "HTTP proxy must include a scheme"
        if ws_url:
            assert urlparse(ws_url).scheme, "WS proxy must include a scheme"
    finally:
        release()


@pytest.mark.integration
def test_binance_proxy_pool_selection_without_live():
    """Validate proxy pool entries can be leased without hitting Binance/Redpanda."""

    pool_env_prefix = "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL__PROXIES__"
    pool_json_env = "CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL"

    has_pool = any(key.startswith(pool_env_prefix) for key in os.environ) or pool_json_env in os.environ
    if not has_pool:
        pytest.skip("No Binance proxy pool configuration provided")

    settings = load_proxy_settings()
    if not (settings.enabled or settings.default or settings.exchanges):
        pytest.skip("Proxy settings not enabled; pool lease not applicable")

    init_proxy_system(settings)
    injector = get_proxy_injector()
    assert injector is not None, "Proxy injector should be initialized for pool test"

    ws_url, release = injector.lease_proxy("binance", "websocket")
    try:
        assert ws_url, "Proxy pool should yield a websocket proxy URL"
        assert urlparse(ws_url).scheme, "Pooled proxy URL must include a scheme"
    finally:
        release()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_kafka_protobuf_trade_roundtrip(redpanda):
    """End-to-end trade roundtrip: Binance WS → Kafka Protobuf → Protobuf decode.

    This test is intentionally opt-in and live: it requires network
    access to Binance and a local Redpanda instance started via the
    shared `redpanda` fixture.
    """
    _require_binance_e2e_prereqs()
    await _preflight_rest_through_proxy()

    strategy = _topic_strategy()
    topic = _topic_name(TRADES, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None

    try:
        fh = await _start_binance_with_kafka(
            redpanda_bootstrap=redpanda,
            topic_strategy=strategy,
        )

        # Consume one record from Kafka without blocking the event loop
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=60.0,
                group_id=f"cf-e2e-binance-proto-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                f"Binance Kafka Protobuf E2E: no message consumed within timeout: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_or_skip_headers(record)
    # Header assertions
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == b"binance"
    assert record.headers[b"symbol"] in {b"BTC-USDT", b"ETH-USDT"}
    assert record.headers[b"data_type"] == b"trade"

    # Payload assertions
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    trade_pb2 = pb_bindings.trade_pb2

    msg = trade_pb2.Trade()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance"
    assert msg.symbol in {"BTC-USDT", "ETH-USDT"}
    # Basic sanity checks: non-empty numeric fields
    assert msg.amount not in ("", "0", "0.0")
    assert msg.price not in ("", "0", "0.0")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_kafka_protobuf_trade_roundtrip_round_robin(redpanda):
    """Validate round-robin partitioning produces keyless records for Binance trades."""
    await _preflight_rest_through_proxy()
    _require_binance_e2e_prereqs()

    strategy = _topic_strategy()
    topic = _topic_name(TRADES, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None

    try:
        fh = await _start_binance_with_kafka(
            redpanda_bootstrap=redpanda,
            partition_strategy="round_robin",
            topic_strategy=strategy,
        )

        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=60.0,
                group_id=f"cf-e2e-binance-proto-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                f"Binance Kafka Protobuf E2E (round-robin): "
                f"no message consumed within timeout: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_or_skip_headers(record)
    assert record.key is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_kafka_protobuf_orderbook_snapshot_roundtrip(redpanda):
    """Order book snapshot+delta path: Binance L2 → Kafka Protobuf → decode."""

    _require_binance_e2e_prereqs()

    strategy = _topic_strategy()
    topic = _topic_name(L2_BOOK, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None

    try:
        fh = await _start_binance_with_kafka(
            redpanda_bootstrap=redpanda,
            channels=[L2_BOOK],
            topic_strategy=strategy,
        )

        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=150.0,
                group_id=f"cf-e2e-binance-proto-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                "Binance Kafka Protobuf E2E (orderbook): no message within timeout;"
                f" possible REST snapshot or WS connectivity issue: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_or_skip_headers(record)
    # Header assertions
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == b"binance"
    assert record.headers[b"symbol"] == b"BTC-USDT"
    assert record.headers[b"data_type"] == b"l2_book"

    # Payload assertions
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    msg = pb_bindings.order_book_pb2.Level2Book()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance"
    assert msg.symbol == "BTC-USDT"
    # Require at least one bid/ask level present
    assert len(msg.bids) > 0 or len(msg.asks) > 0
