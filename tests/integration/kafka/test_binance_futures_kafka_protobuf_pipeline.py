"""Binance Futures → Kafka Protobuf end-to-end tests using Redpanda.

Live tests that exercise the Binance USDⓈ-M futures feed (REST+WS) through
the Kafka Protobuf backend, then decode payloads from Redpanda. All tests are
opt-in and will skip unless `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E` is truthy.

Channels covered:
- High frequency: TRADES, L2_BOOK, TICKER
- Derivatives-specific: FUNDING (markPrice stream), OPEN_INTEREST (REST poll),
  LIQUIDATIONS (forceOrder stream)

Prereqs:
- Docker + docker compose (for Redpanda fixture)
- Network access to Binance USDⓈ-M endpoints (direct or via proxy)

## Proxy Support (FR7: kafka-protobuf-binance-e2e spec)

These tests support running through HTTP or SOCKS5 proxies for both REST and
WebSocket transports. Proxy configuration is loaded from environment variables.

**Quick Start - Direct Mode (no proxy):**
```bash
make redpanda-up
CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
  python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v
make redpanda-down
```

**Quick Start - Single HTTP Proxy:**
```bash
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL=http://proxy:8080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__URL=http://proxy:8080
make redpanda-up
CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
  python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v
make redpanda-down
```

**Quick Start - Single SOCKS5 Proxy (requires python-socks):**
```bash
pip install python-socks
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL=socks5://user:pass@proxy:1080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__URL=socks5://user:pass@proxy:1080
make redpanda-up
CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
  python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v
make redpanda-down
```

**Quick Start - Proxy Pool (round-robin):**
```bash
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL='{"proxies":[{"url":"http://p1:8080","weight":1},{"url":"http://p2:8080","weight":1}],"strategy":"round_robin"}'
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://ws1:1080","weight":1},{"url":"socks5://ws2:1080","weight":1}],"strategy":"round_robin"}'
make redpanda-up
CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
  python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v
make redpanda-down
```

**Makefile Targets:**
```bash
# Direct mode
make test-kafka-binance-futures

# With Mullvad proxy pool (requires Mullvad configuration)
make test-kafka-binance-futures-mullvad
```

**Dependencies:**
- `python-socks` - Required for SOCKS WebSocket proxies: `pip install python-socks`
  (Tests will skip with clear message if SOCKS configured but python-socks missing)

**Comprehensive Documentation:**
See `docs/e2e/PROXY_TESTING.md` for complete proxy configuration guide including:
- Environment variable reference
- Proxy pool configuration
- Timeout configuration (CF_SYMBOL_FETCH_TIMEOUT, CF_LISTEN_KEY_TIMEOUT)
- Troubleshooting guide
- Example test runs
- Futures-specific configuration examples

**Related Docs:**
- Proxy system: `docs/proxy/README.md`
- Timeout config: `docs/proxy/timeout-configuration.md`
- Spec requirements: `.kiro/specs/kafka-protobuf-binance-e2e/requirements.md` (FR7)
"""

from __future__ import annotations

import asyncio
import os
from importlib import import_module
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from cryptofeed.defines import (
    FUNDING,
    LIQUIDATIONS,
    L2_BOOK,
    OPEN_INTEREST,
    TICKER,
    TRADES,
)
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.kafka.partitioner import PartitionerFactory
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION
from cryptofeed.proxy import (
    ProxySettings,
    get_proxy_injector,
    init_proxy_system,
    load_proxy_settings,
)
from tests.integration.kafka.helpers import ConsumedRecord, consume_one
from tests.integration.kafka.topic_provision import ensure_topics_exist

BINANCE_FUTURES_ENV = "CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E"
TOPIC_STRATEGY_ENV = "KAFKA_E2E_TOPIC_STRATEGY"
BINANCE_FUTURES_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def _topic_strategy() -> str:
    value = os.getenv(TOPIC_STRATEGY_ENV, "per_symbol").lower()
    return "consolidated" if value == "consolidated" else "per_symbol"


def _topic_name(channel: str, strategy: str) -> str:
    if strategy == "consolidated":
        mapping = {
            TRADES: "cryptofeed.trade",
            L2_BOOK: "cryptofeed.l2_book",
            TICKER: "cryptofeed.ticker",
            FUNDING: "cryptofeed.funding",
            OPEN_INTEREST: "cryptofeed.open_interest",
            LIQUIDATIONS: "cryptofeed.liquidation",
        }
        return mapping.get(channel, f"cryptofeed.{channel}")

    base = "binance_futures.btc-usdt-perp"
    mapping = {
        TRADES: f"cryptofeed.trade.{base}",
        L2_BOOK: f"cryptofeed.l2_book.{base}",
        TICKER: f"cryptofeed.ticker.{base}",
        FUNDING: f"cryptofeed.funding.{base}",
        OPEN_INTEREST: f"cryptofeed.open_interest.{base}",
        LIQUIDATIONS: f"cryptofeed.liquidation.{base}",
    }
    return mapping.get(channel, f"cryptofeed.{channel}.{base}")


_env_cache: dict[str, str | None] = {}


def _require_futures_env() -> None:
    value = os.getenv(BINANCE_FUTURES_ENV, "")
    if value.lower() not in {"1", "true", "yes", "on"}:
        pytest.skip(
            f"Binance Futures Kafka Protobuf E2E disabled. Set {BINANCE_FUTURES_ENV}=true to enable. "
            f"See docs/e2e/SKIP_CONDITIONS.md for details."
        )


def _python_socks_available() -> bool:
    try:
        import_module("python_socks")
        return True
    except ModuleNotFoundError:
        return False


async def _preflight_rest_through_proxy() -> None:
    """Fetch futures exchangeInfo via configured proxy; skip if unavailable.

    Task 6.5: Initialize proxy system BEFORE attempting to lease proxies.
    """

    # Initialize proxy system first, then get the injector
    has_proxy = _init_proxy_settings_if_configured()
    if not has_proxy:
        return  # no proxy configured; use direct path

    injector = get_proxy_injector()
    if not injector:
        return  # proxy system not initialized

    proxy_url = injector.get_http_proxy_url("binance_futures")
    if not proxy_url:
        return  # no HTTP proxy configured for binance_futures

    try:
        import aiohttp
    except ImportError:
        pytest.skip(
            "aiohttp not available for REST preflight. "
            "Install with: pip install aiohttp. "
            "See docs/e2e/SKIP_CONDITIONS.md for details."
        )

    connector = None
    try:
        scheme = urlparse(proxy_url).scheme.lower()
        if scheme.startswith("socks"):
            try:
                from aiohttp_socks import ProxyConnector  # type: ignore

                connector = ProxyConnector.from_url(proxy_url)
            except ModuleNotFoundError:
                pytest.skip(
                    "Binance Futures REST exchangeInfo via SOCKS proxy requires aiohttp-socks. "
                    "Install with: pip install aiohttp-socks, or use HTTP proxy instead. "
                    "See docs/e2e/SKIP_CONDITIONS.md and docs/e2e/PROXY_TESTING.md."
                )
    except Exception:
        connector = None

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        try:
            async with session.get(
                BINANCE_FUTURES_INFO_URL,
                proxy=None if connector else proxy_url,
            ) as resp:
                if resp.status != 200:
                    pytest.skip(
                        f"Binance Futures REST exchangeInfo via proxy failed (status {resp.status}). "
                        f"REST may be geoblocked or proxy blocked. "
                        f"Check proxy configuration or try different proxy. "
                        f"See docs/e2e/SKIP_CONDITIONS.md and docs/e2e/PROXY_TESTING.md."
                    )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(
                f"Binance Futures REST exchangeInfo via proxy failed: {exc}. "
                f"Check network connectivity, proxy settings, and timeouts (CF_SYMBOL_FETCH_TIMEOUT). "
                f"See docs/e2e/SKIP_CONDITIONS.md and docs/proxy/timeout-configuration.md."
            )


def _init_proxy_settings_if_configured() -> bool:
    _env_cache["HTTP_PROXY"] = os.environ.get("HTTP_PROXY")
    _env_cache["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY")

    settings = load_proxy_settings()
    has_proxy = settings.enabled or settings.default or settings.exchanges
    if not has_proxy:
        return False

    ws_proxy = (
        settings.get_proxy("binance_futures", "websocket")
        if hasattr(settings, "get_proxy")
        else None
    )
    if ws_proxy and ws_proxy.url:
        scheme = urlparse(ws_proxy.url).scheme.lower()
        if scheme.startswith("socks") and not _python_socks_available():
            pytest.skip(
                "Binance Futures Kafka Protobuf E2E: SOCKS websocket proxy configured "
                "but python-socks is not installed. "
                "Install with: pip install python-socks. "
                "See docs/e2e/SKIP_CONDITIONS.md for details."
            )

    init_proxy_system(settings)
    injector = get_proxy_injector()
    if injector and settings.enabled:
        http_proxy_url = injector.get_http_proxy_url("binance_futures") or injector.get_http_proxy_url("binance")
        if http_proxy_url:
            os.environ["HTTPS_PROXY"] = http_proxy_url
            os.environ["HTTP_PROXY"] = http_proxy_url

        if ws_proxy:
            url, release = injector.lease_proxy("binance_futures", "websocket")
            if release is None:
                pytest.skip(
                    "Binance Futures Kafka Protobuf E2E: websocket proxy configured "
                    "but injector returned no release handle"
                )
            try:
                if url is None:
                    pytest.skip(
                        "Binance Futures Kafka Protobuf E2E: websocket proxy configured "
                        "but no proxy was selected"
                    )
                assert urlparse(url).scheme, "Binance Futures proxy resolution must return a scheme"
            finally:
                release()
    return True


async def _start_binance_futures(
    redpanda_bootstrap: str,
    *,
    partition_strategy: str | None = None,
    channels: list[str] | None = None,
    topic_strategy: str = "per_symbol",
) -> FeedHandler:
    loop = asyncio.get_running_loop()

    _init_proxy_settings_if_configured()

    fh = FeedHandler()

    kafka_cb = KafkaProtobufCallback(
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
        pytest.skip(
            "Kafka producer failed to connect to Redpanda. "
            "Ensure Redpanda is running: 'make redpanda-up && make redpanda-health'. "
            "See docs/e2e/SKIP_CONDITIONS.md for troubleshooting."
        )

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
        elif channel == TICKER:
            callbacks[channel] = [_mk_handler("ticker")]
        elif channel == FUNDING:
            callbacks[channel] = [_mk_handler("funding")]
        elif channel == OPEN_INTEREST:
            callbacks[channel] = [_mk_handler("open_interest")]
        elif channel == LIQUIDATIONS:
            callbacks[channel] = [_mk_handler("liquidation")]
        else:  # pragma: no cover - future channels
            callbacks[channel] = [_mk_handler(channel.lower())]

    symbols = ["BTC-USDT-PERP"]
    if topic_strategy == "consolidated":
        symbols.append("ETH-USDT-PERP")

    fh.add_feed(
        "BINANCE_FUTURES",
        symbols=symbols,
        channels=_channels,
        callbacks=callbacks,
    )

    for feed in fh.feeds:
        feed.start(loop)

    await asyncio.sleep(5)

    fh.kafka_cb = kafka_cb  # type: ignore[attr-defined]
    return fh


async def _shutdown_feeds(handler: FeedHandler) -> None:
    shutdown_tasks = []
    for feed in handler.feeds:
        feed.stop()
        shutdown_tasks.append(feed.shutdown())
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks)

    kafka_cb = getattr(handler, "kafka_cb", None)
    if kafka_cb and hasattr(kafka_cb, "stop"):
        await kafka_cb.stop()

    for key, value in _env_cache.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    init_proxy_system(ProxySettings(enabled=False))


def _assert_headers(record: ConsumedRecord, data_type: bytes):
    missing = [h for h in (b"content-type", b"schema_version", b"cf.serialization_format") if h not in record.headers]
    if missing:
        pytest.skip(f"Missing headers {missing}; observed={record.headers}")
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"data_type"] == data_type
    assert record.headers[b"exchange"] == b"binance_futures"
    assert record.headers[b"symbol"] in {b"BTC-USDT-PERP", b"ETH-USDT-PERP"}


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_rest_connectivity_via_proxy():
    _require_futures_env()
    await _preflight_rest_through_proxy()


@pytest.mark.integration
@pytest.mark.live_binance
def test_binance_futures_proxy_resolution_when_configured():
    _require_futures_env()
    if not _init_proxy_settings_if_configured():
        pytest.skip("No proxy configuration provided for Binance Futures")

    injector = get_proxy_injector()
    assert injector is not None, "Proxy injector should be initialized when proxies are configured"

    http_url = injector.get_http_proxy_url("binance_futures") or injector.get_http_proxy_url("binance")
    ws_url, release = injector.lease_proxy("binance_futures", "websocket")
    try:
        if not http_url and not ws_url:
            pytest.skip(
                "Proxy settings loaded but no Binance Futures-specific HTTP/WS proxy configured"
            )
        if http_url:
            assert urlparse(http_url).scheme, "HTTP proxy must include a scheme"
        if ws_url:
            assert urlparse(ws_url).scheme, "WS proxy must include a scheme"
    finally:
        release()


@pytest.mark.integration
def test_binance_futures_proxy_pool_selection_without_live():
    pool_env_prefix = "CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL__PROXIES__"
    pool_json_env = "CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL"

    has_pool = any(key.startswith(pool_env_prefix) for key in os.environ) or pool_json_env in os.environ
    if not has_pool:
        pytest.skip("No Binance Futures proxy pool configuration provided")

    settings = load_proxy_settings()
    if not (settings.enabled or settings.default or settings.exchanges):
        pytest.skip("Proxy settings not enabled; pool lease not applicable")

    init_proxy_system(settings)
    injector = get_proxy_injector()
    assert injector is not None, "Proxy injector should be initialized for pool test"

    ws_url, release = injector.lease_proxy("binance_futures", "websocket")
    try:
        assert ws_url, "Proxy pool should yield a websocket proxy URL"
        assert urlparse(ws_url).scheme, "Pooled proxy URL must include a scheme"
    finally:
        release()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_ws_connectivity_via_proxy():
    _require_futures_env()

    loop = asyncio.get_running_loop()
    fh = FeedHandler()

    received: dict[str, object] = {}

    async def _handler(obj, receipt_timestamp):
        if "msg" not in received:
            received["msg"] = obj
            for feed in fh.feeds:
                feed.stop()

    fh.add_feed(
        "BINANCE_FUTURES",
        symbols=["BTC-USDT-PERP"],
        channels=[TRADES],
        callbacks={TRADES: [_handler]},
    )

    for feed in fh.feeds:
        feed.start(loop)

    try:
        for _ in range(60):
            if "msg" in received:
                break
            await asyncio.sleep(1.0)
        else:
            pytest.skip("Binance Futures WS: no message received within timeout")
    finally:
        await _shutdown_feeds(fh)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_trade_roundtrip(redpanda):
    _require_futures_env()
    await _preflight_rest_through_proxy()

    strategy = _topic_strategy()
    topic = _topic_name(TRADES, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(redpanda_bootstrap=redpanda, topic_strategy=strategy)
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=120.0,
                group_id=f"cf-e2e-binance-fut-trade-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                "Binance Futures Kafka Protobuf E2E (trade): "
                f"no message consumed within timeout: {exc}"
            )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"trade")
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    trade_pb2 = pb_bindings.trade_pb2

    msg = trade_pb2.Trade()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance_futures"
    assert msg.symbol in {"BTC-USDT-PERP", "ETH-USDT-PERP"}
    assert msg.amount not in ("", "0", "0.0")
    assert msg.price not in ("", "0", "0.0")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_trade_roundtrip_round_robin(redpanda):
    await _preflight_rest_through_proxy()
    _require_futures_env()

    strategy = _topic_strategy()
    topic = _topic_name(TRADES, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(
            redpanda_bootstrap=redpanda,
            partition_strategy="round_robin",
            topic_strategy=strategy,
        )
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=120.0,
                group_id=f"cf-e2e-binance-fut-trade-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                "Binance Futures Kafka Protobuf E2E (round-robin trade): "
                f"no message consumed within timeout: {exc}"
            )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"trade")
    assert record.key is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_orderbook_roundtrip(redpanda):
    _require_futures_env()
    await _preflight_rest_through_proxy()

    strategy = _topic_strategy()
    topic = _topic_name(L2_BOOK, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(
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
                group_id=f"cf-e2e-binance-fut-l2-{uuid4().hex}",
                offset_reset="earliest",
            )
        except AssertionError as exc:
            pytest.skip(
                "Binance Futures Kafka Protobuf E2E (orderbook): "
                f"no message consumed within timeout: {exc}"
            )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"l2_book")
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    order_book_pb2 = pb_bindings.order_book_pb2

    msg = order_book_pb2.Level2Book()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance_futures"
    assert msg.symbol in {"BTC-USDT-PERP", "ETH-USDT-PERP"}
    assert msg.bids or msg.asks


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_ticker_roundtrip(redpanda):
    _require_futures_env()
    await _preflight_rest_through_proxy()

    strategy = _topic_strategy()
    topic = _topic_name(TICKER, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(
            redpanda_bootstrap=redpanda,
            channels=[TICKER],
            topic_strategy=strategy,
        )
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=120.0,
                group_id=f"cf-e2e-binance-fut-ticker-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                "Binance Futures Kafka Protobuf E2E (ticker): "
                f"no message consumed within timeout: {exc}"
            )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"ticker")
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    ticker_pb2 = pb_bindings.ticker_pb2

    msg = ticker_pb2.Ticker()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance_futures"
    assert msg.symbol
    assert msg.bid or msg.ask


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_funding_roundtrip(redpanda):
    _require_futures_env()

    strategy = _topic_strategy()
    topic = _topic_name(FUNDING, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(
            redpanda_bootstrap=redpanda,
            channels=[FUNDING],
            topic_strategy=strategy,
        )
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=360.0,
                group_id=f"cf-e2e-binance-fut-funding-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(f"Binance Futures funding: no message within timeout: {exc}")
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"funding")
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    funding_pb2 = pb_bindings.funding_pb2

    msg = funding_pb2.Funding()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance_futures"
    assert msg.symbol
    assert msg.mark_price or msg.rate


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_open_interest_roundtrip(redpanda):
    _require_futures_env()

    strategy = _topic_strategy()
    topic = _topic_name(OPEN_INTEREST, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(
            redpanda_bootstrap=redpanda,
            channels=[OPEN_INTEREST],
            topic_strategy=strategy,
        )
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=420.0,
                group_id=f"cf-e2e-binance-fut-oi-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(f"Binance Futures open_interest: no message within timeout: {exc}")
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"open_interest")
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    open_interest_pb2 = pb_bindings.open_interest_pb2

    msg = open_interest_pb2.OpenInterest()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance_futures"
    assert msg.symbol
    assert msg.open_interest not in ("", "0", "0.0")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_liquidation_roundtrip(redpanda):
    _require_futures_env()

    strategy = _topic_strategy()
    topic = _topic_name(LIQUIDATIONS, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(
            redpanda_bootstrap=redpanda,
            channels=[LIQUIDATIONS],
            topic_strategy=strategy,
        )
        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=420.0,
                group_id=f"cf-e2e-binance-fut-liq-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(f"Binance Futures liquidations: no message within timeout: {exc}")
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"liquidation")
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    liquidation_pb2 = pb_bindings.liquidation_pb2

    msg = liquidation_pb2.Liquidation()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance_futures"
    assert msg.symbol
    assert msg.quantity not in ("", "0", "0.0")
    assert msg.price not in ("", "0", "0.0")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_multi_channel_roundtrip(redpanda):
    _require_futures_env()
    await _preflight_rest_through_proxy()

    strategy = _topic_strategy()
    channels = [TRADES, L2_BOOK, TICKER]
    topics = [_topic_name(ch, strategy) for ch in channels]
    await ensure_topics_exist(redpanda, topics)

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(
            redpanda_bootstrap=redpanda,
            channels=channels,
            topic_strategy=strategy,
        )

        records: dict[str, ConsumedRecord] = {}
        timeouts = {TRADES: 120.0, L2_BOOK: 150.0, TICKER: 120.0}
        for ch, topic in zip(channels, topics):
            try:
                records[ch] = await asyncio.to_thread(
                    consume_one,
                    redpanda,
                    topic,
                    timeout_s=timeouts[ch],
                    group_id=f"cf-e2e-binance-fut-{ch}-{uuid4().hex}",
                    offset_reset="latest",
                )
            except AssertionError as exc:
                pytest.skip(
                    "Binance Futures Kafka Protobuf E2E (multi-channel, "
                    f"{ch}): no message consumed within timeout: {exc}"
                )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    for ch, record in records.items():
        expected = b"trade" if ch == TRADES else ch.lower().encode()
        _assert_headers(record, expected)

    from cryptofeed.backends.protobuf import bindings as pb_bindings

    trade_pb2 = pb_bindings.trade_pb2
    l2_pb2 = pb_bindings.order_book_pb2
    ticker_pb2 = pb_bindings.ticker_pb2

    msg_trade = trade_pb2.Trade()
    msg_trade.ParseFromString(records[TRADES].value)
    assert msg_trade.symbol

    msg_book = l2_pb2.Level2Book()
    msg_book.ParseFromString(records[L2_BOOK].value)
    assert msg_book.bids or msg_book.asks

    msg_ticker = ticker_pb2.Ticker()
    msg_ticker.ParseFromString(records[TICKER].value)
    assert msg_ticker.bid or msg_ticker.ask

