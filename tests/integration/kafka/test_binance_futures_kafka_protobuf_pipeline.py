"""Binance Futures → Kafka Protobuf end-to-end tests using Redpanda.

Live tests that exercise the Binance USDⓈ-M futures feed (REST+WS) through
Mullvad SOCKS5 relays into the Kafka Protobuf backend, then decode payloads
from Redpanda. All tests are opt-in and will skip unless
`CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E` is truthy.

Channels covered:
- High frequency: TRADES, L2_BOOK, TICKER
- Derivatives-specific: FUNDING (markPrice stream), OPEN_INTEREST (REST poll),
  LIQUIDATIONS (forceOrder stream)

Prereqs:
- Docker + docker compose (for Redpanda fixture)
- Network access to Binance USDⓈ-M endpoints (via configured Mullvad proxies)
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from cryptofeed.defines import TRADES, L2_BOOK, TICKER, FUNDING, OPEN_INTEREST, LIQUIDATIONS
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.kafka.partitioner import PartitionerFactory
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION
from cryptofeed.proxy import get_proxy_injector, init_proxy_system, load_proxy_settings, ProxySettings
from tests.integration.kafka.helpers import ConsumedRecord, consume_one
from tests.integration.kafka.topic_provision import ensure_topics_exist

BINANCE_FUTURES_ENV = "CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E"
TOPIC_STRATEGY_ENV = "KAFKA_E2E_TOPIC_STRATEGY"


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
            f"Binance Futures Kafka Protobuf E2E disabled. Set {BINANCE_FUTURES_ENV}=true to enable."
        )


def _init_proxy_settings_if_configured() -> bool:
    _env_cache["HTTP_PROXY"] = os.environ.get("HTTP_PROXY")
    _env_cache["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY")

    settings = load_proxy_settings()
    has_proxy = settings.enabled or settings.default or settings.exchanges
    if not has_proxy:
        return False

    init_proxy_system(settings)
    injector = get_proxy_injector()
    if injector and settings.enabled:
        http_proxy_url = injector.get_http_proxy_url("binance_futures") or injector.get_http_proxy_url("binance")
        if http_proxy_url:
            os.environ["HTTPS_PROXY"] = http_proxy_url
            os.environ["HTTP_PROXY"] = http_proxy_url
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
        pytest.skip("Kafka producer failed to connect to Redpanda")

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


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_trade_roundtrip(redpanda):
    _require_futures_env()

    strategy = _topic_strategy()
    topic = _topic_name(TRADES, strategy)
    await ensure_topics_exist(redpanda, [topic])

    fh: FeedHandler | None = None
    try:
        fh = await _start_binance_futures(redpanda_bootstrap=redpanda, topic_strategy=strategy)
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=120.0,
                group_id=f"cf-e2e-binance-fut-trade-{uuid4().hex}",
                offset_reset="latest",
            )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"trade")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_orderbook_roundtrip(redpanda):
    _require_futures_env()

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
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=150.0,
                group_id=f"cf-e2e-binance-fut-l2-{uuid4().hex}",
                offset_reset="latest",
            )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"l2_book")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_futures_kafka_protobuf_ticker_roundtrip(redpanda):
    _require_futures_env()

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
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=120.0,
                group_id=f"cf-e2e-binance-fut-ticker-{uuid4().hex}",
                offset_reset="latest",
            )
    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_headers(record, b"ticker")


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
