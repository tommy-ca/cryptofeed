"""End-to-end Kafka protobuf roundtrip tests using Redpanda."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
from dataclasses import dataclass

import pytest
from confluent_kafka import Consumer

from cryptofeed.backends.kafka.base import KafkaQueuedMessage
from cryptofeed.backends.kafka.partitioner import PartitionerFactory
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION
from cryptofeed.types import Trade

COMPOSE_FILE = "docker/redpanda.yml"
HOST_BOOTSTRAP = "localhost:19092"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docker_compose_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"Port {host}:{port} not ready after {timeout}s")


@pytest.fixture(scope="session")
def redpanda(request):
    if not _docker_compose_available():
        pytest.skip("docker compose not available")

    # Start Redpanda
    up = subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"],
        capture_output=True,
        text=True,
    )
    if up.returncode != 0:
        pytest.skip(f"failed to start redpanda: {up.stderr.strip()}")

    try:
        _wait_for_port("localhost", 19092, timeout=30)
        time.sleep(3)
    except Exception as exc:  # pragma: no cover - env-specific
        subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "logs"])
        subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "down"], capture_output=True)
        raise exc

    yield HOST_BOOTSTRAP

    subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "down"], capture_output=True)


@dataclass
class _ConsumedRecord:
    value: bytes
    headers: dict[bytes, bytes]
    topic: str
    key: bytes | None


async def _produce_trade(bootstrap: str, trade: Trade) -> None:
    cb = KafkaProtobufCallback(
        bootstrap_servers=[bootstrap],
        producer_factory=None,
        metrics_exporter=None,
    )
    # route per-symbol for easier verification
    cb._topic_strategy = "per_symbol"
    message = KafkaQueuedMessage(data_type="trade", obj=trade, receipt_timestamp=time.time())
    await cb._process_message(message)
    # ensure message leaves client buffer
    cb._producer.flush(2)


def _consume_one(bootstrap: str, topic: str, timeout_s: float = 10.0) -> _ConsumedRecord:
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": "cf-e2e-proto",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    end_time = time.time() + timeout_s
    msg = None
    try:
        while time.time() < end_time:
            msg = consumer.poll(0.5)
            if msg and not msg.error():
                break
    finally:
        consumer.close()
    if msg is None or msg.error():
        raise AssertionError("No message consumed from Kafka")

    def _b(k):
        return k if isinstance(k, bytes) else str(k).encode()

    header_dict = {_b(k): v for k, v in msg.headers() or []}
    return _ConsumedRecord(
        value=msg.value(), headers=header_dict, topic=msg.topic(), key=msg.key()
    )


@pytest.mark.asyncio
async def test_kafka_protobuf_trade_roundtrip(redpanda):
    trade = Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        side="buy",
        amount=1.0,
        price=68000.1,
        timestamp=1700000000.0,
        id="t-1",
        type="spot",
        raw=None,
    )

    await _produce_trade(redpanda, trade)

    topic = "cryptofeed.trade.coinbase.btc-usd"
    record = _consume_one(redpanda, topic)

    # Headers
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == b"coinbase"
    assert record.headers[b"symbol"] == b"BTC-USD"
    assert record.headers[b"data_type"] == b"trade"

    # Payload
    from cryptofeed.proto_bindings import trade_pb2

    msg = trade_pb2.Trade()
    msg.ParseFromString(record.value)
    assert msg.exchange == "coinbase"
    assert msg.symbol == "BTC-USD"
    assert msg.amount == "1.0"
    assert msg.price == "68000.1"


@pytest.mark.asyncio
async def test_kafka_protobuf_partition_key_round_robin(redpanda):
    # round_robin partitioner should emit None key
    trade = Trade(
        exchange="binance",
        symbol="ETH-USDT",
        side="sell",
        amount=0.5,
        price=3500.0,
        timestamp=1700000001.0,
        id="t-2",
        type="spot",
        raw=None,
    )

    cb = KafkaProtobufCallback(
        bootstrap_servers=[redpanda],
        producer_factory=None,
        metrics_exporter=None,
    )
    cb._partitioner = PartitionerFactory.create("round_robin")
    cb._topic_strategy = "per_symbol"

    message = KafkaQueuedMessage(data_type="trade", obj=trade, receipt_timestamp=time.time())
    await cb._process_message(message)

    topic = "cryptofeed.trade.binance.eth-usdt"
    record = _consume_one(redpanda, topic)
    assert record.key is None
