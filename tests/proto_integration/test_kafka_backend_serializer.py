import os
import sys
import asyncio
import types
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _FakeProducer:
    def __init__(self):
        self.sent = []
        # provide minimal client for logging usage
        self.client = types.SimpleNamespace(_client_id="test-client")
        self.serializer = None

    async def start(self):
        return None

    async def stop(self):
        return None

    async def send(self, topic, value, key, partition):
        # capture then return an already-resolved future
        if self.serializer:
            value = self.serializer(value)
        self.sent.append((topic, value, key, partition))
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        fut.set_result(True)
        return fut


@pytest.mark.asyncio
async def test_kafka_backend_uses_value_serializer(monkeypatch):
    from cryptofeed.backends.kafka import TradeKafka, KafkaCallback

    fake = _FakeProducer()

    async def fake_connect(self):
        # bypass real AIOKafkaProducer; install fake and mark running
        fake.serializer = self.producer_config.get('value_serializer')
        self.producer = fake
        self.running = True

    monkeypatch.setattr(KafkaCallback, "_connect", fake_connect, raising=True)

    # Instantiate backend with protobuf serializer for trades
    # Minimal serializer to prove backend calls it
    def simple_serializer(d: dict) -> bytes:
        return f"{d['exchange']}:{d['symbol']}".encode()

    backend = TradeKafka(value_serializer=simple_serializer)

    # Start backend (initializes queue and starts writer)
    loop = asyncio.get_event_loop()
    backend.start(loop=loop, multiprocess=False)

    # Send one message then stop
    payload = {
        'exchange': 'BINANCE',
        'symbol': 'BTC-USDT',
        'side': 'buy',
        'amount': 1.0,
        'price': 100.0,
        'timestamp': 1700000000.0,
    }
    await backend.write(payload)
    # wait for queue to be processed
    await asyncio.wait_for(backend.queue.join(), timeout=2.0)
    # stop loop
    await backend.stop()

    assert len(fake.sent) >= 1
    topic, value, key, partition = fake.sent[0]
    # Value should be protobuf-encoded bytes; decode and assert
    assert value == b"BINANCE:BTC-USDT"
