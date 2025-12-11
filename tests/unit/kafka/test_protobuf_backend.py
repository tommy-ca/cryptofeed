from dataclasses import dataclass, field
import warnings

import pytest

from cryptofeed.backends.kafka.base import KafkaQueuedMessage
from cryptofeed.backends.kafka.callback import KafkaCallback
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION as DEFAULT_SCHEMA_VERSION
from cryptofeed.backends.protobuf.validation import SchemaValidator


class DummyProducer:
    def __init__(self, config):
        self.config = config
        self.messages = []

    def list_topics(self, timeout=None):
        return None

    def produce(self, topic, value, key=None, headers=None, on_delivery=None):
        self.messages.append(
            {
                "topic": topic,
                "value": value,
                "key": key,
                "headers": headers,
            }
        )

    def poll(self, timeout=0.0):
        return 0

    def flush(self, timeout=None):
        return 0


class DummyProducerFactory:
    def __init__(self):
        self.last_producer: DummyProducer | None = None

    def __call__(self, config):
        self.last_producer = DummyProducer(config)
        return self.last_producer


class StubMetrics:
    def __init__(self):
        self.enabled = True
        self.serialization_calls = 0
        self.message_sizes: list[tuple[int, bool]] = []
        self.produce_latency_calls = 0
        self.produced: list[tuple[str, str, str, str]] = []
        self.errors: list[tuple[str, str, str]] = []

    def initialize(self):
        pass

    def record_serialization_latency(self, latency_seconds, data_type: str) -> None:
        self.serialization_calls += 1

    def record_message_size(self, size_bytes: int, data_type: str, compression_enabled: bool) -> None:
        self.message_sizes.append((size_bytes, compression_enabled))

    def record_produce_latency(self, latency_seconds: float, exchange: str, data_type: str) -> None:
        self.produce_latency_calls += 1

    def record_message_produced(self, exchange: str, symbol: str, data_type: str, partition_strategy: str) -> None:
        self.produced.append((exchange, symbol, data_type, partition_strategy))

    def record_produce_error(self, exchange: str, data_type: str, error_type: str) -> None:
        self.errors.append((exchange, data_type, error_type))


@dataclass
class DummyProto:
    initialized: bool = True
    payload: bytes = b"proto-bytes"

    def SerializeToString(self):
        return self.payload

    def IsInitialized(self):
        return self.initialized

    def FindInitializationErrors(self):
        return [] if self.initialized else ["missing_field"]


@dataclass
class DummyData:
    exchange: str = "binance"
    symbol: str = "BTC-USDT"
    timestamp: float = 1234567890.0
    proto: DummyProto = field(default_factory=DummyProto)

    def to_proto(self):
        return self.proto


def _create_callback(producer_factory, metrics_exporter=None):
    return KafkaProtobufCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=producer_factory,
        metrics_exporter=metrics_exporter,
    )


@pytest.mark.asyncio
async def test_protobuf_callback_produces_with_schema_headers():
    factory = DummyProducerFactory()
    stub_metrics = StubMetrics()
    callback = _create_callback(factory, metrics_exporter=stub_metrics)

    message = KafkaQueuedMessage(
        data_type="trade",
        obj=DummyData(),
        receipt_timestamp=123.0,
    )

    await callback._process_message(message)

    producer = factory.last_producer
    assert producer is not None, "producer should have been instantiated"
    assert producer.messages, "producer should receive serialized messages"

    record = producer.messages[0]
    header_dict = {name: value for name, value in record["headers"]}

    assert header_dict[b"schema_version"] == DEFAULT_SCHEMA_VERSION.encode("utf-8")
    assert header_dict[b"content-type"] == b"application/x-protobuf"
    assert header_dict[b"cf.serialization_format"] == b"protobuf"
    assert record["value"] == b"proto-bytes"
    assert stub_metrics.serialization_calls == 1
    assert stub_metrics.message_sizes and stub_metrics.message_sizes[0][0] == len(b"proto-bytes")
    assert stub_metrics.produce_latency_calls == 1
    assert stub_metrics.produced and stub_metrics.produced[0][0] == "binance"
    assert not stub_metrics.errors


@pytest.mark.asyncio
async def test_protobuf_callback_skips_invalid_proto():
    factory = DummyProducerFactory()
    stub_metrics = StubMetrics()
    callback = _create_callback(factory, metrics_exporter=stub_metrics)

    invalid = DummyData(proto=DummyProto(initialized=False))
    message = KafkaQueuedMessage(
        data_type="trade",
        obj=invalid,
        receipt_timestamp=123.0,
    )

    await callback._process_message(message)

    producer = factory.last_producer
    assert producer is not None
    assert producer.messages == [], "invalid messages should not be produced"
    assert stub_metrics.errors
    assert stub_metrics.errors[0][2] == "serialization_error"


@pytest.mark.asyncio
async def test_protobuf_callback_respects_schema_version_override():
    factory = DummyProducerFactory()
    callback = KafkaProtobufCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=factory,
        schema_version="v2",
    )

    message = KafkaQueuedMessage(
        data_type="trade",
        obj=DummyData(),
        receipt_timestamp=0.0,
    )

    await callback._process_message(message)

    headers = {name: value for name, value in factory.last_producer.messages[0]["headers"]}
    assert headers[b"schema_version"] == b"v2"


@pytest.mark.asyncio
async def test_protobuf_callback_emits_single_schema_and_format_headers():
    factory = DummyProducerFactory()
    callback = _create_callback(factory)

    message = KafkaQueuedMessage(
        data_type="trade",
        obj=DummyData(),
        receipt_timestamp=0.0,
    )

    await callback._process_message(message)

    header_names = [name for name, _ in factory.last_producer.messages[0]["headers"]]
    assert header_names.count(b"schema_version") == 1
    assert header_names.count(b"cf.serialization_format") == 1


def test_kafka_callback_protobuf_mode_emits_warning():
    factory = DummyProducerFactory()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=factory,
            serialization_format="protobuf",
            metrics_exporter=StubMetrics(),
        )
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "Expected DeprecationWarning when using KafkaCallback with protobuf serialization"


def test_kafka_callback_protobuf_mode_disallowed_after_cutoff(monkeypatch):
    factory = DummyProducerFactory()
    monkeypatch.setenv("CF_KAFKA_PROTOBUF_CUTOFF", "2020-01-01")

    with pytest.raises(RuntimeError, match="protobuf mode is disabled after"):
        KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=factory,
            serialization_format="protobuf",
            metrics_exporter=StubMetrics(),
        )

    monkeypatch.delenv("CF_KAFKA_PROTOBUF_CUTOFF", raising=False)


def test_kafka_callback_uses_binding_schema_version_by_default():
    factory = DummyProducerFactory()
    cb = KafkaProtobufCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=factory,
    )
    assert cb._schema_version == DEFAULT_SCHEMA_VERSION


def test_schema_version_matches_validator_default():
    validator = SchemaValidator()
    assert getattr(validator, "_expected_version", None) == DEFAULT_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_metrics_recorded_with_protobuf_and_no_cache():
    factory = DummyProducerFactory()
    metrics = StubMetrics()
    callback = KafkaProtobufCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=factory,
        enable_header_precomputation=False,
        enable_partition_key_cache=False,
        metrics_exporter=metrics,
    )
    message = KafkaQueuedMessage(
        data_type="trade",
        obj=DummyData(),
        receipt_timestamp=0.0,
    )

    await callback._process_message(message)

    assert metrics.serialization_calls == 1
    assert metrics.produce_latency_calls == 1
    assert metrics.produced and metrics.produced[0][0] == "binance"


@pytest.mark.asyncio
async def test_protobuf_callback_headers_without_precompute_or_cache():
    factory = DummyProducerFactory()
    callback = KafkaProtobufCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=factory,
        enable_header_precomputation=False,
        enable_partition_key_cache=False,
    )
    message = KafkaQueuedMessage(
        data_type="trade",
        obj=DummyData(),
        receipt_timestamp=0.0,
    )

    await callback._process_message(message)

    headers = factory.last_producer.messages[0]["headers"]
    names = [name for name, _ in headers]
    assert names.count(b"schema_version") == 1
    assert names.count(b"cf.serialization_format") == 1
    assert headers and headers[0][0] == b"content-type"
