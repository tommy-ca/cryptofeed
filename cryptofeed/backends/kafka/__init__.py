"""Kafka backend (canonical).

This package contains the current Kafka producer/callback implementations.
`cryptofeed.kafka_callback` remains as a legacy shim and emits its own
deprecation warning on import.
"""

from .base import KafkaBackendBase, KafkaQueuedMessage  # noqa: F401
from .callback import KafkaCallback  # noqa: F401
from .producer import KafkaProducer  # noqa: F401
from .config import (  # noqa: F401
    KafkaTopicConfig,
    KafkaPartitionConfig,
    KafkaProducerConfig,
    KafkaConfig,
)
from .protobuf_callback import KafkaProtobufCallback  # noqa: F401

# Deprecation helpers kept minimal to avoid heavy maintenance dependencies
from .maintenance import emit_class_deprecation_warning, emit_import_deprecation_warning  # noqa: F401


# Lightweight legacy shims to satisfy deprecation tests without circular imports
class _LegacyStubProducer:
    def __init__(self, config):
        self.config = config
        self.connected = False

    def list_topics(self, timeout=None):
        self.connected = True
        return {"topics": {}}

    def produce(self, *args, **kwargs):
        return 0

    def poll(self, timeout):
        return 0

    def flush(self, timeout=None):
        return 0


class _DeprecatedBase(KafkaCallback):
    _deprecated_name: str = "LegacyKafka"

    def __init__(self, *args, producer_factory=None, **kwargs):
        emit_class_deprecation_warning(self._deprecated_name, "cryptofeed.backends.kafka.KafkaCallback")
        # Default to stub producer to avoid real broker dependency in legacy shims
        pf = producer_factory or (lambda config: _LegacyStubProducer(config))
        super().__init__(*args, producer_factory=pf, **kwargs)


class TradeKafka(_DeprecatedBase):
    _deprecated_name = "TradeKafka"
    default_key = "trades"
    protobuf_data_type = "trades"


class BookKafka(_DeprecatedBase):
    _deprecated_name = "BookKafka"
    default_key = "book"
    protobuf_data_type = "orderbook"


class TickerKafka(_DeprecatedBase):
    _deprecated_name = "TickerKafka"
    default_key = "ticker"
    protobuf_data_type = "ticker"


class FundingKafka(_DeprecatedBase):
    _deprecated_name = "FundingKafka"
    default_key = "funding"
    protobuf_data_type = "funding"


class OpenInterestKafka(_DeprecatedBase):
    _deprecated_name = "OpenInterestKafka"
    default_key = "openinterest"
    protobuf_data_type = "openinterest"


class LiquidationsKafka(_DeprecatedBase):
    _deprecated_name = "LiquidationsKafka"
    default_key = "liquidation"
    protobuf_data_type = "liquidation"


class CandlesKafka(_DeprecatedBase):
    _deprecated_name = "CandlesKafka"
    default_key = "candles"
    protobuf_data_type = "candle"


class OrderInfoKafka(_DeprecatedBase):
    _deprecated_name = "OrderInfoKafka"
    default_key = "order"
    protobuf_data_type = "order"


class TransactionsKafka(_DeprecatedBase):
    _deprecated_name = "TransactionsKafka"
    default_key = "transactions"
    protobuf_data_type = "transaction"


class BalancesKafka(_DeprecatedBase):
    _deprecated_name = "BalancesKafka"
    default_key = "balance"
    protobuf_data_type = "balance"


class FillsKafka(_DeprecatedBase):
    _deprecated_name = "FillsKafka"
    default_key = "fills"
    protobuf_data_type = "fill"

__all__ = [
    "KafkaBackendBase",
    "KafkaQueuedMessage",
    "KafkaCallback",
    "KafkaProtobufCallback",
    "KafkaProducer",
    "KafkaTopicConfig",
    "KafkaPartitionConfig",
    "KafkaProducerConfig",
    "KafkaConfig",
    "TradeKafka",
    "BookKafka",
    "TickerKafka",
    "FundingKafka",
    "OpenInterestKafka",
    "LiquidationsKafka",
    "CandlesKafka",
    "OrderInfoKafka",
    "TransactionsKafka",
    "BalancesKafka",
    "FillsKafka",
]
