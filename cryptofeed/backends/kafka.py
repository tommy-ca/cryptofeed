'''
Legacy Kafka backend maintained for JSON-only deployments.

This module remains supported for environments that rely on the historical
`BackendQueue` + `aiokafka` implementation. It preserves the original per-symbol
topic strategy and JSON serialization behavior while newer deployments should
prefer the refactored callbacks in `cryptofeed.backends.kafka`.

Recommended path for new work:
    from cryptofeed.backends.kafka import KafkaCallback, KafkaProtobufCallback

This module stays focused on backward compatibility—no new functionality will be
added here, but existing behavior will continue to work.
'''
from collections import defaultdict
import asyncio
import logging
import warnings
from typing import Optional, ByteString

from aiokafka import AIOKafkaProducer
from aiokafka.errors import RequestTimedOutError, KafkaConnectionError, NodeNotReadyError
from cryptofeed.json_utils import json

from cryptofeed.backends.backend import BackendBookCallback, BackendCallback, BackendQueue

LOG = logging.getLogger('feedhandler')


class KafkaCallback(BackendQueue):
    def __init__(self, key=None, serialization_format=None, numeric_type=float, none_to=None, **kwargs):
        """
        You can pass configuration options to AIOKafkaProducer as keyword arguments.
        (either individual kwargs, an unpacked dictionary `**config_dict`, or both)
        A full list of configuration parameters can be found at
        https://aiokafka.readthedocs.io/en/stable/api.html#aiokafka.AIOKafkaProducer

        A 'value_serializer' option allows use of other schemas such as Avro, Protobuf etc.
        The default serialization is JSON Bytes

        Example:

            **{'bootstrap_servers': '127.0.0.1:9092',
            'client_id': 'cryptofeed',
            'acks': 1,
            'value_serializer': your_serialization_function}

        (Passing the event loop is already handled)
        """
        self.producer_config = kwargs
        self.producer = None
        self.key: str = key or self.default_key
        self.numeric_type = numeric_type
        self.none_to = none_to
        self.set_serialization_format(serialization_format)
        # Do not allow writer to send messages until connection confirmed
        self.running = False

    async def __call__(self, dtype, receipt_timestamp: float):
        # Use parent class serialization handling (handles both JSON and Protobuf)
        await BackendCallback.__call__(self, dtype, receipt_timestamp)

    def _default_serializer(self, to_bytes: dict | str) -> ByteString:
        if isinstance(to_bytes, dict):
            return json.dumpb(to_bytes)
        elif isinstance(to_bytes, str):
            return to_bytes.encode()
        else:
            raise TypeError(f'{type(to_bytes)} is not a valid Serialization type')

    async def _connect(self):
        if not self.producer:
            loop = asyncio.get_event_loop()
            try:
                config_keys = ', '.join([k for k in self.producer_config.keys()])
                LOG.info(f'{self.__class__.__name__}: Configuring AIOKafka with the following parameters: {config_keys}')
                self.producer = AIOKafkaProducer(**self.producer_config, loop=loop)
            # Quit if invalid config option passed to AIOKafka
            except (TypeError, ValueError) as e:
                LOG.error(f'{self.__class__.__name__}: Invalid AIOKafka configuration: {e.args}{chr(10)}See https://aiokafka.readthedocs.io/en/stable/api.html#aiokafka.AIOKafkaProducer for list of configuration options')
                raise SystemExit
            else:
                while not self.running:
                    try:
                        await self.producer.start()
                    except KafkaConnectionError:
                        LOG.error(f'{self.__class__.__name__}: Unable to bootstrap from host(s)')
                        await asyncio.sleep(10)
                    else:
                        LOG.info(f'{self.__class__.__name__}: "{self.producer.client._client_id}" connected to cluster containing {len(self.producer.client.cluster.brokers())} broker(s)')
                        self.running = True

    def _default_serializer(self, to_bytes: dict | str) -> ByteString:
        if isinstance(to_bytes, dict):
            return json.dumpb(to_bytes)
        elif isinstance(to_bytes, str):
            return to_bytes.encode()
        elif isinstance(to_bytes, bytes):
            return to_bytes
        else:
            raise TypeError(f'{type(to_bytes)} is not a valid Serialization type')

    def topic(self, data: dict | bytes) -> str:
        """Determine topic based on data format and metadata."""
        if isinstance(data, bytes):
            # Protobuf: use data type for hierarchical topic
            data_type = getattr(self, 'protobuf_data_type', self.key)
            return f"cryptofeed.market.{data_type}.protobuf"

        # JSON: use key, exchange, symbol for backward compatibility
        if isinstance(data, dict):
            return f"{self.key}-{data.get('exchange', 'unknown')}-{data.get('symbol', 'unknown')}"

        return self.key

    def partition_key(self, data: dict | bytes) -> Optional[bytes]:
        """Get partition key from symbol when available."""
        if isinstance(data, dict):
            symbol = data.get('symbol')
            if symbol:
                return str(symbol).encode('utf-8')
        return None

    def partition(self, data: dict | bytes) -> Optional[int]:
        return None

    async def writer(self):
        await self._connect()
        while self.running:
            async with self.read_queue() as updates:
                for index in range(len(updates)):
                    message = updates[index]
                    topic = self.topic(message)

                    # Extract key - use symbol from dict or default to key
                    if isinstance(message, dict):
                        raw_key = message.get('symbol') or self.key
                    else:
                        raw_key = self.key

                    key_serializer = self.producer_config.get('key_serializer')
                    if key_serializer:
                        key = raw_key
                    else:
                        key = self._default_serializer(raw_key)

                    # Serialize value based on type
                    value_serializer = self.producer_config.get('value_serializer')

                    if isinstance(message, bytes):
                        # Protobuf: already serialized
                        value = message if not value_serializer else message
                    else:
                        # JSON: serialize dict to bytes
                        value = message if value_serializer else self._default_serializer(message)

                    partition = self.partition(message)
                    try:
                        send_future = await self.producer.send(topic, value, key, partition)
                        await send_future
                    except RequestTimedOutError:
                        LOG.error(f'{self.__class__.__name__}: No response received from server within {self.producer._request_timeout_ms} ms. Messages may not have been delivered')
                    except NodeNotReadyError:
                        LOG.error(f'{self.__class__.__name__}: Node not ready')
                    except Exception as e:
                        LOG.info(f'{self.__class__.__name__}: Encountered an error:{chr(10)}{e}')
        LOG.info(f"{self.__class__.__name__}: sending last messages and closing connection '{self.producer.client._client_id}'")
        await self.producer.stop()


class TradeKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'trades'
    protobuf_data_type = 'trades'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TradeKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class FundingKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'funding'
    protobuf_data_type = 'funding'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FundingKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class BookKafka(KafkaCallback, BackendBookCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'book'
    protobuf_data_type = 'orderbook'

    def __init__(self, *args, snapshots_only=False, snapshot_interval=1000, **kwargs):
        warnings.warn(
            "BookKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.snapshots_only = snapshots_only
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = defaultdict(int)
        super().__init__(*args, **kwargs)


class TickerKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'ticker'
    protobuf_data_type = 'ticker'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TickerKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class OpenInterestKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'open_interest'
    protobuf_data_type = 'open_interest'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OpenInterestKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class LiquidationsKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'liquidations'
    protobuf_data_type = 'liquidation'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LiquidationsKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class CandlesKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'candles'
    protobuf_data_type = 'candles'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CandlesKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class OrderInfoKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'order_info'
    protobuf_data_type = 'order_info'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OrderInfoKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class TransactionsKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'transactions'
    protobuf_data_type = 'transactions'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TransactionsKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class BalancesKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'balances'
    protobuf_data_type = 'balances'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BalancesKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class FillsKafka(KafkaCallback, BackendCallback):
    """DEPRECATED: Use cryptofeed.kafka_callback.KafkaCallback instead."""
    default_key = 'fills'
    protobuf_data_type = 'fills'

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FillsKafka is deprecated. Use cryptofeed.kafka_callback.KafkaCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
