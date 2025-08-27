from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import any_pb2 as _any_pb2
from cryptofeed.v1 import market_data_pb2 as _market_data_pb2
from cryptofeed.v1 import account_data_pb2 as _account_data_pb2
from cryptofeed.v1 import events_pb2 as _events_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class KafkaHeader(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: bytes
    def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...

class KafkaMetadata(_message.Message):
    __slots__ = ("topic", "partition", "offset", "key_bytes", "headers", "timestamp", "attributes")
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    KEY_BYTES_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    topic: str
    partition: int
    offset: int
    key_bytes: bytes
    headers: _containers.RepeatedCompositeFieldContainer[KafkaHeader]
    timestamp: _timestamp_pb2.Timestamp
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, topic: _Optional[str] = ..., partition: _Optional[int] = ..., offset: _Optional[int] = ..., key_bytes: _Optional[bytes] = ..., headers: _Optional[_Iterable[_Union[KafkaHeader, _Mapping]]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class KafkaDataFeedEvent(_message.Message):
    __slots__ = ("event", "metadata")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    event: _events_pb2.DataFeedEvent
    metadata: KafkaMetadata
    def __init__(self, event: _Optional[_Union[_events_pb2.DataFeedEvent, _Mapping]] = ..., metadata: _Optional[_Union[KafkaMetadata, _Mapping]] = ...) -> None: ...

class KafkaRecord(_message.Message):
    __slots__ = ("metadata", "event", "trade", "ticker", "l1_book", "l2_book", "l3_book", "book_delta", "funding", "open_interest", "liquidation", "index", "candle", "order_info", "balance", "transaction", "fill", "position", "custom_event")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    TRADE_FIELD_NUMBER: _ClassVar[int]
    TICKER_FIELD_NUMBER: _ClassVar[int]
    L1_BOOK_FIELD_NUMBER: _ClassVar[int]
    L2_BOOK_FIELD_NUMBER: _ClassVar[int]
    L3_BOOK_FIELD_NUMBER: _ClassVar[int]
    BOOK_DELTA_FIELD_NUMBER: _ClassVar[int]
    FUNDING_FIELD_NUMBER: _ClassVar[int]
    OPEN_INTEREST_FIELD_NUMBER: _ClassVar[int]
    LIQUIDATION_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    CANDLE_FIELD_NUMBER: _ClassVar[int]
    ORDER_INFO_FIELD_NUMBER: _ClassVar[int]
    BALANCE_FIELD_NUMBER: _ClassVar[int]
    TRANSACTION_FIELD_NUMBER: _ClassVar[int]
    FILL_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_EVENT_FIELD_NUMBER: _ClassVar[int]
    metadata: KafkaMetadata
    event: _events_pb2.DataFeedEvent
    trade: _market_data_pb2.Trade
    ticker: _market_data_pb2.Ticker
    l1_book: _market_data_pb2.L1Book
    l2_book: _market_data_pb2.L2Book
    l3_book: _market_data_pb2.L3Book
    book_delta: _market_data_pb2.BookDelta
    funding: _market_data_pb2.Funding
    open_interest: _market_data_pb2.OpenInterest
    liquidation: _market_data_pb2.Liquidation
    index: _market_data_pb2.Index
    candle: _market_data_pb2.Candle
    order_info: _account_data_pb2.OrderInfo
    balance: _account_data_pb2.Balance
    transaction: _account_data_pb2.Transaction
    fill: _account_data_pb2.Fill
    position: _account_data_pb2.Position
    custom_event: _any_pb2.Any
    def __init__(self, metadata: _Optional[_Union[KafkaMetadata, _Mapping]] = ..., event: _Optional[_Union[_events_pb2.DataFeedEvent, _Mapping]] = ..., trade: _Optional[_Union[_market_data_pb2.Trade, _Mapping]] = ..., ticker: _Optional[_Union[_market_data_pb2.Ticker, _Mapping]] = ..., l1_book: _Optional[_Union[_market_data_pb2.L1Book, _Mapping]] = ..., l2_book: _Optional[_Union[_market_data_pb2.L2Book, _Mapping]] = ..., l3_book: _Optional[_Union[_market_data_pb2.L3Book, _Mapping]] = ..., book_delta: _Optional[_Union[_market_data_pb2.BookDelta, _Mapping]] = ..., funding: _Optional[_Union[_market_data_pb2.Funding, _Mapping]] = ..., open_interest: _Optional[_Union[_market_data_pb2.OpenInterest, _Mapping]] = ..., liquidation: _Optional[_Union[_market_data_pb2.Liquidation, _Mapping]] = ..., index: _Optional[_Union[_market_data_pb2.Index, _Mapping]] = ..., candle: _Optional[_Union[_market_data_pb2.Candle, _Mapping]] = ..., order_info: _Optional[_Union[_account_data_pb2.OrderInfo, _Mapping]] = ..., balance: _Optional[_Union[_account_data_pb2.Balance, _Mapping]] = ..., transaction: _Optional[_Union[_account_data_pb2.Transaction, _Mapping]] = ..., fill: _Optional[_Union[_account_data_pb2.Fill, _Mapping]] = ..., position: _Optional[_Union[_account_data_pb2.Position, _Mapping]] = ..., custom_event: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
