from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import any_pb2 as _any_pb2
from cryptofeed.v1 import common_pb2 as _common_pb2
from cryptofeed.v1 import market_data_pb2 as _market_data_pb2
from cryptofeed.v1 import account_data_pb2 as _account_data_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataFeedEvent(_message.Message):
    __slots__ = ("event_id", "channel", "exchange", "symbol", "event_timestamp", "receipt_timestamp", "sequence_number", "trade", "ticker", "l1_book", "l2_book", "l3_book", "book_delta", "funding", "open_interest", "liquidation", "index", "candle", "order_info", "balance", "transaction", "fill", "position", "custom_event")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    EVENT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
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
    event_id: str
    channel: _common_pb2.DataChannel
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    event_timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    sequence_number: int
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
    def __init__(self, event_id: _Optional[str] = ..., channel: _Optional[_Union[_common_pb2.DataChannel, str]] = ..., exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., event_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., sequence_number: _Optional[int] = ..., trade: _Optional[_Union[_market_data_pb2.Trade, _Mapping]] = ..., ticker: _Optional[_Union[_market_data_pb2.Ticker, _Mapping]] = ..., l1_book: _Optional[_Union[_market_data_pb2.L1Book, _Mapping]] = ..., l2_book: _Optional[_Union[_market_data_pb2.L2Book, _Mapping]] = ..., l3_book: _Optional[_Union[_market_data_pb2.L3Book, _Mapping]] = ..., book_delta: _Optional[_Union[_market_data_pb2.BookDelta, _Mapping]] = ..., funding: _Optional[_Union[_market_data_pb2.Funding, _Mapping]] = ..., open_interest: _Optional[_Union[_market_data_pb2.OpenInterest, _Mapping]] = ..., liquidation: _Optional[_Union[_market_data_pb2.Liquidation, _Mapping]] = ..., index: _Optional[_Union[_market_data_pb2.Index, _Mapping]] = ..., candle: _Optional[_Union[_market_data_pb2.Candle, _Mapping]] = ..., order_info: _Optional[_Union[_account_data_pb2.OrderInfo, _Mapping]] = ..., balance: _Optional[_Union[_account_data_pb2.Balance, _Mapping]] = ..., transaction: _Optional[_Union[_account_data_pb2.Transaction, _Mapping]] = ..., fill: _Optional[_Union[_account_data_pb2.Fill, _Mapping]] = ..., position: _Optional[_Union[_account_data_pb2.Position, _Mapping]] = ..., custom_event: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class DataFeedEventBatch(_message.Message):
    __slots__ = ("batch_id", "batch_timestamp", "events", "compression")
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    batch_id: str
    batch_timestamp: _timestamp_pb2.Timestamp
    events: _containers.RepeatedCompositeFieldContainer[DataFeedEvent]
    compression: str
    def __init__(self, batch_id: _Optional[str] = ..., batch_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., events: _Optional[_Iterable[_Union[DataFeedEvent, _Mapping]]] = ..., compression: _Optional[str] = ...) -> None: ...

class SubscriptionRequest(_message.Message):
    __slots__ = ("subscription_id", "exchanges", "symbols", "channels", "start_time", "end_time", "options")
    class OptionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SUBSCRIPTION_ID_FIELD_NUMBER: _ClassVar[int]
    EXCHANGES_FIELD_NUMBER: _ClassVar[int]
    SYMBOLS_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    subscription_id: str
    exchanges: _containers.RepeatedScalarFieldContainer[_common_pb2.Exchange]
    symbols: _containers.RepeatedCompositeFieldContainer[_common_pb2.Symbol]
    channels: _containers.RepeatedScalarFieldContainer[_common_pb2.DataChannel]
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    options: _containers.ScalarMap[str, str]
    def __init__(self, subscription_id: _Optional[str] = ..., exchanges: _Optional[_Iterable[_Union[_common_pb2.Exchange, str]]] = ..., symbols: _Optional[_Iterable[_Union[_common_pb2.Symbol, _Mapping]]] = ..., channels: _Optional[_Iterable[_Union[_common_pb2.DataChannel, str]]] = ..., start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., options: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SubscriptionResponse(_message.Message):
    __slots__ = ("subscription_id", "success", "error_message", "subscription_details", "timestamp")
    SUBSCRIPTION_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SUBSCRIPTION_DETAILS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    subscription_id: str
    success: bool
    error_message: str
    subscription_details: SubscriptionRequest
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, subscription_id: _Optional[str] = ..., success: bool = ..., error_message: _Optional[str] = ..., subscription_details: _Optional[_Union[SubscriptionRequest, _Mapping]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Heartbeat(_message.Message):
    __slots__ = ("heartbeat_id", "timestamp", "healthy", "status")
    class StatusEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEARTBEAT_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    heartbeat_id: str
    timestamp: _timestamp_pb2.Timestamp
    healthy: bool
    status: _containers.ScalarMap[str, str]
    def __init__(self, heartbeat_id: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., healthy: bool = ..., status: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ErrorMessage(_message.Message):
    __slots__ = ("error_code", "error_message", "subscription_id", "timestamp", "context")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SUBSCRIPTION_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    error_code: str
    error_message: str
    subscription_id: str
    timestamp: _timestamp_pb2.Timestamp
    context: _containers.ScalarMap[str, str]
    def __init__(self, error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., subscription_id: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., context: _Optional[_Mapping[str, str]] = ...) -> None: ...
