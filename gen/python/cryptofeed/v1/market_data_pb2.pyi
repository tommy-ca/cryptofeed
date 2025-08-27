from google.protobuf import timestamp_pb2 as _timestamp_pb2
from cryptofeed.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Trade(_message.Message):
    __slots__ = ("exchange", "symbol", "side", "amount", "price", "id", "type", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    side: _common_pb2.Side
    amount: _common_pb2.Decimal
    price: _common_pb2.Decimal
    id: str
    type: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., side: _Optional[_Union[_common_pb2.Side, str]] = ..., amount: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., id: _Optional[str] = ..., type: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Ticker(_message.Message):
    __slots__ = ("exchange", "symbol", "bid", "ask", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    BID_FIELD_NUMBER: _ClassVar[int]
    ASK_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    bid: _common_pb2.Decimal
    ask: _common_pb2.Decimal
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., bid: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., ask: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class L1Book(_message.Message):
    __slots__ = ("exchange", "symbol", "bid_price", "bid_size", "ask_price", "ask_size", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    BID_PRICE_FIELD_NUMBER: _ClassVar[int]
    BID_SIZE_FIELD_NUMBER: _ClassVar[int]
    ASK_PRICE_FIELD_NUMBER: _ClassVar[int]
    ASK_SIZE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    bid_price: _common_pb2.Decimal
    bid_size: _common_pb2.Decimal
    ask_price: _common_pb2.Decimal
    ask_size: _common_pb2.Decimal
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., bid_price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., bid_size: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., ask_price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., ask_size: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class PriceLevel(_message.Message):
    __slots__ = ("price", "size")
    PRICE_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    price: _common_pb2.Decimal
    size: _common_pb2.Decimal
    def __init__(self, price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., size: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ...) -> None: ...

class Order(_message.Message):
    __slots__ = ("order_id", "size")
    ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    order_id: str
    size: _common_pb2.Decimal
    def __init__(self, order_id: _Optional[str] = ..., size: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ...) -> None: ...

class L3PriceLevel(_message.Message):
    __slots__ = ("price", "orders")
    PRICE_FIELD_NUMBER: _ClassVar[int]
    ORDERS_FIELD_NUMBER: _ClassVar[int]
    price: _common_pb2.Decimal
    orders: _containers.RepeatedCompositeFieldContainer[Order]
    def __init__(self, price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., orders: _Optional[_Iterable[_Union[Order, _Mapping]]] = ...) -> None: ...

class L2Book(_message.Message):
    __slots__ = ("exchange", "symbol", "bids", "asks", "sequence_number", "checksum", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    BIDS_FIELD_NUMBER: _ClassVar[int]
    ASKS_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    bids: _containers.RepeatedCompositeFieldContainer[PriceLevel]
    asks: _containers.RepeatedCompositeFieldContainer[PriceLevel]
    sequence_number: int
    checksum: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., bids: _Optional[_Iterable[_Union[PriceLevel, _Mapping]]] = ..., asks: _Optional[_Iterable[_Union[PriceLevel, _Mapping]]] = ..., sequence_number: _Optional[int] = ..., checksum: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class L3Book(_message.Message):
    __slots__ = ("exchange", "symbol", "bids", "asks", "sequence_number", "checksum", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    BIDS_FIELD_NUMBER: _ClassVar[int]
    ASKS_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    bids: _containers.RepeatedCompositeFieldContainer[L3PriceLevel]
    asks: _containers.RepeatedCompositeFieldContainer[L3PriceLevel]
    sequence_number: int
    checksum: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., bids: _Optional[_Iterable[_Union[L3PriceLevel, _Mapping]]] = ..., asks: _Optional[_Iterable[_Union[L3PriceLevel, _Mapping]]] = ..., sequence_number: _Optional[int] = ..., checksum: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class BookDelta(_message.Message):
    __slots__ = ("exchange", "symbol", "bid_changes", "ask_changes", "sequence_number", "checksum", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    BID_CHANGES_FIELD_NUMBER: _ClassVar[int]
    ASK_CHANGES_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    bid_changes: _containers.RepeatedCompositeFieldContainer[PriceLevel]
    ask_changes: _containers.RepeatedCompositeFieldContainer[PriceLevel]
    sequence_number: int
    checksum: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., bid_changes: _Optional[_Iterable[_Union[PriceLevel, _Mapping]]] = ..., ask_changes: _Optional[_Iterable[_Union[PriceLevel, _Mapping]]] = ..., sequence_number: _Optional[int] = ..., checksum: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Funding(_message.Message):
    __slots__ = ("exchange", "symbol", "mark_price", "rate", "next_funding_time", "predicted_rate", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    MARK_PRICE_FIELD_NUMBER: _ClassVar[int]
    RATE_FIELD_NUMBER: _ClassVar[int]
    NEXT_FUNDING_TIME_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_RATE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    mark_price: _common_pb2.Decimal
    rate: _common_pb2.Decimal
    next_funding_time: _timestamp_pb2.Timestamp
    predicted_rate: _common_pb2.Decimal
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., mark_price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., rate: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., next_funding_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., predicted_rate: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class OpenInterest(_message.Message):
    __slots__ = ("exchange", "symbol", "open_interest", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    OPEN_INTEREST_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    open_interest: _common_pb2.Decimal
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., open_interest: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Liquidation(_message.Message):
    __slots__ = ("exchange", "symbol", "side", "quantity", "price", "id", "status", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    QUANTITY_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    side: _common_pb2.Side
    quantity: _common_pb2.Decimal
    price: _common_pb2.Decimal
    id: str
    status: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., side: _Optional[_Union[_common_pb2.Side, str]] = ..., quantity: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., id: _Optional[str] = ..., status: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Index(_message.Message):
    __slots__ = ("exchange", "symbol", "price", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    price: _common_pb2.Decimal
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Candle(_message.Message):
    __slots__ = ("exchange", "symbol", "start_time", "end_time", "interval", "trades", "open", "close", "high", "low", "volume", "closed", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    TRADES_FIELD_NUMBER: _ClassVar[int]
    OPEN_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    LOW_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    CLOSED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    interval: str
    trades: int
    open: _common_pb2.Decimal
    close: _common_pb2.Decimal
    high: _common_pb2.Decimal
    low: _common_pb2.Decimal
    volume: _common_pb2.Decimal
    closed: bool
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., interval: _Optional[str] = ..., trades: _Optional[int] = ..., open: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., close: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., high: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., low: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., volume: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., closed: bool = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...
