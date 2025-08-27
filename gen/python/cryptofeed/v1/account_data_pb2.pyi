from google.protobuf import timestamp_pb2 as _timestamp_pb2
from cryptofeed.v1 import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OrderRequest(_message.Message):
    __slots__ = ("exchange", "symbol", "client_order_id", "side", "type", "price", "amount", "account", "timestamp")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    client_order_id: str
    side: _common_pb2.Side
    type: _common_pb2.OrderType
    price: _common_pb2.Decimal
    amount: _common_pb2.Decimal
    account: str
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., client_order_id: _Optional[str] = ..., side: _Optional[_Union[_common_pb2.Side, str]] = ..., type: _Optional[_Union[_common_pb2.OrderType, str]] = ..., price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., amount: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., account: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class OrderInfo(_message.Message):
    __slots__ = ("exchange", "symbol", "id", "client_order_id", "side", "status", "type", "price", "amount", "remaining", "account", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    REMAINING_FIELD_NUMBER: _ClassVar[int]
    ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    id: str
    client_order_id: str
    side: _common_pb2.Side
    status: _common_pb2.OrderStatus
    type: _common_pb2.OrderType
    price: _common_pb2.Decimal
    amount: _common_pb2.Decimal
    remaining: _common_pb2.Decimal
    account: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., id: _Optional[str] = ..., client_order_id: _Optional[str] = ..., side: _Optional[_Union[_common_pb2.Side, str]] = ..., status: _Optional[_Union[_common_pb2.OrderStatus, str]] = ..., type: _Optional[_Union[_common_pb2.OrderType, str]] = ..., price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., amount: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., remaining: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., account: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Balance(_message.Message):
    __slots__ = ("exchange", "currency", "balance", "reserved", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    CURRENCY_FIELD_NUMBER: _ClassVar[int]
    BALANCE_FIELD_NUMBER: _ClassVar[int]
    RESERVED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    currency: str
    balance: _common_pb2.Decimal
    reserved: _common_pb2.Decimal
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., currency: _Optional[str] = ..., balance: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., reserved: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Transaction(_message.Message):
    __slots__ = ("exchange", "currency", "type", "status", "amount", "id", "fee", "address", "tx_hash", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    CURRENCY_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    FEE_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    TX_HASH_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    currency: str
    type: _common_pb2.TransactionType
    status: _common_pb2.TransactionStatus
    amount: _common_pb2.Decimal
    id: str
    fee: _common_pb2.Decimal
    address: str
    tx_hash: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., currency: _Optional[str] = ..., type: _Optional[_Union[_common_pb2.TransactionType, str]] = ..., status: _Optional[_Union[_common_pb2.TransactionStatus, str]] = ..., amount: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., id: _Optional[str] = ..., fee: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., address: _Optional[str] = ..., tx_hash: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Fill(_message.Message):
    __slots__ = ("exchange", "symbol", "side", "amount", "price", "fee", "fee_currency", "id", "order_id", "liquidity", "type", "account", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    FEE_FIELD_NUMBER: _ClassVar[int]
    FEE_CURRENCY_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ORDER_ID_FIELD_NUMBER: _ClassVar[int]
    LIQUIDITY_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    side: _common_pb2.Side
    amount: _common_pb2.Decimal
    price: _common_pb2.Decimal
    fee: _common_pb2.Decimal
    fee_currency: str
    id: str
    order_id: str
    liquidity: _common_pb2.Liquidity
    type: _common_pb2.OrderType
    account: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., side: _Optional[_Union[_common_pb2.Side, str]] = ..., amount: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., fee: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., fee_currency: _Optional[str] = ..., id: _Optional[str] = ..., order_id: _Optional[str] = ..., liquidity: _Optional[_Union[_common_pb2.Liquidity, str]] = ..., type: _Optional[_Union[_common_pb2.OrderType, str]] = ..., account: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class Position(_message.Message):
    __slots__ = ("exchange", "symbol", "position", "entry_price", "side", "unrealized_pnl", "margin", "leverage", "account", "timestamp", "receipt_timestamp", "raw_data")
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ENTRY_PRICE_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    UNREALIZED_PNL_FIELD_NUMBER: _ClassVar[int]
    MARGIN_FIELD_NUMBER: _ClassVar[int]
    LEVERAGE_FIELD_NUMBER: _ClassVar[int]
    ACCOUNT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    exchange: _common_pb2.Exchange
    symbol: _common_pb2.Symbol
    position: _common_pb2.Decimal
    entry_price: _common_pb2.Decimal
    side: _common_pb2.PositionSide
    unrealized_pnl: _common_pb2.Decimal
    margin: _common_pb2.Decimal
    leverage: _common_pb2.Decimal
    account: str
    timestamp: _timestamp_pb2.Timestamp
    receipt_timestamp: _timestamp_pb2.Timestamp
    raw_data: bytes
    def __init__(self, exchange: _Optional[_Union[_common_pb2.Exchange, str]] = ..., symbol: _Optional[_Union[_common_pb2.Symbol, _Mapping]] = ..., position: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., entry_price: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., side: _Optional[_Union[_common_pb2.PositionSide, str]] = ..., unrealized_pnl: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., margin: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., leverage: _Optional[_Union[_common_pb2.Decimal, _Mapping]] = ..., account: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., receipt_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...
