from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Exchange(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXCHANGE_UNSPECIFIED: _ClassVar[Exchange]
    EXCHANGE_ASCENDEX: _ClassVar[Exchange]
    EXCHANGE_ASCENDEX_FUTURES: _ClassVar[Exchange]
    EXCHANGE_BEQUANT: _ClassVar[Exchange]
    EXCHANGE_BITFINEX: _ClassVar[Exchange]
    EXCHANGE_BITHUMB: _ClassVar[Exchange]
    EXCHANGE_BITMEX: _ClassVar[Exchange]
    EXCHANGE_BINANCE: _ClassVar[Exchange]
    EXCHANGE_BINANCE_US: _ClassVar[Exchange]
    EXCHANGE_BINANCE_TR: _ClassVar[Exchange]
    EXCHANGE_BINANCE_FUTURES: _ClassVar[Exchange]
    EXCHANGE_BINANCE_DELIVERY: _ClassVar[Exchange]
    EXCHANGE_BITDOTCOM: _ClassVar[Exchange]
    EXCHANGE_BITFLYER: _ClassVar[Exchange]
    EXCHANGE_BITGET: _ClassVar[Exchange]
    EXCHANGE_BITSTAMP: _ClassVar[Exchange]
    EXCHANGE_BLOCKCHAIN: _ClassVar[Exchange]
    EXCHANGE_BYBIT: _ClassVar[Exchange]
    EXCHANGE_COINBASE: _ClassVar[Exchange]
    EXCHANGE_CRYPTODOTCOM: _ClassVar[Exchange]
    EXCHANGE_DELTA: _ClassVar[Exchange]
    EXCHANGE_DERIBIT: _ClassVar[Exchange]
    EXCHANGE_DYDX: _ClassVar[Exchange]
    EXCHANGE_EXX: _ClassVar[Exchange]
    EXCHANGE_FMFW: _ClassVar[Exchange]
    EXCHANGE_GATEIO: _ClassVar[Exchange]
    EXCHANGE_GATEIO_FUTURES: _ClassVar[Exchange]
    EXCHANGE_GEMINI: _ClassVar[Exchange]
    EXCHANGE_HITBTC: _ClassVar[Exchange]
    EXCHANGE_HUOBI: _ClassVar[Exchange]
    EXCHANGE_HUOBI_DM: _ClassVar[Exchange]
    EXCHANGE_HUOBI_SWAP: _ClassVar[Exchange]
    EXCHANGE_INDEPENDENT_RESERVE: _ClassVar[Exchange]
    EXCHANGE_KRAKEN: _ClassVar[Exchange]
    EXCHANGE_KRAKEN_FUTURES: _ClassVar[Exchange]
    EXCHANGE_KUCOIN: _ClassVar[Exchange]
    EXCHANGE_OKCOIN: _ClassVar[Exchange]
    EXCHANGE_OKX: _ClassVar[Exchange]
    EXCHANGE_PHEMEX: _ClassVar[Exchange]
    EXCHANGE_POLONIEX: _ClassVar[Exchange]
    EXCHANGE_PROBIT: _ClassVar[Exchange]
    EXCHANGE_UPBIT: _ClassVar[Exchange]

class Side(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SIDE_UNSPECIFIED: _ClassVar[Side]
    SIDE_BUY: _ClassVar[Side]
    SIDE_SELL: _ClassVar[Side]
    SIDE_BID: _ClassVar[Side]
    SIDE_ASK: _ClassVar[Side]

class OrderType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ORDER_TYPE_UNSPECIFIED: _ClassVar[OrderType]
    ORDER_TYPE_LIMIT: _ClassVar[OrderType]
    ORDER_TYPE_MARKET: _ClassVar[OrderType]
    ORDER_TYPE_STOP_LIMIT: _ClassVar[OrderType]
    ORDER_TYPE_STOP_MARKET: _ClassVar[OrderType]
    ORDER_TYPE_MAKER_OR_CANCEL: _ClassVar[OrderType]
    ORDER_TYPE_FILL_OR_KILL: _ClassVar[OrderType]
    ORDER_TYPE_IMMEDIATE_OR_CANCEL: _ClassVar[OrderType]
    ORDER_TYPE_GOOD_TIL_CANCELED: _ClassVar[OrderType]
    ORDER_TYPE_TRIGGER_LIMIT: _ClassVar[OrderType]
    ORDER_TYPE_TRIGGER_MARKET: _ClassVar[OrderType]
    ORDER_TYPE_MARGIN_LIMIT: _ClassVar[OrderType]
    ORDER_TYPE_MARGIN_MARKET: _ClassVar[OrderType]

class OrderStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ORDER_STATUS_UNSPECIFIED: _ClassVar[OrderStatus]
    ORDER_STATUS_OPEN: _ClassVar[OrderStatus]
    ORDER_STATUS_PENDING: _ClassVar[OrderStatus]
    ORDER_STATUS_FILLED: _ClassVar[OrderStatus]
    ORDER_STATUS_PARTIAL: _ClassVar[OrderStatus]
    ORDER_STATUS_CANCELLED: _ClassVar[OrderStatus]
    ORDER_STATUS_UNFILLED: _ClassVar[OrderStatus]
    ORDER_STATUS_EXPIRED: _ClassVar[OrderStatus]
    ORDER_STATUS_SUSPENDED: _ClassVar[OrderStatus]
    ORDER_STATUS_FAILED: _ClassVar[OrderStatus]
    ORDER_STATUS_SUBMITTING: _ClassVar[OrderStatus]
    ORDER_STATUS_CANCELLING: _ClassVar[OrderStatus]
    ORDER_STATUS_CLOSED: _ClassVar[OrderStatus]

class InstrumentType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INSTRUMENT_TYPE_UNSPECIFIED: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_CURRENCY: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_FUTURES: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_PERPETUAL: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_OPTION: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_OPTION_COMBO: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_FUTURE_COMBO: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_SPOT: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_CALL: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_PUT: _ClassVar[InstrumentType]
    INSTRUMENT_TYPE_FX: _ClassVar[InstrumentType]

class Liquidity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LIQUIDITY_UNSPECIFIED: _ClassVar[Liquidity]
    LIQUIDITY_MAKER: _ClassVar[Liquidity]
    LIQUIDITY_TAKER: _ClassVar[Liquidity]

class PositionSide(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    POSITION_SIDE_UNSPECIFIED: _ClassVar[PositionSide]
    POSITION_SIDE_LONG: _ClassVar[PositionSide]
    POSITION_SIDE_SHORT: _ClassVar[PositionSide]
    POSITION_SIDE_BOTH: _ClassVar[PositionSide]

class TransactionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSACTION_TYPE_UNSPECIFIED: _ClassVar[TransactionType]
    TRANSACTION_TYPE_DEPOSIT: _ClassVar[TransactionType]
    TRANSACTION_TYPE_WITHDRAWAL: _ClassVar[TransactionType]

class TransactionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSACTION_STATUS_UNSPECIFIED: _ClassVar[TransactionStatus]
    TRANSACTION_STATUS_PENDING: _ClassVar[TransactionStatus]
    TRANSACTION_STATUS_CONFIRMED: _ClassVar[TransactionStatus]
    TRANSACTION_STATUS_FAILED: _ClassVar[TransactionStatus]

class DataChannel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATA_CHANNEL_UNSPECIFIED: _ClassVar[DataChannel]
    DATA_CHANNEL_L1_BOOK: _ClassVar[DataChannel]
    DATA_CHANNEL_L2_BOOK: _ClassVar[DataChannel]
    DATA_CHANNEL_L3_BOOK: _ClassVar[DataChannel]
    DATA_CHANNEL_TRADES: _ClassVar[DataChannel]
    DATA_CHANNEL_TICKER: _ClassVar[DataChannel]
    DATA_CHANNEL_FUNDING: _ClassVar[DataChannel]
    DATA_CHANNEL_OPEN_INTEREST: _ClassVar[DataChannel]
    DATA_CHANNEL_LIQUIDATIONS: _ClassVar[DataChannel]
    DATA_CHANNEL_INDEX: _ClassVar[DataChannel]
    DATA_CHANNEL_CANDLES: _ClassVar[DataChannel]
    DATA_CHANNEL_ORDER_INFO: _ClassVar[DataChannel]
    DATA_CHANNEL_TRANSACTIONS: _ClassVar[DataChannel]
    DATA_CHANNEL_BALANCES: _ClassVar[DataChannel]
    DATA_CHANNEL_FILLS: _ClassVar[DataChannel]
    DATA_CHANNEL_POSITIONS: _ClassVar[DataChannel]
EXCHANGE_UNSPECIFIED: Exchange
EXCHANGE_ASCENDEX: Exchange
EXCHANGE_ASCENDEX_FUTURES: Exchange
EXCHANGE_BEQUANT: Exchange
EXCHANGE_BITFINEX: Exchange
EXCHANGE_BITHUMB: Exchange
EXCHANGE_BITMEX: Exchange
EXCHANGE_BINANCE: Exchange
EXCHANGE_BINANCE_US: Exchange
EXCHANGE_BINANCE_TR: Exchange
EXCHANGE_BINANCE_FUTURES: Exchange
EXCHANGE_BINANCE_DELIVERY: Exchange
EXCHANGE_BITDOTCOM: Exchange
EXCHANGE_BITFLYER: Exchange
EXCHANGE_BITGET: Exchange
EXCHANGE_BITSTAMP: Exchange
EXCHANGE_BLOCKCHAIN: Exchange
EXCHANGE_BYBIT: Exchange
EXCHANGE_COINBASE: Exchange
EXCHANGE_CRYPTODOTCOM: Exchange
EXCHANGE_DELTA: Exchange
EXCHANGE_DERIBIT: Exchange
EXCHANGE_DYDX: Exchange
EXCHANGE_EXX: Exchange
EXCHANGE_FMFW: Exchange
EXCHANGE_GATEIO: Exchange
EXCHANGE_GATEIO_FUTURES: Exchange
EXCHANGE_GEMINI: Exchange
EXCHANGE_HITBTC: Exchange
EXCHANGE_HUOBI: Exchange
EXCHANGE_HUOBI_DM: Exchange
EXCHANGE_HUOBI_SWAP: Exchange
EXCHANGE_INDEPENDENT_RESERVE: Exchange
EXCHANGE_KRAKEN: Exchange
EXCHANGE_KRAKEN_FUTURES: Exchange
EXCHANGE_KUCOIN: Exchange
EXCHANGE_OKCOIN: Exchange
EXCHANGE_OKX: Exchange
EXCHANGE_PHEMEX: Exchange
EXCHANGE_POLONIEX: Exchange
EXCHANGE_PROBIT: Exchange
EXCHANGE_UPBIT: Exchange
SIDE_UNSPECIFIED: Side
SIDE_BUY: Side
SIDE_SELL: Side
SIDE_BID: Side
SIDE_ASK: Side
ORDER_TYPE_UNSPECIFIED: OrderType
ORDER_TYPE_LIMIT: OrderType
ORDER_TYPE_MARKET: OrderType
ORDER_TYPE_STOP_LIMIT: OrderType
ORDER_TYPE_STOP_MARKET: OrderType
ORDER_TYPE_MAKER_OR_CANCEL: OrderType
ORDER_TYPE_FILL_OR_KILL: OrderType
ORDER_TYPE_IMMEDIATE_OR_CANCEL: OrderType
ORDER_TYPE_GOOD_TIL_CANCELED: OrderType
ORDER_TYPE_TRIGGER_LIMIT: OrderType
ORDER_TYPE_TRIGGER_MARKET: OrderType
ORDER_TYPE_MARGIN_LIMIT: OrderType
ORDER_TYPE_MARGIN_MARKET: OrderType
ORDER_STATUS_UNSPECIFIED: OrderStatus
ORDER_STATUS_OPEN: OrderStatus
ORDER_STATUS_PENDING: OrderStatus
ORDER_STATUS_FILLED: OrderStatus
ORDER_STATUS_PARTIAL: OrderStatus
ORDER_STATUS_CANCELLED: OrderStatus
ORDER_STATUS_UNFILLED: OrderStatus
ORDER_STATUS_EXPIRED: OrderStatus
ORDER_STATUS_SUSPENDED: OrderStatus
ORDER_STATUS_FAILED: OrderStatus
ORDER_STATUS_SUBMITTING: OrderStatus
ORDER_STATUS_CANCELLING: OrderStatus
ORDER_STATUS_CLOSED: OrderStatus
INSTRUMENT_TYPE_UNSPECIFIED: InstrumentType
INSTRUMENT_TYPE_CURRENCY: InstrumentType
INSTRUMENT_TYPE_FUTURES: InstrumentType
INSTRUMENT_TYPE_PERPETUAL: InstrumentType
INSTRUMENT_TYPE_OPTION: InstrumentType
INSTRUMENT_TYPE_OPTION_COMBO: InstrumentType
INSTRUMENT_TYPE_FUTURE_COMBO: InstrumentType
INSTRUMENT_TYPE_SPOT: InstrumentType
INSTRUMENT_TYPE_CALL: InstrumentType
INSTRUMENT_TYPE_PUT: InstrumentType
INSTRUMENT_TYPE_FX: InstrumentType
LIQUIDITY_UNSPECIFIED: Liquidity
LIQUIDITY_MAKER: Liquidity
LIQUIDITY_TAKER: Liquidity
POSITION_SIDE_UNSPECIFIED: PositionSide
POSITION_SIDE_LONG: PositionSide
POSITION_SIDE_SHORT: PositionSide
POSITION_SIDE_BOTH: PositionSide
TRANSACTION_TYPE_UNSPECIFIED: TransactionType
TRANSACTION_TYPE_DEPOSIT: TransactionType
TRANSACTION_TYPE_WITHDRAWAL: TransactionType
TRANSACTION_STATUS_UNSPECIFIED: TransactionStatus
TRANSACTION_STATUS_PENDING: TransactionStatus
TRANSACTION_STATUS_CONFIRMED: TransactionStatus
TRANSACTION_STATUS_FAILED: TransactionStatus
DATA_CHANNEL_UNSPECIFIED: DataChannel
DATA_CHANNEL_L1_BOOK: DataChannel
DATA_CHANNEL_L2_BOOK: DataChannel
DATA_CHANNEL_L3_BOOK: DataChannel
DATA_CHANNEL_TRADES: DataChannel
DATA_CHANNEL_TICKER: DataChannel
DATA_CHANNEL_FUNDING: DataChannel
DATA_CHANNEL_OPEN_INTEREST: DataChannel
DATA_CHANNEL_LIQUIDATIONS: DataChannel
DATA_CHANNEL_INDEX: DataChannel
DATA_CHANNEL_CANDLES: DataChannel
DATA_CHANNEL_ORDER_INFO: DataChannel
DATA_CHANNEL_TRANSACTIONS: DataChannel
DATA_CHANNEL_BALANCES: DataChannel
DATA_CHANNEL_FILLS: DataChannel
DATA_CHANNEL_POSITIONS: DataChannel

class Decimal(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: str
    def __init__(self, value: _Optional[str] = ...) -> None: ...

class Symbol(_message.Message):
    __slots__ = ("base", "quote", "symbol", "type")
    BASE_FIELD_NUMBER: _ClassVar[int]
    QUOTE_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    base: str
    quote: str
    symbol: str
    type: InstrumentType
    def __init__(self, base: _Optional[str] = ..., quote: _Optional[str] = ..., symbol: _Optional[str] = ..., type: _Optional[_Union[InstrumentType, str]] = ...) -> None: ...
