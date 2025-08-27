"""
Trade adapter for converting between legacy Trade objects and protobuf Trade messages.
"""

from decimal import Decimal
from typing import Optional
import time

from google.protobuf.timestamp_pb2 import Timestamp

# Import legacy types
try:
    from cryptofeed.types import Trade as LegacyTrade
except ImportError:
    # Fallback for testing without Cython extensions
    LegacyTrade = None

# Import protobuf types (these would be generated)
try:
    from cryptofeed.proto.v1.market_data_pb2 import Trade as ProtoTrade
    from cryptofeed.proto.v1.common_pb2 import Exchange, Side, Decimal as ProtoDecimal, Symbol
except ImportError:
    # Mock for development without generated protobuf code
    ProtoTrade = None
    Exchange = None
    Side = None
    ProtoDecimal = None
    Symbol = None


class TradeAdapter:
    """Adapter for converting between legacy Trade and protobuf Trade messages."""
    
    # Exchange mapping from string to protobuf enum
    EXCHANGE_MAPPING = {
        'ASCENDEX': 'EXCHANGE_ASCENDEX',
        'ASCENDEX_FUTURES': 'EXCHANGE_ASCENDEX_FUTURES', 
        'BEQUANT': 'EXCHANGE_BEQUANT',
        'BITFINEX': 'EXCHANGE_BITFINEX',
        'BITHUMB': 'EXCHANGE_BITHUMB',
        'BITMEX': 'EXCHANGE_BITMEX',
        'BINANCE': 'EXCHANGE_BINANCE',
        'BINANCE_US': 'EXCHANGE_BINANCE_US',
        'BINANCE_TR': 'EXCHANGE_BINANCE_TR',
        'BINANCE_FUTURES': 'EXCHANGE_BINANCE_FUTURES',
        'BINANCE_DELIVERY': 'EXCHANGE_BINANCE_DELIVERY',
        'BIT.COM': 'EXCHANGE_BITDOTCOM',
        'BITFLYER': 'EXCHANGE_BITFLYER',
        'BITGET': 'EXCHANGE_BITGET',
        'BITSTAMP': 'EXCHANGE_BITSTAMP',
        'BLOCKCHAIN': 'EXCHANGE_BLOCKCHAIN',
        'BYBIT': 'EXCHANGE_BYBIT',
        'COINBASE': 'EXCHANGE_COINBASE',
        'CRYPTO.COM': 'EXCHANGE_CRYPTODOTCOM',
        'DELTA': 'EXCHANGE_DELTA',
        'DERIBIT': 'EXCHANGE_DERIBIT',
        'DYDX': 'EXCHANGE_DYDX',
        'EXX': 'EXCHANGE_EXX',
        'FMFW': 'EXCHANGE_FMFW',
        'GATEIO': 'EXCHANGE_GATEIO',
        'GATEIO_FUTURES': 'EXCHANGE_GATEIO_FUTURES',
        'GEMINI': 'EXCHANGE_GEMINI',
        'HITBTC': 'EXCHANGE_HITBTC',
        'HUOBI': 'EXCHANGE_HUOBI',
        'HUOBI_DM': 'EXCHANGE_HUOBI_DM',
        'HUOBI_SWAP': 'EXCHANGE_HUOBI_SWAP',
        'INDEPENDENT_RESERVE': 'EXCHANGE_INDEPENDENT_RESERVE',
        'KRAKEN': 'EXCHANGE_KRAKEN',
        'KRAKEN_FUTURES': 'EXCHANGE_KRAKEN_FUTURES',
        'KUCOIN': 'EXCHANGE_KUCOIN',
        'OKCOIN': 'EXCHANGE_OKCOIN',
        'OKX': 'EXCHANGE_OKX',
        'PHEMEX': 'EXCHANGE_PHEMEX',
        'POLONIEX': 'EXCHANGE_POLONIEX',
        'PROBIT': 'EXCHANGE_PROBIT',
        'UPBIT': 'EXCHANGE_UPBIT'
    }
    
    # Side mapping
    SIDE_MAPPING = {
        'buy': 'SIDE_BUY',
        'sell': 'SIDE_SELL',
        'bid': 'SIDE_BID',
        'ask': 'SIDE_ASK'
    }
    
    @classmethod
    def to_protobuf(cls, legacy_trade: 'LegacyTrade') -> 'ProtoTrade':
        """Convert legacy Trade to protobuf Trade message."""
        if ProtoTrade is None:
            raise ImportError("Protobuf classes not available")
            
        proto_trade = ProtoTrade()
        
        # Set exchange
        exchange_name = cls.EXCHANGE_MAPPING.get(legacy_trade.exchange, 'EXCHANGE_UNSPECIFIED')
        if Exchange:
            proto_trade.exchange = Exchange.Value(exchange_name)
        
        # Set symbol
        if legacy_trade.symbol:
            proto_trade.symbol.symbol = legacy_trade.symbol
            # Parse symbol parts if possible (e.g., "BTC-USD" -> base="BTC", quote="USD")
            if '-' in legacy_trade.symbol:
                parts = legacy_trade.symbol.split('-', 1)
                proto_trade.symbol.base = parts[0]
                proto_trade.symbol.quote = parts[1]
        
        # Set side
        side_name = cls.SIDE_MAPPING.get(legacy_trade.side, 'SIDE_UNSPECIFIED')
        if Side:
            proto_trade.side = Side.Value(side_name)
        
        # Set amount and price as string decimals
        if legacy_trade.amount is not None:
            proto_trade.amount.value = str(legacy_trade.amount)
        if legacy_trade.price is not None:
            proto_trade.price.value = str(legacy_trade.price)
        
        # Set other fields
        proto_trade.id = legacy_trade.id or ""
        proto_trade.type = legacy_trade.type or ""
        
        # Set timestamp
        if legacy_trade.timestamp:
            proto_trade.timestamp.FromSeconds(int(legacy_trade.timestamp))
            # Handle fractional seconds
            nanos = int((legacy_trade.timestamp - int(legacy_trade.timestamp)) * 1e9)
            proto_trade.timestamp.nanos = nanos
        
        # Set receipt timestamp to current time
        proto_trade.receipt_timestamp.FromSeconds(int(time.time()))
        
        return proto_trade
    
    @classmethod
    def from_protobuf(cls, proto_trade: 'ProtoTrade') -> 'LegacyTrade':
        """Convert protobuf Trade message to legacy Trade."""
        if LegacyTrade is None:
            raise ImportError("Legacy Trade class not available")
            
        # Reverse mapping for exchange
        exchange_name = Exchange.Name(proto_trade.exchange) if Exchange else "UNKNOWN"
        exchange = exchange_name.replace("EXCHANGE_", "") if exchange_name.startswith("EXCHANGE_") else exchange_name
        
        # Reverse mapping for side  
        side_name = Side.Name(proto_trade.side) if Side else "unknown"
        side = side_name.replace("SIDE_", "").lower() if side_name.startswith("SIDE_") else side_name.lower()
        
        # Convert amounts back to Decimal
        amount = Decimal(proto_trade.amount.value) if proto_trade.amount.value else None
        price = Decimal(proto_trade.price.value) if proto_trade.price.value else None
        
        # Convert timestamp back to float
        timestamp = None
        if proto_trade.timestamp.seconds or proto_trade.timestamp.nanos:
            timestamp = proto_trade.timestamp.seconds + (proto_trade.timestamp.nanos / 1e9)
        
        return LegacyTrade(
            exchange=exchange,
            symbol=proto_trade.symbol.symbol,
            side=side,
            amount=amount,
            price=price,
            timestamp=timestamp,
            id=proto_trade.id if proto_trade.id else None,
            type=proto_trade.type if proto_trade.type else None
        )
    
    @classmethod
    def create_protobuf_from_dict(cls, trade_dict: dict) -> 'ProtoTrade':
        """Create protobuf Trade from dictionary (for backward compatibility)."""
        if ProtoTrade is None:
            raise ImportError("Protobuf classes not available")
            
        proto_trade = ProtoTrade()
        
        # Map dictionary fields to protobuf
        if 'exchange' in trade_dict:
            exchange_name = cls.EXCHANGE_MAPPING.get(trade_dict['exchange'], 'EXCHANGE_UNSPECIFIED')
            if Exchange:
                proto_trade.exchange = Exchange.Value(exchange_name)
        
        if 'symbol' in trade_dict:
            proto_trade.symbol.symbol = trade_dict['symbol']
            if '-' in trade_dict['symbol']:
                parts = trade_dict['symbol'].split('-', 1)
                proto_trade.symbol.base = parts[0]
                proto_trade.symbol.quote = parts[1]
        
        if 'side' in trade_dict:
            side_name = cls.SIDE_MAPPING.get(trade_dict['side'], 'SIDE_UNSPECIFIED')
            if Side:
                proto_trade.side = Side.Value(side_name)
        
        if 'amount' in trade_dict:
            proto_trade.amount.value = str(trade_dict['amount'])
        
        if 'price' in trade_dict:
            proto_trade.price.value = str(trade_dict['price'])
        
        if 'id' in trade_dict:
            proto_trade.id = str(trade_dict['id'])
        
        if 'type' in trade_dict:
            proto_trade.type = str(trade_dict['type'])
        
        if 'timestamp' in trade_dict:
            timestamp = float(trade_dict['timestamp'])
            proto_trade.timestamp.FromSeconds(int(timestamp))
            nanos = int((timestamp - int(timestamp)) * 1e9)
            proto_trade.timestamp.nanos = nanos
        
        return proto_trade
    
    @classmethod
    def to_dict(cls, proto_trade: 'ProtoTrade') -> dict:
        """Convert protobuf Trade to dictionary (for backward compatibility)."""
        exchange_name = Exchange.Name(proto_trade.exchange) if Exchange else "UNKNOWN"
        exchange = exchange_name.replace("EXCHANGE_", "") if exchange_name.startswith("EXCHANGE_") else exchange_name
        
        side_name = Side.Name(proto_trade.side) if Side else "unknown"
        side = side_name.replace("SIDE_", "").lower() if side_name.startswith("SIDE_") else side_name.lower()
        
        timestamp = None
        if proto_trade.timestamp.seconds or proto_trade.timestamp.nanos:
            timestamp = proto_trade.timestamp.seconds + (proto_trade.timestamp.nanos / 1e9)
        
        return {
            'exchange': exchange,
            'symbol': proto_trade.symbol.symbol,
            'side': side,
            'amount': proto_trade.amount.value,
            'price': proto_trade.price.value,
            'id': proto_trade.id if proto_trade.id else None,
            'type': proto_trade.type if proto_trade.type else None,
            'timestamp': timestamp
        }
