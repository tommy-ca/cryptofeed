"""Backpack message adapters converting raw payloads to cryptofeed types."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Iterable, List, Optional, Sequence

from cryptofeed.defines import ASK, BID, BUY, SELL
from cryptofeed.types import Candle, OrderBook, OrderInfo, Position, Trade, Ticker

from .errors import (
    BackpackOrderBookGap,
    BackpackOrderBookMissingSnapshot,
    BackpackPayloadError,
)


def _microseconds_to_seconds(value: Optional[int | float | str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise BackpackPayloadError(f"Invalid timestamp value: {value}") from exc
    if isinstance(value, float):
        return value
    if value >= 1_000_000_000_000:
        return float(value) / 1_000_000.0
    if value >= 1_000_000:
        return float(value) / 1_000.0
    return float(value)


@dataclass(slots=True)
class TradePayload:
    symbol: str
    price: Decimal
    amount: Decimal
    side: Optional[str]
    trade_id: Optional[str]
    sequence: Optional[int]
    timestamp: Optional[float]
    raw: dict


class BackpackTradeAdapter:
    """Convert Backpack trade payloads into cryptofeed Trade objects."""

    def __init__(self, exchange: str):
        self._exchange = exchange

    def parse(self, payload: dict, *, normalized_symbol: str) -> Trade:
        trade = self._parse_payload(payload, normalized_symbol)
        return Trade(
            exchange=self._exchange,
            symbol=trade.symbol,
            side=trade.side,
            amount=trade.amount,
            price=trade.price,
            timestamp=trade.timestamp or 0.0,
            id=trade.trade_id,
            raw=payload,
        )

    def _parse_payload(self, payload: dict, normalized_symbol: str) -> TradePayload:
        try:
            price_raw = payload.get("price") or payload.get("p")
            amount_raw = payload.get("size") or payload.get("q")
            if price_raw is None or amount_raw is None:
                raise KeyError("price/size missing")
            price = Decimal(str(price_raw))
            amount = Decimal(str(amount_raw))
        except (InvalidOperation, KeyError) as exc:
            raise BackpackPayloadError(f"Invalid trade payload: {payload}") from exc

        timestamp_raw = payload.get("timestamp") or payload.get("ts")
        sequence_raw = payload.get("sequence") or payload.get("s")
        trade_id_raw = payload.get("id") or payload.get("t")

        sequence = None
        if sequence_raw is not None:
            try:
                sequence = int(sequence_raw)
            except (TypeError, ValueError):
                sequence = None

        side = self._normalize_side(payload.get("side"))

        return TradePayload(
            symbol=normalized_symbol,
            price=price,
            amount=amount,
            side=side,
            trade_id=str(trade_id_raw) if trade_id_raw is not None else None,
            sequence=sequence,
            timestamp=_microseconds_to_seconds(timestamp_raw),
            raw=payload,
        )

    @staticmethod
    def _normalize_side(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        upper = str(value).upper()
        if upper in {"BUY", "BID", "LONG"}:
            return BUY
        if upper in {"SELL", "ASK", "SHORT"}:
            return SELL
        return upper


class BackpackOrderBookAdapter:
    """Maintain Backpack order book state and emit cryptofeed OrderBook objects."""

    def __init__(self, exchange: str, *, max_depth: int = 0):
        self._exchange = exchange
        self._max_depth = max_depth
        self._books: Dict[str, OrderBook] = {}
        self._sequences: Dict[str, Optional[int]] = {}

    def apply_snapshot(
        self,
        *,
        normalized_symbol: str,
        bids: Iterable[Iterable],
        asks: Iterable[Iterable],
        timestamp: Optional[int | float | str] = None,
        sequence: Optional[int] = None,
        raw: Optional[dict] = None,
    ) -> OrderBook:
        bids_processed = self._normalize_side_levels(bids, side=BID)
        asks_processed = self._normalize_side_levels(asks, side=ASK)

        order_book = OrderBook(
            exchange=self._exchange,
            symbol=normalized_symbol,
            bids=bids_processed,
            asks=asks_processed,
            max_depth=self._max_depth,
        )
        order_book.timestamp = _microseconds_to_seconds(timestamp)
        order_book.sequence_number = self._coerce_sequence(sequence)
        order_book.raw = raw
        self._books[normalized_symbol] = order_book
        self._sequences[normalized_symbol] = order_book.sequence_number
        self._trim_depth(order_book)
        return order_book

    def apply_delta(
        self,
        *,
        normalized_symbol: str,
        bids: Iterable[Iterable] | None,
        asks: Iterable[Iterable] | None,
        timestamp: Optional[int | float | str],
        sequence: Optional[int],
        raw: Optional[dict],
    ) -> OrderBook:
        if normalized_symbol not in self._books:
            raise BackpackOrderBookMissingSnapshot(f"No snapshot for symbol {normalized_symbol}")

        book = self._books[normalized_symbol]
        next_sequence = self._coerce_sequence(sequence)
        last_sequence = self._sequences.get(normalized_symbol)
        if last_sequence is not None and next_sequence is not None:
            if next_sequence <= last_sequence or next_sequence - last_sequence > 1:
                raise BackpackOrderBookGap(
                    symbol=normalized_symbol,
                    expected=last_sequence + 1 if last_sequence is not None else None,
                    actual=next_sequence,
                )

        delta_bids = self._normalize_delta_levels(bids)
        delta_asks = self._normalize_delta_levels(asks)

        if delta_bids:
            self._update_levels(book.book.bids, delta_bids)
        if delta_asks:
            self._update_levels(book.book.asks, delta_asks)

        book.timestamp = _microseconds_to_seconds(timestamp)
        book.sequence_number = next_sequence if next_sequence is not None else last_sequence
        self._sequences[normalized_symbol] = book.sequence_number
        book.delta = {BID: delta_bids, ASK: delta_asks}
        book.raw = raw
        self._trim_depth(book)
        return book

    def has_snapshot(self, normalized_symbol: str) -> bool:
        return normalized_symbol in self._books

    def clear(self, normalized_symbol: str) -> None:
        self._books.pop(normalized_symbol, None)
        self._sequences.pop(normalized_symbol, None)

    def _normalize_side_levels(
        self, levels: Iterable[Iterable], *, side: str
    ) -> Dict[Decimal, Decimal]:
        if levels is None:
            return {}
        result: Dict[Decimal, Decimal] = {}
        for level in levels:
            try:
                price = Decimal(str(level[0]))
                size = Decimal(str(level[1]))
            except (InvalidOperation, IndexError) as exc:
                raise BackpackPayloadError(f"Invalid {side} level: {level}") from exc
            if size < 0:
                raise BackpackPayloadError(f"Negative size in {side} level: {level}")
            if size == 0:
                continue
            result[price] = size
        return result

    def _normalize_delta_levels(
        self, levels: Iterable[Iterable] | None
    ) -> List[tuple[Decimal, Decimal]]:
        if not levels:
            return []
        normalized: List[tuple[Decimal, Decimal]] = []
        for level in levels:
            try:
                price = Decimal(str(level[0]))
                size = Decimal(str(level[1]))
            except (InvalidOperation, IndexError) as exc:
                raise BackpackPayloadError(f"Invalid order book delta level: {level}") from exc
            normalized.append((price, size))
        return normalized

    @staticmethod
    def _update_levels(price_map: Dict[Decimal, Decimal], delta: Sequence[tuple[Decimal, Decimal]]) -> None:
        for price, size in delta:
            if size == 0:
                if price in price_map:
                    del price_map[price]
            else:
                price_map[price] = size

    def _trim_depth(self, book: OrderBook) -> None:
        if not self._max_depth or self._max_depth <= 0:
            return
        bids = sorted(book.book.bids.items(), key=lambda item: item[0], reverse=True)
        asks = sorted(book.book.asks.items(), key=lambda item: item[0])
        if len(bids) > self._max_depth:
            trimmed = bids[: self._max_depth]
            book.book.bids.clear()
            book.book.bids.update(trimmed)
        if len(asks) > self._max_depth:
            trimmed = asks[: self._max_depth]
            book.book.asks.clear()
            book.book.asks.update(trimmed)

    @staticmethod
    def _coerce_sequence(sequence: Optional[int]) -> Optional[int]:
        if sequence is None:
            return None
        try:
            return int(sequence)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None


class BackpackCandleAdapter:
    """Convert Backpack candle payloads to cryptofeed Candle objects."""

    def __init__(self, exchange: str):
        self._exchange = exchange

    def parse(self, payload: dict, *, normalized_symbol: str) -> Candle:
        start = self._normalize_epoch(
            payload.get("startTime")
            or payload.get("start")
            or payload.get("openTime")
            or payload.get("t")
        )
        stop = self._normalize_epoch(
            payload.get("endTime")
            or payload.get("stop")
            or payload.get("closeTime")
            or payload.get("T")
            or start
        )
        interval = str(
            payload.get("interval")
            or payload.get("resolution")
            or payload.get("granularity")
            or payload.get("period")
            or "1m"
        )

        open_price = self._decimal_field(payload, ["open", "o"])
        close_price = self._decimal_field(payload, ["close", "c"])
        high_price = self._decimal_field(payload, ["high", "h"])
        low_price = self._decimal_field(payload, ["low", "l"])
        volume = self._decimal_field(payload, ["volume", "v"])

        trades = self._int_field(payload, ["trades", "tradeCount", "n"])
        closed = bool(payload.get("closed", payload.get("isClosed", True)))
        timestamp = _microseconds_to_seconds(payload.get("timestamp") or payload.get("ts") or stop)

        return Candle(
            self._exchange,
            normalized_symbol,
            start,
            stop,
            interval,
            trades,
            open_price,
            close_price,
            high_price,
            low_price,
            volume,
            closed,
            timestamp,
            raw=payload,
        )

    @staticmethod
    def _normalize_epoch(value: Optional[int | float | str]) -> float:
        if value is None:
            return 0.0
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise BackpackPayloadError(f"Invalid epoch value: {value}") from exc
        if isinstance(value, float) and value < 1_000_000_000_000:
            return value
        if isinstance(value, float):
            return value / 1_000.0
        if value >= 1_000_000_000_000:
            return float(value) / 1_000.0
        return float(value)

    @staticmethod
    def _decimal_field(payload: dict, keys: Sequence[str]) -> Decimal:
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return Decimal(str(payload[key]))
                except InvalidOperation as exc:
                    raise BackpackPayloadError(f"Invalid decimal field '{key}': {payload[key]}") from exc
        raise BackpackPayloadError(f"Missing decimal fields {keys}")

    @staticmethod
    def _int_field(payload: dict, keys: Sequence[str]) -> Optional[int]:
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return int(payload[key])
                except (TypeError, ValueError):
                    return None
        return None


class BackpackOrderAdapter:
    """Convert Backpack private order payloads to OrderInfo objects."""

    def __init__(self, exchange: str):
        self._exchange = exchange

    def parse(self, payload: dict, *, normalized_symbol: str) -> OrderInfo:
        order_id = self._string_field(payload, ["id", "orderId", "order_id"])
        side = self._normalize_side(payload.get("side"))
        status = str(payload.get("status", "UNKNOWN")).upper()
        order_type = str(payload.get("type", "LIMIT"))
        price = self._decimal_field(payload, ["price", "p", "avgPrice", "stopPrice"], default=Decimal("0"))
        amount = self._decimal_field(payload, ["size", "amount", "q"], default=Decimal("0"))
        remaining = self._derive_remaining(payload, amount)
        client_order_id = payload.get("clientOrderId") or payload.get("client_order_id")
        account = payload.get("accountId") or payload.get("account")
        timestamp = _microseconds_to_seconds(
            payload.get("timestamp")
            or payload.get("ts")
            or payload.get("updatedAt")
            or payload.get("createdAt")
        )

        return OrderInfo(
            self._exchange,
            normalized_symbol,
            order_id,
            side,
            status,
            order_type,
            price,
            amount,
            remaining,
            timestamp,
            client_order_id=client_order_id,
            account=str(account) if account is not None else None,
            raw=payload,
        )

    @staticmethod
    def _string_field(payload: dict, keys: Sequence[str]) -> str:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return str(value)
        raise BackpackPayloadError(f"Missing required string fields {keys}")

    @staticmethod
    def _decimal_field(payload: dict, keys: Sequence[str], *, default: Optional[Decimal] = None) -> Decimal:
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return Decimal(str(payload[key]))
                except InvalidOperation as exc:
                    raise BackpackPayloadError(f"Invalid decimal field '{key}': {payload[key]}") from exc
        if default is not None:
            return default
        raise BackpackPayloadError(f"Missing decimal fields {keys}")

    @staticmethod
    def _normalize_side(side: Optional[str]) -> str:
        if side is None:
            return "UNKNOWN"
        upper = str(side).upper()
        if upper in {"BUY", BUY}:
            return BUY
        if upper in {"SELL", SELL}:
            return SELL
        return upper

    @staticmethod
    def _derive_remaining(payload: dict, amount: Decimal) -> Optional[Decimal]:
        remaining_keys = ["remaining", "remainingSize", "leavesQty", "leavesSize"]
        for key in remaining_keys:
            if key in payload and payload[key] is not None:
                try:
                    return Decimal(str(payload[key]))
                except InvalidOperation:
                    return None

        filled = payload.get("filledSize") or payload.get("filled")
        if filled is None:
            return None
        try:
            filled_decimal = Decimal(str(filled))
        except InvalidOperation:
            return None
        remaining = amount - filled_decimal
        return remaining if remaining >= 0 else Decimal("0")


class BackpackPositionAdapter:
    """Convert Backpack private position payloads to Position objects."""

    def __init__(self, exchange: str):
        self._exchange = exchange

    def parse(self, payload: dict, *, normalized_symbol: str) -> Position:
        size = self._decimal_field(payload, ["size", "position", "quantity"])
        entry = self._decimal_field(payload, ["entryPrice", "avgEntryPrice", "price"])
        side = str(payload.get("side", "BOTH")).upper()
        pnl = self._decimal_optional(payload, ["unrealizedPnl", "pnl", "unrealisedPnl"])
        timestamp = _microseconds_to_seconds(payload.get("timestamp") or payload.get("ts") or payload.get("updatedAt"))

        return Position(
            self._exchange,
            normalized_symbol,
            size,
            entry,
            side,
            pnl,
            timestamp,
            raw=payload,
        )

    @staticmethod
    def _decimal_field(payload: dict, keys: Sequence[str]) -> Decimal:
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return Decimal(str(payload[key]))
                except InvalidOperation as exc:
                    raise BackpackPayloadError(f"Invalid decimal field '{key}': {payload[key]}") from exc
        raise BackpackPayloadError(f"Missing decimal fields {keys}")

    @staticmethod
    def _decimal_optional(payload: dict, keys: Sequence[str]) -> Optional[Decimal]:
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return Decimal(str(payload[key]))
                except InvalidOperation:
                    return None
        return None


class BackpackTickerAdapter:
    """Convert Backpack ticker payloads into cryptofeed Ticker objects."""

    def __init__(self, exchange: str):
        self._exchange = exchange

    def parse(self, payload: dict, *, normalized_symbol: str) -> Ticker:
        last_raw = payload.get("last") or payload.get("price")
        last_price = Decimal(str(last_raw)) if last_raw is not None else Decimal("0")
        bid_val = payload.get("bestBid") or payload.get("bid")
        ask_val = payload.get("bestAsk") or payload.get("ask")
        bid_dec = Decimal(str(bid_val)) if bid_val is not None else last_price
        ask_dec = Decimal(str(ask_val)) if ask_val is not None else last_price
        timestamp = _microseconds_to_seconds(payload.get("timestamp") or payload.get("ts")) or 0.0

        return Ticker(
            exchange=self._exchange,
            symbol=normalized_symbol,
            bid=bid_dec,
            ask=ask_dec,
            timestamp=timestamp,
            raw=payload,
        )
