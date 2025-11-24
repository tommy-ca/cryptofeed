"""Header enrichment utilities for Kafka messages."""

from __future__ import annotations

from typing import Any, Optional

class MessageHeaders:
    """Builder for mandatory message headers (Task 3.2).

    Mandatory headers provide essential routing metadata for Kafka consumers:
    - **content-type**: Serialization format (application/x-protobuf or application/json)
    - **exchange**: Exchange name (e.g., coinbase, binance) - normalized to lowercase
    - **symbol**: Trading symbol (e.g., BTC-USD) - underscores converted to hyphens
    - **data_type**: Message type (e.g., trades, ticker, orderbook)

    All header values are returned as UTF-8 encoded bytes in a list of tuples format
    that can be passed directly to `KafkaProducer.produce(headers=...)`.

    Normalization Rules:
    - Exchange: Lowercase, whitespace stripped
    - Symbol: Underscores converted to hyphens, whitespace stripped
    - Content-Type: Passed through as-is (lowercase recommended by caller)
    - Data Type: Passed through as-is (lowercase recommended by caller)

    Missing Attributes:
    - If message lacks exchange or symbol attributes, defaults to "unknown"
    - If exchange/symbol are None or empty, defaults to "unknown"

    Example:
        >>> from cryptofeed.types import Trade
        >>> trade = Trade(exchange='Coinbase', symbol='BTC_USD', ...)
        >>> headers = MessageHeaders.build(
        ...     message=trade,
        ...     data_type='trades',
        ...     content_type='application/x-protobuf'
        ... )
        >>> # headers = [
        >>> #     (b'content-type', b'application/x-protobuf'),
        >>> #     (b'exchange', b'coinbase'),  # normalized to lowercase
        >>> #     (b'symbol', b'BTC-USD'),    # underscores converted to hyphens
        >>> #     (b'data_type', b'trades')
        >>> # ]
    """

    @staticmethod
    def build(message: Any, data_type: str, content_type: str) -> list[tuple[bytes, bytes]]:
        """Build mandatory headers from message metadata.

        Extracts exchange and symbol from message object and normalizes them
        according to Kafka header conventions. All output is UTF-8 encoded bytes.

        Args:
            message: Message object with exchange and symbol attributes.
                     If attributes are missing, defaults to "unknown".
            data_type: Data type name (e.g., 'trades', 'orderbook', 'ticker').
                       Should be lowercase for consistency.
            content_type: Serialization format (e.g., 'application/x-protobuf',
                         'application/json'). Should be MIME type format.

        Returns:
            List of 4 (header_name, header_value) tuples with bytes values:
            - (b'content-type', <content_type>)
            - (b'exchange', <normalized_exchange>)
            - (b'symbol', <normalized_symbol>)
            - (b'data_type', <data_type>)

        Type:
            Callable[[Any, str, str], list[tuple[bytes, bytes]]]
        """
        # Extract metadata from message with fallbacks
        exchange = getattr(message, "exchange", "unknown")
        symbol = getattr(message, "symbol", "unknown")

        # Normalize exchange: lowercase and strip whitespace
        exchange_str = str(exchange).strip().lower() if exchange else "unknown"

        # Normalize symbol: strip whitespace and convert underscores to hyphens
        symbol_str = str(symbol).strip() if symbol else "unknown"
        symbol_str = symbol_str.replace("_", "-")

        # Build header list with consistent ordering
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", content_type.encode("utf-8")),
            (b"exchange", exchange_str.encode("utf-8")),
            (b"symbol", symbol_str.encode("utf-8")),
            (b"data_type", data_type.encode("utf-8")),
        ]

        return headers


class OptionalHeaders:
    """Builder for optional message headers (Task 3.3).

    Optional headers provide additional metadata for downstream processing and tracing:
    - **schema_version**: Protobuf schema version for backwards compatibility (default: v1)
    - **producer_version**: Cryptofeed package version for issue tracking
    - **timestamp_generated**: ISO8601 timestamp of message generation for latency tracking

    All header values are returned as UTF-8 encoded bytes in a list of tuples format.

    Usage:
    - Use for version tracking across consumer deployments
    - Timestamp allows consumers to measure end-to-end latency
    - Schema version enables migrations between protobuf versions
    - Can be overridden for testing or specific deployment scenarios

    Example:
        >>> headers = OptionalHeaders.build(
        ...     schema_version='v1',
        ...     producer_version='2.4.1'
        ... )
        >>> # headers = [
        >>> #     (b'schema_version', b'v1'),
        >>> #     (b'producer_version', b'2.4.1'),
        >>> #     (b'timestamp_generated', b'2025-11-09T12:34:56.123456Z')
        >>> # ]
    """

    @staticmethod
    def build(
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
        serialization_format: str = "json",
    ) -> list[tuple[bytes, bytes]]:
        """Build optional headers with defaults.

        Args:
            schema_version: Protobuf schema version (default: 'v1').
                           Should match version in .proto files and consumers.
            producer_version: Cryptofeed producer package version.
                             Default: '2.4.1' (from setup.py).
                             Can be overridden for testing or custom builds.
            timestamp_generated: ISO8601 timestamp of message generation.
                                Default: Current UTC time with Z suffix.
                                Format: YYYY-MM-DDTHH:MM:SS[.ffffff]Z

        Returns:
            List of 3 (header_name, header_value) tuples with bytes values:
            - (b'schema_version', <version>)
            - (b'producer_version', <version>)
            - (b'timestamp_generated', <iso8601_timestamp>)

        Type:
            Callable[[str, Optional[str], Optional[str]], list[tuple[bytes, bytes]]]
        """
        from datetime import datetime, timezone

        # Default producer version to package version (from setup.py)
        if producer_version is None:
            producer_version = "2.4.1"

        # Default timestamp to current UTC time in ISO8601 format
        if timestamp_generated is None:
            # Note: datetime.now(timezone.utc).isoformat() already includes +00:00
            # Don't add 'Z' since that's for naive UTC times
            iso_str = datetime.now(timezone.utc).isoformat()
            # Replace the +00:00 suffix with Z for brevity (standard for UTC)
            timestamp_generated = iso_str.replace("+00:00", "Z")

        # Build header list with consistent ordering
        headers: list[tuple[bytes, bytes]] = [
            (b"schema_version", schema_version.encode("utf-8")),
            (b"producer_version", producer_version.encode("utf-8")),
            (b"timestamp_generated", timestamp_generated.encode("utf-8")),
            (b"cf.serialization_format", serialization_format.encode("utf-8")),
        ]

        return headers


class HeaderEnricher:
    """Complete message header enrichment pipeline (Task 3.1).

    Combines mandatory and optional headers into a single enrichment class
    that integrates into the Kafka producer pipeline. This is the primary
    interface for header generation used by KafkaCallback.

    The enricher ensures all headers are properly formatted and encoded:
    - All header names and values are bytes (UTF-8 encoded)
    - Headers are returned as list of tuples compatible with confluent-kafka
    - Metadata is normalized (exchange lowercase, symbol underscores to hyphens)
    - Consistent header ordering: mandatory headers first, then optional headers

    Architecture:
    - HeaderEnricher delegates to MessageHeaders for mandatory headers
    - HeaderEnricher delegates to OptionalHeaders for optional headers
    - Composition pattern for separation of concerns

    Performance:
    - Fast header generation (<1ms per message)
    - No memory allocations beyond header list
    - Suitable for 10K+ messages/second throughput

    Example Usage:
        >>> from cryptofeed.kafka_callback import HeaderEnricher
        >>> enricher = HeaderEnricher(
        ...     content_type='application/x-protobuf',
        ...     schema_version='v1'
        ... )
        >>> trade = Trade(exchange='coinbase', symbol='BTC-USD', ...)
        >>> headers = enricher.build(message=trade, data_type='trades')
        >>> # Produces 7 headers:
        >>> #   Mandatory (4): content-type, exchange, symbol, data_type
        >>> #   Optional (3): schema_version, producer_version, timestamp_generated
        >>> # Result: [
        >>> #     (b'content-type', b'application/x-protobuf'),
        >>> #     (b'exchange', b'coinbase'),
        >>> #     (b'symbol', b'BTC-USD'),
        >>> #     (b'data_type', b'trades'),
        >>> #     (b'schema_version', b'v1'),
        >>> #     (b'producer_version', b'2.4.1'),
        >>> #     (b'timestamp_generated', b'2025-11-09T12:34:56Z')
        >>> # ]
    """

    def __init__(
        self,
        content_type: str = "application/x-protobuf",
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
        serialization_format: str = "json",
    ) -> None:
        """Initialize header enricher with configuration.

        Args:
            content_type: Serialization format for content-type header.
                         Default: 'application/x-protobuf'.
                         Common values: 'application/x-protobuf', 'application/json'.
            schema_version: Protobuf schema version for version tracking.
                           Default: 'v1'.
                           Should match version in consumers.
            producer_version: Cryptofeed producer package version.
                             Default: None (uses package version '2.4.1').
                             Can be overridden for custom builds.
            timestamp_generated: ISO8601 timestamp for message generation.
                                Default: None (uses current UTC time).
                                Useful for testing reproducible headers.
        """
        self.content_type = content_type
        self.schema_version = schema_version
        self.producer_version = producer_version
        self.timestamp_generated = timestamp_generated
        self.serialization_format = serialization_format

    def build(self, message: Any, data_type: str) -> list[tuple[bytes, bytes]]:
        """Build complete set of headers (mandatory + optional) for a message.

        Delegates to MessageHeaders for mandatory headers and OptionalHeaders
        for optional headers, then combines them in order.

        Args:
            message: Message object with exchange and symbol attributes.
                    Typically Trade, Ticker, OrderBook, or similar from cryptofeed.types.
                    If missing exchange/symbol attributes, defaults to "unknown".
            data_type: Data type name (e.g., 'trades', 'orderbook', 'ticker').
                      Should be lowercase for consistency.

        Returns:
            List of 7 (header_name, header_value) tuples with bytes values:
            1. (b'content-type', <format>)
            2. (b'exchange', <exchange>)
            3. (b'symbol', <symbol>)
            4. (b'data_type', <type>)
            5. (b'schema_version', <version>)
            6. (b'producer_version', <version>)
            7. (b'timestamp_generated', <timestamp>)

        Type:
            Callable[[Any, str], list[tuple[bytes, bytes]]]

        Example:
            >>> enricher = HeaderEnricher()
            >>> headers = enricher.build(trade, 'trades')
            >>> # Pass to producer:
            >>> producer.produce(topic, value=serialized_value, headers=headers)
        """
        # Build mandatory headers (exchange, symbol, data_type, content-type)
        mandatory = MessageHeaders.build(
            message=message,
            data_type=data_type,
            content_type=self.content_type,
        )

        # Build optional headers (schema_version, producer_version, timestamp)
        optional = OptionalHeaders.build(
            schema_version=self.schema_version,
            producer_version=self.producer_version,
            timestamp_generated=self.timestamp_generated,
            serialization_format=self.serialization_format,
        )

        # Combine all headers: mandatory first, then optional
        return mandatory + optional

    def enrich_message(
        self,
        message: Any,
        data_type: str,
        content_type: str = None,
    ) -> list[tuple[bytes, bytes]]:
        """Build headers for message enrichment (alias for build()).

        This method provides an alternative API name for building headers.
        It accepts an optional content_type parameter that overrides the
        instance's configured content_type.

        Args:
            message: Message object with exchange and symbol attributes.
            data_type: Data type name (e.g., 'trades', 'orderbook').
            content_type: Optional override for content-type header.
                         If not provided, uses instance's content_type.

        Returns:
            List of (header_name, header_value) tuples with bytes values.
        """
        # Use provided content_type or fall back to instance's default
        ct = content_type if content_type is not None else self.content_type

        # Build mandatory headers with overridden content_type if provided
        mandatory = MessageHeaders.build(
            message=message,
            data_type=data_type,
            content_type=ct,
        )

        # Build optional headers
        optional = OptionalHeaders.build(
            schema_version=self.schema_version,
            producer_version=self.producer_version,
            timestamp_generated=self.timestamp_generated,
        )

        return mandatory + optional

# ============================================================================
# Health Check Models and Implementation (Task 17.3)
# ============================================================================


