"""
Topic naming strategies and management utilities for Kafka topics.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class TopicStrategy(Enum):
    """Topic naming strategies for Kafka topics."""

    CONSOLIDATED = "consolidated"
    PER_SYMBOL = "per_symbol"


class TopicManager:
    """Manages topic naming strategies for Kafka topics."""

    # Supported data types (normalized to singular form for topic naming)
    SUPPORTED_DATA_TYPES = {
        "trade",
        "trades",  # plural form supported for backward compatibility
        "orderbook",
        "ticker",
        "candle",
        "funding",
        "liquidation",
        "index",
        "openinterest",
        "fill",
        "balance",
        "position",
        "margin",
        "order",
        "transaction",
    }

    STRATEGIES = {"consolidated", "per_symbol"}

    @staticmethod
    def validate_strategy(strategy: str) -> None:
        if strategy not in TopicManager.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Supported strategies: {', '.join(sorted(TopicManager.STRATEGIES))}"
            )

    @staticmethod
    def validate_data_type(data_type: str) -> None:
        if data_type not in TopicManager.SUPPORTED_DATA_TYPES:
            sorted_types = ", ".join(sorted(TopicManager.SUPPORTED_DATA_TYPES))
            raise ValueError(
                f"Unsupported data type: {data_type}. Supported types: {sorted_types}"
            )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # Lowercase for topic stability; normalize common separators.
        return str(symbol).lower().replace("_", "-").replace("/", "-")

    @staticmethod
    def _normalize_exchange(exchange: str) -> str:
        return str(exchange).lower()

    @staticmethod
    def get_topic(
        data_type: str,
        symbol: str,
        exchange: str,
        strategy: str = "consolidated",
        prefix: Optional[str] = None,
    ) -> str:
        TopicManager.validate_strategy(strategy)
        TopicManager.validate_data_type(data_type)

        strategy = strategy.lower()
        prefix_clean = prefix.strip() if prefix else ""

        if strategy == TopicStrategy.CONSOLIDATED.value:
            topic_body = f"cryptofeed.{data_type}"
        else:
            # Validate required fields for per_symbol strategy
            if not symbol:
                raise ValueError("Symbol is required for per_symbol topic strategy")
            if not exchange:
                raise ValueError("Exchange is required for per_symbol topic strategy")

            normalized_exchange = TopicManager._normalize_exchange(exchange)
            normalized_symbol = TopicManager._normalize_symbol(symbol)
            topic_body = (
                f"cryptofeed.{data_type}.{normalized_exchange}.{normalized_symbol}"
            )

        if prefix_clean:
            return f"{prefix_clean}.{topic_body}"
        return topic_body


__all__ = ["TopicStrategy", "TopicManager"]
