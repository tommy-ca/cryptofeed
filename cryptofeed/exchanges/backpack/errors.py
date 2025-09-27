"""Custom exceptions for the Backpack native exchange integration."""
from __future__ import annotations

from dataclasses import dataclass


class BackpackPayloadError(RuntimeError):
    """Raised when an inbound Backpack payload fails validation."""


class BackpackUnknownChannel(RuntimeError):
    """Raised when the router cannot map a payload to a known channel."""


class BackpackOrderBookMissingSnapshot(RuntimeError):
    """Raised when a delta arrives before an initial order book snapshot."""


@dataclass(slots=True)
class BackpackOrderBookGap(RuntimeError):
    """Raised when an order book delta sequence reveals a gap requiring resync."""

    symbol: str
    expected: int | None
    actual: int | None

    def __str__(self) -> str:  # pragma: no cover - defensive stringification
        return (
            f"Backpack order book gap for {self.symbol}: "
            f"expected {self.expected}, received {self.actual}"
        )


class BackpackRouterError(RuntimeError):
    """Generic error raised by the Backpack message router."""

