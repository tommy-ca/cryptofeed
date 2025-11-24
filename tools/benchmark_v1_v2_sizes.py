"""Quick size comparison between v1 and v2 protobuf payloads.

Usage:
    python tools/benchmark_v1_v2_sizes.py

Outputs byte lengths for a representative Trade message encoded with
legacy v1 helpers (string decimals) and v2 helpers (native doubles).
"""

from decimal import Decimal

from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf
from cryptofeed.backends.protobuf_helpers_v2 import serialize_to_protobuf_v2


class _Trade:
    def __init__(self):
        self.exchange = "coinbase"
        self.symbol = "BTC-USD"
        self.side = "buy"
        self.id = "sample-1"
        self.price = Decimal("68000.12345678")
        self.amount = Decimal("0.25000000")
        self.timestamp = 1700000000.123456


def main() -> None:
    trade = _Trade()

    v1_bytes = serialize_to_protobuf(trade)
    v2_bytes = serialize_to_protobuf_v2(trade)

    reduction = 100 * (1 - len(v2_bytes) / len(v1_bytes)) if len(v1_bytes) else 0

    print("v1 bytes:", len(v1_bytes))
    print("v2 bytes:", len(v2_bytes))
    print(f"size reduction: {reduction:.2f}%")


if __name__ == "__main__":
    main()
