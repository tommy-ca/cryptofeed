from typing import Tuple, Any

BASE_KNOWN_QUOTES = (
    "USDT", "USD", "USDC", "BUSD", "FDUSD", "TUSD",
    "BTC", "ETH", "EUR", "GBP", "JPY", "AUD", "CAD"
)
# Prefer longer matches first to avoid partial suffix collisions (e.g., FDUSD vs USD)
KNOWN_QUOTES = tuple(sorted(BASE_KNOWN_QUOTES, key=len, reverse=True))

def split_base_quote_concat(symbol: str) -> Tuple[str, str]:
    base, quote = symbol, ""
    for q in KNOWN_QUOTES:
        if symbol.endswith(q) and len(symbol) > len(q):
            return symbol[:-len(q)], q
    return base, quote


def split_hyphen_symbol(inst_id: str) -> Tuple[str, str]:
    parts = inst_id.split('-')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return inst_id, ""


def make_decimal(cmn, val: Any):
    d = cmn.Decimal()
    d.value = str(val)
    return d
