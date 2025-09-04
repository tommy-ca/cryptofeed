import os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cryptofeed.proto_mappers import util

@pytest.mark.parametrize('symbol, expected', [
    ('BTCUSDT', ('BTC','USDT')),
    ('ETHUSD', ('ETH','USD')),
    ('BTCUSD', ('BTC','USD')),
    ('BTCUSDC', ('BTC','USDC')),
    ('BTCFDUSD', ('BTC','FDUSD')),
    ('BTC-USD', ('BTC','USD')),
])
def test_split_base_quote(symbol, expected):
    if '-' in symbol:
        base, quote = util.split_hyphen_symbol(symbol)
    else:
        base, quote = util.split_base_quote_concat(symbol)
    assert (base, quote) == expected
