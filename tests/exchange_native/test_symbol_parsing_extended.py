import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cryptofeed.proto_mappers import util


def test_concat_split_extended_quotes():
    cases = [
        ('BTCTRY', ('BTC','TRY')),
        ('ETHEUR', ('ETH','EUR')),
        ('BNBUSDC', ('BNB','USDC')),
        ('SOLBRL', ('SOL','BRL')),
        ('XRPDAI', ('XRP','DAI')),
        ('BTCBIDR', ('BTC','BIDR')),
        ('ETHBVND', ('ETH','BVND')),
        ('BTCUSDD', ('BTC','USDD')),
        ('BTCUSTC', ('BTC','USTC')),
    ]
    for sym, expected in cases:
        base, quote = util.split_base_quote_concat(sym)
        assert (base, quote) == expected

