import os, sys, importlib.util, types
import pytest

def project_root(start: str) -> str:
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, 'gen', 'python')) and os.path.isfile(os.path.join(cur, 'buf.yaml')):
            return cur
        nxt = os.path.abspath(os.path.join(cur, os.pardir))
        if nxt == cur:
            return cur
        cur = nxt

def load_pb2(root: str, rel_path: str):
    gen_root = os.path.join(root, 'gen', 'python')
    if gen_root not in sys.path:
        sys.path.insert(0, gen_root)
    cf_pkg = types.ModuleType('cryptofeed'); cf_pkg.__path__=[os.path.join(gen_root,'cryptofeed')]
    sys.modules['cryptofeed']=cf_pkg
    path = os.path.join(root, rel_path)
    spec = importlib.util.spec_from_file_location('_pb2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

ROOT = project_root(os.path.dirname(__file__))


@pytest.mark.parametrize(
    'mapper_mod, pb2_path, build_pair',
    [
        (
            'cryptofeed.proto_mappers.binance',
            'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py',
            lambda pb2, md: (
                (lambda d: (d, 100))(pb2.DepthUpdate(symbol='BTCUSDT')),
                (lambda d: (d, 101))(pb2.DepthUpdate(symbol='BTCUSDT')),
            ),
        ),
        (
            'cryptofeed.proto_mappers.okx',
            'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py',
            lambda pb2, md: (
                (lambda o: (o, 200))(pb2.OrderBook(inst_id='BTC-USDT')),
                (lambda o: (o, 201))(pb2.OrderBook(inst_id='BTC-USDT')),
            ),
        ),
        (
            'cryptofeed.proto_mappers.bybit',
            'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py',
            lambda pb2, md: (
                (lambda o: (o, 300))(pb2.OrderBook(symbol='BTCUSDT')),
                (lambda o: (o, 301))(pb2.OrderBook(symbol='BTCUSDT')),
            ),
        ),
        (
            'cryptofeed.proto_mappers.bitget',
            'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py',
            lambda pb2, md: (
                (lambda o: (o, 400))(pb2.OrderBook(inst_id='BTCUSDT')),
                (lambda o: (o, 401))(pb2.OrderBook(inst_id='BTCUSDT')),
            ),
        ),
    ],
)
def test_sequence_ordering_progression(mapper_mod, pb2_path, build_pair):
    mapper = __import__(mapper_mod, fromlist=['dummy'])
    pb2 = load_pb2(ROOT, pb2_path)
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    first, first_seq = build_pair(pb2, md)[0]
    second, second_seq = build_pair(pb2, md)[1]

    # attach trivial one-level entries so mapping populates
    plb = md.PriceLevel(); plb.price.value='1'; plb.size.value='1'
    pla = md.PriceLevel(); pla.price.value='2'; pla.size.value='2'
    for ob in (first, second):
        if hasattr(ob, 'bids'):
            ob.bids.append(plb)
            ob.asks.append(pla)

    # set sequence ids using exchange-specific field names
    if first.__class__.__name__ == 'DepthUpdate':
        first.final_update_id = first_seq
        second.final_update_id = second_seq
        to_delta = getattr(mapper, 'to_common_book_delta_from_depth')
    else:
        # OrderBook variants
        if hasattr(first, 'seq_id'):
            first.seq_id = first_seq; second.seq_id = second_seq
        elif hasattr(first, 'seq'):
            first.seq = first_seq; second.seq = second_seq
        to_delta = getattr(mapper, 'to_common_book_delta')

    d1 = to_delta(first, md)
    d2 = to_delta(second, md)
    assert d1.sequence_number < d2.sequence_number
