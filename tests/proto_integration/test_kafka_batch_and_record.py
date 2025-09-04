import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ensure_gen_on_path():
    gen_root = os.path.join(ROOT, "gen", "python")
    if gen_root not in sys.path:
        sys.path.insert(0, gen_root)
    import types
    cf_pkg = types.ModuleType('cryptofeed')
    cf_pkg.__path__ = [os.path.join(gen_root, 'cryptofeed')]
    sys.modules['cryptofeed'] = cf_pkg


def test_data_feed_event_batch_roundtrip():
    _ensure_gen_on_path()
    from cryptofeed.v1 import events_pb2 as ev
    from cryptofeed.v1 import common_pb2 as common

    e1 = ev.DataFeedEvent(event_id='e1', channel=common.DATA_CHANNEL_TRADES)
    e2 = ev.DataFeedEvent(event_id='e2', channel=common.DATA_CHANNEL_TICKER)
    batch = ev.DataFeedEventBatch(batch_id='b1')
    batch.events.extend([e1, e2])

    data = batch.SerializeToString()
    out = ev.DataFeedEventBatch()
    out.ParseFromString(data)

    assert out.batch_id == 'b1'
    assert len(out.events) == 2
    assert out.events[0].event_id == 'e1'
    assert out.events[1].event_id == 'e2'


def test_kafka_record_event_and_ticker_payloads():
    _ensure_gen_on_path()
    from cryptofeed.v1 import kafka_pb2 as kafka
    from cryptofeed.v1 import events_pb2 as ev
    from cryptofeed.v1 import market_data_pb2 as md
    from cryptofeed.v1 import common_pb2 as common

    # Event payload
    rec1 = kafka.KafkaRecord()
    rec1.metadata.topic = 'crypto.events'
    rec1.event.CopyFrom(ev.DataFeedEvent(event_id='evt-123'))
    assert rec1.WhichOneof('payload') == 'event'

    # Direct ticker payload
    rec2 = kafka.KafkaRecord()
    rec2.metadata.topic = 'crypto.ticker'
    tick = md.Ticker(exchange=common.EXCHANGE_BINANCE)
    rec2.ticker.CopyFrom(tick)

    data = rec2.SerializeToString()
    out = kafka.KafkaRecord()
    out.ParseFromString(data)
    assert out.WhichOneof('payload') == 'ticker'
    assert out.ticker.exchange == common.EXCHANGE_BINANCE

