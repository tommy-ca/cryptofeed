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


def test_kafka_wrappers_roundtrip():
    _ensure_gen_on_path()

    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from cryptofeed.v1 import kafka_pb2 as kafka

    # Build a minimal event
    e = ev.DataFeedEvent(
        event_id='evt-1',
        channel=common.DATA_CHANNEL_TICKER,
        exchange=common.EXCHANGE_OKX,
    )

    # Wrap in KafkaDataFeedEvent
    kde = kafka.KafkaDataFeedEvent()
    kde.event.CopyFrom(e)
    kde.metadata.topic = 'crypto.events'
    kde.metadata.partition = 3
    kde.metadata.offset = 42
    kde.metadata.headers.add(key='schema.version', value=b'v1')

    data = kde.SerializeToString()
    out = kafka.KafkaDataFeedEvent()
    out.ParseFromString(data)

    assert out.event.event_id == 'evt-1'
    assert out.event.channel == common.DATA_CHANNEL_TICKER
    assert out.metadata.topic == 'crypto.events'
    assert out.metadata.partition == 3
    assert out.metadata.offset == 42
    assert any(h.key == 'schema.version' and h.value == b'v1' for h in out.metadata.headers)

