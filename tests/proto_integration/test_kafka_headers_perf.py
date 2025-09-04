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


def test_kafka_data_feed_event_headers_and_size():
    _ensure_gen_on_path()
    from cryptofeed.v1 import kafka_pb2 as kafka
    from cryptofeed.v1 import events_pb2 as ev

    evt = ev.DataFeedEvent(event_id='evt-1')
    wrapped = kafka.KafkaDataFeedEvent()
    wrapped.event.CopyFrom(evt)
    wrapped.metadata.topic = 'crypto.events'
    wrapped.metadata.headers.add(key='schema.version', value=b'v1')
    wrapped.metadata.headers.add(key='content.type', value=b'application/x-protobuf')

    data = wrapped.SerializeToString()
    # sanity size bound for minimal envelope
    assert len(data) < 2048

    out = kafka.KafkaDataFeedEvent()
    out.ParseFromString(data)
    keys = {h.key: h.value for h in out.metadata.headers}
    assert keys['schema.version'] == b'v1'
    assert keys['content.type'] == b'application/x-protobuf'

