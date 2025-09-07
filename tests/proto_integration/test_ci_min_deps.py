import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def test_proto_ci_installs_minimal_deps_only():
    path = os.path.join(ROOT, '.github', 'workflows', 'proto-ci.yml')
    data = open(path, 'r', encoding='utf-8').read()
    assert '-r requirements.txt' not in data
    assert 'pip install protobuf pytest' in data

