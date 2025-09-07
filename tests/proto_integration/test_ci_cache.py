import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def test_proto_ci_uses_pip_cache():
    path = os.path.join(ROOT, '.github', 'workflows', 'proto-ci.yml')
    data = open(path, 'r', encoding='utf-8').read()
    assert 'actions/cache@v4' in data
    assert 'path: ~/.cache/pip' in data
    assert "hashFiles('**/requirements.txt')" in data


def test_proto_contracts_uses_pip_cache():
    path = os.path.join(ROOT, '.github', 'workflows', 'proto-contracts.yml')
    data = open(path, 'r', encoding='utf-8').read()
    assert 'actions/cache@v4' in data
    assert 'path: ~/.cache/pip' in data
    assert "hashFiles('**/requirements.txt')" in data

