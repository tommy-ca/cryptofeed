import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def test_publish_workflow_has_required_steps_and_inputs():
    path = os.path.join(ROOT, '.github', 'workflows', 'bsr-publish.yml')
    data = open(path, 'r', encoding='utf-8').read()
    # Inputs present
    assert 'workflow_dispatch' in data
    assert 'version:' in data and 'dry_run:' in data
    assert 'create:' in data and 'visibility:' in data
    # Auth and checks
    assert 'buf registry whoami' in data
    assert 'buf lint' in data and 'buf build' in data and 'buf breaking' in data
    # Push with tag
    assert 'buf push --tag' in data

def test_bsr_smoke_workflow_exists():
    path = os.path.join(ROOT, '.github', 'workflows', 'bsr-smoke.yml')
    data = open(path, 'r', encoding='utf-8').read()
    assert 'buf registry whoami' in data
    assert 'buf lint' in data and 'buf build' in data

