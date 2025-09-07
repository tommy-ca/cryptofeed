import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def test_makefile_has_bsr_smoke_target():
    mk = open(os.path.join(ROOT, 'Makefile'), 'r', encoding='utf-8').read()
    assert '\nbsr-smoke:' in mk

