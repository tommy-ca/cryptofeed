import os
import importlib.util
import sys
import types


def project_root(start: str) -> str:
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, 'gen', 'python')) and os.path.isfile(os.path.join(cur, 'buf.yaml')):
            return cur
        nxt = os.path.abspath(os.path.join(cur, os.pardir))
        if nxt == cur:
            return cur
        cur = nxt


def ensure_gen_on_path(root: str) -> None:
    gen_root = os.path.join(root, 'gen', 'python')
    if gen_root not in sys.path:
        sys.path.insert(0, gen_root)
    cf_pkg = types.ModuleType('cryptofeed')
    cf_pkg.__path__ = [os.path.join(gen_root, 'cryptofeed')]
    sys.modules['cryptofeed'] = cf_pkg


def load_pb2(root: str, rel_path: str):
    ensure_gen_on_path(root)
    path = os.path.join(root, rel_path)
    spec = importlib.util.spec_from_file_location('_pb2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod
