from __future__ import annotations

import ast
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_install_requires() -> list[str]:
    setup_path = _project_root() / "setup.py"
    tree = ast.parse(setup_path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "install_requires":
            return ast.literal_eval(node.value)

    raise AssertionError("install_requires not found in setup.py")


def test_setup_install_requires_contains_proxy_dependencies():
    install_requires = _read_install_requires()
    assert any(dep.startswith("pydantic>=") for dep in install_requires)
    assert any(dep.startswith("pydantic-settings>=") for dep in install_requires)


def test_requirements_file_contains_proxy_dependencies():
    requirements_path = _project_root() / "requirements.txt"
    requirements = [line.strip() for line in requirements_path.read_text().splitlines() if line.strip()]

    assert any(line.startswith("pydantic>=") for line in requirements)
    assert any(line.startswith("pydantic-settings>=") for line in requirements)
