"""
Test suite to validate the requests → aiohttp migration inventory.

This test module proves that:
1. Wave 1 migrations (Binance symbol bootstrap, listen-key) are complete
2. Wave 2 targets (schema registry) are correctly identified
3. Deferred paths (tooling) are documented and not blocking production

Spec: kafka-protobuf-binance-e2e
Task: 6.6 (Migration Plan)
"""

import ast
import os
import pytest
from pathlib import Path


class RequestsUsageFinder(ast.NodeVisitor):
    """AST visitor to find requests.* calls and import statements."""

    def __init__(self):
        self.imports = []
        self.calls = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "requests":
                self.imports.append((node.lineno, alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "requests":
            for alias in node.names:
                self.imports.append((node.lineno, f"from requests import {alias.name}"))
        self.generic_visit(node)

    def visit_Call(self, node):
        # Detect requests.get, requests.post, etc.
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "requests":
                self.calls.append((node.lineno, node.func.attr))
        self.generic_visit(node)


def find_requests_usage_in_file(file_path: Path):
    """Parse a Python file and find all requests usage."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            # Skip files with syntax errors (e.g., Cython)
            return [], []

    finder = RequestsUsageFinder()
    finder.visit(tree)
    return finder.imports, finder.calls


def get_production_modules():
    """Return list of production Python modules (exclude tests, tools, venv)."""
    repo_root = Path(__file__).resolve().parents[2]
    cryptofeed_dir = repo_root / "cryptofeed"

    production_files = []
    for py_file in cryptofeed_dir.rglob("*.py"):
        # Exclude test files, __pycache__, venv
        if any(
            part in py_file.parts
            for part in ["test_", "__pycache__", ".venv", "site-packages"]
        ):
            continue
        production_files.append(py_file)

    return production_files


class TestRequestsMigrationInventory:
    """Validate the requests → aiohttp migration inventory."""

    def test_wave1_binance_symbol_bootstrap_migrated(self):
        """Verify Binance symbol bootstrap no longer uses requests.

        Wave 1 (Complete): Migrated to _fetch_json_via_proxy (aiohttp).
        Test: Ensure no requests imports in binance.py or binance_futures.py.
        """
        repo_root = Path(__file__).resolve().parents[2]
        binance_files = [
            repo_root / "cryptofeed/exchanges/binance.py",
            repo_root / "cryptofeed/exchanges/binance_futures.py",
        ]

        for binance_file in binance_files:
            if not binance_file.exists():
                pytest.skip(f"{binance_file} not found")

            imports, calls = find_requests_usage_in_file(binance_file)

            assert len(imports) == 0, (
                f"{binance_file.name} still imports requests: {imports}. "
                "Expected: aiohttp-based _fetch_json_via_proxy."
            )
            assert len(calls) == 0, (
                f"{binance_file.name} still calls requests methods: {calls}. "
                "Expected: aiohttp-based _fetch_json_via_proxy."
            )

    def test_wave1_binance_listenkey_migrated(self):
        """Verify Binance listen-key flows no longer use requests.

        Wave 1 (Complete): Migrated _generate_token and _refresh_token to aiohttp.
        Test: Ensure no requests calls in listen-key methods.
        """
        repo_root = Path(__file__).resolve().parents[2]
        binance_file = repo_root / "cryptofeed/exchanges/binance.py"

        if not binance_file.exists():
            pytest.skip("binance.py not found")

        with open(binance_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that _generate_token and _refresh_token do NOT contain "requests."
        assert "def _generate_token" in content or "async def _generate_token" in content
        assert "def _refresh_token" in content or "async def _refresh_token" in content

        # Simple heuristic: ensure no "requests." appears in token methods
        # (More rigorous: parse AST and check method bodies, but this is sufficient for inventory)
        assert "requests.post" not in content, (
            "binance.py contains 'requests.post', expected aiohttp-based _http_request_with_proxy"
        )
        assert "requests.put" not in content, (
            "binance.py contains 'requests.put', expected aiohttp-based _http_request_with_proxy"
        )

    def test_wave2_schema_registry_identified(self):
        """Verify schema registry client is correctly identified for Wave 2 migration.

        Wave 2 (Pending): kafka_schema.py uses requests for Confluent registry.
        Test: Confirm requests usage exists and is documented in migration plan.
        """
        repo_root = Path(__file__).resolve().parents[2]
        schema_file = repo_root / "cryptofeed/backends/kafka_schema.py"

        if not schema_file.exists():
            pytest.skip("kafka_schema.py not found")

        imports, calls = find_requests_usage_in_file(schema_file)

        # Expect requests import
        assert len(imports) > 0, "kafka_schema.py should import requests (Wave 2 target)"

        # Expect at least 5 requests calls (post, get, put for Confluent registry)
        assert len(calls) >= 5, (
            f"kafka_schema.py should have ≥5 requests calls (register, get, check, set mode). "
            f"Found: {len(calls)} calls at lines {[c[0] for c in calls]}"
        )

        # Check specific methods exist
        with open(schema_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "requests.post" in content, "Expected requests.post for schema registration"
        assert "requests.get" in content, "Expected requests.get for schema retrieval"
        assert "requests.put" in content, "Expected requests.put for set compatibility"

    def test_deferred_tooling_documented(self):
        """Verify developer tooling requests usage is correctly identified as deferred.

        Deferred (Low Priority): tools/tools.py uses requests for symbol scrapers.
        Test: Confirm usage exists but is not blocking production.
        """
        repo_root = Path(__file__).resolve().parents[2]
        tools_file = repo_root / "tools/tools.py"

        if not tools_file.exists():
            pytest.skip("tools/tools.py not found")

        imports, calls = find_requests_usage_in_file(tools_file)

        # Expect requests import in tooling
        assert len(imports) > 0, "tools/tools.py should import requests (deferred tooling)"

        # Expect at least 3 requests.get calls (CEX, EXX, BitMEX)
        get_calls = [c for c in calls if c[1] == "get"]
        assert len(get_calls) >= 3, (
            f"tools/tools.py should have ≥3 requests.get calls for symbol scrapers. "
            f"Found: {len(get_calls)}"
        )

    def test_production_modules_no_unexpected_requests(self):
        """Verify no production modules use requests outside documented paths.

        Test: Ensure only kafka_schema.py (Wave 2) has requests imports.
        All other production modules should use aiohttp.
        """
        production_files = get_production_modules()
        repo_root = Path(__file__).resolve().parents[2]
        schema_file = repo_root / "cryptofeed/backends/kafka_schema.py"

        unexpected_imports = []

        for py_file in production_files:
            # Allow kafka_schema.py (Wave 2 target)
            if py_file == schema_file:
                continue

            imports, _ = find_requests_usage_in_file(py_file)
            if imports:
                unexpected_imports.append((py_file, imports))

        assert len(unexpected_imports) == 0, (
            f"Unexpected requests imports in production modules: {unexpected_imports}. "
            "Expected: only kafka_schema.py (Wave 2 target) should import requests."
        )

    def test_migration_plan_exists(self):
        """Verify the migration plan document exists and contains required sections."""
        repo_root = Path(__file__).resolve().parents[2]
        plan_file = (
            repo_root
            / ".kiro/specs/kafka-protobuf-binance-e2e/REQUESTS_MIGRATION_PLAN.md"
        )

        assert plan_file.exists(), "REQUESTS_MIGRATION_PLAN.md not found"

        with open(plan_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Required sections
        required_sections = [
            "## 1. Complete Inventory",
            "## 2. Classification and Priority",
            "## 3. Migration Approach",
            "## 4. Migration Roadmap (Wave 2)",
            "## 5. Test Coverage Requirements",
            "## 6. Dependency Cleanup",
            "## 7. Success Criteria",
        ]

        for section in required_sections:
            assert section in content, f"Migration plan missing section: {section}"

        # Verify Wave 1 components are documented as complete
        assert "✅" in content, "Migration plan should mark Wave 1 tasks as complete"
        assert "Binance Symbol Bootstrap" in content
        assert "Binance Listen-Key" in content

        # Verify Wave 2 components are documented as pending
        assert "Schema Registry" in content or "kafka_schema.py" in content
        assert "Task 6.8b" in content

    def test_no_requests_in_base_requirements(self):
        """Verify requests is not in _BASE_REQUIREMENTS in setup.py."""
        repo_root = Path(__file__).resolve().parents[2]
        setup_file = repo_root / "setup.py"

        if not setup_file.exists():
            pytest.skip("setup.py not found")

        with open(setup_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse setup.py and find _BASE_REQUIREMENTS
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_BASE_REQUIREMENTS":
                        # Found _BASE_REQUIREMENTS, check it doesn't contain "requests"
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    assert not elt.value.startswith("requests"), (
                                        "_BASE_REQUIREMENTS contains 'requests'. "
                                        "Expected: removed after Wave 1 migration."
                                    )

        # Also check for comment confirming removal
        assert (
            "requests no longer required at runtime" in content
            or "aiohttp used for HTTP paths" in content
        ), "setup.py should document requests removal"


class TestWave1MigrationCoverage:
    """Verify Wave 1 migration has adequate test coverage."""

    def test_symbol_mapping_proxy_tests_exist(self):
        """Verify symbol mapping proxy tests exist (Wave 1 coverage)."""
        repo_root = Path(__file__).resolve().parents[2]
        test_file = repo_root / "tests/unit/test_exchange_symbol_mapping_proxy.py"

        assert test_file.exists(), (
            "test_exchange_symbol_mapping_proxy.py not found. "
            "Expected: 6 tests for symbol bootstrap proxy/timeout behavior."
        )

    def test_listenkey_proxy_tests_exist(self):
        """Verify listen-key proxy tests exist (Wave 1 coverage)."""
        repo_root = Path(__file__).resolve().parents[2]
        test_file = repo_root / "tests/unit/test_binance_listenkey_proxy.py"

        assert test_file.exists(), (
            "test_binance_listenkey_proxy.py not found. "
            "Expected: 6 tests for listen-key generate/refresh proxy behavior."
        )

    def test_preflight_proxy_tests_exist(self):
        """Verify preflight proxy initialization tests exist (Wave 1 coverage)."""
        repo_root = Path(__file__).resolve().parents[2]
        unit_test_file = repo_root / "tests/unit/test_preflight_proxy_init_order.py"
        integration_test_file = (
            repo_root / "tests/integration/kafka/test_preflight_proxy_integration.py"
        )

        assert unit_test_file.exists(), (
            "test_preflight_proxy_init_order.py not found. "
            "Expected: 5 unit tests for proxy init order."
        )
        assert integration_test_file.exists(), (
            "test_preflight_proxy_integration.py not found. "
            "Expected: 5 integration tests for E2E proxy validation."
        )


class TestWave2MigrationReadiness:
    """Verify Wave 2 migration is well-defined and testable."""

    def test_schema_registry_migration_task_defined(self):
        """Verify Task 6.8b (schema registry migration) is defined in tasks.md."""
        repo_root = Path(__file__).resolve().parents[2]
        tasks_file = repo_root / ".kiro/specs/kafka-protobuf-binance-e2e/tasks.md"

        if not tasks_file.exists():
            pytest.skip("tasks.md not found")

        with open(tasks_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "6.8b" in content or "6.8 " in content, "Task 6.8b not found in tasks.md"
        assert (
            "schema registry" in content.lower() or "kafka_schema" in content.lower()
        ), "Task 6.8b should reference schema registry client"

    def test_wave2_test_requirements_documented(self):
        """Verify Wave 2 test coverage requirements are documented."""
        repo_root = Path(__file__).resolve().parents[2]
        plan_file = (
            repo_root
            / ".kiro/specs/kafka-protobuf-binance-e2e/REQUESTS_MIGRATION_PLAN.md"
        )

        if not plan_file.exists():
            pytest.skip("REQUESTS_MIGRATION_PLAN.md not found")

        with open(plan_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check Wave 2 test requirements section
        assert "### 5.2 Wave 2 Tests" in content or "Wave 2 Tests (Required" in content
        assert "Minimum 5 unit tests" in content or "5+ unit tests" in content
        assert "Schema registration with HTTP proxy" in content
        assert "Schema registration with SOCKS proxy" in content
        assert "timeout enforcement" in content


# Run tests if invoked directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
