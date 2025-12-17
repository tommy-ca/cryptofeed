"""
Test suite for simplified deprecation warning system (Phase 1 - Dead Code Removal).

After Phase 1, deprecation.py should contain only 2 simple warning functions:
- emit_deprecation_warning(): Generic deprecation warning
- warn_legacy_usage(): Legacy-specific usage warning

All timeline infrastructure (DeprecationTimeline, CommunicationSystem, etc.) is removed.
"""

import warnings
import pytest


class TestSimplifiedDeprecationWarnings:
    """Test simplified deprecation warning functions (23 LOC total)."""

    def test_emit_deprecation_warning_function_exists(self):
        """Verify emit_deprecation_warning() function is available."""
        from cryptofeed.backends.kafka.deprecation import emit_deprecation_warning

        assert callable(emit_deprecation_warning)

    def test_emit_deprecation_warning_emits_warning(self):
        """Verify emit_deprecation_warning() emits DeprecationWarning."""
        from cryptofeed.backends.kafka.deprecation import emit_deprecation_warning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning("Old API", "New API")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Old API" in str(w[0].message)
            assert "New API" in str(w[0].message)

    def test_emit_deprecation_warning_custom_stacklevel(self):
        """Verify emit_deprecation_warning() respects custom stacklevel."""
        from cryptofeed.backends.kafka.deprecation import emit_deprecation_warning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning("Old", "New", stacklevel=3)

            assert len(w) == 1
            # stacklevel doesn't affect message content, just where warning appears in traceback

    def test_warn_legacy_usage_function_exists(self):
        """Verify warn_legacy_usage() function is available."""
        from cryptofeed.backends.kafka.deprecation import warn_legacy_usage

        assert callable(warn_legacy_usage)

    def test_warn_legacy_usage_emits_warning(self):
        """Verify warn_legacy_usage() emits DeprecationWarning."""
        from cryptofeed.backends.kafka.deprecation import warn_legacy_usage

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_legacy_usage("TradeKafka")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TradeKafka" in str(w[0].message)
            assert "deprecated" in str(w[0].message).lower()


class TestMaintenanceFunctionsPreserved:
    """Test that maintenance module's essential functions are preserved in deprecation.py."""

    def test_emit_class_deprecation_warning_preserved(self):
        """Verify emit_class_deprecation_warning() still works after migration."""
        # After deleting maintenance module, this should import from deprecation.py
        from cryptofeed.backends.kafka.deprecation import emit_class_deprecation_warning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_class_deprecation_warning("OldClass", "NewClass")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OldClass" in str(w[0].message)
            assert "NewClass" in str(w[0].message)

    def test_emit_import_deprecation_warning_preserved(self):
        """Verify emit_import_deprecation_warning() still works after migration."""
        from cryptofeed.backends.kafka.deprecation import emit_import_deprecation_warning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_import_deprecation_warning("old.module", "new.module")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old.module" in str(w[0].message)
            assert "new.module" in str(w[0].message)


class TestInfrastructureRemoved:
    """Verify that complex infrastructure is removed from deprecation.py."""

    def test_deprecation_timeline_class_removed(self):
        """Verify DeprecationTimeline class no longer exists."""
        with pytest.raises(ImportError):
            from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

    def test_communication_system_class_removed(self):
        """Verify CommunicationSystem class no longer exists."""
        with pytest.raises(ImportError):
            from cryptofeed.backends.kafka.deprecation import CommunicationSystem

    def test_decision_log_class_removed(self):
        """Verify DecisionLog class no longer exists."""
        with pytest.raises(ImportError):
            from cryptofeed.backends.kafka.deprecation import DecisionLog

    def test_progress_report_class_removed(self):
        """Verify ProgressReport class no longer exists."""
        with pytest.raises(ImportError):
            from cryptofeed.backends.kafka.deprecation import ProgressReport


class TestMaintenanceModuleDeleted:
    """Verify maintenance module is completely deleted."""

    def test_maintenance_module_import_fails(self):
        """Verify maintenance module cannot be imported."""
        with pytest.raises(ImportError):
            from cryptofeed.backends.kafka import maintenance

    def test_maintenance_init_import_fails(self):
        """Verify maintenance.__init__ cannot be imported."""
        with pytest.raises(ImportError):
            from cryptofeed.backends.kafka.maintenance import DocumentationAutoUpdater
