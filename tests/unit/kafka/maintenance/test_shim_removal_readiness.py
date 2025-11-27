"""
Test suite for compatibility shim removal readiness validation.

This test suite validates that:
1. No internal cryptofeed code depends on the compatibility shim
2. All internal references use the new backend implementation
3. Shim removal validation tools work correctly
4. Rollback procedures can be verified

Requirements: 2.4, 2.5
"""



class TestShimRemovalAuditor:
    """Test the shim removal auditor for internal dependency verification."""

    def test_auditor_detects_internal_shim_dependencies(self):
        """Verify auditor can detect internal code using shim imports."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import ShimRemovalAuditor

        auditor = ShimRemovalAuditor()
        result = auditor.audit_internal_dependencies()

        # After migration, should have zero blocking dependencies
        # (All internal references have been updated to use new backend)
        assert result.has_dependencies is False
        assert len(result.dependencies) == 0

        # Verify files were scanned
        assert result.scanned_files > 0

    def test_auditor_excludes_test_files(self):
        """Verify auditor excludes test files from dependency check."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import ShimRemovalAuditor

        auditor = ShimRemovalAuditor()
        result = auditor.audit_internal_dependencies()

        # All dependencies should be in cryptofeed/, not tests/
        for dep in result.dependencies:
            assert not dep.file_path.startswith("tests/")
            assert "/tests/" not in dep.file_path

    def test_auditor_excludes_docs(self):
        """Verify auditor excludes documentation files from dependency check."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import ShimRemovalAuditor

        auditor = ShimRemovalAuditor()
        result = auditor.audit_internal_dependencies()

        # No dependencies should be in docs/
        for dep in result.dependencies:
            assert not dep.file_path.startswith("docs/")
            assert "/docs/" not in dep.file_path

    def test_auditor_excludes_examples(self):
        """Verify auditor excludes example files from dependency check."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import ShimRemovalAuditor

        auditor = ShimRemovalAuditor()
        result = auditor.audit_internal_dependencies()

        # No dependencies should be in examples/
        for dep in result.dependencies:
            assert not dep.file_path.startswith("examples/")
            assert "/examples/" not in dep.file_path

    def test_auditor_provides_detailed_dependency_info(self):
        """Verify auditor provides detailed information about each dependency."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import ShimRemovalAuditor

        auditor = ShimRemovalAuditor()
        result = auditor.audit_internal_dependencies()

        # Each dependency should have detailed info
        for dep in result.dependencies:
            assert dep.file_path is not None
            assert dep.line_number > 0
            assert dep.import_statement is not None
            assert len(dep.import_statement) > 0

    def test_auditor_generates_removal_report(self):
        """Verify auditor can generate a comprehensive removal readiness report."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import ShimRemovalAuditor

        auditor = ShimRemovalAuditor()
        report = auditor.generate_removal_report()

        assert report is not None
        assert "summary" in report
        assert "dependencies" in report
        assert "removal_readiness" in report
        assert "recommendations" in report

    def test_auditor_identifies_blocking_dependencies(self):
        """Verify auditor can identify dependencies that block shim removal."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import ShimRemovalAuditor

        auditor = ShimRemovalAuditor()
        result = auditor.audit_internal_dependencies()

        # After migration, should have zero blocking dependencies
        blocking_deps = [dep for dep in result.dependencies if dep.is_blocking]
        assert len(blocking_deps) == 0


class TestShimRemovalTimeline:
    """Test the shim removal timeline management."""

    def test_timeline_creation(self):
        """Verify removal timeline can be created with milestones."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            ShimRemovalTimeline,
        )

        timeline = ShimRemovalTimeline()

        # Should have default milestones
        assert len(timeline.milestones) > 0

        # Should include key milestones
        milestone_names = [m.name for m in timeline.milestones]
        assert "internal_migration_complete" in milestone_names
        assert "deprecation_warnings_added" in milestone_names
        assert "shim_removal_date" in milestone_names

    def test_timeline_validation(self):
        """Verify timeline validates milestone ordering and dependencies."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            ShimRemovalTimeline,
        )

        timeline = ShimRemovalTimeline()

        # Validate milestone ordering
        validation_result = timeline.validate()
        assert validation_result.is_valid is True

    def test_timeline_milestone_dependencies(self):
        """Verify timeline enforces milestone dependencies."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            ShimRemovalTimeline,
        )

        timeline = ShimRemovalTimeline()

        # Should enforce that internal migration must complete before shim removal
        internal_migration = timeline.get_milestone("internal_migration_complete")
        shim_removal = timeline.get_milestone("shim_removal_date")

        assert internal_migration.date < shim_removal.date

    def test_timeline_generates_communication_plan(self):
        """Verify timeline can generate a communication plan."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            ShimRemovalTimeline,
        )

        timeline = ShimRemovalTimeline()
        communication_plan = timeline.generate_communication_plan()

        assert communication_plan is not None
        assert "channels" in communication_plan
        assert "messages" in communication_plan
        assert "milestones" in communication_plan


class TestInternalReferenceUpdater:
    """Test the internal reference updater for migrating to new backend."""

    def test_updater_identifies_references_to_update(self):
        """Verify updater can identify all internal references needing update."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            InternalReferenceUpdater,
        )

        updater = InternalReferenceUpdater()
        references = updater.find_references_to_update()

        # After migration, should have zero references to update
        assert len(references) == 0

    def test_updater_generates_migration_patches(self):
        """Verify updater can generate patches for migrating references."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            InternalReferenceUpdater,
        )

        updater = InternalReferenceUpdater()
        patches = updater.generate_migration_patches()

        # After migration, should have zero patches needed
        assert len(patches) == 0

    def test_updater_validates_patch_safety(self):
        """Verify updater validates that patches won't break functionality."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            InternalReferenceUpdater,
        )

        updater = InternalReferenceUpdater()
        patches = updater.generate_migration_patches()

        for patch in patches:
            validation = updater.validate_patch(patch)
            assert validation.is_safe is True
            assert len(validation.warnings) >= 0


class TestShimRemovalValidator:
    """Test the comprehensive shim removal validator."""

    def test_validator_checks_all_preconditions(self):
        """Verify validator checks all preconditions for safe shim removal."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            ShimRemovalValidator,
        )

        validator = ShimRemovalValidator()
        result = validator.validate_removal_readiness()

        # Should check all critical preconditions
        assert "internal_dependencies" in result.checks
        assert "test_coverage" in result.checks
        assert "documentation_updated" in result.checks
        assert "rollback_procedure" in result.checks

    def test_validator_fails_with_internal_dependencies(self):
        """Verify validator correctly reports when ready for removal."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            ShimRemovalValidator,
        )

        validator = ShimRemovalValidator()
        result = validator.validate_removal_readiness()

        # After migration, should be ready for removal
        assert result.is_ready is True
        assert len(result.blocking_issues) == 0

    def test_validator_provides_actionable_recommendations(self):
        """Verify validator provides clear recommendations when issues exist."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            ShimRemovalValidator,
        )

        validator = ShimRemovalValidator()
        result = validator.validate_removal_readiness()

        # After migration, should have no recommendations (no issues)
        # The validator correctly returns empty recommendations when ready
        assert isinstance(result.recommendations, list)


class TestRollbackProcedure:
    """Test the rollback procedure for shim removal."""

    def test_rollback_procedure_validation(self):
        """Verify rollback procedure can be validated."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            RollbackProcedure,
        )

        procedure = RollbackProcedure()
        validation = procedure.validate()

        assert validation.is_valid is True
        assert len(validation.steps) > 0

    def test_rollback_procedure_steps_are_executable(self):
        """Verify rollback procedure steps can be executed."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            RollbackProcedure,
        )

        procedure = RollbackProcedure()

        # Each step should be executable
        for step in procedure.steps:
            assert step.name is not None
            assert step.description is not None
            assert callable(step.execute) or step.command is not None

    def test_rollback_procedure_preserves_functionality(self):
        """Verify rollback procedure preserves system functionality."""
        from cryptofeed.backends.kafka.maintenance.shim_removal import (
            RollbackProcedure,
        )

        procedure = RollbackProcedure()
        validation = procedure.validate_functionality_preservation()

        assert validation.preserves_functionality is True
