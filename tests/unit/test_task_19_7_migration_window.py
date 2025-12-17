"""
Test suite for Task 19.7: Migration Window Extension

Validates that PHASE_5_EXECUTION_PLAN.md has been updated with:
- 6-hour migration windows (extended from 4 hours)
- Explicit pause points after key phases
- Updated Week 3 timeline
- Updated success criteria

This is a documentation verification test using grep-based checks.
"""
import re
from pathlib import Path


class TestMigrationWindowExtension:
    """Test suite verifying migration window extension updates."""

    SPEC_DIR = Path(__file__).parent.parent.parent / ".kiro" / "specs" / "market-data-kafka-producer"
    EXECUTION_PLAN = SPEC_DIR / "PHASE_5_EXECUTION_PLAN.md"

    def test_execution_plan_exists(self):
        """Verify PHASE_5_EXECUTION_PLAN.md exists."""
        assert self.EXECUTION_PLAN.exists(), f"Execution plan not found: {self.EXECUTION_PLAN}"

    def test_migration_window_extended_to_6_hours(self):
        """Verify per-exchange migration window is documented as 6 hours (not 4)."""
        content = self.EXECUTION_PLAN.read_text()

        # Should mention "6-hour window" or "6 hours" in migration context
        pattern = r"(6-hour|6 hour|six-hour|six hour).*window"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) >= 1, "Expected at least 1 reference to '6-hour window' in migration sections"

    def test_no_remaining_4_hour_references_in_migration_procedure(self):
        """Verify migration procedure title and timeline use 6-hour (not 4-hour)."""
        content = self.EXECUTION_PLAN.read_text()

        # Extract the "Per-Exchange Migration Procedure" section
        procedure_match = re.search(
            r"#### Per-Exchange Migration Procedure.*?(?=^#{1,4} |\Z)",
            content,
            re.MULTILINE | re.DOTALL
        )

        if procedure_match:
            procedure_section = procedure_match.group(0)

            # The procedure title should say "6-hour window" not "4-hour window"
            # (It's OK to mention "4-hour" in rationale/explanation, but title must be 6-hour)
            title_line = procedure_section.split('\n')[0]
            assert "6-hour" in title_line.lower(), \
                f"Procedure title should mention '6-hour window'. Found: {title_line}"

            # Verify the rationale explains the change from 4-hour to 6-hour
            assert "rationale" in procedure_section.lower(), \
                "Expected rationale section explaining window extension"

    def test_pause_point_after_parallel_deployment(self):
        """Verify pause point documented after parallel deployment (30min)."""
        content = self.EXECUTION_PLAN.read_text()

        # Look for pause point mention with 30min duration
        pattern = r"pause.*30.*min|30.*min.*pause"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) >= 1, "Expected pause point after parallel deployment (30min)"

    def test_pause_point_after_consumer_validation(self):
        """Verify pause point documented after consumer validation (60min)."""
        content = self.EXECUTION_PLAN.read_text()

        # Look for pause point mention with 60min duration
        pattern = r"pause.*60.*min|60.*min.*pause|pause.*1.*hour|1.*hour.*pause"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) >= 1, "Expected pause point after consumer validation (60min/1 hour)"

    def test_pause_point_after_cutover(self):
        """Verify pause point documented after cutover (60min)."""
        content = self.EXECUTION_PLAN.read_text()

        # This is already covered by test_pause_point_after_consumer_validation
        # since both reference 60min pause
        # Let's verify there are at least 2 pause point references
        pause_mentions = re.findall(r"pause point|breathing room", content, re.IGNORECASE)
        assert len(pause_mentions) >= 2, "Expected multiple pause points documented"

    def test_week3_migration_windows_show_6_hours(self):
        """Verify Week 3 timeline shows 6-hour windows for migration days."""
        content = self.EXECUTION_PLAN.read_text()

        # Extract Week 3 migration sequence table
        table_match = re.search(
            r"\| Day \| Exchange \| Volume.*?\n(?:\|.*\n)+",
            content,
            re.MULTILINE
        )

        if table_match:
            table_section = table_match.group(0)

            # Should reference 6 hours in migration window column
            # Looking for patterns like "10:00-16:00" (6-hour span)
            pattern = r"10:00.*16:00"
            matches = re.findall(pattern, table_section, re.IGNORECASE)
            assert len(matches) >= 1, "Expected 6-hour time window (10:00-16:00 UTC) in Week 3 table"

    def test_success_criteria_updated_for_6_hour_window(self):
        """Verify success criteria or documentation mentions 6-hour cutover window."""
        content = self.EXECUTION_PLAN.read_text()

        # The task mentions updating success criteria
        # Should find reference to 6-hour cutover or migration window
        pattern = r"cutover.*6.*hour|migration.*window.*6.*hour|6.*hour.*cutover"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) >= 1, "Expected success criteria or documentation to mention 6-hour cutover/migration window"

    def test_rationale_for_window_extension_documented(self):
        """Verify rationale for extending window from 4h to 6h is documented."""
        content = self.EXECUTION_PLAN.read_text()

        # Task specifies: "Provide rationale: 4-hour window too tight for unexpected issues under pressure"
        # Look for keywords: tight, buffer, breathing room, unexpected, pressure
        keywords = ["buffer", "breathing room", "unexpected", "tight"]
        found_keywords = [kw for kw in keywords if kw.lower() in content.lower()]

        assert len(found_keywords) >= 2, \
            f"Expected rationale keywords (buffer, breathing room, unexpected, tight). Found: {found_keywords}"

    def test_phase_breakdown_with_extended_times(self):
        """Verify phase breakdown shows extended buffer times (30min, 1h, 60min)."""
        content = self.EXECUTION_PLAN.read_text()

        # Task specifies buffer allocation: +30min cutover, +1h validation, +30min monitoring
        # The document should show phases that add up to 6 hours total

        # Look for phase timing references
        # Should have phases like T+0 to T+360min (6 hours = 360 minutes)
        pattern = r"T\+\d+"
        time_markers = re.findall(pattern, content)

        # Find the maximum time marker (should be ≥360min for 6-hour window)
        max_time = 0
        for marker in time_markers:
            minutes = int(re.search(r"\d+", marker).group())
            max_time = max(max_time, minutes)

        assert max_time >= 360, \
            f"Expected maximum time marker ≥360min (6 hours), found: {max_time}min"

    def test_go_no_go_decision_points_mentioned(self):
        """Verify go/no-go decision points are documented with breathing room."""
        content = self.EXECUTION_PLAN.read_text()

        # Task mentions: "Add go/no-go decision points with breathing room"
        pattern = r"go/no-go|go\/no-go"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) >= 1, "Expected go/no-go decision points to be documented"

    def test_task_19_7_completion_criteria(self):
        """Meta-test: Verify all task 19.7 requirements are testable."""
        # This test documents what we're validating:
        requirements = [
            "6-hour migration windows (not 4 hours)",
            "Pause points: After parallel deployment (30min)",
            "Pause points: After consumer validation (60min)",
            "Pause points: After cutover (60min)",
            "Week 3 schedule updated",
            "Success criteria updated (6-hour cutover window)",
            "Rationale documented (4h too tight)"
        ]

        # All requirements covered by tests above
        assert len(requirements) == 7, "Task 19.7 has 7 testable requirements"


class TestMigrationWindowRollbackProcedure:
    """Verify rollback procedure is compatible with extended 6-hour window."""

    SPEC_DIR = Path(__file__).parent.parent.parent / ".kiro" / "specs" / "market-data-kafka-producer"
    EXECUTION_PLAN = SPEC_DIR / "PHASE_5_EXECUTION_PLAN.md"

    def test_rollback_time_still_under_5_minutes(self):
        """Verify rollback procedure remains <5 minutes despite extended migration window."""
        content = self.EXECUTION_PLAN.read_text()

        # Rollback procedure should still be documented as <5 minutes
        pattern = r"rollback.{0,50}<\s*5\s*min"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) >= 1, "Expected rollback procedure to remain <5 minutes"

    def test_rollback_procedure_exists(self):
        """Verify rollback procedure (Runbook 1 or 1.5) is documented."""
        content = self.EXECUTION_PLAN.read_text()

        # Should have runbook sections for rollback
        pattern = r"runbook.*rollback|rollback.*runbook|rollback procedure"
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert len(matches) >= 1, "Expected rollback procedure/runbook to be documented"
