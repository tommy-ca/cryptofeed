"""
Test suite for Task 19.4: Partial Rollback Procedure Documentation

This test suite validates that the partial rollback procedure documentation
exists and contains all required elements.

BLOCKER: CRIT-3 (Multi-agent review finding)
- Current rollback procedure assumes full rollback of all exchanges
- No procedure for rolling back single exchange while keeping others on new topics
- Scenario: Coinbase migrated successfully, Binance fails - rollback only Binance or all?
"""

import pytest
import re
from pathlib import Path


class TestPartialRollbackDocumentation:
    """Verify partial rollback procedure documentation completeness."""

    @pytest.fixture
    def execution_plan_path(self):
        """Path to PHASE_5_EXECUTION_PLAN.md."""
        return Path(".kiro/specs/market-data-kafka-producer/PHASE_5_EXECUTION_PLAN.md")

    @pytest.fixture
    def execution_plan_content(self, execution_plan_path):
        """Read execution plan content."""
        assert execution_plan_path.exists(), "PHASE_5_EXECUTION_PLAN.md not found"
        return execution_plan_path.read_text()

    def test_partial_rollback_section_exists(self, execution_plan_content):
        """Verify 'Partial Rollback Procedure' section exists."""
        # Look for section header (### or #### level)
        pattern = r"#{3,4}\s+.*Partial Rollback"
        assert re.search(pattern, execution_plan_content, re.IGNORECASE), \
            "Missing 'Partial Rollback Procedure' section in execution plan"

    def test_decision_tree_present(self, execution_plan_content):
        """Verify decision tree for partial vs full rollback is present."""
        # Look for decision tree keywords
        decision_keywords = [
            "decision tree",
            "partial.*full",
            "rollback.*decision",
            "when to use"
        ]

        found_decision_logic = any(
            re.search(keyword, execution_plan_content, re.IGNORECASE)
            for keyword in decision_keywords
        )

        assert found_decision_logic, \
            "Missing decision tree for partial vs full rollback"

    def test_per_exchange_rollback_steps(self, execution_plan_content):
        """Verify per-exchange rollback steps are documented."""
        required_steps = [
            "identify.*consumer.*exchange",  # Identify affected consumers
            "revert.*consumer",               # Revert consumer config
            "partition.*strategy",            # Update partition strategy
            "isolated|isolation"              # Validate exchange isolation (flexible match)
        ]

        for step in required_steps:
            assert re.search(step, execution_plan_content, re.IGNORECASE), \
                f"Missing required step in per-exchange rollback: {step}"

    def test_example_scenarios_present(self, execution_plan_content):
        """Verify example scenarios are provided."""
        # Look for specific scenario examples
        scenario_keywords = [
            "coinbase.*binance",     # Example exchanges mentioned
            "day 1.*day 2",          # Timeline scenarios
            "exchange.*fail",        # Failure scenarios
            "keep.*healthy"          # Keep successful migrations
        ]

        found_scenarios = sum(
            1 for keyword in scenario_keywords
            if re.search(keyword, execution_plan_content, re.IGNORECASE)
        )

        assert found_scenarios >= 2, \
            "Missing clear example scenarios (expected at least 2 keywords matched)"

    def test_validation_steps_included(self, execution_plan_content):
        """Verify validation steps for partial rollback are included."""
        validation_keywords = [
            "validate.*rollback",
            "verify.*exchange",
            "check.*lag",
            "confirm.*isolation"
        ]

        found_validation = sum(
            1 for keyword in validation_keywords
            if re.search(keyword, execution_plan_content, re.IGNORECASE)
        )

        assert found_validation >= 2, \
            "Missing validation steps for partial rollback (expected at least 2)"

    def test_consumer_identification_documented(self, execution_plan_content):
        """Verify procedure for identifying affected consumer instances is documented."""
        identification_keywords = [
            "filter.*exchange",
            "consumer.*routing",
            "affected.*instance",
            "exchange.*header"
        ]

        found_identification = any(
            re.search(keyword, execution_plan_content, re.IGNORECASE)
            for keyword in identification_keywords
        )

        assert found_identification, \
            "Missing procedure for identifying affected consumer instances"

    def test_partial_vs_full_rollback_criteria(self, execution_plan_content):
        """Verify criteria for choosing partial vs full rollback are documented."""
        criteria_keywords = [
            "when.*partial",
            "when.*full",
            "criteria.*rollback",
            "scope.*rollback"
        ]

        found_criteria = sum(
            1 for keyword in criteria_keywords
            if re.search(keyword, execution_plan_content, re.IGNORECASE)
        )

        assert found_criteria >= 1, \
            "Missing criteria for partial vs full rollback decision"

    def test_rollback_isolation_procedure(self, execution_plan_content):
        """Verify procedure for isolating rollback to single exchange is documented."""
        # Should document how to rollback one exchange without affecting others
        isolation_keywords = [
            "isolat.*exchange",
            "without affecting",
            "keep.*running",
            "exclude.*exchange"
        ]

        found_isolation = any(
            re.search(keyword, execution_plan_content, re.IGNORECASE)
            for keyword in isolation_keywords
        )

        assert found_isolation, \
            "Missing procedure for isolating rollback to single exchange"

    def test_multiple_example_scenarios(self, execution_plan_content):
        """Verify multiple concrete examples are provided."""
        # Extract section after "Partial Rollback" heading
        partial_section_match = re.search(
            r"#{3,4}\s+.*Partial Rollback.*?(?=#{3,4}|\Z)",
            execution_plan_content,
            re.IGNORECASE | re.DOTALL
        )

        if partial_section_match:
            partial_section = partial_section_match.group(0)

            # Count code blocks or numbered/bulleted lists (likely examples)
            code_blocks = len(re.findall(r"```", partial_section))
            lists = len(re.findall(r"^\s*[-*\d]+\.", partial_section, re.MULTILINE))

            assert code_blocks >= 2 or lists >= 4, \
                f"Expected at least 2 code blocks or 4 list items (examples), found {code_blocks} blocks and {lists} lists"

    def test_decision_flow_structure(self, execution_plan_content):
        """Verify decision flow or flowchart structure is present."""
        # Look for decision flow indicators
        flow_indicators = [
            r"if.*then",
            r"yes.*no",
            r"→",  # Arrow indicating flow
            r"step \d+",
            r"scenario [a-z]"
        ]

        found_flow = sum(
            1 for indicator in flow_indicators
            if re.search(indicator, execution_plan_content, re.IGNORECASE)
        )

        assert found_flow >= 2, \
            "Missing clear decision flow structure (expected at least 2 flow indicators)"


class TestPartialRollbackIntegration:
    """Test integration of partial rollback with existing procedures."""

    @pytest.fixture
    def execution_plan_content(self):
        """Read execution plan content."""
        path = Path(".kiro/specs/market-data-kafka-producer/PHASE_5_EXECUTION_PLAN.md")
        return path.read_text()

    def test_references_full_rollback_procedure(self, execution_plan_content):
        """Verify partial rollback references the full rollback procedure."""
        # Should reference Runbook 1 or the full rollback section
        reference_keywords = [
            r"runbook\s+1",
            "full rollback",
            "see.*rollback.*procedure",
            r"section.*rollback"
        ]

        found_reference = any(
            re.search(keyword, execution_plan_content, re.IGNORECASE)
            for keyword in reference_keywords
        )

        assert found_reference, \
            "Partial rollback should reference full rollback procedure"

    def test_references_per_exchange_migration(self, execution_plan_content):
        """Verify partial rollback references per-exchange migration procedure."""
        # Should reference Week 3 migration or per-exchange section
        reference_keywords = [
            "week 3",
            "per-exchange migration",
            "migration.*procedure",
            "runbook 2"
        ]

        found_reference = any(
            re.search(keyword, execution_plan_content, re.IGNORECASE)
            for keyword in reference_keywords
        )

        assert found_reference, \
            "Partial rollback should reference per-exchange migration procedure"

    def test_consistent_timeline_format(self, execution_plan_content):
        """Verify partial rollback uses consistent timeline format (T+Xmin)."""
        # Look for timeline format like "T+0min", "T+1min", etc.
        partial_section_match = re.search(
            r"#{3,4}\s+.*Partial Rollback.*?(?=#{3,4}|\Z)",
            execution_plan_content,
            re.IGNORECASE | re.DOTALL
        )

        if partial_section_match:
            partial_section = partial_section_match.group(0)
            timeline_markers = re.findall(r"T[+\-]\d+min", partial_section)

            assert len(timeline_markers) >= 3, \
                f"Expected at least 3 timeline markers (T+Xmin format), found {len(timeline_markers)}"

    def test_maintains_5min_target(self, execution_plan_content):
        """Verify partial rollback maintains <5min target from full rollback."""
        # Should mention 5 minute or similar time constraint
        partial_section_match = re.search(
            r"#{3,4}\s+.*Partial Rollback.*?(?=#{3,4}|\Z)",
            execution_plan_content,
            re.IGNORECASE | re.DOTALL
        )

        if partial_section_match:
            partial_section = partial_section_match.group(0)
            assert re.search(r"<?\s*5\s*min", partial_section, re.IGNORECASE), \
                "Partial rollback should maintain <5 minute target"


class TestPartialRollbackCompleteness:
    """Test completeness of partial rollback procedure."""

    @pytest.fixture
    def execution_plan_content(self):
        """Read execution plan content."""
        path = Path(".kiro/specs/market-data-kafka-producer/PHASE_5_EXECUTION_PLAN.md")
        return path.read_text()

    def test_all_required_elements_present(self, execution_plan_content):
        """Verify all required elements from task specification are present."""
        required_elements = {
            "section_header": r"#{3,4}\s+.*Partial Rollback",
            "decision_tree": r"partial.*full|decision",
            "per_exchange_steps": r"identify.*consumer.*exchange",
            "validation_steps": r"validate.*rollback|verify",
            "examples": r"coinbase.*binance|exchange.*fail"
        }

        missing_elements = []
        for element_name, pattern in required_elements.items():
            if not re.search(pattern, execution_plan_content, re.IGNORECASE):
                missing_elements.append(element_name)

        assert not missing_elements, \
            f"Missing required elements in partial rollback: {', '.join(missing_elements)}"

    def test_addresses_crit3_blocker(self, execution_plan_content):
        """Verify procedure addresses CRIT-3 blocker scenario."""
        # CRIT-3: No procedure for rolling back single exchange while keeping others
        # Should mention scenario where one exchange fails but others stay healthy

        crit3_keywords = [
            r"single.*exchange.*fail",
            r"single.*failed.*exchange",
            r"keep.*healthy",
            r"remain.*consolidated"
        ]

        found_crit3_coverage = any(
            re.search(keyword, execution_plan_content, re.IGNORECASE)
            for keyword in crit3_keywords
        )

        assert found_crit3_coverage, \
            "Partial rollback doesn't address CRIT-3 scenario (single exchange failure)"

    def test_minimum_content_length(self, execution_plan_content):
        """Verify partial rollback section has sufficient content."""
        # Extract partial rollback section
        partial_section_match = re.search(
            r"#{3,4}\s+.*Partial Rollback.*?(?=#{3,4}|\Z)",
            execution_plan_content,
            re.IGNORECASE | re.DOTALL
        )

        assert partial_section_match, "Partial rollback section not found"

        partial_section = partial_section_match.group(0)
        word_count = len(partial_section.split())

        assert word_count >= 300, \
            f"Partial rollback section too short: {word_count} words (expected at least 300)"
