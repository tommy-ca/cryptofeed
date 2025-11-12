"""Tests for Task 19: Producer Tuning Guide Documentation.

This test suite validates the producer tuning guide documentation
by verifying:
1. File exists and has sufficient content
2. All required sections are present
3. Configuration examples are valid
4. Use case profiles cover all scenarios
5. Performance tuning checklist is complete
"""

import pytest
from pathlib import Path


class TestProducerTuningGuideContent:
    """Test that producer tuning guide has all required sections."""

    @pytest.fixture
    def tuning_guide_path(self) -> Path:
        """Return path to tuning guide."""
        return Path(__file__).parent.parent / "docs" / "kafka" / "producer-tuning.md"

    def test_tuning_guide_exists(self, tuning_guide_path):
        """Test that producer tuning guide file exists."""
        assert tuning_guide_path.exists(), f"Tuning guide not found at {tuning_guide_path}"

    def test_tuning_guide_has_minimum_length(self, tuning_guide_path):
        """Test that tuning guide has at least 1000 lines of content."""
        content = tuning_guide_path.read_text()
        lines = content.split('\n')
        assert len(lines) >= 1000, f"Tuning guide should have 1000+ lines, found {len(lines)}"

    def test_tuning_guide_has_executive_summary(self, tuning_guide_path):
        """Test that tuning guide includes executive summary section."""
        content = tuning_guide_path.read_text()
        assert "Executive Summary" in content or "Summary" in content, \
            "Tuning guide missing Executive Summary section"

    def test_tuning_guide_has_configuration_reference(self, tuning_guide_path):
        """Test that tuning guide documents configuration parameters."""
        content = tuning_guide_path.read_text()
        # Check for key configuration parameters
        required_configs = [
            "batch.size",
            "linger.ms",
            "buffer.memory",
            "compression.type",
            "acks",
            "retries",
            "max.in.flight.requests",
            "request.timeout.ms"
        ]
        for config in required_configs:
            assert config in content, f"Tuning guide missing documentation for {config}"

    def test_tuning_guide_explains_batch_size(self, tuning_guide_path):
        """Test batch.size parameter documentation."""
        content = tuning_guide_path.read_text()
        assert "batch.size" in content
        assert "latency" in content.lower() or "throughput" in content.lower(), \
            "batch.size explanation should mention latency-throughput trade-off"

    def test_tuning_guide_explains_linger_ms(self, tuning_guide_path):
        """Test linger.ms parameter documentation."""
        content = tuning_guide_path.read_text()
        assert "linger.ms" in content or "linger" in content
        assert "wait" in content.lower() or "delay" in content.lower(), \
            "linger.ms explanation should mention waiting/delay"

    def test_tuning_guide_explains_compression(self, tuning_guide_path):
        """Test compression.type parameter documentation."""
        content = tuning_guide_path.read_text()
        assert "compression" in content.lower()
        # Should mention specific compression algorithms
        compression_types = ["snappy", "lz4", "gzip", "none"]
        found_types = sum(1 for t in compression_types if t in content.lower())
        assert found_types >= 2, "compression.type explanation should mention multiple algorithms"

    def test_tuning_guide_has_use_case_profiles(self, tuning_guide_path):
        """Test that tuning guide includes all use case profiles."""
        content = tuning_guide_path.read_text()
        # Check for all required use case profiles
        profiles = [
            "Latency-Sensitive",
            "Throughput-Optimized",
            "Balanced",
            "High-Reliability"
        ]
        for profile in profiles:
            assert profile in content, f"Tuning guide missing use case profile: {profile}"

    def test_latency_sensitive_profile_documented(self, tuning_guide_path):
        """Test latency-sensitive profile has proper documentation."""
        content = tuning_guide_path.read_text()
        assert "Latency-Sensitive" in content
        # Should mention minimal batching or immediate flush
        assert ("batch" in content.lower() or "immediate" in content.lower() or
                "flush" in content.lower()), \
            "Latency-sensitive profile should mention batching strategy"

    def test_throughput_optimized_profile_documented(self, tuning_guide_path):
        """Test throughput-optimized profile has proper documentation."""
        content = tuning_guide_path.read_text()
        assert "Throughput-Optimized" in content
        # Should mention large batches or linger time
        assert ("batch" in content.lower() or "linger" in content.lower()), \
            "Throughput-optimized profile should mention batching or linger strategy"

    def test_tuning_guide_has_performance_checklist(self, tuning_guide_path):
        """Test that tuning guide includes performance tuning checklist."""
        content = tuning_guide_path.read_text()
        assert "Checklist" in content or "checklist" in content.lower(), \
            "Tuning guide missing performance tuning checklist"
        # Check for key checklist items
        checklist_items = ["bottleneck", "Adjust", "Measure", "Validate"]
        for item in checklist_items:
            assert item in content, f"Checklist missing item: {item}"

    def test_tuning_guide_has_monitoring_driven_optimization(self, tuning_guide_path):
        """Test tuning guide explains monitoring-driven optimization workflow."""
        content = tuning_guide_path.read_text()
        assert "Monitor" in content or "Prometheus" in content or "metrics" in content, \
            "Tuning guide should mention monitoring-driven optimization"

    def test_tuning_guide_covers_common_scenarios(self, tuning_guide_path):
        """Test that tuning guide covers common tuning scenarios."""
        content = tuning_guide_path.read_text()
        scenarios = [
            "latency",
            "throughput",
            "memory",
            "cpu",
            "reliability"
        ]
        for scenario in scenarios:
            assert scenario.lower() in content.lower(), \
                f"Tuning guide should cover scenario: {scenario}"

    def test_tuning_guide_includes_p99_latency_target(self, tuning_guide_path):
        """Test that tuning guide mentions p99 latency target."""
        content = tuning_guide_path.read_text()
        assert "p99" in content or "P99" in content or "99th percentile" in content.lower(), \
            "Tuning guide should mention p99 latency target"

    def test_tuning_guide_includes_throughput_target(self, tuning_guide_path):
        """Test that tuning guide mentions throughput targets."""
        content = tuning_guide_path.read_text()
        assert "100k" in content or "100,000" in content or "throughput" in content.lower(), \
            "Tuning guide should mention throughput targets"

    def test_tuning_guide_has_configuration_examples(self, tuning_guide_path):
        """Test that tuning guide includes configuration examples."""
        content = tuning_guide_path.read_text()
        # Should have YAML or code examples
        assert ("yaml" in content.lower() or "```" in content or
                "batch_size" in content or "acks" in content), \
            "Tuning guide should include configuration examples"

    def test_tuning_guide_explains_acks_setting(self, tuning_guide_path):
        """Test that acks setting is properly documented."""
        content = tuning_guide_path.read_text()
        assert "acks" in content
        # Should mention different ack values: 0, 1, all
        ack_values = ["acks: 0", "acks: 1", "acks: all", "acks=0", "acks=1", "acks=all"]
        found = any(val in content for val in ack_values)
        assert found, "acks documentation should mention different values (0, 1, all)"


class TestProducerTuningGuideUsability:
    """Test that tuning guide is usable and practical."""

    @pytest.fixture
    def tuning_guide_path(self) -> Path:
        """Return path to tuning guide."""
        return Path(__file__).parent.parent / "docs" / "kafka" / "producer-tuning.md"

    def test_tuning_guide_has_table_of_contents(self, tuning_guide_path):
        """Test that tuning guide has clear structure."""
        content = tuning_guide_path.read_text()
        # Should have clear sections with headers
        assert content.count('#') >= 5, "Tuning guide should have multiple sections"

    def test_tuning_guide_has_examples_for_each_profile(self, tuning_guide_path):
        """Test that each use case profile includes configuration example."""
        content = tuning_guide_path.read_text()
        # Count sections and examples
        profile_count = content.count("profile") + content.count("Profile")
        example_count = content.count("example") + content.count("Example")
        assert example_count >= profile_count * 0.5, \
            "Should have examples for use case profiles"

    def test_tuning_guide_has_step_by_step_procedures(self, tuning_guide_path):
        """Test that tuning guide includes step-by-step procedures."""
        content = tuning_guide_path.read_text()
        # Check for numbered steps or bullet points
        has_steps = ("1." in content or "Step 1" in content or
                     "step 1" in content.lower() or "- " in content)
        assert has_steps, "Tuning guide should include step-by-step procedures"


class TestProducerTuningGuidePractical:
    """Test that tuning guide is practical and actionable."""

    @pytest.fixture
    def tuning_guide_path(self) -> Path:
        """Return path to tuning guide."""
        return Path(__file__).parent.parent / "docs" / "kafka" / "producer-tuning.md"

    def test_tuning_guide_recommends_default_values(self, tuning_guide_path):
        """Test that tuning guide provides default value guidance."""
        content = tuning_guide_path.read_text()
        assert "default" in content.lower(), \
            "Tuning guide should explain default values"

    def test_tuning_guide_explains_impact_of_changes(self, tuning_guide_path):
        """Test that tuning guide explains impact of configuration changes."""
        content = tuning_guide_path.read_text()
        assert ("impact" in content.lower() or "trade-off" in content.lower() or
                "tradeoff" in content.lower()), \
            "Tuning guide should explain configuration impacts"

    def test_tuning_guide_warns_about_common_mistakes(self, tuning_guide_path):
        """Test that tuning guide mentions common tuning mistakes."""
        content = tuning_guide_path.read_text()
        assert ("avoid" in content.lower() or "warning" in content.lower() or
                "common" in content.lower() or "mistake" in content.lower()), \
            "Tuning guide should mention common mistakes or warnings"
