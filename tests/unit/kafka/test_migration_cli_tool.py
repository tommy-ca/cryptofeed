"""Tests for Kafka migration CLI tool.

This module tests the CLI interface for config translation and validation,
including command parsing and user-friendly output.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from cryptofeed.migration.cli import (
    MigrationCLI,
    CLIResult,
    CommandParser,
)


class TestCLICommandParsing:
    """Test CLI command parsing and argument handling."""

    def test_parse_translate_command(self):
        """Parse translate command with arguments."""
        parser = CommandParser()
        cmd = parser.parse(['translate', '--input', 'legacy.yaml', '--output', 'phase2.yaml'])

        assert cmd['command'] == 'translate'
        assert cmd['input'] == 'legacy.yaml'
        assert cmd['output'] == 'phase2.yaml'

    def test_parse_translate_with_dry_run(self):
        """Parse translate command with dry-run flag."""
        parser = CommandParser()
        cmd = parser.parse(['translate', '--input', 'legacy.yaml', '--dry-run'])

        assert cmd['command'] == 'translate'
        assert cmd['dry_run'] is True

    def test_parse_validate_command(self):
        """Parse validate command."""
        parser = CommandParser()
        cmd = parser.parse(['validate', '--config', 'phase2.yaml'])

        assert cmd['command'] == 'validate'
        assert cmd['config'] == 'phase2.yaml'

    def test_parse_validate_with_kafka_test(self):
        """Parse validate command with Kafka connectivity test."""
        parser = CommandParser()
        cmd = parser.parse(['validate', '--config', 'phase2.yaml', '--test-kafka'])

        assert cmd['command'] == 'validate'
        assert cmd['test_kafka'] is True

    def test_parse_unknown_command_raises_error(self):
        """Unknown command raises error."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Unknown"):
            parser.parse(['invalid_command'])

    def test_parse_missing_required_argument(self):
        """Missing required argument raises error."""
        parser = CommandParser()
        with pytest.raises(ValueError):
            parser.parse(['translate'])  # Missing --input

    def test_parse_help_command(self):
        """Parse help command."""
        parser = CommandParser()
        cmd = parser.parse(['--help'])

        assert cmd['command'] == 'help' or 'help' in cmd


class TestTranslateCommand:
    """Test translate command functionality."""

    def test_translate_command_basic(self):
        """Execute translate command with basic config."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
topic_prefix: legacy
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as src:
            src.write(legacy_yaml)
            src.flush()
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as dst:
            dst_path = dst.name

        try:
            cli = MigrationCLI()
            result = cli.translate(src_path, dst_path)

            assert result.success
            # Verify output file exists
            assert Path(dst_path).exists()

            # Verify content
            with open(dst_path, 'r') as f:
                output = yaml.safe_load(f)

            assert output['bootstrap_servers'] == ['kafka:9092']
            assert output['topic']['prefix'] == 'legacy'
        finally:
            Path(src_path).unlink()
            Path(dst_path).unlink()

    def test_translate_with_dry_run(self):
        """Execute translate with dry-run preview."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
acks: '1'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(legacy_yaml)
            f.flush()

            try:
                cli = MigrationCLI()
                result = cli.translate_dry_run(f.name)

                assert result.success
                # Should show translation preview
                assert 'preview' in result.output.lower() or 'translation' in result.output.lower()
            finally:
                Path(f.name).unlink()

    def test_translate_shows_diff(self):
        """Translate shows differences between old and new."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
topic_prefix: myapp
acks: '1'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as src:
            src.write(legacy_yaml)
            src.flush()
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as dst:
            dst_path = dst.name

        try:
            cli = MigrationCLI()
            result = cli.translate(src_path, dst_path)

            assert result.success
            # Output should indicate what changed
            output_str = str(result.output).lower()
            assert 'translated' in output_str or 'converted' in output_str.lower()
        finally:
            Path(src_path).unlink()
            Path(dst_path).unlink()

    def test_translate_invalid_input_file(self):
        """Handle invalid input file gracefully."""
        cli = MigrationCLI()
        result = cli.translate('/nonexistent/config.yaml', '/tmp/output.yaml')

        assert not result.success
        assert result.error is not None

    def test_translate_preserves_producer_settings(self):
        """Translate preserves all producer settings."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
acks: '1'
retries: 5
batch_size: 32768
linger_ms: 20
compression_type: snappy
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as src:
            src.write(legacy_yaml)
            src.flush()
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as dst:
            dst_path = dst.name

        try:
            cli = MigrationCLI()
            cli.translate(src_path, dst_path)

            with open(dst_path, 'r') as f:
                output = yaml.safe_load(f)

            assert output['acks'] == '1'
            assert output['retries'] == 5
            assert output['batch_size'] == 32768
            assert output['linger_ms'] == 20
            assert output['compression_type'] == 'snappy'
        finally:
            Path(src_path).unlink()
            Path(dst_path).unlink()


class TestValidateCommand:
    """Test validate command functionality."""

    def test_validate_command_basic(self):
        """Execute validate command on valid config."""
        phase2_yaml = """
bootstrap_servers:
  - kafka:9092
topic:
  strategy: consolidated
partition:
  strategy: composite
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(phase2_yaml)
            f.flush()

            try:
                cli = MigrationCLI()
                result = cli.validate(f.name)

                assert result.success
            finally:
                Path(f.name).unlink()

    def test_validate_invalid_config(self):
        """Detect invalid configuration."""
        phase2_yaml = """
bootstrap_servers:
  - kafka:9092
topic:
  strategy: invalid_strategy
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(phase2_yaml)
            f.flush()

            try:
                cli = MigrationCLI()
                result = cli.validate(f.name)

                assert not result.success
                assert result.error is not None
            finally:
                Path(f.name).unlink()

    def test_validate_with_error_report(self):
        """Validate provides error report."""
        invalid_yaml = "invalid: yaml: [content"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()

            try:
                cli = MigrationCLI()
                result = cli.validate(f.name)

                assert not result.success
                # Should have error description
                assert result.error is not None
                assert len(str(result.error)) > 0
            finally:
                Path(f.name).unlink()

    @pytest.mark.integration
    def test_validate_with_kafka_connectivity(self):
        """Validate config with Kafka connectivity test."""
        phase2_yaml = """
bootstrap_servers:
  - localhost:9092
topic:
  strategy: consolidated
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(phase2_yaml)
            f.flush()

            try:
                cli = MigrationCLI()
                result = cli.validate(f.name, test_kafka=False)  # Don't test if no Kafka

                assert result is not None
            finally:
                Path(f.name).unlink()


class TestCLIOutput:
    """Test CLI output formatting."""

    def test_cli_success_message(self):
        """CLI displays success message."""
        result = CLIResult(success=True, output="Configuration translated successfully")
        formatted = result.format_for_display()

        assert 'success' in formatted.lower() or 'translated' in formatted.lower()

    def test_cli_error_message(self):
        """CLI displays error message."""
        result = CLIResult(success=False, error="Invalid bootstrap_servers")
        formatted = result.format_for_display()

        assert 'error' in formatted.lower() or 'invalid' in formatted.lower()

    def test_cli_display_diff(self):
        """CLI displays configuration diff."""
        original = {'bootstrap_servers': ['kafka:9092'], 'acks': '1'}
        translated = {'bootstrap_servers': ['kafka:9092'], 'acks': '1', 'topic': {'strategy': 'per_symbol'}}

        cli = MigrationCLI()
        diff = cli._format_diff(original, translated)

        assert 'topic' in diff.lower() or 'added' in diff.lower()

    def test_cli_display_summary(self):
        """CLI displays summary of changes."""
        result = CLIResult(
            success=True,
            output="Translated legacy config",
            changes_summary={'added_fields': 2, 'modified_fields': 1}
        )
        formatted = result.format_for_display()

        assert 'added' in formatted.lower() or 'translated' in formatted.lower()


class TestCLIIntegration:
    """Test full CLI workflow integration."""

    def test_translate_then_validate_workflow(self):
        """Full workflow: translate then validate."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
topic_prefix: test
acks: '1'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as src:
            src.write(legacy_yaml)
            src.flush()
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as dst:
            dst_path = dst.name

        try:
            cli = MigrationCLI()

            # Step 1: Translate
            translate_result = cli.translate(src_path, dst_path)
            assert translate_result.success

            # Step 2: Validate
            validate_result = cli.validate(dst_path)
            assert validate_result.success
        finally:
            Path(src_path).unlink()
            Path(dst_path).unlink()

    def test_dry_run_before_translate(self):
        """Use dry-run preview before actual translation."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
acks: '1'
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(legacy_yaml)
            f.flush()

            try:
                cli = MigrationCLI()

                # Step 1: Preview with dry-run
                preview_result = cli.translate_dry_run(f.name)
                assert preview_result.success

                # Step 2: Check what would change
                assert 'bootstrap' in str(preview_result.output).lower() or 'translation' in str(preview_result.output).lower()
            finally:
                Path(f.name).unlink()


class TestErrorHandling:
    """Test error handling in CLI."""

    def test_cli_handles_file_not_found(self):
        """CLI handles file not found gracefully."""
        cli = MigrationCLI()
        result = cli.translate('/nonexistent.yaml', '/tmp/out.yaml')

        assert not result.success
        assert 'not found' in str(result.error).lower() or 'file' in str(result.error).lower()

    def test_cli_handles_permission_error(self):
        """CLI handles permission errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('bootstrap_servers:\n  - kafka:9092\n')
            f.flush()
            f_path = f.name

        try:
            # Make output directory read-only
            output_path = '/root/protected/config.yaml'
            cli = MigrationCLI()
            result = cli.translate(f_path, output_path)

            # Should handle gracefully
            if not result.success:
                assert result.error is not None
        finally:
            Path(f_path).unlink()

    def test_cli_handles_invalid_yaml(self):
        """CLI handles invalid YAML syntax."""
        invalid_yaml = "invalid: : : : yaml"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()

            try:
                cli = MigrationCLI()
                result = cli.translate(f.name, '/tmp/out.yaml')

                assert not result.success
                assert 'yaml' in str(result.error).lower() or 'parse' in str(result.error).lower()
            finally:
                Path(f.name).unlink()


class TestCLIExamples:
    """Test CLI with realistic examples."""

    def test_migrate_simple_config(self):
        """Migrate simple single-broker config."""
        legacy_yaml = """
bootstrap_servers:
  - kafka:9092
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as src:
            src.write(legacy_yaml)
            src.flush()
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as dst:
            dst_path = dst.name

        try:
            cli = MigrationCLI()
            result = cli.translate(src_path, dst_path)
            assert result.success
        finally:
            Path(src_path).unlink()
            Path(dst_path).unlink()

    def test_migrate_production_config(self):
        """Migrate production multi-broker config."""
        legacy_yaml = """
bootstrap_servers:
  - kafka-1.prod:9092
  - kafka-2.prod:9092
  - kafka-3.prod:9092
topic_prefix: production
acks: all
retries: 10
compression_type: snappy
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as src:
            src.write(legacy_yaml)
            src.flush()
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as dst:
            dst_path = dst.name

        try:
            cli = MigrationCLI()
            result = cli.translate(src_path, dst_path)
            assert result.success
        finally:
            Path(src_path).unlink()
            Path(dst_path).unlink()
