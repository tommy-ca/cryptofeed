"""CLI tool for Kafka configuration migration.

This module provides a command-line interface for migrating legacy Kafka
configurations to Phase 2 format, with validation and dry-run capabilities.

Commands:
  translate   Translate legacy config to Phase 2 format
  validate    Validate Phase 2 configuration
  help        Show help message

Usage:
  python tools/migrate-kafka-config.py translate --input legacy.yaml --output phase2.yaml
  python tools/migrate-kafka-config.py translate --input legacy.yaml --dry-run
  python tools/migrate-kafka-config.py validate --config phase2.yaml
  python tools/migrate-kafka-config.py validate --config phase2.yaml --test-kafka
"""

import sys
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import argparse

from cryptofeed.migration.config_translator import ConfigTranslator
from cryptofeed.migration.config_validator import ConfigValidator


@dataclass
class CLIResult:
    """Result of a CLI command execution."""

    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    changes_summary: Optional[Dict[str, Any]] = None

    def format_for_display(self) -> str:
        """Format result for terminal output."""
        if self.success:
            lines = []
            if self.output:
                lines.append(self.output)
            if self.changes_summary:
                lines.append(self._format_summary())
            return "\n".join(lines)
        else:
            return f"Error: {self.error}"

    def _format_summary(self) -> str:
        """Format changes summary."""
        lines = ["Summary of changes:"]
        for key, value in self.changes_summary.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class CommandParser:
    """Parse CLI command-line arguments."""

    def parse(self, args: List[str]) -> Dict[str, Any]:
        """Parse command-line arguments.

        Args:
            args: Command-line arguments (without program name)

        Returns:
            Dictionary with parsed command and arguments

        Raises:
            ValueError: If arguments are invalid
        """
        if not args:
            raise ValueError("No command specified. Use 'help' for usage information.")

        command = args[0]

        if command == "help":
            return {"command": "help"}
        elif command == "--help":
            return {"command": "help"}
        elif command == "translate":
            return self._parse_translate(args[1:])
        elif command == "validate":
            return self._parse_validate(args[1:])
        else:
            raise ValueError(f"Unknown command: {command}")

    def _parse_translate(self, args: List[str]) -> Dict[str, Any]:
        """Parse translate command arguments."""
        parser = argparse.ArgumentParser(prog="migrate-kafka-config translate")
        parser.add_argument("--input", "-i", required=True, help="Input legacy config YAML")
        parser.add_argument("--output", "-o", help="Output Phase 2 config YAML")
        parser.add_argument("--dry-run", action="store_true", help="Preview changes without saving")

        try:
            parsed = parser.parse_args(args)
            result = {
                "command": "translate",
                "input": parsed.input,
                "output": parsed.output,
                "dry_run": parsed.dry_run,
            }
            return result
        except SystemExit as e:
            raise ValueError(f"Invalid arguments: {e}")

    def _parse_validate(self, args: List[str]) -> Dict[str, Any]:
        """Parse validate command arguments."""
        parser = argparse.ArgumentParser(prog="migrate-kafka-config validate")
        parser.add_argument("--config", "-c", required=True, help="Config YAML file to validate")
        parser.add_argument("--test-kafka", action="store_true", help="Test Kafka connectivity")

        try:
            parsed = parser.parse_args(args)
            result = {
                "command": "validate",
                "config": parsed.config,
                "test_kafka": parsed.test_kafka,
            }
            return result
        except SystemExit as e:
            raise ValueError(f"Invalid arguments: {e}")


class MigrationCLI:
    """CLI interface for Kafka configuration migration.

    This class provides the main command implementations for translating
    and validating Kafka configurations.

    Example:
        >>> cli = MigrationCLI()
        >>> result = cli.translate('legacy.yaml', 'phase2.yaml')
        >>> print(result.format_for_display())
    """

    def __init__(self):
        """Initialize CLI."""
        self.translator = ConfigTranslator()
        self.validator = ConfigValidator()

    def translate(
        self,
        input_file: str,
        output_file: str,
    ) -> CLIResult:
        """Translate legacy config to Phase 2 format and save.

        Args:
            input_file: Path to legacy config YAML
            output_file: Path to save Phase 2 config YAML

        Returns:
            CLIResult with success status and output
        """
        try:
            # Translate
            phase2_dict = self.translator.translate_yaml_file(input_file)

            # Save
            self.translator.save_yaml(phase2_dict, output_file)

            # Get diff for summary
            import yaml
            with open(input_file, 'r') as f:
                legacy_dict = yaml.safe_load(f)

            diff_output = self._format_diff(legacy_dict, phase2_dict)

            return CLIResult(
                success=True,
                output=f"Configuration translated successfully!\n"
                        f"Input:  {input_file}\n"
                        f"Output: {output_file}\n\n"
                        f"{diff_output}",
            )
        except FileNotFoundError as e:
            return CLIResult(success=False, error=f"File not found: {e}")
        except Exception as e:
            return CLIResult(success=False, error=f"Translation failed: {e}")

    def translate_dry_run(self, input_file: str) -> CLIResult:
        """Preview translation without saving.

        Args:
            input_file: Path to legacy config YAML

        Returns:
            CLIResult with dry-run preview
        """
        try:
            import yaml
            with open(input_file, 'r') as f:
                legacy_dict = yaml.safe_load(f)

            result = self.translator.dry_run(legacy_dict)

            output_lines = [
                "Dry-run: Configuration Translation Preview",
                "=" * 50,
                result["summary"],
                "\n" + "=" * 50,
                "\nNo changes saved (use --dry-run removed to save)",
            ]

            return CLIResult(
                success=True,
                output="\n".join(output_lines),
            )
        except Exception as e:
            return CLIResult(success=False, error=f"Dry-run failed: {e}")

    def validate(self, config_file: str, test_kafka: bool = False) -> CLIResult:
        """Validate Phase 2 configuration.

        Args:
            config_file: Path to Phase 2 config YAML
            test_kafka: Whether to test Kafka connectivity

        Returns:
            CLIResult with validation results
        """
        try:
            result = self.validator.validate_yaml_file(config_file)

            if result.is_valid:
                output = "Configuration is valid!\n\n"
                output += result.format_report()
                return CLIResult(success=True, output=output)
            else:
                error_report = result.format_report()
                return CLIResult(success=False, error=error_report)
        except Exception as e:
            return CLIResult(success=False, error=f"Validation failed: {e}")

    def _format_diff(
        self,
        original: Dict[str, Any],
        translated: Dict[str, Any],
    ) -> str:
        """Format diff between original and translated config.

        Args:
            original: Original configuration
            translated: Translated configuration

        Returns:
            Formatted diff string
        """
        lines = ["Key differences:"]

        # Check bootstrap_servers
        if original.get("bootstrap_servers") != translated.get("bootstrap_servers"):
            lines.append(f"- bootstrap_servers: {translated.get('bootstrap_servers')}")

        # Check topic prefix
        orig_prefix = original.get("topic_prefix", "cryptofeed")
        trans_prefix = translated.get("topic", {}).get("prefix", "cryptofeed")
        if orig_prefix != trans_prefix:
            lines.append(f"- topic.prefix: {orig_prefix} → {trans_prefix}")

        # Check acks
        orig_acks = original.get("acks", "1")
        trans_acks = translated.get("acks", "all")
        if orig_acks != trans_acks:
            lines.append(f"- acks: {orig_acks} → {trans_acks}")

        # Show new fields
        new_fields = {
            "topic": "Topic configuration (consolidated/per_symbol)",
            "partition": "Partition strategy configuration",
            "idempotence": "Idempotence setting",
        }
        for field, description in new_fields.items():
            if field in translated and field not in original:
                lines.append(f"+ {field}: {description}")

        return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if argv is None:
        argv = sys.argv[1:]

    try:
        parser = CommandParser()
        cmd = parser.parse(argv)

        if cmd["command"] == "help":
            print(__doc__)
            return 0

        cli = MigrationCLI()

        if cmd["command"] == "translate":
            if cmd["dry_run"]:
                result = cli.translate_dry_run(cmd["input"])
            else:
                result = cli.translate(cmd["input"], cmd["output"])

        elif cmd["command"] == "validate":
            result = cli.validate(cmd["config"], test_kafka=cmd.get("test_kafka", False))

        print(result.format_for_display())
        return 0 if result.success else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
