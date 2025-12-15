"""
CLI utility to migrate legacy Kafka configuration files to modern KafkaConfig format.

Usage:
    python -m cryptofeed.tools.kafka_config_migrate --input legacy.yaml --output modern.yaml

Options:
    --input PATH     Legacy YAML config (required)
    --output PATH    Output path for translated config (default: stdout)
    --backup         Create .bak backup of input file (default: true when output overwrites input)
    --dry-run        Show translated config without writing files
    --pretty         Pretty-print YAML with indentation
"""

from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

from tools.migrate_kafka_config import (
    detect_legacy_config,
    translate_legacy_config,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping/object")
    return data


def dump_yaml(data: Dict[str, Any], pretty: bool) -> str:
    return yaml.safe_dump(data, sort_keys=False, indent=2 if pretty else 0)


def migrate_file(
    input_path: Path,
    output_path: Path | None,
    backup: bool,
    dry_run: bool,
    pretty: bool,
) -> int:
    legacy_config = load_yaml(input_path)
    if not detect_legacy_config(legacy_config):
        print("Input already appears to be modern KafkaConfig; no migration done.", file=sys.stderr)
        return 1

    result = translate_legacy_config(legacy_config)
    modern_dict = result.modern_config.model_dump()

    if backup and output_path and output_path.resolve() == input_path.resolve():
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copyfile(input_path, backup_path)
        print(f"Backup created: {backup_path}", file=sys.stderr)

    rendered = dump_yaml(modern_dict, pretty=pretty)

    if dry_run or output_path is None:
        print(rendered)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
        print(f"Translated configuration written to {output_path}", file=sys.stderr)

    if result.unmapped_options:
        print(
            f"Warning: Unmapped legacy options: {', '.join(sorted(result.unmapped_options))}",
            file=sys.stderr,
        )

    for warning in result.warnings:
        print(f"Note: {warning}", file=sys.stderr)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy Kafka configs to modern format.")
    parser.add_argument("--input", "-i", required=True, help="Legacy YAML config path")
    parser.add_argument("--output", "-o", help="Output path (default: stdout)")
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create .bak when output overwrites input (default: true)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print translated config to stdout without writing files",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print YAML with indentation",
    )

    args = parser.parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    return migrate_file(
        input_path=input_path,
        output_path=output_path,
        backup=args.backup,
        dry_run=args.dry_run,
        pretty=args.pretty,
    )


if __name__ == "__main__":
    sys.exit(main())
