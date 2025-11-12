#!/usr/bin/env python3
"""Kafka configuration migration tool for Phase 2 upgrade.

This tool helps operators migrate legacy Kafka configurations to the new
Phase 2 KafkaCallback format, with validation and dry-run capabilities.

Usage:
    python tools/migrate-kafka-config.py translate --input legacy.yaml --output phase2.yaml
    python tools/migrate-kafka-config.py translate --input legacy.yaml --dry-run
    python tools/migrate-kafka-config.py validate --config phase2.yaml
    python tools/migrate-kafka-config.py validate --config phase2.yaml --test-kafka
    python tools/migrate-kafka-config.py --help

Commands:
    translate   Translate legacy config to Phase 2 format
    validate    Validate Phase 2 configuration
    help        Show help message

Options:
    --input FILE, -i FILE       Input legacy config YAML (required for translate)
    --output FILE, -o FILE      Output Phase 2 config YAML (optional, for translate)
    --config FILE, -c FILE      Config YAML file (required for validate)
    --dry-run                   Preview changes without saving (for translate)
    --test-kafka                Test Kafka connectivity (for validate)
    --help                      Show this help message
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import cryptofeed
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptofeed.migration.cli import main

if __name__ == "__main__":
    sys.exit(main())
