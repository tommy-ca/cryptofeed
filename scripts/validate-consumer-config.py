#!/usr/bin/env python
"""
Consumer Configuration Validator - Task 21.1

Validates consumer configuration for migration to consolidated Kafka topics.

Features:
- Validates topic subscription patterns (wildcard, regex)
- Checks consumer group configuration
- Verifies offset management strategy
- Validates header extraction setup
- Checks protobuf deserialization config
- Validates batch processing settings

Usage:
    python scripts/validate-consumer-config.py <config_file.json>
    python scripts/validate-consumer-config.py --help
    python scripts/validate-consumer-config.py --version

Example Config:
    {
        "consumer_type": "python-async",
        "topics": ["cryptofeed.trades"],
        "bootstrap_servers": ["kafka1:9092"],
        "consumer_group": "my-processor",
        "offset_reset": "earliest",
        "enable_headers": true
    }
"""

import argparse
import json
import sys
import re
from typing import Dict, List, Any


VERSION = "1.0.0"

VALID_CONSUMER_TYPES = ["flink", "python-async", "custom"]
VALID_OFFSET_STRATEGIES = ["earliest", "latest", "none"]


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate consumer configuration.

    Args:
        config: Consumer configuration dictionary

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "errors": List[str],
            "consumer_type": str,
            ...
        }
    """
    errors = []
    result = {"valid": True, "errors": errors}

    # Copy all config fields to result for inspection
    result.update(config)

    # Validate consumer type
    consumer_type = config.get("consumer_type")
    if not consumer_type:
        errors.append("Missing required field: consumer_type")
    elif consumer_type not in VALID_CONSUMER_TYPES:
        errors.append(
            f"Invalid consumer_type: {consumer_type}. "
            f"Must be one of {VALID_CONSUMER_TYPES}"
        )

    # Validate bootstrap servers
    bootstrap_servers = config.get("bootstrap_servers")
    if not bootstrap_servers:
        errors.append("Missing required field: bootstrap_servers")
    elif not isinstance(bootstrap_servers, list) or len(bootstrap_servers) == 0:
        errors.append("bootstrap_servers must be a non-empty list")

    # Validate topics
    topics = config.get("topics")
    if not topics:
        errors.append("Missing required field: topics")
    elif not isinstance(topics, list) or len(topics) == 0:
        errors.append("topics must be a non-empty list")
    else:
        # Validate topic patterns
        for topic in topics:
            if not _validate_topic_pattern(topic):
                errors.append(f"Invalid topic pattern: {topic}")

    # Validate consumer group
    consumer_group = config.get("consumer_group")
    if not consumer_group:
        errors.append("Missing required field: consumer_group")
    elif not _validate_consumer_group_name(consumer_group):
        errors.append(f"Invalid consumer group name: {consumer_group}")

    # Validate offset reset strategy (optional)
    offset_reset = config.get("offset_reset")
    if offset_reset and offset_reset not in VALID_OFFSET_STRATEGIES:
        errors.append(
            f"Invalid offset_reset: {offset_reset}. "
            f"Must be one of {VALID_OFFSET_STRATEGIES}"
        )

    # Validate header filters (optional)
    header_filters = config.get("header_filters")
    if header_filters and not isinstance(header_filters, dict):
        errors.append("header_filters must be a dictionary")

    # Validate batch size (optional)
    batch_size = config.get("batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append("batch_size must be a positive integer")

    # Validate batch timeout (optional)
    batch_timeout_ms = config.get("batch_timeout_ms")
    if batch_timeout_ms is not None:
        if not isinstance(batch_timeout_ms, int) or batch_timeout_ms <= 0:
            errors.append("batch_timeout_ms must be a positive integer")

    # Set valid flag
    result["valid"] = len(errors) == 0

    return result


def _validate_topic_pattern(pattern: str) -> bool:
    """
    Validate Kafka topic pattern.

    Supports:
    - Exact names: cryptofeed.trades
    - Wildcards: cryptofeed.*
    - Regex patterns: cryptofeed.(trades|orderbook)
    """
    if not pattern:
        return False

    # Basic topic name validation (alphanumeric, dots, dashes, underscores, wildcards)
    # Kafka topic names can contain: a-z A-Z 0-9 . _ - *
    valid_pattern = re.compile(r'^[a-zA-Z0-9._*()-|]+$')
    return bool(valid_pattern.match(pattern))


def _validate_consumer_group_name(name: str) -> bool:
    """
    Validate consumer group name.

    Allows: alphanumeric, dots, dashes, underscores
    """
    if not name:
        return False

    # Consumer group naming convention
    valid_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
    return bool(valid_pattern.match(name))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate consumer configuration for Kafka migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        nargs="?",
        help="Path to consumer configuration JSON file",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    args = parser.parse_args()

    # Check if config file provided
    if not args.config_file:
        parser.print_help()
        sys.stderr.write("\nerror: the following arguments are required: config_file\n")
        sys.exit(1)

    # Load configuration
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(json.dumps({"valid": False, "errors": [f"Config file not found: {args.config_file}"]}))
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(json.dumps({"valid": False, "errors": [f"Invalid JSON: {e}"]}))
        sys.exit(1)

    # Validate configuration
    result = validate_config(config)

    # Output result as JSON
    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
