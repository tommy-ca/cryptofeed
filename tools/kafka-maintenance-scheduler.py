#!/usr/bin/env python
"""
Kafka Maintenance Scheduler CLI Tool (Task 6.3).

Command-line interface for managing automated Kafka backend maintenance tasks.
"""

import argparse
import sys
import logging
from pathlib import Path
import json

from cryptofeed.backends.kafka.maintenance.scheduler import MaintenanceScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_once(args):
    """Run scheduler once, executing all due tasks."""
    scheduler = MaintenanceScheduler(args.config)
    results = scheduler.run_once()

    print(f"\nExecuted {len(results)} tasks")
    for i, result in enumerate(results, 1):
        status = "SUCCESS" if result.success else "FAILED"
        print(f"Task {i}: {status}")
        if result.errors:
            print(f"  Errors: {', '.join(result.errors)}")

    return 0 if all(r.success for r in results) else 1


def run_forever(args):
    """Run scheduler continuously."""
    scheduler = MaintenanceScheduler(args.config)
    print(f"Starting scheduler (check interval: {args.interval}s)")
    print("Press Ctrl+C to stop")

    try:
        scheduler.run_forever(check_interval_seconds=args.interval)
    except KeyboardInterrupt:
        print("\nScheduler stopped")

    return 0


def status(args):
    """Show scheduler status."""
    scheduler = MaintenanceScheduler(args.config)
    status_data = scheduler.get_status()

    print("\n=== Maintenance Scheduler Status ===")
    print(f"Total tasks: {status_data['total_tasks']}")
    print(f"Enabled tasks: {status_data['enabled_tasks']}")
    print(f"Completed tasks: {status_data['completed_tasks']}")
    print(f"Failed tasks: {status_data['failed_tasks']}")
    print(f"Pending tasks: {status_data['pending_tasks']}")

    print("\n=== Task Details ===")
    for task in status_data["task_details"]:
        print(f"\nTask ID: {task['task_id']}")
        print(f"  Type: {task['task_type']}")
        print(f"  Status: {task['status']}")
        print(f"  Enabled: {task['enabled']}")
        print(f"  Last run: {task['last_run'] or 'Never'}")
        print(f"  Next run: {task['next_run'] or 'N/A'}")
        if task['failure_count'] > 0:
            print(f"  Failures: {task['failure_count']}")
            print(f"  Last error: {task['last_error']}")

    if args.json:
        json_path = Path(args.json)
        with open(json_path, "w") as f:
            json.dump(status_data, f, indent=2)
        print(f"\nStatus written to {json_path}")

    return 0


def validate_config(args):
    """Validate configuration file."""
    from cryptofeed.backends.kafka.maintenance.scheduler import ScheduleConfig

    try:
        config = ScheduleConfig.from_yaml(args.config)
        errors = config.validate()

        if errors:
            print("Configuration validation FAILED:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print(f"Configuration validation PASSED ({len(config.tasks)} tasks)")
            return 0

    except Exception as e:
        print(f"Configuration loading FAILED: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kafka Maintenance Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once
  %(prog)s run-once --config config/maintenance.yaml

  # Run continuously
  %(prog)s run-forever --config config/maintenance.yaml --interval 60

  # Show status
  %(prog)s status --config config/maintenance.yaml

  # Validate configuration
  %(prog)s validate --config config/maintenance.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # run-once command
    run_once_parser = subparsers.add_parser(
        "run-once", help="Run scheduler once, executing all due tasks"
    )
    run_once_parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration YAML file",
    )

    # run-forever command
    run_forever_parser = subparsers.add_parser(
        "run-forever", help="Run scheduler continuously"
    )
    run_forever_parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration YAML file",
    )
    run_forever_parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status", help="Show scheduler status"
    )
    status_parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration YAML file",
    )
    status_parser.add_argument(
        "--json",
        help="Write status to JSON file",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration file"
    )
    validate_parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    if args.command == "run-once":
        return run_once(args)
    elif args.command == "run-forever":
        return run_forever(args)
    elif args.command == "status":
        return status(args)
    elif args.command == "validate":
        return validate_config(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
