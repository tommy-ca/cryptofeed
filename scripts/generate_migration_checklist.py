#!/usr/bin/env python
"""
Migration Checklist Generator - Task 23

Generates per-exchange migration checklists and documentation templates.

Features:
- Per-exchange checklist generation
- 5-phase migration workflow
- 8 success criteria validation
- Markdown export

Usage:
    python scripts/generate_migration_checklist.py coinbase
    python scripts/generate_migration_checklist.py binance --output checklist.md
    python scripts/generate_migration_checklist.py --help
"""

import argparse
import json
import sys
from typing import Dict, Any


VERSION = "1.0.0"


class MigrationChecklistGenerator:
    """Generates migration checklists and documentation."""

    SUCCESS_CRITERIA = [
        "consumer_lag",
        "error_rate",
        "data_completeness",
        "no_duplicates",
        "latency_p99",
        "downstream_storage",
        "monitoring",
        "no_incidents",
    ]

    def generate_checklist(self, exchange: str) -> Dict[str, Any]:
        """
        Generate migration checklist for exchange.

        Args:
            exchange: Exchange name

        Returns:
            Checklist dictionary
        """
        return {
            "exchange": exchange,
            "pre_migration": {
                "tasks": [
                    "Review baseline metrics",
                    "Verify monitoring dashboard operational",
                    "Notify stakeholders",
                    "Confirm rollback procedure ready",
                    "Confirm QA team available",
                ]
            },
            "consumer_cutover": {
                "tasks": [
                    "Update consumer subscriptions",
                    "Deploy updated consumers",
                    "Verify consumers started successfully",
                    "Validate consumer lag <5s",
                ]
            },
            "validation": {
                "success_criteria": {
                    criterion: {"threshold": self._get_threshold(criterion)}
                    for criterion in self.SUCCESS_CRITERIA
                }
            },
            "monitoring": {
                "tasks": [
                    "Monitor for 1 hour (passive observation)",
                    "Review metrics and identify anomalies",
                    "Document migration results",
                ]
            },
            "post_migration": {
                "tasks": [
                    "Create post-migration report",
                    "Update stakeholders",
                    "Schedule next exchange migration",
                ]
            },
        }

    def _get_threshold(self, criterion: str) -> str:
        """
        Get threshold for success criterion.

        Args:
            criterion: Criterion name

        Returns:
            Threshold description
        """
        thresholds = {
            "consumer_lag": "<5 seconds",
            "error_rate": "<0.1%",
            "data_completeness": "100%",
            "no_duplicates": "0 duplicates",
            "latency_p99": "<5ms",
            "downstream_storage": "100%",
            "monitoring": "Dashboard healthy",
            "no_incidents": "0 incidents",
        }
        return thresholds.get(criterion, "unknown")

    def export_markdown(self, exchange: str, output_path: str) -> None:
        """
        Export checklist as markdown.

        Args:
            exchange: Exchange name
            output_path: Path to output markdown file
        """
        checklist = self.generate_checklist(exchange)

        markdown = f"# Migration Checklist: {exchange.capitalize()}\n\n"

        # Pre-Migration
        markdown += "## Pre-Migration\n\n"
        for task in checklist["pre_migration"]["tasks"]:
            markdown += f"- [ ] {task}\n"
        markdown += "\n"

        # Consumer Cutover
        markdown += "## Consumer Cutover\n\n"
        for task in checklist["consumer_cutover"]["tasks"]:
            markdown += f"- [ ] {task}\n"
        markdown += "\n"

        # Validation
        markdown += "## Validation\n\n"
        markdown += "### Success Criteria\n\n"
        for criterion, config in checklist["validation"]["success_criteria"].items():
            markdown += f"- [ ] {criterion.replace('_', ' ').title()}: {config['threshold']}\n"
        markdown += "\n"

        # Monitoring
        markdown += "## Monitoring\n\n"
        for task in checklist["monitoring"]["tasks"]:
            markdown += f"- [ ] {task}\n"
        markdown += "\n"

        # Post-Migration
        markdown += "## Post-Migration\n\n"
        for task in checklist["post_migration"]["tasks"]:
            markdown += f"- [ ] {task}\n"

        with open(output_path, 'w') as f:
            f.write(markdown)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migration checklist generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "exchange",
        help="Exchange name (e.g., coinbase, binance)",
    )

    parser.add_argument(
        "--output",
        help="Output markdown file path",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    args = parser.parse_args()

    # Generate checklist
    generator = MigrationChecklistGenerator()
    checklist = generator.generate_checklist(args.exchange)

    # Output or export
    if args.output:
        generator.export_markdown(args.exchange, args.output)
        print(json.dumps({"status": "exported", "path": args.output}))
    else:
        print(json.dumps(checklist, indent=2))


if __name__ == "__main__":
    main()
