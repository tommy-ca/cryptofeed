#!/usr/bin/env python
"""
Go/No-Go Decision Engine - Task 23

Automated go/no-go decision support based on success criteria metrics.

Decision Criteria:
- Consumer lag <5 seconds
- Error rate <0.1%
- Data completeness 100%
- Latency p99 <5ms
- All other success criteria passed

Usage:
    python scripts/go_nogo_decision.py metrics.json
    python scripts/go_nogo_decision.py --help
"""

import argparse
import json
import sys
from typing import Dict, Any


VERSION = "1.0.0"


class GoNoGoDecisionEngine:
    """Automated go/no-go decision engine."""

    SUCCESS_CRITERIA = {
        "consumer_lag_seconds": {"threshold": 5.0, "operator": "less_than"},
        "error_rate_percent": {"threshold": 0.1, "operator": "less_than"},
        "data_completeness_percent": {"threshold": 100.0, "operator": "equal"},
        "latency_p99_ms": {"threshold": 5.0, "operator": "less_than"},
        "no_duplicates": {"threshold": 0, "operator": "equal"},
        "downstream_storage_percent": {"threshold": 100.0, "operator": "equal"},
    }

    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate metrics against success criteria.

        Args:
            metrics: Metrics dictionary

        Returns:
            Decision result with go/no-go recommendation
        """
        failed_criteria = []
        passed_criteria = []

        for criterion, config in self.SUCCESS_CRITERIA.items():
            if criterion not in metrics:
                continue

            value = metrics[criterion]
            threshold = config["threshold"]
            operator = config["operator"]

            passed = self._check_criterion(value, threshold, operator)

            if passed:
                passed_criteria.append(criterion)
            else:
                failed_criteria.append(criterion)

        all_criteria_passed = len(failed_criteria) == 0

        decision = {
            "go_nogo": "GO" if all_criteria_passed else "NO-GO",
            "all_criteria_passed": all_criteria_passed,
            "passed_criteria": passed_criteria,
            "failed_criteria": failed_criteria,
            "metrics": metrics,
        }

        # Generate recommendation
        if all_criteria_passed:
            decision["recommendation"] = "✅ All success criteria passed. Proceed to next exchange migration."
        else:
            decision["recommendation"] = f"❌ {len(failed_criteria)} criteria failed. Execute rollback procedure."

        return decision

    def _check_criterion(self, value: float, threshold: float, operator: str) -> bool:
        """
        Check if criterion is met.

        Args:
            value: Measured value
            threshold: Threshold value
            operator: Comparison operator (less_than, equal, greater_than)

        Returns:
            True if criterion passed
        """
        if operator == "less_than":
            return value < threshold
        elif operator == "equal":
            return value == threshold
        elif operator == "greater_than":
            return value > threshold
        else:
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Go/no-go decision engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "metrics_file",
        nargs="?",
        help="Path to metrics JSON file",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    args = parser.parse_args()

    if not args.metrics_file:
        parser.print_help()
        sys.stderr.write("\nerror: the following arguments are required: metrics_file\n")
        sys.exit(1)

    # Load metrics
    try:
        with open(args.metrics_file, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(json.dumps({"status": "failed", "error": f"Metrics file not found: {args.metrics_file}"}))
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "failed", "error": f"Invalid JSON: {e}"}))
        sys.exit(1)

    # Evaluate decision
    engine = GoNoGoDecisionEngine()
    decision = engine.evaluate(metrics)

    # Output decision as JSON
    print(json.dumps(decision, indent=2))

    # Exit with appropriate code
    if decision["go_nogo"] == "GO":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
