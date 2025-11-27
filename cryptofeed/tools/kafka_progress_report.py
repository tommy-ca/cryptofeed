#!/usr/bin/env python
"""
Generate monthly progress report for Kafka backend migration.

Usage:
    python -m cryptofeed.tools.kafka_progress_report [--output PATH]

This tool generates a progress report based on tracked usage statistics,
migration percentage, and timeline recommendations.
"""

import argparse
from datetime import datetime
from pathlib import Path

from cryptofeed.backends.kafka.deprecation import ProgressReport, DeprecationTimeline


def generate_report(output_path: str | None = None) -> str:
    """
    Generate progress report and save to file.

    Args:
        output_path: Optional custom output path. If None, uses default location.

    Returns:
        Path to generated report
    """
    # Create report instance
    # In production, this would load from persisted usage data
    report = ProgressReport()

    # Load timeline for milestone context
    timeline = DeprecationTimeline.load_default()

    # Generate report content
    report_date = datetime.now()
    month_str = report_date.strftime("%Y-%m")

    lines = [
        f"# Kafka Backend Migration Progress - {month_str}",
        "",
        f"**Report Date:** {report_date.date()}",
        f"**Reporting Period:** {report_date.strftime('%B %Y')}",
        "",
        "## Summary",
        "",
    ]

    # Migration statistics
    migration_pct = report.get_migration_percentage()
    legacy_stats = report.get_legacy_usage_stats()
    modern_stats = report.get_modern_usage_stats()

    lines.extend(
        [
            f"**Migration Completion:** {migration_pct:.1f}%",
            f"**Legacy Usage Events:** {legacy_stats['total_legacy_usage']}",
            f"**Modern Usage Events:** {modern_stats['total_modern_usage']}",
            "",
            "### Migration Progress Bar",
            "",
            "```",
            f"[{'█' * int(migration_pct / 5)}{'░' * (20 - int(migration_pct / 5))}] {migration_pct:.1f}%",
            "```",
            "",
        ]
    )

    # Timeline status
    lines.extend(
        [
            "## Timeline Status",
            "",
            timeline.to_markdown(),
            "",
        ]
    )

    # Usage breakdown
    if legacy_stats["total_legacy_usage"] > 0:
        lines.extend(
            [
                "## Legacy Usage Breakdown",
                "",
                "**Classes Still In Use:**",
                "",
            ]
        )
        for class_name in legacy_stats["classes_used"]:
            lines.append(f"- `{class_name}`")
        lines.append("")

    if modern_stats["total_modern_usage"] > 0:
        lines.extend(
            [
                "## Modern Backend Adoption",
                "",
                "**Classes In Use:**",
                "",
            ]
        )
        for class_name in modern_stats["classes_used"]:
            lines.append(f"- `{class_name}`")
        lines.append("")

    # Timeline recommendation
    recommendation = report.get_timeline_recommendation()
    lines.extend(
        [
            "## Timeline Recommendation",
            "",
            f"**Extend Timeline:** {'Yes' if recommendation.should_extend_timeline else 'No'}",
            "",
            f"**Reason:** {recommendation.reason}",
            "",
        ]
    )

    if recommendation.should_extend_timeline:
        lines.append(f"**Recommended Extension:** {recommendation.recommended_extension_days} days")
        lines.append("")

    # Notable events (placeholder for manual additions)
    lines.extend(
        [
            "## Notable Events",
            "",
            "_(No notable events recorded this period)_",
            "",
            "<!-- Add critical issues, milestones, or significant migration events here -->",
            "",
        ]
    )

    # Action items
    lines.extend(
        [
            "## Action Items",
            "",
            "- [ ] Review migration progress with team",
            "- [ ] Update deprecation timeline if recommended",
            "- [ ] Address any blocking issues for legacy users",
            "- [ ] Communicate timeline updates if needed",
            "",
        ]
    )

    # References
    lines.extend(
        [
            "## References",
            "",
            "- Deprecation Timeline: `docs/kafka/deprecation-timeline.md`",
            "- Migration Guide: `docs/kafka/migration-guide-phase2-maintenance.md`",
            "- Decision Log: `docs/kafka/decisions/`",
            "",
            "---",
            "",
            f"_Report generated automatically on {report_date.isoformat()}_",
        ]
    )

    report_content = "\n".join(lines)

    # Determine output path
    if output_path is None:
        output_path = f"docs/kafka/progress-reports/{month_str}-progress.md"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write report
    with open(output_file, "w") as f:
        f.write(report_content)

    return str(output_file)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Kafka backend migration progress report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report with default path
  python -m cryptofeed.tools.kafka_progress_report

  # Generate report to custom location
  python -m cryptofeed.tools.kafka_progress_report --output /tmp/progress.md
        """,
    )

    parser.add_argument("--output", "-o", help="Output file path (default: docs/kafka/progress-reports/YYYY-MM-progress.md)")

    args = parser.parse_args()

    try:
        output_path = generate_report(args.output)
        print(f"✅ Progress report generated: {output_path}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        raise


if __name__ == "__main__":
    main()
