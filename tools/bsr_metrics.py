#!/usr/bin/env python
"""BSR Metrics Collection and Reporting Tool.

Automates collection of Buf Schema Registry metrics for the crypto-market-data module:
- Module downloads (daily)
- Dependent modules (weekly)
- Version distribution (daily)
- Adoption trends (monthly)

Usage:
    python tools/bsr_metrics.py --collect              # Collect metrics now
    python tools/bsr_metrics.py --report json          # Generate JSON report
    python tools/bsr_metrics.py --report markdown      # Generate Markdown report
    python tools/bsr_metrics.py --dashboard            # Generate dashboard HTML
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any


class BSRMetricsCollector:
    """Collects metrics from Buf Schema Registry for crypto-market-data module."""

    # Metric definitions with collection frequency
    METRIC_DEFINITIONS = {
        "module_downloads": {
            "display_name": "Total Downloads",
            "unit": "count",
            "frequency": "daily",
            "description": "Total number of module downloads from BSR",
        },
        "version_adoption": {
            "display_name": "Latest Version Adoption %",
            "unit": "percentage",
            "frequency": "daily",
            "description": "Percentage of consumers using latest version",
        },
        "active_consumers": {
            "display_name": "Active Dependent Modules",
            "unit": "count",
            "frequency": "weekly",
            "description": "Number of modules/projects actively depending on this module",
        },
        "download_trends": {
            "display_name": "Download Trend",
            "unit": "count",
            "frequency": "daily",
            "description": "Daily download count trend",
        },
        "version_distribution": {
            "display_name": "Version Distribution",
            "unit": "percentage",
            "frequency": "daily",
            "description": "Distribution of consumers across versions",
        },
    }

    # Review cadence definitions
    REVIEW_CADENCE = {
        "daily": {
            "metrics": ["module_downloads", "version_adoption", "download_trends"],
            "audience": "DevOps Team",
            "sla": "Review within 4 hours",
            "purpose": "Real-time monitoring",
        },
        "weekly": {
            "metrics": ["active_consumers", "version_distribution"],
            "audience": "Schema Team",
            "sla": "Review within 24 hours",
            "purpose": "Adoption and dependency analysis",
        },
        "monthly": {
            "metrics": ["adoption_trends", "sla_compliance"],
            "audience": "Engineering Leadership",
            "sla": "Review within 5 days",
            "purpose": "Strategic alignment and roadmap",
        },
    }

    # Alerting thresholds
    ALERTING_THRESHOLDS = {
        "low_adoption": {
            "metric": "version_adoption",
            "threshold": 0.60,  # 60% adoption
            "severity": "warning",
            "description": "Latest version adoption below 60%",
        },
        "critical_adoption": {
            "metric": "version_adoption",
            "threshold": 0.40,  # 40% adoption
            "severity": "critical",
            "description": "Latest version adoption below 40%",
        },
        "no_activity": {
            "metric": "module_downloads",
            "threshold": 0,  # 0 downloads in 24h
            "severity": "warning",
            "description": "No downloads in past 24 hours",
        },
        "low_dependent_growth": {
            "metric": "active_consumers",
            "threshold": 5,
            "severity": "info",
            "description": "Fewer than 5 active dependents",
        },
    }

    def __init__(
        self,
        owner: str = "tommyk",
        module: str = "crypto-market-data",
        namespace: str | None = None,
    ):
        """Initialize metrics collector.

        Args:
            owner: BSR owner/organization
            module: Module name
            namespace: Full namespace (optional, constructed from owner/module if not provided)
        """
        self.owner = owner
        self.module = module
        self.namespace = namespace or f"buf.build/{owner}/{module}"
        self.collection_time = datetime.now()
        self.metrics: dict[str, Any] = {}

    def collect_downloads(self, period: str = "30d", time_series: bool = False) -> dict:
        """Collect download metrics from BSR.

        Args:
            period: Time period for metrics (1d, 7d, 30d, 90d)
            time_series: Include time series data

        Returns:
            Dictionary with download statistics
        """
        return {
            "total_downloads": 0,  # Would query BSR API
            "period": period,
            "frequency": self.METRIC_DEFINITIONS["module_downloads"]["frequency"],
            "last_24h": 0,
            "last_7d": 0,
            "last_30d": 0,
            "time_series": [] if time_series else None,
            "collected_at": self.collection_time.isoformat(),
        }

    def collect_dependents(self) -> dict:
        """Collect dependent modules metrics.

        Returns:
            Dictionary with dependent module statistics
        """
        return {
            "total_dependents": 0,  # Would query BSR API
            "by_language": {
                "go": 0,
                "python": 0,
                "typescript": 0,
                "other": 0,
            },
            "frequency": self.METRIC_DEFINITIONS["active_consumers"]["frequency"],
            "collected_at": self.collection_time.isoformat(),
        }

    def collect_versions(self) -> dict:
        """Collect version distribution metrics.

        Returns:
            Dictionary with version statistics
        """
        return {
            "latest_version": "v0.1.0",
            "total_versions": 1,
            "adoption_by_version": {
                "v0.1.0": 1.0,  # 100% adoption
            },
            "frequency": self.METRIC_DEFINITIONS["version_adoption"]["frequency"],
            "collected_at": self.collection_time.isoformat(),
        }

    def collect_all_metrics(self) -> dict:
        """Collect all metrics.

        Returns:
            Dictionary with all collected metrics
        """
        self.metrics = {
            "downloads": self.collect_downloads(time_series=True),
            "dependents": self.collect_dependents(),
            "versions": self.collect_versions(),
        }
        return self.metrics

    def generate_report(self, format: str = "json") -> dict | str:
        """Generate metrics report.

        Args:
            format: Report format (json, markdown, html)

        Returns:
            Report in requested format
        """
        if not self.metrics:
            self.collect_all_metrics()

        report = {
            "module": self.module,
            "owner": self.owner,
            "namespace": self.namespace,
            "generated_at": datetime.now().isoformat(),
            "metrics": self.metrics,
            "review_cadence": self.REVIEW_CADENCE,
            "alerting_thresholds": self.ALERTING_THRESHOLDS,
        }

        if format == "json":
            return report
        elif format == "markdown":
            return self._format_markdown(report)
        elif format == "html":
            return self._format_html(report)
        else:
            return report

    def _format_markdown(self, report: dict) -> str:
        """Format report as Markdown.

        Args:
            report: Report dictionary

        Returns:
            Markdown formatted report
        """
        lines = [
            "# BSR Metrics Report",
            f"\n**Module**: `{report['module']}`",
            f"**Owner**: `{report['owner']}`",
            f"**Namespace**: `{report['namespace']}`",
            f"**Generated**: {report['generated_at']}",
            "\n## Metrics Summary",
            "\n### Downloads",
            f"- Last 24h: {report['metrics']['downloads']['last_24h']}",
            f"- Last 7d: {report['metrics']['downloads']['last_7d']}",
            f"- Last 30d: {report['metrics']['downloads']['last_30d']}",
            "\n### Versions",
            f"- Latest: {report['metrics']['versions']['latest_version']}",
            f"- Total: {report['metrics']['versions']['total_versions']}",
            "\n### Dependents",
            f"- Total: {report['metrics']['dependents']['total_dependents']}",
            "\n## Review Cadence",
            self._cadence_markdown(report["review_cadence"]),
            "\n## Alerting Thresholds",
            self._thresholds_markdown(report["alerting_thresholds"]),
        ]
        return "\n".join(lines)

    def _cadence_markdown(self, cadence: dict) -> str:
        """Format review cadence as Markdown."""
        lines = []
        for freq, details in cadence.items():
            lines.append(f"\n### {freq.capitalize()}")
            lines.append(f"- **Audience**: {details['audience']}")
            lines.append(f"- **SLA**: {details['sla']}")
            lines.append(f"- **Purpose**: {details['purpose']}")
            lines.append(f"- **Metrics**: {', '.join(details['metrics'])}")
        return "\n".join(lines)

    def _thresholds_markdown(self, thresholds: dict) -> str:
        """Format alerting thresholds as Markdown."""
        lines = []
        for alert_name, alert_config in thresholds.items():
            lines.append(f"\n#### {alert_name} ({alert_config['severity'].upper()})")
            lines.append(f"- **Metric**: {alert_config['metric']}")
            lines.append(f"- **Threshold**: {alert_config['threshold']}")
            lines.append(f"- **Description**: {alert_config['description']}")
        return "\n".join(lines)

    def _format_html(self, report: dict) -> str:
        """Format report as HTML dashboard.

        Args:
            report: Report dictionary

        Returns:
            HTML formatted report
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>BSR Metrics Dashboard - {report['module']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BSR Metrics Dashboard</h1>
        <p><strong>Module</strong>: {report['module']}</p>
        <p><strong>Owner</strong>: {report['owner']}</p>
        <p><strong>Generated</strong>: {report['generated_at']}</p>
    </div>
    <h2>Key Metrics</h2>
    <div class="metric">
        <div class="metric-value">{report['metrics']['downloads']['last_24h']}</div>
        <div class="metric-label">Downloads (24h)</div>
    </div>
    <div class="metric">
        <div class="metric-value">{report['metrics']['dependents']['total_dependents']}</div>
        <div class="metric-label">Active Dependents</div>
    </div>
    <div class="metric">
        <div class="metric-value">v{report['metrics']['versions']['latest_version']}</div>
        <div class="metric-label">Latest Version</div>
    </div>
    <h2>Review Cadence</h2>
    <table>
        <tr><th>Frequency</th><th>Audience</th><th>SLA</th><th>Purpose</th></tr>
"""
        for freq, details in report["review_cadence"].items():
            html += f"""        <tr>
            <td>{freq.capitalize()}</td>
            <td>{details['audience']}</td>
            <td>{details['sla']}</td>
            <td>{details['purpose']}</td>
        </tr>
"""
        html += """    </table>
</body>
</html>
"""
        return html

    def save_report(self, filepath: Path | str, format: str = "json") -> None:
        """Save report to file.

        Args:
            filepath: Path to save report
            format: Report format
        """
        report = self.generate_report(format)
        filepath = Path(filepath)

        if format == "json":
            filepath.write_text(json.dumps(report, indent=2))
        else:
            filepath.write_text(report)


def main() -> None:
    """CLI entry point for metrics collection and reporting."""
    parser = argparse.ArgumentParser(
        description="BSR Metrics Collection and Reporting Tool"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect metrics now",
    )
    parser.add_argument(
        "--report",
        choices=["json", "markdown", "html"],
        default="json",
        help="Generate report in format (default: json)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate dashboard HTML",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--owner",
        default="tommyk",
        help="BSR owner (default: tommyk)",
    )
    parser.add_argument(
        "--module",
        default="crypto-market-data",
        help="Module name (default: crypto-market-data)",
    )

    args = parser.parse_args()

    collector = BSRMetricsCollector(owner=args.owner, module=args.module)

    if args.collect or args.report or args.dashboard:
        report_format = "html" if args.dashboard else args.report
        report = collector.generate_report(format=report_format)

        if args.output:
            collector.save_report(args.output, format=report_format)
            print(f"Report saved to {args.output}")
        else:
            if isinstance(report, dict):
                print(json.dumps(report, indent=2))
            else:
                print(report)


if __name__ == "__main__":
    main()
