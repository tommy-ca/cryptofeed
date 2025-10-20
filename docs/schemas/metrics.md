# BSR Metrics Monitoring Setup

**Status**: ✅ Implementation Ready
**Phase**: Phase 3 - Operational Improvements
**Component**: Task 9.1 - BSR Metrics Monitoring

## Overview

This document defines the automated metrics collection system for the crypto-market-data module on the Buf Schema Registry (BSR). The system collects and reports on module usage, adoption, and health metrics to enable informed governance decisions.

## Architecture

```
BSR API
  ↓
BSRMetricsCollector (tools/bsr_metrics.py)
  ├─ collect_downloads()     → Daily downloads
  ├─ collect_dependents()    → Dependent modules
  ├─ collect_versions()      → Version adoption
  └─ generate_report()       → JSON/Markdown/HTML
       ↓
  Reports & Dashboard
```

## Metric Definitions

### 1. Module Downloads (Daily)

**Display Name**: Total Downloads
**Unit**: Count
**Frequency**: Daily
**Collection Method**: BSR API `/modules/{owner}/{name}/analytics/downloads`
**Description**: Total number of module downloads from BSR

**Sub-metrics**:
- `last_24h`: Downloads in past 24 hours
- `last_7d`: Downloads in past 7 days
- `last_30d`: Downloads in past 30 days
- `time_series`: Hourly/daily download data

**SLA**: Data available within 1 hour of collection

---

### 2. Version Adoption (Daily)

**Display Name**: Latest Version Adoption %
**Unit**: Percentage
**Frequency**: Daily
**Collection Method**: BSR API `/modules/{owner}/{name}/versions`
**Description**: Percentage of consumers using the latest version

**Sub-metrics**:
- `adoption_by_version`: Distribution across all versions
- `latest_version`: Current latest version tag
- `previous_version_adoption`: Adoption of previous major version

**Target**: ≥ 80% adoption within 30 days of release

---

### 3. Active Dependents (Weekly)

**Display Name**: Active Dependent Modules
**Unit**: Count
**Frequency**: Weekly
**Collection Method**: BSR API `/modules/{owner}/{name}/dependents`
**Description**: Number of modules/projects actively depending on this module

**Sub-metrics**:
- `by_language`: Count by programming language (Go, Python, TypeScript, etc.)
- `by_phase`: Count by adoption phase (beta, stable, production)

**SLA**: Data refreshed weekly on Monday 08:00 UTC

---

### 4. Download Trends (Daily)

**Display Name**: Download Trend
**Unit**: Count
**Frequency**: Daily
**Collection Method**: Time-series aggregation from downloads metric
**Description**: Daily download count trend for pattern analysis

**Analysis**:
- Week-over-week growth rate
- Seasonal patterns
- Anomaly detection (unexplained drops)

---

### 5. Version Distribution (Daily)

**Display Name**: Version Distribution
**Unit**: Percentage
**Frequency**: Daily
**Collection Method**: BSR API download metrics by version
**Description**: Distribution of consumers across all available versions

**Insights**:
- Version migration speed
- Legacy version adoption
- Breaking change impact

---

## Collection Frequency & Cadence

### Daily Metrics (Automatic)

Collected automatically every 24 hours at **02:00 UTC**:
- Module downloads
- Version adoption
- Download trends
- Version distribution

**Tool**: `python tools/bsr_metrics.py --collect`

---

### Weekly Metrics Review

Triggered every **Monday at 09:00 UTC**:

**Audience**: Schema Team
**SLA**: Review and respond within 24 hours
**Purpose**: Adoption and dependency analysis

**Review Focus**:
- Active dependents growth
- Version adoption progress
- Consumer feedback queue
- Governance escalations

---

### Monthly Strategic Review

Triggered on the **1st of each month at 09:00 UTC**:

**Audience**: Engineering Leadership
**SLA**: Review and report within 5 business days
**Purpose**: Strategic alignment and roadmap decisions

**Review Focus**:
- Long-term adoption trends
- SLA compliance (governance, response times)
- Consumer satisfaction metrics
- Resource allocation for next quarter

---

## Alerting Thresholds

### Warning Alerts

Triggered when conditions are met; team notified via Slack/email.

| Alert | Metric | Threshold | SLA Response |
|-------|--------|-----------|--------------|
| Low Adoption | Version Adoption % | < 60% | Within 2 days |
| No Activity | Downloads (24h) | 0 downloads | Within 4 hours |
| Low Dependents | Active Dependents | < 5 | Within 5 days |

### Critical Alerts

Escalated immediately to on-call maintainer.

| Alert | Metric | Threshold | SLA Response |
|-------|--------|-----------|--------------|
| Critical Adoption | Version Adoption % | < 40% | Within 2 hours |
| Adoption Cliff | Week-over-week change | < -50% | Within 1 hour |
| BSR API Down | Metrics Collection | Failed 2 consecutive cycles | Immediate |

---

## Report Generation

### JSON Reports

Machine-readable format for CI/CD integration.

```bash
python tools/bsr_metrics.py --report json --output metrics.json
```

**Contents**:
- All collected metrics
- Timestamps and frequencies
- Review cadence definitions
- Alerting thresholds

**Usage**: Automated CI/CD pipelines, dashboards, alerting systems

---

### Markdown Reports

Human-readable format for documentation and team review.

```bash
python tools/bsr_metrics.py --report markdown --output metrics_report.md
```

**Contents**:
- Formatted metrics summary
- Review cadence table
- Threshold definitions
- Action items

**Usage**: Weekly team reviews, status reports

---

### HTML Dashboard

Interactive dashboard for monitoring and visualization.

```bash
python tools/bsr_metrics.py --report html --output dashboard.html
```

**Features**:
- Real-time metrics display
- Chart visualization (trends, distribution)
- Threshold indicators (green/yellow/red)
- Review schedule

**Usage**: Open in browser for live monitoring

---

## Review Workflows

### Daily Monitoring (DevOps Team)

**When**: Daily at 03:00 UTC (1 hour after collection)
**SLA**: Acknowledge within 4 hours
**Action**: Investigate any warnings

```
Check metrics.json
├─ Downloads: Verify no drop > 10%
├─ Version adoption: Monitor if trending down
└─ No activity alerts: Verify BSR status
```

---

### Weekly Team Review (Schema Team)

**When**: Monday 10:00 UTC (schema team meeting)
**Duration**: 15-30 minutes
**SLA**: Complete within 24 hours
**Outputs**:
- Adoption report
- Dependency analysis
- Issue prioritization

**Agenda**:
1. Review weekly metrics report
2. Identify adoption blockers
3. Plan consumer engagement
4. Escalate if needed

---

### Monthly Strategic Review (Leadership)

**When**: 1st of month, 14:00 UTC
**Duration**: 1-2 hours
**SLA**: Complete within 5 business days
**Outputs**:
- Strategic recommendations
- Resource allocation decisions
- Quarterly roadmap adjustments

**Agenda**:
1. Review trend analysis (30 days)
2. Adoption velocity vs targets
3. Consumer satisfaction trends
4. Governance effectiveness
5. Roadmap implications

---

## Configuration

### CLI Usage

```bash
# Collect metrics now
python tools/bsr_metrics.py --collect

# Generate JSON report
python tools/bsr_metrics.py --report json

# Generate Markdown report
python tools/bsr_metrics.py --report markdown

# Generate HTML dashboard
python tools/bsr_metrics.py --report html --output dashboard.html

# Custom module and output
python tools/bsr_metrics.py \
  --owner myorg \
  --module mymodule \
  --report html \
  --output my_dashboard.html
```

### Automated Collection (Cron)

Add to crontab for daily automated collection:

```cron
# Collect BSR metrics daily at 02:00 UTC
0 2 * * * cd /path/to/cryptofeed && python tools/bsr_metrics.py --collect >> /var/log/bsr_metrics.log 2>&1
```

---

## Integration Points

### CI/CD Pipeline

```yaml
# .github/workflows/metrics.yml
name: BSR Metrics Collection

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 02:00 UTC

jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Collect BSR metrics
        run: python tools/bsr_metrics.py --collect
      - name: Generate report
        run: python tools/bsr_metrics.py --report json --output metrics.json
      - name: Upload to dashboard
        run: |
          # Push to metrics database, dashboard, or monitoring system
```

### Monitoring Dashboard

Integrate with existing monitoring systems:
- **Grafana**: Import JSON metrics for visualization
- **Datadog**: Push metrics to Datadog API
- **CloudWatch**: Store in AWS CloudWatch
- **Custom**: Process JSON reports in internal systems

---

## Metric Targets

### Adoption Goals

| Metric | 1 Month | 3 Months | 6 Months |
|--------|---------|----------|----------|
| Version Adoption | 80% | 90% | 95% |
| Active Dependents | 10 | 25 | 50 |
| Download Growth | Baseline | +50% | +100% |

### Response Time SLAs

| Issue Type | Audience | SLA |
|-----------|----------|-----|
| Bug report | Team | 2 business days |
| Adoption blocker | Team | 3 business days |
| Breaking change request | Leadership | 5 business days |
| Strategic issue | Leadership | 10 business days |

---

## Troubleshooting

### BSR API Unavailable

If metrics collection fails:
1. Check BSR status page
2. Verify credentials/permissions
3. Retry after 1 hour
4. Escalate if persists > 2 hours

### Metrics Collection Tool

```bash
# Test metrics collector
python -c "from tools.bsr_metrics import BSRMetricsCollector; collector = BSRMetricsCollector(); print(collector.generate_report('json'))"

# Run tests
python -m pytest tests/proto_integration/test_bsr_metrics.py -v
```

---

## Future Enhancements

- **Real-time Monitoring**: WebSocket streaming of metrics
- **Predictive Analytics**: ML-based adoption forecasting
- **Custom Dashboards**: Per-stakeholder views
- **Automated Actions**: Self-healing based on thresholds
- **Multi-module Tracking**: Monitor dependencies tree

---

## Related Documents

- `governance.md` - Governance processes and workflows
- `RELEASE_v0.1.0.md` - v0.1.0 baseline release details
- `tools/bsr_metrics.py` - Metrics collection implementation

---

**Last Updated**: 2025-10-20
**Maintained By**: Schema Team
**Review Frequency**: Quarterly

