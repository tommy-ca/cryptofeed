# Kafka Backend Migration Progress Reports

This directory contains monthly progress reports tracking the migration from legacy Kafka backend to the modern implementation.

## Report Format

Each report is named `YYYY-MM-progress.md` and includes:

- **Migration Percentage:** Proportion of modern vs legacy usage
- **Legacy Usage Stats:** Which classes are still in use
- **Modern Usage Stats:** Adoption of new backend
- **Timeline Recommendation:** Should timeline be adjusted?
- **Notable Events:** Critical issues, milestones reached, etc.

## Automation

Progress reports are generated automatically using:

```bash
python -m cryptofeed.tools.kafka_progress_report
```

This command:
1. Loads usage statistics from deprecation tracking
2. Calculates migration percentage
3. Generates timeline recommendations
4. Creates markdown report with visualizations

## Manual Report Generation

To generate a report manually:

```python
from cryptofeed.backends.kafka.deprecation import ProgressReport
from datetime import datetime

# Load tracked usage data
report = ProgressReport()
# ... record usage events ...

# Generate markdown report
markdown = report.to_markdown()
with open(f"docs/kafka/progress-reports/{datetime.now().strftime('%Y-%m')}-progress.md", "w") as f:
    f.write(markdown)
```

## Report Schedule

- **Generation:** First Monday of each month
- **Review:** Platform engineering team
- **Publication:** Within 3 business days of generation
- **Timeline Updates:** If recommendation suggests adjustment

## Current Reports

Reports are listed chronologically (newest first):

- (No reports yet - first report expected after Phase 5 completion)

## References

- Deprecation Timeline: `docs/kafka/deprecation-timeline.md`
- Progress Tracking API: `cryptofeed/backends/kafka/deprecation.py`
- Usage Analytics: `cryptofeed/backends/kafka/analytics.py` (future)
