# Per-Exchange Migration Automation (Task 23)

Automated tools for Week 3 per-exchange consumer migration during the market-data-kafka-producer Blue-Green cutover.

## Overview

These scripts automate the per-exchange migration workflow defined in `PHASE_5_EXECUTION_PLAN.md`, enabling safe, gradual migration of exchanges from legacy per-symbol topics to consolidated topics.

## Migration Sequence

Exchanges are migrated in volume order with 1 exchange per business day:

| Day | Exchange | Volume | Migration Window |
|-----|----------|--------|------------------|
| Mon | Coinbase | Highest | 10:00-16:00 UTC (6 hours) |
| Tue | Binance | High | 10:00-16:00 UTC (6 hours) |
| Wed | OKX | Medium | 10:00-16:00 UTC (6 hours) |
| Thu | Kraken + Bybit | Medium | 10:00-16:00 UTC (6 hours) |
| Fri | Remaining | Low-Medium | 10:00-16:00 UTC (6 hours) |

## Scripts

### 1. `migrate_exchange.py` - Migration Orchestrator

Orchestrates the complete 5-phase migration workflow with 3 pause points.

**Usage:**
```bash
# Dry-run (simulate without changes)
python scripts/migrate_exchange.py coinbase --dry-run

# Production migration
python scripts/migrate_exchange.py coinbase

# With custom configuration
python scripts/migrate_exchange.py binance --config migration-config.json
```

**Configuration Example (`migration-config.json`):**
```json
{
  "exchange": "coinbase",
  "migration_window_hours": 6,
  "validation_checks": [
    "lag", "error_rate", "data_completeness",
    "no_duplicates", "latency_p99", "downstream_storage",
    "monitoring", "no_incidents"
  ],
  "rollback_timeout_seconds": 300
}
```

**Output:**
```json
{
  "status": "success",
  "exchange": "coinbase",
  "phases_executed": ["pre_migration", "consumer_cutover", "validation", "monitoring", "post_migration"],
  "phase_timings": {
    "pre_migration_minutes": 30.0,
    "consumer_cutover_minutes": 90.0,
    "validation_minutes": 210.0,
    "monitoring_minutes": 90.0,
    "post_migration_minutes": 60.0
  },
  "pause_points_encountered": 3,
  "rollback_triggered": false
}
```

### 2. `validate_exchange_migration.py` - Validation Checker

Validates migration success criteria for an exchange.

**Usage:**
```bash
# Validate all criteria
python scripts/validate_exchange_migration.py coinbase

# Validate specific check
python scripts/validate_exchange_migration.py coinbase --check lag
python scripts/validate_exchange_migration.py coinbase --check error_rate
python scripts/validate_exchange_migration.py coinbase --check data_completeness
```

**Success Criteria Checked:**
1. Consumer lag <5 seconds
2. Error rate <0.1%
3. Data completeness 100%
4. No duplicates
5. Latency p99 <5ms
6. Downstream storage 100%
7. Monitoring dashboard functional
8. No production incidents

**Output:**
```json
{
  "status": "success",
  "checks": {
    "consumer_lag": {"status": "success", "lag_seconds": 3.2, "threshold_seconds": 5.0},
    "error_rate": {"status": "success", "error_rate_percent": 0.05, "threshold_percent": 0.1},
    "data_completeness": {"status": "success", "match_rate_percent": 100.0}
  },
  "exchange": "coinbase"
}
```

### 3. `track_migration_status.py` - Status Tracker

Tracks migration status across all exchanges and generates dashboard data.

**Usage:**
```bash
# Record exchange status
python scripts/track_migration_status.py --record coinbase completed

# Generate dashboard
python scripts/track_migration_status.py --dashboard

# Export JSON report
python scripts/track_migration_status.py --export migration-report.json
```

**Dashboard Output:**
```json
{
  "exchanges_completed": 2,
  "exchanges_in_progress": 1,
  "exchanges_pending": 2,
  "total_exchanges": 5,
  "exchanges": {
    "coinbase": {"status": "completed", "metrics": {"lag_seconds": 2.3}},
    "binance": {"status": "in_progress", "metrics": {"lag_seconds": 1.8}},
    "okx": {"status": "pending"}
  }
}
```

### 4. `go_nogo_decision.py` - Decision Engine

Automated go/no-go decision support based on success criteria metrics.

**Usage:**
```bash
# Evaluate metrics
python scripts/go_nogo_decision.py metrics.json
```

**Metrics Input (`metrics.json`):**
```json
{
  "consumer_lag_seconds": 3.2,
  "error_rate_percent": 0.05,
  "data_completeness_percent": 100.0,
  "latency_p99_ms": 4.2
}
```

**Decision Output:**
```json
{
  "go_nogo": "GO",
  "all_criteria_passed": true,
  "passed_criteria": ["consumer_lag_seconds", "error_rate_percent", "data_completeness_percent"],
  "failed_criteria": [],
  "recommendation": "✅ All success criteria passed. Proceed to next exchange migration."
}
```

### 5. `rollback_exchange.py` - Rollback Executor

Executes partial rollback for a single failed exchange (Runbook 1.5).

**Usage:**
```bash
# Dry-run rollback
python scripts/rollback_exchange.py coinbase --dry-run

# Production rollback
python scripts/rollback_exchange.py coinbase
```

**Output:**
```json
{
  "status": "success",
  "exchange": "coinbase",
  "consumer_reverted": true,
  "validation": {
    "consumer_lag_decreasing": true,
    "error_rate_normalized": true
  },
  "incident_report": {
    "exchange": "coinbase",
    "trigger": "Validation failure",
    "timestamp": "2025-11-26T10:30:00Z",
    "rollback_duration_seconds": 245
  },
  "duration_seconds": 245
}
```

### 6. `generate_migration_checklist.py` - Checklist Generator

Generates per-exchange migration checklists and documentation.

**Usage:**
```bash
# Generate JSON checklist
python scripts/generate_migration_checklist.py coinbase

# Export Markdown checklist
python scripts/generate_migration_checklist.py coinbase --output checklist-coinbase.md
```

**Markdown Output:**
```markdown
# Migration Checklist: Coinbase

## Pre-Migration
- [ ] Review baseline metrics
- [ ] Verify monitoring dashboard operational
- [ ] Notify stakeholders
- [ ] Confirm rollback procedure ready
- [ ] Confirm QA team available

## Consumer Cutover
- [ ] Update consumer subscriptions
- [ ] Deploy updated consumers
- [ ] Verify consumers started successfully
- [ ] Validate consumer lag <5s

## Validation
### Success Criteria
- [ ] Consumer Lag: <5 seconds
- [ ] Error Rate: <0.1%
- [ ] Data Completeness: 100%
...
```

## Typical Daily Workflow

### Morning (T-30min before migration window)

```bash
# 1. Generate migration checklist
python scripts/generate_migration_checklist.py coinbase --output checklist-coinbase.md

# 2. Verify baseline metrics
python scripts/validate_exchange_migration.py coinbase

# 3. Record migration start
python scripts/track_migration_status.py --record coinbase in_progress
```

### Migration Execution (10:00-16:00 UTC)

```bash
# 4. Execute migration
python scripts/migrate_exchange.py coinbase

# Output shows pause points:
# - Pause Point 1 (T+90min): Review cutover metrics
# - Pause Point 2 (T+210min): Go/no-go decision after validation
# - Pause Point 3 (T+300min): Final approval
```

### Post-Migration Validation (T+300min)

```bash
# 5. Validate success criteria
python scripts/validate_exchange_migration.py coinbase > coinbase-validation.json

# 6. Automated go/no-go decision
python scripts/go_nogo_decision.py coinbase-validation.json

# If GO:
python scripts/track_migration_status.py --record coinbase completed

# If NO-GO:
python scripts/rollback_exchange.py coinbase
python scripts/track_migration_status.py --record coinbase failed
```

### End of Day

```bash
# 7. Generate daily dashboard
python scripts/track_migration_status.py --dashboard > daily-dashboard.json

# 8. Export migration report
python scripts/track_migration_status.py --export migration-report-day1.json
```

## Rollback Scenarios

### Partial Rollback (Single Exchange Failure)

When one exchange fails but others are healthy:

```bash
# Rollback only the failed exchange
python scripts/rollback_exchange.py binance

# Healthy exchanges (coinbase) remain on new topics
# Failed exchange (binance) reverts to legacy topics
```

**When to Use:**
- ✅ Single exchange failure
- ✅ Other exchanges healthy (lag <5s, error <0.1%)
- ✅ Failed exchange isolated (no cascading failures)
- ✅ Failed exchange <10% of total volume

**When to Use Full Rollback:**
- ❌ Multiple exchanges failing
- ❌ Cascading failures or infrastructure issues
- ❌ First exchange migration fails (Day 1)

## Testing

All automation scripts have comprehensive unit tests:

```bash
# Run all Task 23 tests
python -m pytest tests/unit/test_task_23_exchange_migration.py -v

# 32 tests covering:
# - Migration orchestration
# - Validation checks
# - Status tracking
# - Go/no-go decisions
# - Rollback automation
# - Documentation generation
```

## Dependencies

All scripts are self-contained with standard library dependencies only:
- `json` - Configuration and output
- `argparse` - CLI argument parsing
- `datetime` - Timestamp generation
- `typing` - Type hints

## Integration with PHASE_5_EXECUTION_PLAN.md

These scripts implement the Week 3 per-exchange migration workflow from `PHASE_5_EXECUTION_PLAN.md`:

- **Section 2.3**: Week 3 Daily Checklist (lines 834-875)
- **Runbook 1.5**: Partial Rollback Procedure (lines 1428-1733)
- **Success Criteria**: 10 Measurable Metrics (lines 1975-2247)

## Support

For issues or questions:
- Review `PHASE_5_EXECUTION_PLAN.md` for detailed execution procedures
- Check `OPERATIONAL_RUNBOOK.md` for operational procedures
- Escalate to L2 (DevOps + Engineering On-Call) if automated scripts fail

## Version

- **Version**: 1.0.0
- **Created**: November 26, 2025
- **Status**: Production Ready
- **Tests**: 32 passing
