# Kafka Backend Deprecation Runbook

## Overview
This runbook provides operational procedures for managing the deprecation lifecycle of legacy Kafka backend classes during the migration to the modern implementation.

**Audience**: Platform engineers, SREs, DevOps teams
**Estimated time**: 15-30 minutes per procedure
**Prerequisites**: Access to production logs, monitoring dashboards, Kafka cluster

## Table of Contents
1. [Monitoring Deprecation Warnings](#monitoring-deprecation-warnings)
2. [Tracking Migration Progress](#tracking-migration-progress)
3. [Handling User Reports](#handling-user-reports)
4. [Emergency Rollback Procedures](#emergency-rollback-procedures)

---

## Monitoring Deprecation Warnings

### Purpose
Track and analyze deprecation warning frequency to inform migration timelines and identify slow-migrating deployments.

### Procedure

1. **Access Monitoring Dashboard**
   ```bash
   # View deprecation warning metrics
   kubectl logs -l app=cryptofeed --tail=1000 | grep "DeprecationWarning"
   ```

2. **Check Warning Statistics**
   ```python
   from cryptofeed.backends.kafka.maintenance.integration import MaintenanceCoordinator

   coordinator = MaintenanceCoordinator()
   analytics = coordinator.deprecation_monitoring.get_migration_analytics()

   print(f"Legacy usage: {analytics['legacy_usage']['total_legacy_usage']}")
   print(f"Modern usage: {analytics['modern_usage']['total_modern_usage']}")
   print(f"Migration percentage: {analytics['migration_percentage']}%")
   ```

3. **Review Usage Patterns**
   - Open Grafana dashboard: `Kafka Backend Migration Progress`
   - Check panels:
     - Legacy vs Modern Usage (7-day trend)
     - Top Legacy Components (by usage count)
     - Migration Velocity (daily percentage change)

4. **Identify Slow Migrations**
   - If migration percentage < 50% after 4 weeks, investigate:
     - Check application logs for blocking issues
     - Review user feedback channels
     - Contact teams with high legacy usage

### Expected Outcomes
- Clear visibility into migration progress
- Identification of blockers or issues
- Data-driven timeline adjustments

### Troubleshooting
**Issue**: No deprecation warnings logged
**Solution**: Verify logging configuration, check that legacy classes are being imported

**Issue**: Warning spam in logs
**Solution**: Consider rate-limiting warnings or adjusting log levels

---

## Tracking Migration Progress

### Purpose
Generate regular migration progress reports for stakeholders and identify acceleration opportunities.

### Procedure

1. **Run Automated Progress Report**
   ```bash
   python tools/kafka-maintenance-scheduler.py run-once \
     --config config/maintenance-tasks.yaml
   ```

2. **Review Report Output**
   ```bash
   # Check latest deprecation report
   cat /var/log/cryptofeed/deprecation-reports/latest.json | jq .
   ```

3. **Analyze Migration Velocity**
   - Calculate daily migration rate: `(current_percentage - previous_percentage) / days`
   - Estimate completion date: `days_remaining = (100 - current_percentage) / daily_rate`

4. **Update Timeline if Needed**
   ```python
   from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

   timeline = DeprecationTimeline()
   recommendation = analytics['timeline_recommendation']

   if recommendation.should_extend_timeline:
       print(f"Recommended extension: {recommendation.recommended_weeks} weeks")
       print(f"Reason: {recommendation.reason}")
       # Update timeline in communication system
   ```

5. **Communicate Status**
   - Post update to team Slack channel
   - Update confluence page with latest metrics
   - Flag any blockers or risks to stakeholders

### Expected Outcomes
- Weekly progress reports generated automatically
- Timely timeline adjustments based on data
- Transparent communication with stakeholders

### Troubleshooting
**Issue**: Report generation fails
**Solution**: Check Kafka connectivity, verify scheduler configuration

**Issue**: Migration velocity slowing
**Solution**: Reach out to heavy legacy users, offer migration support

---

## Handling User Reports

### Purpose
Respond to user-reported issues related to deprecation warnings or migration challenges.

### Procedure

1. **Triage User Report**
   - Severity: Critical (production down), High (degraded), Medium (warnings), Low (questions)
   - Component: Legacy backend, modern backend, migration tool, documentation

2. **Gather Context**
   ```bash
   # Request from user:
   # - Full stack trace
   # - Configuration file (sanitized)
   # - Cryptofeed version
   # - Python version
   ```

3. **Common Issues and Solutions**

   **Issue**: "DeprecationWarning flooding logs"
   ```python
   # Solution: Add warning filter
   import warnings
   warnings.filterwarnings('once', category=DeprecationWarning, module='cryptofeed.backends.kafka')
   ```

   **Issue**: "Migration guide doesn't match my configuration"
   ```bash
   # Solution: Use configuration migration tool
   python tools/migrate-kafka-config.py \
     --legacy-config config/old-kafka.yaml \
     --output config/new-kafka.yaml
   ```

   **Issue**: "New backend doesn't support feature X"
   ```text
   # Solution:
   # 1. Check if feature deprecated or renamed
   # 2. Review migration guide feature mapping
   # 3. If no equivalent, file feature request with justification
   ```

4. **Escalation Path**
   - Tier 1 (First responder): Check documentation, provide guides
   - Tier 2 (Backend team): Debug configuration, provide workarounds
   - Tier 3 (Core maintainers): Code fixes, architecture decisions

### Expected Outcomes
- Timely resolution of user issues
- Improved documentation based on common questions
- User confidence in migration process

### Troubleshooting
**Issue**: User stuck on migration for >1 week
**Solution**: Schedule pair programming session, offer hands-on support

---

## Emergency Rollback Procedures

### Purpose
Safely rollback to legacy backend if critical production issues arise during migration.

### Procedure

1. **Assess Situation**
   - Is production traffic impacted? (Yes → Immediate rollback)
   - Can issue be mitigated? (Partial rollback, traffic shifting)
   - What is blast radius? (All exchanges, specific symbols)

2. **Execute Rollback**
   ```python
   # Option 1: Configuration rollback (fast)
   # Revert to previous configuration
   kubectl apply -f config/kafka-legacy.yaml
   kubectl rollout restart deployment/cryptofeed

   # Option 2: Code rollback (slower but complete)
   git revert <commit-hash>
   docker build -t cryptofeed:rollback .
   kubectl set image deployment/cryptofeed cryptofeed=cryptofeed:rollback
   ```

3. **Verify Rollback**
   ```bash
   # Check that legacy backend is active
   kubectl logs -l app=cryptofeed --tail=100 | grep "Kafka backend"

   # Verify messages flowing
   kafka-console-consumer --bootstrap-server kafka:9092 \
     --topic cryptofeed.trades --from-beginning --max-messages 10
   ```

4. **Root Cause Analysis**
   - Collect logs and metrics from failure period
   - Identify triggering configuration or code change
   - Document issue in incident report
   - Create action items to prevent recurrence

5. **Post-Rollback Actions**
   - Notify stakeholders of rollback and reason
   - Update migration timeline if needed
   - Schedule post-mortem meeting
   - Plan safer migration approach

### Expected Outcomes
- Production restored within 5 minutes
- Root cause identified
- Prevention plan created

### Troubleshooting
**Issue**: Rollback doesn't restore service
**Solution**: Check Kafka cluster health, verify network connectivity

**Issue**: Data loss during rollback
**Solution**: Review exactly-once semantics configuration, check consumer offsets

---

## Success Criteria

### Metrics to Monitor
- Deprecation warning frequency (decreasing over time)
- Migration percentage (increasing toward 100%)
- User-reported issues (low and decreasing)
- Rollback frequency (zero or near-zero)

### Targets
- Migration velocity: >10% per week
- User issue resolution: <48 hours
- Rollback incidents: 0
- Production availability: >99.9%

## Related Documentation
- [Migration Guide](../kafka/MIGRATION_GUIDE.md)
- [Troubleshooting Guide](../kafka/TROUBLESHOOTING.md)
- [Timeline Documentation](../kafka/deprecation-timeline.md)

## Changelog
- 2025-11-26: Initial runbook created (Task 6.3)
