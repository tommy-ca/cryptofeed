# Kafka Migration Execution Runbook

## Overview
This runbook provides step-by-step procedures for executing the migration from legacy to modern Kafka backend in production environments with zero downtime.

**Audience**: DevOps engineers, SREs, Release managers
**Estimated time**: 2-4 hours per deployment
**Prerequisites**: Tested migration in staging, approval from stakeholders, rollback plan prepared

## Table of Contents
1. [Pre-Migration Checklist](#pre-migration-checklist)
2. [Blue-Green Deployment](#blue-green-deployment)
3. [Canary Rollout](#canary-rollout)
4. [Validation and Verification](#validation-and-verification)
5. [Rollback Procedures](#rollback-procedures)

---

## Pre-Migration Checklist

### Purpose
Ensure all prerequisites are met before executing production migration.

### Checklist

- [ ] **Configuration Prepared**
  ```bash
  # Validate new configuration
  python tools/migrate-kafka-config.py \
    --legacy-config config/prod-kafka-legacy.yaml \
    --output config/prod-kafka-modern.yaml \
    --validate

  # Compare configurations
  diff config/prod-kafka-legacy.yaml config/prod-kafka-modern.yaml
  ```

- [ ] **Staging Environment Tested**
  ```bash
  # Deploy to staging
  kubectl apply -f config/staging/kafka-modern.yaml -n staging

  # Run integration tests
  pytest tests/integration/ -v --env=staging

  # Validate message delivery
  kafka-console-consumer --bootstrap-server staging-kafka:9092 \
    --topic cryptofeed.trades --from-beginning --max-messages 100
  ```

- [ ] **Rollback Plan Documented**
  - Identify rollback triggers (error rate > 1%, latency > 100ms)
  - Prepare rollback commands (documented below)
  - Test rollback in staging environment

- [ ] **Monitoring and Alerts Configured**
  ```bash
  # Verify Prometheus scraping new metrics
  curl http://prometheus:9090/api/v1/targets | grep cryptofeed

  # Test alert rules
  promtool check rules config/alerts/kafka-backend.yaml

  # Verify Grafana dashboards
  curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
    http://grafana:3000/api/dashboards/uid/kafka-migration
  ```

- [ ] **Stakeholder Approval**
  - [ ] Product owner approval
  - [ ] Engineering lead sign-off
  - [ ] Operations team ready
  - [ ] Communication plan in place

- [ ] **Communication Sent**
  - [ ] Internal team notified (Slack, email)
  - [ ] External stakeholders informed (if applicable)
  - [ ] Maintenance window scheduled (if needed)

### Expected Outcomes
- All checklist items completed
- Team ready for migration
- Rollback plan tested and ready

---

## Blue-Green Deployment

### Purpose
Deploy modern backend alongside legacy backend with zero downtime using blue-green deployment strategy.

### Procedure

1. **Deploy Green (Modern) Environment**
   ```bash
   # Create new deployment with modern backend
   kubectl apply -f config/prod/kafka-modern-deployment.yaml

   # Wait for pods to be ready
   kubectl rollout status deployment/cryptofeed-modern

   # Verify pod health
   kubectl get pods -l app=cryptofeed,version=modern
   ```

2. **Configure Traffic Splitting (0% to Modern)**
   ```bash
   # Initial state: 100% traffic to legacy (blue)
   kubectl apply -f config/prod/service-blue-green-0percent.yaml

   # Verify routing
   curl http://cryptofeed-service/health | jq .backend_version
   # Should return "legacy"
   ```

3. **Shift 10% Traffic to Modern (Canary)**
   ```bash
   # Update service to route 10% to modern
   kubectl apply -f config/prod/service-blue-green-10percent.yaml

   # Monitor for 15 minutes
   watch -n 10 'curl -s http://prometheus:9090/api/v1/query?query=kafka_publish_latency_seconds{version="modern",quantile="0.99"}'
   ```

4. **Validate 10% Traffic**
   ```bash
   # Check error rate
   ERROR_RATE=$(curl -s 'http://prometheus:9090/api/v1/query?query=rate(kafka_publish_errors_total{version="modern"}[5m])/rate(kafka_publish_attempts_total{version="modern"}[5m])*100' | jq -r '.data.result[0].value[1]')

   if (( $(echo "$ERROR_RATE > 0.1" | bc -l) )); then
     echo "ERROR: Error rate too high ($ERROR_RATE%), rolling back"
     # Execute rollback (see section below)
   else
     echo "OK: Error rate acceptable ($ERROR_RATE%)"
   fi
   ```

5. **Gradually Increase Traffic**
   ```bash
   # 25% modern
   kubectl apply -f config/prod/service-blue-green-25percent.yaml
   sleep 900  # Wait 15 minutes

   # 50% modern
   kubectl apply -f config/prod/service-blue-green-50percent.yaml
   sleep 900

   # 75% modern
   kubectl apply -f config/prod/service-blue-green-75percent.yaml
   sleep 900

   # 100% modern
   kubectl apply -f config/prod/service-blue-green-100percent.yaml
   ```

6. **Monitor During Transition**
   ```bash
   # Real-time dashboard
   watch -n 5 'python tools/kafka-maintenance-scheduler.py status --config config/prod-health.yaml'

   # Alert channel
   # Monitor Slack #kafka-migration for alerts
   ```

7. **Decommission Legacy (Blue) Environment**
   ```bash
   # After 24 hours of 100% modern traffic with no issues
   kubectl scale deployment/cryptofeed-legacy --replicas=0

   # After 7 days
   kubectl delete deployment/cryptofeed-legacy
   ```

### Expected Outcomes
- Modern backend handling 100% traffic
- Error rate < 0.1%
- Latency p99 < 5ms
- Zero downtime during transition

### Troubleshooting
**Issue**: Error rate spike during traffic shift
**Solution**: Pause traffic increase, investigate errors, consider rollback

**Issue**: Latency degradation
**Solution**: Check Kafka broker health, review network latency, optimize batch settings

---

## Canary Rollout

### Purpose
Alternative deployment strategy using canary pods for gradual rollout.

### Procedure

1. **Deploy Canary Pod**
   ```bash
   # Deploy single canary pod
   kubectl apply -f config/prod/kafka-modern-canary.yaml

   # Verify canary health
   kubectl get pods -l app=cryptofeed,canary=true
   ```

2. **Route 1% Traffic to Canary**
   ```bash
   # Use service mesh (Istio) for precise traffic control
   kubectl apply -f config/prod/virtualservice-canary-1percent.yaml
   ```

3. **Monitor Canary for 1 Hour**
   ```bash
   # Canary-specific metrics
   curl -s 'http://prometheus:9090/api/v1/query?query=kafka_publish_latency_seconds{pod=~".*canary.*",quantile="0.99"}'

   # Compare to baseline
   python scripts/compare-canary-metrics.py \
     --canary-pod cryptofeed-canary-xyz \
     --baseline-pod cryptofeed-legacy-abc
   ```

4. **Promote Canary to Full Rollout**
   ```bash
   # If canary successful, promote to full deployment
   kubectl apply -f config/prod/kafka-modern-deployment-full.yaml

   # Follow blue-green procedure above for gradual traffic shift
   ```

### Expected Outcomes
- Canary pod performs identically to baseline
- Issues detected early with minimal blast radius
- Confidence in full rollout

---

## Validation and Verification

### Purpose
Verify migration success and validate system behavior.

### Procedure

1. **Functional Validation**
   ```bash
   # End-to-end message flow test
   python scripts/test-message-flow.py \
     --bootstrap-servers kafka.prod:9092 \
     --topic cryptofeed.trades \
     --count 1000

   # Verify message delivery
   kafka-console-consumer --bootstrap-server kafka.prod:9092 \
     --topic cryptofeed.trades \
     --from-beginning --max-messages 10
   ```

2. **Performance Validation**
   ```bash
   # Latency check
   python scripts/measure-latency.py --duration=300  # 5 minutes

   # Expected: p99 < 5ms
   ```

3. **Data Integrity Validation**
   ```python
   # Compare message schemas
   from cryptofeed.backends.kafka.maintenance.integration import MaintenanceCoordinator

   coordinator = MaintenanceCoordinator()
   status = coordinator.get_system_status()

   # Verify no data loss
   assert status['messages_published'] == status['messages_delivered']
   ```

4. **Configuration Validation**
   ```bash
   # Verify new configuration applied
   kubectl get configmap cryptofeed-config -o yaml

   # Check environment variables
   kubectl exec -it cryptofeed-pod -- env | grep KAFKA
   ```

5. **Documentation Update**
   ```bash
   # Update deployment docs
   git add docs/deployment/kafka-backend.md
   git commit -m "docs: update Kafka backend to modern implementation"

   # Update architecture diagrams
   git add docs/architecture/kafka-flow.png
   git commit -m "docs: update Kafka architecture diagram"
   ```

### Expected Outcomes
- All validations pass
- Performance meets SLAs
- Documentation up to date

---

## Rollback Procedures

### Purpose
Safely rollback to legacy backend if issues arise.

### Rollback Triggers

**Automatic Rollback** (if configured):
- Error rate > 1% for 5 minutes
- Latency p99 > 100ms for 10 minutes
- Availability < 99% for 5 minutes

**Manual Rollback Decision**:
- Critical bugs discovered
- Data integrity issues
- Stakeholder request
- Unforeseen production impact

### Rollback Procedure

1. **Immediate Traffic Shift**
   ```bash
   # Shift 100% traffic back to legacy (blue)
   kubectl apply -f config/prod/service-blue-green-0percent.yaml

   # Verify routing
   curl http://cryptofeed-service/health | jq .backend_version
   # Should return "legacy"
   ```

2. **Verify Legacy Functionality**
   ```bash
   # Health check
   python tools/kafka-maintenance-scheduler.py run-once \
     --config config/prod-health-legacy.yaml

   # Message flow test
   python scripts/test-message-flow.py --implementation=legacy
   ```

3. **Investigate Root Cause**
   ```bash
   # Collect logs
   kubectl logs -l app=cryptofeed,version=modern --tail=1000 > rollback-logs.txt

   # Export metrics
   curl -s 'http://prometheus:9090/api/v1/query_range?query=kafka_publish_errors_total{version="modern"}&start=...' > rollback-metrics.json

   # Create incident report
   python scripts/generate-incident-report.py \
     --start-time "2025-11-26T10:00:00Z" \
     --end-time "2025-11-26T11:00:00Z" \
     --output incident-rollback.md
   ```

4. **Communication**
   ```text
   # Notify stakeholders
   Subject: [INCIDENT] Kafka Backend Migration Rollback

   Team,

   We have rolled back the Kafka backend migration due to [reason].

   Timeline:
   - 10:00 - Migration started
   - 10:30 - [Issue detected]
   - 10:35 - Rollback initiated
   - 10:40 - Rollback complete, traffic restored

   Current status: All systems operational on legacy backend

   Next steps:
   - Root cause analysis scheduled for [date/time]
   - Revised migration plan to be shared by [date]

   Questions? Contact [team lead]
   ```

5. **Schedule Post-Mortem**
   - Within 24 hours of rollback
   - Blameless retrospective
   - Document learnings and improvements
   - Update migration plan

### Expected Outcomes
- Service restored within 5 minutes
- No data loss
- Root cause identified
- Prevention plan created

---

## Success Criteria

### Migration Success Metrics
- **Downtime**: Zero
- **Error Rate**: <0.1% throughout migration
- **Latency**: p99 < 5ms maintained
- **Data Loss**: Zero messages lost
- **Rollback**: Not required

### Post-Migration Targets (30 days)
- **Availability**: >99.9%
- **Performance**: All SLAs met
- **User Issues**: <5 migration-related tickets
- **Team Confidence**: High (>8/10 survey score)

## Related Documentation
- [Migration Guide](../kafka/MIGRATION_GUIDE.md)
- [Configuration Migration Tool](../kafka/config-translation-examples.md)
- [Rollback Procedures](../kafka/rollback-procedures.md)
- [Health Monitoring Runbook](kafka-backend-health-monitoring.md)

## Changelog
- 2025-11-26: Initial runbook created (Task 6.3)
