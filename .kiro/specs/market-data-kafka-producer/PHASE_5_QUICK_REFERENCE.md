# Phase 5 Execution - Quick Reference Guide

**Status**: READY FOR EXECUTION
**Timeline**: 4 weeks + 2 weeks standby
**Strategy**: Blue-Green Cutover

---

## 🚀 Quick Start

### Pre-Execution Checklist (5 minutes)

```bash
# 1. Verify all Phase 5 materials present
ls -1 .kiro/specs/market-data-kafka-producer/PHASE_5_*.md
# Expected: 4 files (DESIGN, TASKS, MIGRATION_PLAN, EXECUTION_PLAN)

# 2. Verify tests passing
pytest tests/ -v --tb=short
# Expected: 493+ tests passing (100%)

# 3. Verify branch status
git status
# Expected: On branch 'next', clean working tree

# 4. Review execution plan
cat .kiro/specs/market-data-kafka-producer/PHASE_5_EXECUTION_PLAN.md
```

---

## 📋 Git Workflow (4 Atomic Commits)

### Commit 1: Specification Finalization (30 min)

```bash
git add .kiro/specs/market-data-kafka-producer/spec.json
git commit -m "docs(spec): Finalize Phase 5 execution specification"
```

**Changes**:
- spec.json: status → "phase-5-ready-for-execution"
- Add execution_plan reference
- Update success criteria

### Commit 2: Execution Materials (1 hour)

```bash
git add .kiro/specs/market-data-kafka-producer/PHASE_5_*.md
git commit -m "docs(phase5): Complete execution support materials"
```

**Changes**:
- PHASE_5_EXECUTION_PLAN.md (NEW - 2,109 lines)
- PHASE_5_DESIGN.md (mark FINAL)
- PHASE_5_TASKS.md (mark FINAL)
- PHASE_5_MIGRATION_PLAN.md (mark FINAL)

### Commit 3: Team Handoff (2 hours)

```bash
git add .kiro/specs/market-data-kafka-producer/handoff/
git commit -m "docs(handoff): Phase 5 execution team handoff materials"
```

**Changes**:
- 8 handoff documents (Week 1-4 guides, runbooks, procedures)

### Commit 4: PR Preparation (1 hour)

```bash
git push origin next
gh pr create --base main --head next \
  --title "Phase 5 Execution Materials - Production Ready" \
  --body-file .kiro/specs/market-data-kafka-producer/PR_DESCRIPTION.md
```

---

## 📅 Weekly Timeline

### Week 1: Infrastructure Setup (Tasks 20-21)

**Goal**: Deploy new backend, setup monitoring, validate infrastructure

**Days**:
- Mon: Kafka topic creation scripts (Task A)
- Tue: Deployment verification (Task B)
- Wed: Consumer templates (Task C, Part 1)
- Thu: Consumer templates + monitoring (Task C+D)
- Fri: Monitoring completion + validation (Task D)

**Exit Criteria**:
- [ ] All topics created (O(20))
- [ ] Staging + production deployed (100%)
- [ ] Consumer templates working (3 types)
- [ ] Monitoring operational (Prometheus + Grafana)

### Week 2: Consumer Validation (Tasks 22-23)

**Goal**: Validate consumer subscriptions, deploy monitoring dashboard

**Days**:
- Mon-Tue: Consumer subscription updates (staging)
- Wed-Thu: Monitoring dashboard deployment (production)
- Fri: Week 2 validation + Week 3 prep

**Exit Criteria**:
- [ ] All consumer types validated
- [ ] Monitoring dashboard operational
- [ ] Alert rules configured and tested
- [ ] Week 3 migration plan approved

### Week 3: Per-Exchange Migration (Tasks 24-25) - CRITICAL

**Goal**: Migrate consumers incrementally, 1 exchange/day

**Days**:
- Mon: Coinbase (10:00-14:00 UTC)
- Tue: Binance (10:00-14:00 UTC)
- Wed: OKX (10:00-14:00 UTC)
- Thu: Kraken + Bybit (10:00-14:00 UTC)
- Fri: Remaining exchanges (10:00-16:00 UTC)

**Exit Criteria**:
- [ ] All exchanges migrated
- [ ] All success criteria met (per exchange)
- [ ] Zero rollbacks (or documented and resolved)
- [ ] Monitoring shows stable metrics

### Week 4: Stabilization (Tasks 26-28)

**Goal**: Monitor stability, cleanup legacy, validate final success

**Days**:
- Mon-Wed: 72-hour stability monitoring
- Thu: Legacy topic decommissioning
- Fri: Post-migration validation

**Exit Criteria**:
- [ ] 72-hour stability (no P0/P1 incidents)
- [ ] Legacy topics archived and deleted
- [ ] Post-migration validation complete
- [ ] All success criteria met (10/10)

### Weeks 5-6: Legacy Standby (Task 29)

**Goal**: Maintain rollback capability, execute final cleanup

**Timeline**:
- Week 5: Keep 10% legacy producers on standby
- Week 6: Final cleanup + postmortem publication

---

## ✅ Success Criteria (10 Metrics)

| # | Criterion | Target | Validation |
|---|-----------|--------|------------|
| 1 | Message Loss | Zero | Message count ratio 1:1 (±0.1%) |
| 2 | Consumer Lag | <5s | Prometheus metric |
| 3 | Error Rate | <0.1% | DLQ ratio |
| 4 | Latency (p99) | <5ms | Percentile calculation |
| 5 | Throughput | ≥100k msg/s | Messages/sec metric |
| 6 | Data Integrity | 100% | Hash validation (1000 messages) |
| 7 | Monitoring | Functional | Dashboard + alerts operational |
| 8 | Rollback Time | <5min | Procedure execution time |
| 9 | Topic Count | O(20) | vs O(10K+) legacy |
| 10 | Headers Present | 100% | All messages have headers |

---

## 🚨 Emergency Procedures

### Rollback (<5 minutes)

**Trigger**: Error rate >1% OR consumer lag >30s OR P0/P1 incident

```bash
# Step 1: Pause new topic production (T+0min)
kubectl set env deployment/kafka-producer KAFKA_CALLBACK_ENABLED=false

# Step 2: Revert consumer subscriptions (T+1min)
kubectl apply -f k8s/consumers-legacy-config.yaml

# Step 3: Redeploy consumers (T+2min)
kubectl rollout restart deployment/kafka-consumers

# Step 4: Verify consumers reconnected (T+3min)
kubectl logs -l app=kafka-consumers --tail=100 | grep "Subscribed to topics"

# Step 5: Monitor consumer lag (T+4min)
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group cryptofeed-consumers

# Step 6: Confirm rollback success (T+5min)
# Expected: Consumer lag decreasing, error rate <0.1%
```

### Escalation

- **L1 (SRE)**: Slack #sre-oncall, PagerDuty (<5min response)
- **L2 (DevOps + Engineering)**: Slack #eng-oncall, PagerDuty (<5min response)
- **L3 (Engineering Lead)**: Slack #eng-leads, Email (<10min response)

---

## 📊 Monitoring Dashboard

### Key Metrics

1. **Message Throughput**: ≥100k msg/s (per exchange)
2. **Latency Distribution**: p50, p95, p99 (<5ms target)
3. **Error Rate**: DLQ ratio (<0.1%)
4. **Consumer Lag**: All groups <5s
5. **Kafka Broker Metrics**: CPU, memory, disk
6. **Producer Health**: Circuit breaker status
7. **Topic Partition Metrics**: Partition distribution
8. **Alert History**: Recent alert firing

### Dashboard URL

```
Production: http://grafana.example.com/d/kafka-producer
Staging: http://grafana-staging.example.com/d/kafka-producer
```

---

## 📞 Team Responsibilities

| Team | Week 1 | Week 2 | Week 3 | Week 4 |
|------|--------|--------|--------|--------|
| **DevOps** | Infrastructure (A-B) | - | - | Legacy cleanup |
| **Engineering** | Consumer templates (C) | Consumer prep | Migration | - |
| **SRE** | Monitoring (D) | Dashboard deploy | Migration support | Stability |
| **QA** | Testing (A.5-D.5) | Validation | Per-exchange validation | Post-migration |

---

## 🎯 Daily Standup (Week 1-4)

**Time**: 10:00 UTC daily (15 minutes)
**Channel**: #data-engineering Slack
**Agenda**:
1. Yesterday's progress (what completed?)
2. Today's plan (what executing?)
3. Blockers (what needs escalation?)
4. Success metrics (are we on track?)

---

## 📚 Reference Documents

### Execution Materials

- **PHASE_5_EXECUTION_PLAN.md**: Strategic execution plan (2,109 lines) - Full details
- **PHASE_5_DESIGN.md**: Technical design (1,549 lines) - Task specifications
- **PHASE_5_TASKS.md**: Implementation tasks (1,291 lines) - Sub-task details
- **PHASE_5_MIGRATION_PLAN.md**: Week-by-week guide (382 lines) - Migration procedures
- **PHASE_5_QUICK_REFERENCE.md**: This document - Quick reference

### Core Specification

- **spec.json**: Metadata and phase status
- **requirements.md**: Functional and non-functional requirements
- **design.md**: Architecture and components
- **tasks.md**: 28 tasks across 5 phases

### Operational Guides

- **handoff/WEEK_1_DEPLOYMENT_GUIDE.md**: Week 1 procedures
- **handoff/OPERATIONAL_RUNBOOK.md**: Deployment + rollback procedures
- **handoff/ESCALATION_MATRIX.md**: On-call escalation
- **handoff/TEAM_RESPONSIBILITIES.md**: Team ownership

---

## 🔍 Validation Commands

### Pre-Migration

```bash
# Check tests passing
pytest tests/ -v --tb=short | grep -E "passed|failed"

# Check Kafka cluster health
kafka-broker-api-versions.sh --bootstrap-server localhost:9092

# Check topic list (legacy)
kafka-topics.sh --bootstrap-server localhost:9092 --list | grep cryptofeed | wc -l
```

### During Migration (Week 3)

```bash
# Check consumer lag (per exchange)
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group cryptofeed-consumers

# Check error rate (DLQ)
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic cryptofeed.dlq --from-beginning | wc -l

# Check message headers
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic cryptofeed.trades --max-messages 10 | jq '.headers'
```

### Post-Migration (Week 4)

```bash
# Validate latency (Prometheus)
curl -s 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=histogram_quantile(0.99, kafka_producer_latency_bucket)'

# Validate throughput (Prometheus)
curl -s 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=rate(kafka_producer_messages_total[1m])'

# Validate topic count
kafka-topics.sh --bootstrap-server localhost:9092 --list | grep -E "^cryptofeed\." | wc -l
# Expected: ~20 (vs 10K+ legacy)
```

---

## 🎓 Training & Preparation

### Week 0 (Before Execution)

**DevOps Team**:
- [ ] Review Kafka topic creation scripts (Task A)
- [ ] Test deployment verification procedures (Task B)
- [ ] Practice rollback procedure (<5 minutes target)

**Engineering Team**:
- [ ] Review consumer templates (Task C)
- [ ] Test consumer subscriptions in staging
- [ ] Understand partition strategies

**SRE Team**:
- [ ] Review monitoring setup (Task D)
- [ ] Test Prometheus + Grafana configuration
- [ ] Understand alert rules and escalation

**QA Team**:
- [ ] Review validation checklists (per exchange)
- [ ] Understand success criteria (10 metrics)
- [ ] Practice validation procedures

---

## 🏁 Go/No-Go Checklist

**Before Week 1 Execution**:

- [ ] All Phase 1-4 code merged to main
- [ ] 493+ tests passing (100% pass rate)
- [ ] Kafka cluster ready (3+ brokers, healthy)
- [ ] Monitoring infrastructure ready (Prometheus + Grafana)
- [ ] Consumer applications ready for redeployment
- [ ] On-call rotation scheduled (L1/L2/L3)
- [ ] Stakeholders notified (timeline + expected impact)
- [ ] Staging cluster available for validation
- [ ] Rollback procedure tested (<5 minutes)
- [ ] Team handoff materials reviewed
- [ ] Communication plan finalized (Slack, email)
- [ ] Git workflow approved (4 atomic commits)

**Go/No-Go Decision**: [PENDING REVIEW]

---

## 📈 Expected Outcomes

### Operational Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Topic Count | O(10K+) | O(20) | 99.8% reduction |
| Message Size | JSON (100%) | Protobuf (37%) | 63% reduction |
| Latency (p99) | Unknown | <5ms | Validated |
| Throughput | Unknown | 150k+ msg/s | Validated |
| Monitoring | None | 9 metrics | New capability |
| Partition Strategies | 1 | 4 | +3 options |
| Configuration | Dict | Pydantic | Type-safe |

### Migration Benefits

- **Infrastructure**: Reduced Kafka metadata (99.8% fewer topics)
- **Performance**: Lower latency (<5ms p99), higher throughput (150k+ msg/s)
- **Monitoring**: Comprehensive observability (9 metrics, dashboard, alerts)
- **Reliability**: Exactly-once semantics, circuit breaker, DLQ handling
- **Developer Experience**: Type-safe configuration, clear migration guides

---

**Document Version**: 1.0.0
**Created**: November 13, 2025
**Status**: READY FOR TEAM REFERENCE
**Next Review**: Pre-Week 1 execution kickoff

**Quick Links**:
- Full Execution Plan: PHASE_5_EXECUTION_PLAN.md
- Technical Design: PHASE_5_DESIGN.md
- Implementation Tasks: PHASE_5_TASKS.md
- Migration Plan: PHASE_5_MIGRATION_PLAN.md

