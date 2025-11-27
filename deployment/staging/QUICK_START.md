# Task 20 Quick Start Guide

**For Operations Team - Staging Deployment**

---

## Prerequisites (5 minutes)

1. **Configure environment variables**:
   ```bash
   cd /path/to/cryptofeed
   cp .env.production.template .env.production
   # Edit .env.production with actual values
   source .env.production
   ```

2. **Verify all required variables set**:
   ```bash
   ./scripts/validate-environment.sh
   ```
   Expected: All checkmarks green

---

## Deployment (6 hours)

### Step 1: Pre-Deployment Validation (30 min)

```bash
./scripts/validate-staging-deployment.sh
```

**Expected**: All checks pass
**If fails**: Stop, fix issues, retry

---

### Step 2: Run Deployment (6 hours)

```bash
./scripts/deploy-staging-kafka-callback.sh
```

**Follow prompts**:
- Confirm readiness
- Deploy to 10%, monitor 2 hours
- Expand to 50%, monitor 2 hours
- Complete to 100%

**Monitoring**:
```bash
# In separate terminal
./scripts/health-check-staging.sh --interval 30
```

**Watch Grafana**: `${GRAFANA_URL}/d/kafka-producer-staging`

---

### Step 3: Post-Deployment Validation (1 hour)

```bash
./scripts/validate-post-deployment.sh
```

**Expected**: All checks pass
**If fails**: Consider rollback

---

## Rollback (if needed, <5 min)

```bash
./scripts/rollback-staging-deployment.sh
```

**Triggers**:
- Error rate >0.1%
- Latency p99 >5ms
- Broker CPU >90%
- Consumer errors

---

## Success Criteria

- [ ] Error rate <0.1% for 2 hours
- [ ] Latency p99 <5ms for 2 hours
- [ ] Broker CPU <80% for 2-4 hours
- [ ] Broker Memory <80% for 2-4 hours
- [ ] Message headers validated
- [ ] Protobuf serialization confirmed
- [ ] Consumer teams validated

---

## Key Files

- **Configuration**: `deployment/staging/kafka-callback-config.yaml`
- **Runbook**: `deployment/staging/DEPLOYMENT_RUNBOOK.md`
- **Summary**: `deployment/staging/TASK20_IMPLEMENTATION_SUMMARY.md`

---

## Support

**Issues?**
- Check logs in deployment/rollback directories
- Review `DEPLOYMENT_RUNBOOK.md` troubleshooting section
- Contact on-call engineer

---

**Total Time**: 6-7 hours deployment + 2-4 hours monitoring
**Rollback Time**: <5 minutes
**Ready?** Review `DEPLOYMENT_RUNBOOK.md` for detailed instructions
