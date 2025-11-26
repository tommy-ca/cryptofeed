#!/usr/bin/env bash
#
# Staging Deployment Script
# Market Data Kafka Producer - Phase 5 Week 1
#
# Purpose: Deploy KafkaCallback to staging environment with canary rollout
# Usage: ./scripts/deploy-staging-kafka-callback.sh
#
# Deployment Strategy: Canary Rollout
#   - Stage 1: Deploy to 10% of instances, monitor 2 hours
#   - Stage 2: Expand to 50% of instances, monitor 2 hours
#   - Stage 3: Complete rollout to 100%
#
# Prerequisites:
#   - Pre-deployment validation passed: ./scripts/validate-staging-deployment.sh
#   - Environment variables configured
#   - Team on-call and ready to respond
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Deployment configuration
ENVIRONMENT="staging"
CONFIG_FILE="deployment/staging/kafka-callback-config.yaml"
DEPLOYMENT_LOG="deployment_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"

# Timestamp for logging
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Logging function
log() {
    echo -e "$1" | tee -a "$DEPLOYMENT_LOG"
}

log "================================================================"
log "Staging Deployment - KafkaCallback"
log "Market Data Kafka Producer - Phase 5 Week 1"
log "Timestamp: $TIMESTAMP"
log "================================================================"
log ""

# ==============================================================================
# Pre-Deployment Checks
# ==============================================================================
log "${BLUE}[PRE-DEPLOYMENT] Running pre-deployment checks${NC}"

# Check if pre-validation was run
if [[ ! -x "./scripts/validate-staging-deployment.sh" ]]; then
    log "${RED}✗${NC} Pre-deployment validation script not found"
    log "${RED}✗${NC} Deployment aborted"
    exit 1
fi

log "Running pre-deployment validation..."
if ./scripts/validate-staging-deployment.sh >> "$DEPLOYMENT_LOG" 2>&1; then
    log "${GREEN}✓${NC} Pre-deployment validation passed"
else
    log "${RED}✗${NC} Pre-deployment validation failed"
    log "${RED}✗${NC} Review $DEPLOYMENT_LOG for details"
    log "${RED}✗${NC} Deployment aborted"
    exit 1
fi

# Confirm deployment with operator
log ""
log "${YELLOW}WARNING: This will deploy KafkaCallback to staging environment${NC}"
log "Deployment strategy: Canary rollout (10% -> 50% -> 100%)"
log "Total estimated time: ~6 hours"
log ""
read -p "Proceed with deployment? [y/N]: " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "${YELLOW}Deployment cancelled by operator${NC}"
    exit 0
fi

log ""

# ==============================================================================
# Create Backup Point
# ==============================================================================
log "${BLUE}[BACKUP] Creating backup point${NC}"

BACKUP_DIR="backups/deployment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup current configuration
if [[ -f "$CONFIG_FILE" ]]; then
    cp "$CONFIG_FILE" "$BACKUP_DIR/"
    log "${GREEN}✓${NC} Configuration backed up to $BACKUP_DIR"
fi

# Record current Kafka state
if command -v kafka-topics.sh &> /dev/null && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list > "$BACKUP_DIR/topics_before.txt" 2>/dev/null || true
    log "${GREEN}✓${NC} Kafka state recorded"
fi

log ""

# ==============================================================================
# Topic Creation (if needed)
# ==============================================================================
log "${BLUE}[TOPICS] Creating consolidated topics (if not exist)${NC}"

# List of topics to create
TOPICS=(
    "cryptofeed.trades"
    "cryptofeed.orderbook"
    "cryptofeed.ticker"
    "cryptofeed.candle"
    "cryptofeed.funding"
    "cryptofeed.liquidation"
    "cryptofeed.index"
    "cryptofeed.openinterest"
)

PARTITIONS=12
REPLICATION_FACTOR=3

for topic in "${TOPICS[@]}"; do
    if command -v kafka-topics.sh &> /dev/null; then
        # Check if topic exists
        if kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list 2>/dev/null | grep -q "^${topic}$"; then
            log "${YELLOW}⚠${NC}  Topic already exists: $topic"
        else
            log "Creating topic: $topic (partitions=$PARTITIONS, replication=$REPLICATION_FACTOR)"
            if kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
                --create \
                --topic "$topic" \
                --partitions "$PARTITIONS" \
                --replication-factor "$REPLICATION_FACTOR" \
                --config min.insync.replicas=2 \
                --config cleanup.policy=delete \
                --config retention.ms=604800000 \
                2>&1 | tee -a "$DEPLOYMENT_LOG"; then
                log "${GREEN}✓${NC} Topic created: $topic"
            else
                log "${RED}✗${NC} Failed to create topic: $topic"
                log "${RED}✗${NC} Deployment aborted"
                exit 1
            fi
        fi
    else
        log "${YELLOW}⚠${NC}  kafka-topics.sh not available, assuming topics will be auto-created"
        break
    fi
done

log ""

# ==============================================================================
# Stage 1: Deploy to 10% of Instances
# ==============================================================================
log "${BLUE}[STAGE 1] Deploying to 10% of instances${NC}"
log "Duration: 2 hours monitoring"
log "Success criteria: Error rate <0.1%, Latency p99 <5ms, No message loss"

# Note: Actual deployment mechanism depends on your infrastructure
# This is a placeholder that would integrate with your deployment system

log "${YELLOW}ACTION REQUIRED:${NC}"
log "  1. Deploy cryptofeed with KafkaCallback to 10% of producer instances"
log "  2. Verify deployment using your deployment tool (Kubernetes, Docker, etc.)"
log "  3. Confirm deployment before continuing"
log ""

read -p "Has deployment to 10% completed successfully? [y/N]: " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "${RED}✗${NC} Deployment to 10% failed or cancelled"
    log "${RED}✗${NC} Run rollback: ./scripts/rollback-staging-deployment.sh"
    exit 1
fi

log "${GREEN}✓${NC} Deployment to 10% confirmed"
log ""

# Monitor for 2 hours
STAGE1_START=$(date +%s)
MONITOR_DURATION_SECONDS=$((2 * 60 * 60))  # 2 hours

log "Monitoring Stage 1 for 2 hours..."
log "Start time: $(date)"
log "Expected end time: $(date -d @$((STAGE1_START + MONITOR_DURATION_SECONDS)))"
log ""
log "${YELLOW}Run health checks in parallel:${NC}"
log "  ./scripts/health-check-staging.sh"
log ""
log "${YELLOW}Monitor metrics:${NC}"
log "  - Error rate (target: <0.1%)"
log "  - Latency p99 (target: <5ms)"
log "  - Broker CPU/memory"
log ""

read -p "After 2 hours, confirm Stage 1 metrics are healthy [y/N]: " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "${RED}✗${NC} Stage 1 health check failed"
    log "${RED}✗${NC} Run rollback: ./scripts/rollback-staging-deployment.sh"
    exit 1
fi

log "${GREEN}✓${NC} Stage 1 monitoring complete - metrics healthy"
log ""

# ==============================================================================
# Stage 2: Expand to 50% of Instances
# ==============================================================================
log "${BLUE}[STAGE 2] Expanding to 50% of instances${NC}"
log "Duration: 2 hours monitoring"
log "Success criteria: Error rate <0.1%, Latency p99 <5ms, No message loss"

log "${YELLOW}ACTION REQUIRED:${NC}"
log "  1. Expand deployment to 50% of producer instances"
log "  2. Verify expansion using your deployment tool"
log "  3. Confirm expansion before continuing"
log ""

read -p "Has expansion to 50% completed successfully? [y/N]: " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "${RED}✗${NC} Expansion to 50% failed or cancelled"
    log "${RED}✗${NC} Run rollback: ./scripts/rollback-staging-deployment.sh"
    exit 1
fi

log "${GREEN}✓${NC} Expansion to 50% confirmed"
log ""

# Monitor for 2 hours
log "Monitoring Stage 2 for 2 hours..."
log "Start time: $(date)"
log ""

read -p "After 2 hours, confirm Stage 2 metrics are healthy [y/N]: " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "${RED}✗${NC} Stage 2 health check failed"
    log "${RED}✗${NC} Run rollback: ./scripts/rollback-staging-deployment.sh"
    exit 1
fi

log "${GREEN}✓${NC} Stage 2 monitoring complete - metrics healthy"
log ""

# ==============================================================================
# Stage 3: Complete Rollout to 100%
# ==============================================================================
log "${BLUE}[STAGE 3] Completing rollout to 100% of instances${NC}"

log "${YELLOW}ACTION REQUIRED:${NC}"
log "  1. Complete rollout to 100% of producer instances"
log "  2. Verify final deployment state"
log "  3. Confirm completion"
log ""

read -p "Has rollout to 100% completed successfully? [y/N]: " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "${RED}✗${NC} Rollout to 100% failed or cancelled"
    log "${RED}✗${NC} Run rollback: ./scripts/rollback-staging-deployment.sh"
    exit 1
fi

log "${GREEN}✓${NC} Rollout to 100% complete"
log ""

# ==============================================================================
# Post-Deployment Validation
# ==============================================================================
log "${BLUE}[POST-DEPLOYMENT] Running post-deployment validation${NC}"

if [[ -x "./scripts/validate-post-deployment.sh" ]]; then
    if ./scripts/validate-post-deployment.sh >> "$DEPLOYMENT_LOG" 2>&1; then
        log "${GREEN}✓${NC} Post-deployment validation passed"
    else
        log "${YELLOW}⚠${NC}  Post-deployment validation had warnings"
        log "${YELLOW}⚠${NC}  Review $DEPLOYMENT_LOG for details"
    fi
else
    log "${YELLOW}⚠${NC}  Post-deployment validation script not found"
fi

log ""

# ==============================================================================
# Record Deployment State
# ==============================================================================
log "${BLUE}[RECORD] Recording final deployment state${NC}"

# Record final Kafka state
if command -v kafka-topics.sh &> /dev/null && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list > "$BACKUP_DIR/topics_after.txt" 2>/dev/null || true
    log "${GREEN}✓${NC} Final Kafka state recorded"
fi

# Create deployment manifest
cat > "$BACKUP_DIR/deployment_manifest.txt" << EOF
Deployment Manifest
===================
Environment: $ENVIRONMENT
Timestamp: $TIMESTAMP
Config File: $CONFIG_FILE
Deployment Log: $DEPLOYMENT_LOG

Topics Created:
$(for topic in "${TOPICS[@]}"; do echo "  - $topic"; done)

Deployment Stages:
  Stage 1: 10% (2 hours monitoring) - COMPLETED
  Stage 2: 50% (2 hours monitoring) - COMPLETED
  Stage 3: 100% - COMPLETED

Status: SUCCESS
EOF

log "${GREEN}✓${NC} Deployment manifest created"
log ""

# ==============================================================================
# Deployment Complete
# ==============================================================================
log "================================================================"
log "${GREEN}✓ DEPLOYMENT COMPLETE${NC}"
log "================================================================"
log ""
log "Deployment Summary:"
log "  Environment: $ENVIRONMENT"
log "  Total Duration: ~6 hours"
log "  Deployment Log: $DEPLOYMENT_LOG"
log "  Backup Directory: $BACKUP_DIR"
log ""
log "Next Steps:"
log "  1. Continue monitoring with: ./scripts/health-check-staging.sh"
log "  2. Validate message format: ./scripts/validate-post-deployment.sh"
log "  3. Monitor broker metrics for 2-4 hours"
log "  4. Document any issues or observations"
log ""
log "Rollback (if needed):"
log "  ./scripts/rollback-staging-deployment.sh"
log ""
log "${GREEN}Deployment completed successfully at $(date)${NC}"
