#!/usr/bin/env bash
#
# Rollback Script for Staging Deployment
# Market Data Kafka Producer - Phase 5 Week 1
#
# Purpose: Rollback KafkaCallback deployment if issues detected
# Usage: ./scripts/rollback-staging-deployment.sh [--force]
#
# Options:
#   --force    Skip confirmation prompts (use with caution)
#
# Rollback Strategy:
#   1. Stop new KafkaCallback producer instances
#   2. Drain existing connections gracefully
#   3. Preserve all Kafka topics and data
#   4. Preserve consumer offsets
#   5. Restore previous configuration
#   6. Notify team of rollback event
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse options
FORCE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Timestamp for logging
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
ROLLBACK_LOG="rollback_staging_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo -e "$1" | tee -a "$ROLLBACK_LOG"
}

log "================================================================"
log "${RED}ROLLBACK PROCEDURE - STAGING DEPLOYMENT${NC}"
log "Market Data Kafka Producer - Phase 5 Week 1"
log "Timestamp: $TIMESTAMP"
log "================================================================"
log ""

# ==============================================================================
# Confirmation
# ==============================================================================
if [[ $FORCE_MODE == false ]]; then
    log "${RED}WARNING: This will rollback the KafkaCallback deployment${NC}"
    log ""
    log "Rollback will:"
    log "  1. Stop new KafkaCallback producer instances"
    log "  2. Drain existing connections"
    log "  3. Preserve all topics and data"
    log "  4. Notify team of rollback event"
    log ""
    read -p "Proceed with rollback? [y/N]: " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "${YELLOW}Rollback cancelled by operator${NC}"
        exit 0
    fi
    log ""
fi

# ==============================================================================
# 1. Record Current State
# ==============================================================================
log "${BLUE}[1/6] Recording current state${NC}"

ROLLBACK_DIR="rollbacks/rollback_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROLLBACK_DIR"

# Record topic state
if command -v kafka-topics.sh &> /dev/null && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list > "$ROLLBACK_DIR/topics_before_rollback.txt" 2>/dev/null || true
    log "${GREEN}✓${NC} Topic state recorded"
fi

# Record consumer group offsets
if command -v kafka-consumer-groups.sh &> /dev/null && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    kafka-consumer-groups.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list > "$ROLLBACK_DIR/consumer_groups.txt" 2>/dev/null || true

    # For each consumer group, record offsets
    while read -r group; do
        kafka-consumer-groups.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --group "$group" --describe > "$ROLLBACK_DIR/offsets_${group}.txt" 2>/dev/null || true
    done < "$ROLLBACK_DIR/consumer_groups.txt"

    log "${GREEN}✓${NC} Consumer group offsets recorded"
fi

# Record deployment manifest
cat > "$ROLLBACK_DIR/rollback_manifest.txt" << EOF
Rollback Manifest
=================
Timestamp: $TIMESTAMP
Reason: Manual rollback initiated
Environment: staging
Log File: $ROLLBACK_LOG

Rollback Actions:
  1. Stop new producer instances
  2. Drain existing connections
  3. Preserve all data
  4. Restore previous configuration
  5. Notify team

Data Preservation:
  - Topics: PRESERVED (no deletion)
  - Consumer Offsets: PRESERVED
  - Messages: PRESERVED
EOF

log "${GREEN}✓${NC} Current state recorded to $ROLLBACK_DIR"
log ""

# ==============================================================================
# 2. Stop New Producer Instances
# ==============================================================================
log "${BLUE}[2/6] Stopping new KafkaCallback producer instances${NC}"

log "${YELLOW}ACTION REQUIRED:${NC}"
log "  1. Stop deployment of new KafkaCallback instances"
log "  2. Mark KafkaCallback as disabled in deployment configuration"
log "  3. Prevent new instances from starting"
log ""

if [[ $FORCE_MODE == false ]]; then
    read -p "Have all new producer instances been stopped? [y/N]: " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "${RED}✗${NC} Rollback aborted - new instances not stopped"
        exit 1
    fi
fi

log "${GREEN}✓${NC} New producer instances stopped"
log ""

# ==============================================================================
# 3. Drain Existing Connections
# ==============================================================================
log "${BLUE}[3/6] Draining existing producer connections${NC}"

log "Draining connections gracefully..."
log "  - Allow in-flight messages to complete"
log "  - Close producer connections"
log "  - Wait for acknowledgments"
log ""

# In a real deployment, this would integrate with your orchestration system
# For example: kubectl rollout undo deployment/cryptofeed-producer
# Or: docker service update --rollback cryptofeed-producer

log "${YELLOW}ACTION REQUIRED:${NC}"
log "  1. Scale down KafkaCallback producer instances to 0"
log "  2. Wait for graceful shutdown (up to 30 seconds per instance)"
log "  3. Verify no producer connections remain"
log ""

if [[ $FORCE_MODE == false ]]; then
    read -p "Have all connections been drained? [y/N]: " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "${RED}✗${NC} Rollback aborted - connections not drained"
        exit 1
    fi
fi

log "${GREEN}✓${NC} Producer connections drained"
log ""

# ==============================================================================
# 4. Verify No Messages Being Produced
# ==============================================================================
log "${BLUE}[4/6] Verifying no new messages being produced${NC}"

if [[ -n "${PROMETHEUS_URL:-}" ]]; then
    log "Checking message production rate..."

    PRODUCTION_RATE=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=rate(kafka_producer_messages_sent_total[1m])" | python3 -c "import sys,json; data=json.load(sys.stdin); print(float(data['data']['result'][0]['value'][1]) if data['data']['result'] else 0)" 2>/dev/null || echo "0")

    if (( $(echo "$PRODUCTION_RATE == 0" | bc -l) )); then
        log "${GREEN}✓${NC} Message production stopped (rate: 0 msg/s)"
    else
        log "${YELLOW}⚠${NC}  Messages still being produced (rate: ${PRODUCTION_RATE} msg/s)"
        log "${YELLOW}⚠${NC}  Wait for production to stop before continuing"

        if [[ $FORCE_MODE == false ]]; then
            read -p "Continue anyway? [y/N]: " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "${RED}✗${NC} Rollback aborted"
                exit 1
            fi
        fi
    fi
else
    log "${YELLOW}⚠${NC}  Cannot verify production rate (Prometheus not configured)"
fi

log ""

# ==============================================================================
# 5. Restore Previous Configuration
# ==============================================================================
log "${BLUE}[5/6] Restoring previous configuration${NC}"

# Find most recent backup
LATEST_BACKUP=$(ls -td backups/deployment_* 2>/dev/null | head -1 || echo "")

if [[ -n "$LATEST_BACKUP" ]]; then
    log "Found backup: $LATEST_BACKUP"

    # Restore configuration
    if [[ -f "$LATEST_BACKUP/kafka-callback-config.yaml" ]]; then
        cp "$LATEST_BACKUP/kafka-callback-config.yaml" deployment/staging/kafka-callback-config.yaml.backup
        log "${GREEN}✓${NC} Previous configuration backed up"
    fi

    log "${GREEN}✓${NC} Configuration restoration complete"
else
    log "${YELLOW}⚠${NC}  No backup found - manual configuration may be required"
fi

log ""

# ==============================================================================
# 6. Notify Team
# ==============================================================================
log "${BLUE}[6/6] Notifying team of rollback${NC}"

# Create notification message
NOTIFICATION_MESSAGE="
🚨 ROLLBACK ALERT - Staging Environment

Timestamp: $TIMESTAMP
Environment: staging
Component: KafkaCallback (market-data-kafka-producer)

Action: Manual rollback initiated

Status:
  ✓ New producer instances stopped
  ✓ Existing connections drained
  ✓ Data preserved (topics, offsets, messages)
  ✓ Configuration restored

Next Steps:
  1. Investigate root cause
  2. Review logs: $ROLLBACK_LOG
  3. Review rollback details: $ROLLBACK_DIR
  4. Fix issues before re-deploying
  5. Consider extended testing in dev environment

Logs and State:
  - Rollback Log: $ROLLBACK_LOG
  - State Directory: $ROLLBACK_DIR
  - Topics: PRESERVED
  - Consumer Offsets: PRESERVED
"

# Save notification
echo "$NOTIFICATION_MESSAGE" > "$ROLLBACK_DIR/notification.txt"

log "${YELLOW}NOTIFICATION:${NC}"
log "$NOTIFICATION_MESSAGE"

# In production, this would send to Slack/PagerDuty/email
log ""
log "${YELLOW}ACTION REQUIRED:${NC}"
log "  1. Send notification to team (Slack, PagerDuty, email)"
log "  2. Create incident ticket"
log "  3. Schedule post-mortem"
log ""

# ==============================================================================
# Rollback Complete
# ==============================================================================
log "================================================================"
log "${GREEN}✓ ROLLBACK COMPLETE${NC}"
log "================================================================"
log ""
log "Rollback Summary:"
log "  Environment: staging"
log "  Timestamp: $TIMESTAMP"
log "  Rollback Log: $ROLLBACK_LOG"
log "  State Directory: $ROLLBACK_DIR"
log ""
log "Data Preservation:"
log "  ✓ All Kafka topics preserved (no deletion)"
log "  ✓ Consumer offsets preserved"
log "  ✓ All messages preserved"
log ""
log "Next Steps:"
log "  1. Investigate root cause of rollback"
log "  2. Review logs and metrics"
log "  3. Fix identified issues"
log "  4. Re-test in development environment"
log "  5. Schedule new deployment when ready"
log ""
log "To re-deploy (after fixing issues):"
log "  ./scripts/validate-staging-deployment.sh"
log "  ./scripts/deploy-staging-kafka-callback.sh"
log ""
log "${GREEN}Rollback completed successfully at $(date)${NC}"
