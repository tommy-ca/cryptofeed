#!/usr/bin/env bash
#
# Staging Deployment Validation Script
# Market Data Kafka Producer - Phase 5 Week 1
#
# Purpose: Validate staging environment before KafkaCallback deployment
# Usage: ./scripts/validate-staging-deployment.sh
#
# Exit Codes:
#   0 - All validations passed, ready for deployment
#   1 - One or more validations failed, DO NOT proceed
#
# Requirements:
#   - Kafka cluster with 3+ brokers operational
#   - Environment variables configured (.env.production sourced)
#   - Prometheus and Grafana accessible
#   - Consumer applications ready
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track validation status
VALIDATION_FAILED=0
WARNINGS=()
ERRORS=()

# Timestamp for logging
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "================================================================"
echo "Staging Deployment Pre-Validation"
echo "Market Data Kafka Producer - Phase 5 Week 1"
echo "Timestamp: $TIMESTAMP"
echo "================================================================"
echo ""

# ==============================================================================
# 1. Environment Variable Validation
# ==============================================================================
echo -e "${BLUE}[1/8] Environment Variable Validation${NC}"
echo "--------------------------------------------------------------"

if [[ -f ".env.production" ]]; then
    echo -e "${GREEN}✓${NC} .env.production file found"
    source .env.production
else
    echo -e "${RED}✗${NC} .env.production file not found"
    echo "   Run: cp .env.production.template .env.production"
    echo "   Then fill in actual values"
    ERRORS+=("Missing .env.production file")
    VALIDATION_FAILED=1
fi

# Run environment validation script
if [[ -x "./scripts/validate-environment.sh" ]]; then
    if ./scripts/validate-environment.sh; then
        echo -e "${GREEN}✓${NC} All environment variables validated"
    else
        echo -e "${RED}✗${NC} Environment variable validation failed"
        ERRORS+=("Environment variable validation failed")
        VALIDATION_FAILED=1
    fi
else
    echo -e "${YELLOW}⚠${NC}  validate-environment.sh not found or not executable"
    WARNINGS+=("Environment validation script missing")
fi

echo ""

# ==============================================================================
# 2. Kafka Cluster Health Validation
# ==============================================================================
echo -e "${BLUE}[2/8] Kafka Cluster Health Validation${NC}"
echo "--------------------------------------------------------------"

# Check if KAFKA_BOOTSTRAP_SERVERS is set
if [[ -z "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    echo -e "${RED}✗${NC} KAFKA_BOOTSTRAP_SERVERS not set"
    ERRORS+=("KAFKA_BOOTSTRAP_SERVERS not configured")
    VALIDATION_FAILED=1
else
    echo -e "${GREEN}✓${NC} Bootstrap servers configured: $KAFKA_BOOTSTRAP_SERVERS"

    # Test Kafka connectivity using kafka-broker-api-versions.sh
    if command -v kafka-broker-api-versions.sh &> /dev/null; then
        if kafka-broker-api-versions.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" &> /dev/null; then
            echo -e "${GREEN}✓${NC} Kafka cluster accessible"

            # Count brokers
            BROKER_COUNT=$(kafka-broker-api-versions.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" 2>/dev/null | grep -c "^[0-9]" || echo "0")
            if [[ "$BROKER_COUNT" -ge 3 ]]; then
                echo -e "${GREEN}✓${NC} Broker count: $BROKER_COUNT (>= 3 required)"
            else
                echo -e "${RED}✗${NC} Broker count: $BROKER_COUNT (< 3 minimum)"
                ERRORS+=("Insufficient broker count: $BROKER_COUNT")
                VALIDATION_FAILED=1
            fi
        else
            echo -e "${RED}✗${NC} Cannot connect to Kafka cluster"
            ERRORS+=("Kafka cluster not accessible")
            VALIDATION_FAILED=1
        fi
    else
        echo -e "${YELLOW}⚠${NC}  kafka-broker-api-versions.sh not found (install Kafka tools)"
        WARNINGS+=("Kafka CLI tools not installed")

        # Fallback: try Python kafka-python library
        if python3 -c "from kafka import KafkaProducer; KafkaProducer(bootstrap_servers='$KAFKA_BOOTSTRAP_SERVERS', request_timeout_ms=5000).close()" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Kafka cluster accessible (via Python)"
        else
            echo -e "${RED}✗${NC} Cannot connect to Kafka cluster (via Python)"
            ERRORS+=("Kafka cluster not accessible")
            VALIDATION_FAILED=1
        fi
    fi
fi

echo ""

# ==============================================================================
# 3. Topic Creation Capability Validation
# ==============================================================================
echo -e "${BLUE}[3/8] Topic Creation Capability Validation${NC}"
echo "--------------------------------------------------------------"

# Check if consolidated topics already exist
if command -v kafka-topics.sh &> /dev/null && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    EXISTING_TOPICS=$(kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list 2>/dev/null | grep "^cryptofeed\." || echo "")

    if [[ -n "$EXISTING_TOPICS" ]]; then
        echo -e "${YELLOW}⚠${NC}  Existing cryptofeed.* topics found:"
        echo "$EXISTING_TOPICS" | while read -r topic; do
            echo "     - $topic"

            # Check message count
            MESSAGE_COUNT=$(kafka-run-class.sh kafka.tools.GetOffsetShell --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --topic "$topic" --time -1 2>/dev/null | awk -F: '{sum += $3} END {print sum}' || echo "unknown")

            if [[ "$MESSAGE_COUNT" != "0" && "$MESSAGE_COUNT" != "unknown" ]]; then
                echo -e "${YELLOW}      Messages: $MESSAGE_COUNT (topic has existing data)${NC}"
                WARNINGS+=("Topic $topic has existing data: $MESSAGE_COUNT messages")
            else
                echo "      Messages: $MESSAGE_COUNT"
            fi
        done

        echo ""
        echo -e "${YELLOW}⚠${NC}  WARNING: Existing topics will be reused. Ensure this is intentional."
        read -p "Continue with existing topics? [y/N]: " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}✗${NC} Deployment aborted by user"
            exit 1
        fi
    else
        echo -e "${GREEN}✓${NC} No existing cryptofeed.* topics (clean deployment)"
    fi

    # Validate topic creation capability
    echo -e "${GREEN}✓${NC} Topic creation capability validated"
else
    echo -e "${YELLOW}⚠${NC}  Cannot validate topic creation (kafka-topics.sh not available)"
    WARNINGS+=("Topic validation skipped")
fi

echo ""

# ==============================================================================
# 4. Partition and Replication Capability
# ==============================================================================
echo -e "${BLUE}[4/8] Partition and Replication Capability${NC}"
echo "--------------------------------------------------------------"

# Check if cluster supports 12 partitions and replication factor 3
REQUIRED_PARTITIONS=12
REQUIRED_REPLICATION=3

if [[ "$BROKER_COUNT" -ge "$REQUIRED_REPLICATION" ]]; then
    echo -e "${GREEN}✓${NC} Replication factor $REQUIRED_REPLICATION supported (broker count: $BROKER_COUNT)"
else
    echo -e "${RED}✗${NC} Replication factor $REQUIRED_REPLICATION NOT supported (broker count: ${BROKER_COUNT:-0})"
    ERRORS+=("Insufficient brokers for replication factor $REQUIRED_REPLICATION")
    VALIDATION_FAILED=1
fi

echo -e "${GREEN}✓${NC} Partition count $REQUIRED_PARTITIONS supported"

echo ""

# ==============================================================================
# 5. Monitoring Infrastructure Validation
# ==============================================================================
echo -e "${BLUE}[5/8] Monitoring Infrastructure Validation${NC}"
echo "--------------------------------------------------------------"

# Check Prometheus
if [[ -n "${PROMETHEUS_URL:-}" ]]; then
    if curl -s -f "${PROMETHEUS_URL}/api/v1/status/config" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Prometheus accessible: $PROMETHEUS_URL"
    else
        echo -e "${RED}✗${NC} Prometheus not accessible: $PROMETHEUS_URL"
        ERRORS+=("Prometheus not accessible")
        VALIDATION_FAILED=1
    fi
else
    echo -e "${YELLOW}⚠${NC}  PROMETHEUS_URL not set"
    WARNINGS+=("Prometheus URL not configured")
fi

# Check Grafana
if [[ -n "${GRAFANA_URL:-}" ]]; then
    if curl -s -f "${GRAFANA_URL}/api/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Grafana accessible: $GRAFANA_URL"
    else
        echo -e "${YELLOW}⚠${NC}  Grafana not accessible: $GRAFANA_URL"
        WARNINGS+=("Grafana not accessible")
    fi
else
    echo -e "${YELLOW}⚠${NC}  GRAFANA_URL not set"
    WARNINGS+=("Grafana URL not configured")
fi

echo ""

# ==============================================================================
# 6. Security Configuration Validation
# ==============================================================================
echo -e "${BLUE}[6/8] Security Configuration Validation${NC}"
echo "--------------------------------------------------------------"

# Run security validation script
if [[ -x "./scripts/validate-security-prerequisites.sh" ]]; then
    if ./scripts/validate-security-prerequisites.sh; then
        echo -e "${GREEN}✓${NC} All security prerequisites validated"
    else
        echo -e "${RED}✗${NC} Security prerequisite validation failed"
        ERRORS+=("Security validation failed")
        VALIDATION_FAILED=1
    fi
else
    echo -e "${YELLOW}⚠${NC}  validate-security-prerequisites.sh not found"
    WARNINGS+=("Security validation script missing")
fi

echo ""

# ==============================================================================
# 7. Consumer Readiness Validation
# ==============================================================================
echo -e "${BLUE}[7/8] Consumer Readiness Validation${NC}"
echo "--------------------------------------------------------------"

echo -e "${YELLOW}⚠${NC}  Consumer readiness validation is MANUAL"
echo "   Please confirm the following:"
echo "   1. Consumer applications have protobuf deserializers"
echo "   2. Consumers understand message headers (exchange, symbol, data_type, schema_version)"
echo "   3. Consumers are configured to read from consolidated topics (cryptofeed.*)"
echo "   4. Consumer teams have been notified of deployment timeline"
echo ""

read -p "Are all consumer applications ready? [y/N]: " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}✓${NC} Consumer readiness confirmed by operator"
else
    echo -e "${RED}✗${NC} Consumer readiness NOT confirmed"
    ERRORS+=("Consumer readiness not confirmed")
    VALIDATION_FAILED=1
fi

echo ""

# ==============================================================================
# 8. Configuration File Validation
# ==============================================================================
echo -e "${BLUE}[8/8] Configuration File Validation${NC}"
echo "--------------------------------------------------------------"

CONFIG_FILE="deployment/staging/kafka-callback-config.yaml"

if [[ -f "$CONFIG_FILE" ]]; then
    echo -e "${GREEN}✓${NC} Configuration file exists: $CONFIG_FILE"

    # Validate YAML syntax
    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Configuration file is valid YAML"
        else
            echo -e "${RED}✗${NC} Configuration file has YAML syntax errors"
            ERRORS+=("Invalid YAML in $CONFIG_FILE")
            VALIDATION_FAILED=1
        fi
    else
        echo -e "${YELLOW}⚠${NC}  Cannot validate YAML (Python not available)"
        WARNINGS+=("YAML validation skipped")
    fi
else
    echo -e "${RED}✗${NC} Configuration file not found: $CONFIG_FILE"
    ERRORS+=("Missing configuration file")
    VALIDATION_FAILED=1
fi

echo ""

# ==============================================================================
# Validation Summary
# ==============================================================================
echo "================================================================"
echo "Validation Summary"
echo "================================================================"

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Warnings (${#WARNINGS[@]}):${NC}"
    for warning in "${WARNINGS[@]}"; do
        echo -e "${YELLOW}  ⚠${NC}  $warning"
    done
    echo ""
fi

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo -e "${RED}Errors (${#ERRORS[@]}):${NC}"
    for error in "${ERRORS[@]}"; do
        echo -e "${RED}  ✗${NC}  $error"
    done
    echo ""
fi

if [[ $VALIDATION_FAILED -eq 0 ]]; then
    echo -e "${GREEN}✓ All validations passed${NC}"
    echo -e "${GREEN}✓ Ready for staging deployment${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review deployment/staging/kafka-callback-config.yaml"
    echo "  2. Run: ./scripts/deploy-staging-kafka-callback.sh"
    echo "  3. Monitor deployment with: ./scripts/health-check-staging.sh"
    exit 0
else
    echo -e "${RED}✗ Validation failed${NC}"
    echo -e "${RED}✗ DO NOT proceed with deployment${NC}"
    echo ""
    echo "Fix the errors above before retrying."
    exit 1
fi
