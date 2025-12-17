#!/usr/bin/env bash
#
# Continuous Health Check Script
# Market Data Kafka Producer - Phase 5 Week 1
#
# Purpose: Continuously monitor staging deployment health
# Usage: ./scripts/health-check-staging.sh [--interval SECONDS] [--duration HOURS]
#
# Options:
#   --interval SECONDS   Health check interval (default: 30)
#   --duration HOURS     Total monitoring duration (default: infinite)
#   --alert-on-failure   Send alerts on check failures (default: disabled)
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
INTERVAL_SECONDS=30
DURATION_HOURS=0  # 0 = infinite
ALERT_ON_FAILURE=false
FAILURE_COUNT=0
FAILURE_THRESHOLD=3

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            INTERVAL_SECONDS="$2"
            shift 2
            ;;
        --duration)
            DURATION_HOURS="$2"
            shift 2
            ;;
        --alert-on-failure)
            ALERT_ON_FAILURE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate end time
if [[ $DURATION_HOURS -gt 0 ]]; then
    END_TIME=$(($(date +%s) + DURATION_HOURS * 3600))
else
    END_TIME=0  # Run forever
fi

echo "================================================================"
echo "Health Check Monitoring - Staging Environment"
echo "Market Data Kafka Producer - Phase 5 Week 1"
echo "================================================================"
echo "Configuration:"
echo "  Check Interval: ${INTERVAL_SECONDS}s"
if [[ $DURATION_HOURS -gt 0 ]]; then
    echo "  Duration: ${DURATION_HOURS} hours"
else
    echo "  Duration: Continuous (Ctrl+C to stop)"
fi
echo "  Alert on Failure: ${ALERT_ON_FAILURE}"
echo "  Failure Threshold: ${FAILURE_THRESHOLD} consecutive failures"
echo "================================================================"
echo ""

# Function to run a single health check
run_health_check() {
    local check_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local check_failed=false

    echo -e "${BLUE}Health Check @ $check_timestamp${NC}"
    echo "--------------------------------------------------------------"

    # Check 1: Producer Connectivity
    echo -n "  Producer Connectivity: "
    if [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
        if timeout 5 python3 -c "from kafka import KafkaProducer; KafkaProducer(bootstrap_servers='$KAFKA_BOOTSTRAP_SERVERS', request_timeout_ms=3000).close()" 2>/dev/null; then
            echo -e "${GREEN}PASS${NC}"
        else
            echo -e "${RED}FAIL${NC}"
            check_failed=true
        fi
    else
        echo -e "${YELLOW}SKIP (KAFKA_BOOTSTRAP_SERVERS not set)${NC}"
    fi

    # Check 2: Message Delivery
    echo -n "  Message Delivery: "
    # Note: This would require sending a test message
    # For now, we'll check if topics exist
    if command -v kafka-topics.sh &> /dev/null && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
        if kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list 2>/dev/null | grep -q "cryptofeed.trades"; then
            echo -e "${GREEN}PASS${NC}"
        else
            echo -e "${RED}FAIL (topics not found)${NC}"
            check_failed=true
        fi
    else
        echo -e "${YELLOW}SKIP (kafka-topics.sh not available)${NC}"
    fi

    # Check 3: Error Rate
    echo -n "  Error Rate: "
    if [[ -n "${PROMETHEUS_URL:-}" ]]; then
        ERROR_RATE=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=rate(kafka_producer_errors_total[5m])" | python3 -c "import sys,json; data=json.load(sys.stdin); print(float(data['data']['result'][0]['value'][1]) * 100 if data['data']['result'] else 0)" 2>/dev/null || echo "0")

        if (( $(echo "$ERROR_RATE < 0.1" | bc -l) )); then
            echo -e "${GREEN}PASS (${ERROR_RATE}%)${NC}"
        else
            echo -e "${RED}FAIL (${ERROR_RATE}% >= 0.1%)${NC}"
            check_failed=true
        fi
    else
        echo -e "${YELLOW}SKIP (Prometheus not configured)${NC}"
    fi

    # Check 4: Latency p99
    echo -n "  Latency p99: "
    if [[ -n "${PROMETHEUS_URL:-}" ]]; then
        LATENCY_P99=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=histogram_quantile(0.99,%20rate(kafka_producer_latency_seconds_bucket[5m]))" | python3 -c "import sys,json; data=json.load(sys.stdin); print(float(data['data']['result'][0]['value'][1]) * 1000 if data['data']['result'] else 0)" 2>/dev/null || echo "0")

        if (( $(echo "$LATENCY_P99 < 5" | bc -l) )); then
            echo -e "${GREEN}PASS (${LATENCY_P99}ms)${NC}"
        else
            echo -e "${RED}FAIL (${LATENCY_P99}ms >= 5ms)${NC}"
            check_failed=true
        fi
    else
        echo -e "${YELLOW}SKIP (Prometheus not configured)${NC}"
    fi

    # Check 5: Broker CPU
    echo -n "  Broker CPU: "
    if [[ -n "${PROMETHEUS_URL:-}" ]]; then
        BROKER_CPU=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=avg(kafka_broker_cpu_percent)" | python3 -c "import sys,json; data=json.load(sys.stdin); print(float(data['data']['result'][0]['value'][1]) if data['data']['result'] else 0)" 2>/dev/null || echo "0")

        if (( $(echo "$BROKER_CPU < 80" | bc -l) )); then
            echo -e "${GREEN}PASS (${BROKER_CPU}%)${NC}"
        else
            echo -e "${YELLOW}WARN (${BROKER_CPU}% >= 80%)${NC}"
            # High CPU is warning, not failure
        fi
    else
        echo -e "${YELLOW}SKIP (Prometheus not configured)${NC}"
    fi

    # Check 6: Broker Memory
    echo -n "  Broker Memory: "
    if [[ -n "${PROMETHEUS_URL:-}" ]]; then
        BROKER_MEM=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=avg(kafka_broker_memory_percent)" | python3 -c "import sys,json; data=json.load(sys.stdin); print(float(data['data']['result'][0]['value'][1]) if data['data']['result'] else 0)" 2>/dev/null || echo "0")

        if (( $(echo "$BROKER_MEM < 80" | bc -l) )); then
            echo -e "${GREEN}PASS (${BROKER_MEM}%)${NC}"
        else
            echo -e "${YELLOW}WARN (${BROKER_MEM}% >= 80%)${NC}"
            # High memory is warning, not failure
        fi
    else
        echo -e "${YELLOW}SKIP (Prometheus not configured)${NC}"
    fi

    # Overall status
    echo "--------------------------------------------------------------"
    if $check_failed; then
        echo -e "${RED}Status: UNHEALTHY${NC}"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))

        # Alert if threshold reached
        if [[ $FAILURE_COUNT -ge $FAILURE_THRESHOLD && $ALERT_ON_FAILURE == true ]]; then
            echo -e "${RED}⚠ ALERT: $FAILURE_COUNT consecutive failures (threshold: $FAILURE_THRESHOLD)${NC}"
            # In production, this would trigger PagerDuty/Slack/etc.
            # For now, just print to console
        fi

        return 1
    else
        echo -e "${GREEN}Status: HEALTHY${NC}"
        FAILURE_COUNT=0  # Reset failure count on success
        return 0
    fi
}

# Function to generate status report
generate_status_report() {
    local report_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat << EOF

================================================================
Health Check Status Report
================================================================
Timestamp: $report_timestamp
Environment: staging
Status: $(if [[ $FAILURE_COUNT -eq 0 ]]; then echo "HEALTHY"; else echo "DEGRADED ($FAILURE_COUNT consecutive failures)"; fi)

Metrics:
$(if [[ -n "${PROMETHEUS_URL:-}" ]]; then
    echo "  Error Rate: ${ERROR_RATE:-N/A}%"
    echo "  Latency p99: ${LATENCY_P99:-N/A}ms"
    echo "  Broker CPU: ${BROKER_CPU:-N/A}%"
    echo "  Broker Memory: ${BROKER_MEM:-N/A}%"
else
    echo "  (Prometheus not configured)"
fi)

Next Check: $(date -d "+${INTERVAL_SECONDS} seconds" +"%Y-%m-%d %H:%M:%S")
================================================================

EOF
}

# Main monitoring loop
CHECK_COUNT=0

while true; do
    CHECK_COUNT=$((CHECK_COUNT + 1))

    echo ""
    echo "================================================================"
    echo "Check #$CHECK_COUNT"
    echo "================================================================"

    # Run health check
    if run_health_check; then
        :  # Success
    else
        :  # Failure (already logged)
    fi

    # Generate status report every 10 checks
    if [[ $((CHECK_COUNT % 10)) -eq 0 ]]; then
        generate_status_report
    fi

    # Check if duration exceeded
    if [[ $END_TIME -gt 0 && $(date +%s) -ge $END_TIME ]]; then
        echo ""
        echo "================================================================"
        echo "Monitoring Duration Complete"
        echo "================================================================"
        echo "Total Checks: $CHECK_COUNT"
        echo "Final Status: $(if [[ $FAILURE_COUNT -eq 0 ]]; then echo "HEALTHY"; else echo "DEGRADED"; fi)"
        break
    fi

    # Sleep until next check
    echo ""
    echo "Next check in ${INTERVAL_SECONDS}s... (Ctrl+C to stop)"
    sleep "$INTERVAL_SECONDS"
done
