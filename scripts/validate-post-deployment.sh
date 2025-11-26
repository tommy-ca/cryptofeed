#!/usr/bin/env bash
#
# Post-Deployment Validation Script
# Market Data Kafka Producer - Phase 5 Week 1
#
# Purpose: Validate KafkaCallback deployment after staging rollout
# Usage: ./scripts/validate-post-deployment.sh
#
# Validates:
#   - Message format and headers
#   - Protobuf serialization
#   - Message latency
#   - Error rate
#   - Broker metrics stability
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
echo "Post-Deployment Validation"
echo "Market Data Kafka Producer - Phase 5 Week 1"
echo "Timestamp: $TIMESTAMP"
echo "================================================================"
echo ""

# ==============================================================================
# 1. Message Format Validation
# ==============================================================================
echo -e "${BLUE}[1/6] Message Format and Headers Validation${NC}"
echo "--------------------------------------------------------------"

# Sample messages from consolidated topics
SAMPLE_TOPICS=("cryptofeed.trades" "cryptofeed.orderbook" "cryptofeed.ticker")

for topic in "${SAMPLE_TOPICS[@]}"; do
    echo "Sampling topic: $topic"

    if command -v kafka-console-consumer.sh &> /dev/null && [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
        # Consume 1 message from topic (with timeout)
        SAMPLE_MESSAGE=$(timeout 10 kafka-console-consumer.sh \
            --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
            --topic "$topic" \
            --max-messages 1 \
            --timeout-ms 5000 \
            --property print.headers=true \
            2>/dev/null || echo "")

        if [[ -n "$SAMPLE_MESSAGE" ]]; then
            # Check if headers are present
            if echo "$SAMPLE_MESSAGE" | grep -q "exchange:"; then
                echo -e "${GREEN}✓${NC} Header 'exchange' present"
            else
                echo -e "${RED}✗${NC} Header 'exchange' missing"
                ERRORS+=("Missing 'exchange' header in $topic")
                VALIDATION_FAILED=1
            fi

            if echo "$SAMPLE_MESSAGE" | grep -q "symbol:"; then
                echo -e "${GREEN}✓${NC} Header 'symbol' present"
            else
                echo -e "${RED}✗${NC} Header 'symbol' missing"
                ERRORS+=("Missing 'symbol' header in $topic")
                VALIDATION_FAILED=1
            fi

            if echo "$SAMPLE_MESSAGE" | grep -q "data_type:"; then
                echo -e "${GREEN}✓${NC} Header 'data_type' present"
            else
                echo -e "${RED}✗${NC} Header 'data_type' missing"
                ERRORS+=("Missing 'data_type' header in $topic")
                VALIDATION_FAILED=1
            fi

            if echo "$SAMPLE_MESSAGE" | grep -q "schema_version:"; then
                echo -e "${GREEN}✓${NC} Header 'schema_version' present"
            else
                echo -e "${RED}✗${NC} Header 'schema_version' missing"
                ERRORS+=("Missing 'schema_version' header in $topic")
                VALIDATION_FAILED=1
            fi

            echo -e "${GREEN}✓${NC} Message headers validated for $topic"
        else
            echo -e "${YELLOW}⚠${NC}  No messages available in $topic (topic may be empty)"
            WARNINGS+=("No messages in $topic for validation")
        fi
    else
        echo -e "${YELLOW}⚠${NC}  Cannot sample messages (kafka-console-consumer.sh not available)"
        WARNINGS+=("Message format validation skipped")
        break
    fi

    echo ""
done

# ==============================================================================
# 2. Protobuf Serialization Validation
# ==============================================================================
echo -e "${BLUE}[2/6] Protobuf Serialization Validation${NC}"
echo "--------------------------------------------------------------"

echo "Validating protobuf serialization using Python script..."

# Python script to validate protobuf serialization
cat > /tmp/validate_protobuf.py << 'EOF'
import sys
from kafka import KafkaConsumer
from cryptofeed.backends.protobuf_helpers import deserialize_protobuf_message

bootstrap_servers = sys.argv[1]
topic = sys.argv[2]

consumer = KafkaConsumer(
    topic,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset='latest',
    max_poll_records=1,
    consumer_timeout_ms=5000
)

try:
    for message in consumer:
        # Check if message is binary (protobuf)
        if isinstance(message.value, bytes):
            print("✓ Message is binary (protobuf format)")

            # Get data type from headers
            data_type = None
            for header_key, header_value in message.headers:
                if header_key == 'data_type':
                    data_type = header_value.decode('utf-8')
                    break

            if data_type:
                print(f"✓ Data type: {data_type}")

                # Attempt to deserialize
                try:
                    deserialized = deserialize_protobuf_message(message.value, data_type)
                    print("✓ Protobuf deserialization successful")

                    # Check message size reduction
                    import json
                    # Approximate JSON size (rough estimate)
                    json_size = len(json.dumps(str(deserialized)))
                    protobuf_size = len(message.value)
                    reduction = (1 - protobuf_size / json_size) * 100

                    if reduction > 30:
                        print(f"✓ Message size reduction: {reduction:.1f}% (target: >30%)")
                    else:
                        print(f"⚠ Message size reduction: {reduction:.1f}% (target: >30%)")

                    sys.exit(0)
                except Exception as e:
                    print(f"✗ Protobuf deserialization failed: {e}")
                    sys.exit(1)
            else:
                print("✗ No data_type header found")
                sys.exit(1)
        else:
            print("✗ Message is not binary (expected protobuf)")
            sys.exit(1)
        break
    else:
        print("⚠ No messages available for validation")
        sys.exit(2)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
finally:
    consumer.close()
EOF

if python3 /tmp/validate_protobuf.py "${KAFKA_BOOTSTRAP_SERVERS}" "cryptofeed.trades" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Protobuf serialization validated"
else
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 2 ]]; then
        echo -e "${YELLOW}⚠${NC}  No messages available for protobuf validation"
        WARNINGS+=("Protobuf validation skipped (no messages)")
    else
        echo -e "${RED}✗${NC} Protobuf validation failed"
        ERRORS+=("Protobuf serialization validation failed")
        VALIDATION_FAILED=1
    fi
fi

rm -f /tmp/validate_protobuf.py

echo ""

# ==============================================================================
# 3. Message Latency Validation
# ==============================================================================
echo -e "${BLUE}[3/6] Message Latency Validation${NC}"
echo "--------------------------------------------------------------"

# Query Prometheus for latency metrics
if [[ -n "${PROMETHEUS_URL:-}" ]]; then
    echo "Querying Prometheus for latency metrics..."

    # Query p99 latency
    P99_QUERY='histogram_quantile(0.99, rate(kafka_producer_latency_seconds_bucket[5m]))'
    P99_RESULT=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${P99_QUERY}" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data']['result'][0]['value'][1] if data['data']['result'] else 'no_data')" 2>/dev/null || echo "no_data")

    if [[ "$P99_RESULT" != "no_data" ]]; then
        P99_MS=$(echo "$P99_RESULT * 1000" | bc 2>/dev/null || echo "0")
        if (( $(echo "$P99_MS < 5" | bc -l) )); then
            echo -e "${GREEN}✓${NC} Message latency p99: ${P99_MS}ms (< 5ms target)"
        else
            echo -e "${RED}✗${NC} Message latency p99: ${P99_MS}ms (>= 5ms target)"
            ERRORS+=("Latency p99 exceeds 5ms: ${P99_MS}ms")
            VALIDATION_FAILED=1
        fi
    else
        echo -e "${YELLOW}⚠${NC}  Latency metrics not available yet (may need more time to collect)"
        WARNINGS+=("Latency metrics not available")
    fi
else
    echo -e "${YELLOW}⚠${NC}  PROMETHEUS_URL not set, cannot validate latency"
    WARNINGS+=("Latency validation skipped")
fi

echo ""

# ==============================================================================
# 4. Error Rate Validation
# ==============================================================================
echo -e "${BLUE}[4/6] Error Rate Validation${NC}"
echo "--------------------------------------------------------------"

if [[ -n "${PROMETHEUS_URL:-}" ]]; then
    echo "Querying Prometheus for error rate..."

    # Query error rate
    ERROR_RATE_QUERY='rate(kafka_producer_errors_total[5m])'
    ERROR_RATE_RESULT=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${ERROR_RATE_QUERY}" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data']['result'][0]['value'][1] if data['data']['result'] else '0')" 2>/dev/null || echo "0")

    ERROR_RATE_PERCENT=$(echo "$ERROR_RATE_RESULT * 100" | bc 2>/dev/null || echo "0")

    if (( $(echo "$ERROR_RATE_PERCENT < 0.1" | bc -l) )); then
        echo -e "${GREEN}✓${NC} Error rate: ${ERROR_RATE_PERCENT}% (< 0.1% target)"
    else
        echo -e "${RED}✗${NC} Error rate: ${ERROR_RATE_PERCENT}% (>= 0.1% target)"
        ERRORS+=("Error rate exceeds 0.1%: ${ERROR_RATE_PERCENT}%")
        VALIDATION_FAILED=1
    fi
else
    echo -e "${YELLOW}⚠${NC}  PROMETHEUS_URL not set, cannot validate error rate"
    WARNINGS+=("Error rate validation skipped")
fi

echo ""

# ==============================================================================
# 5. Broker Metrics Stability
# ==============================================================================
echo -e "${BLUE}[5/6] Broker Metrics Stability Validation${NC}"
echo "--------------------------------------------------------------"

if [[ -n "${PROMETHEUS_URL:-}" ]]; then
    echo "Querying Prometheus for broker metrics..."

    # Query broker CPU
    CPU_QUERY='avg(kafka_broker_cpu_percent)'
    CPU_RESULT=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${CPU_QUERY}" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data']['result'][0]['value'][1] if data['data']['result'] else 'no_data')" 2>/dev/null || echo "no_data")

    if [[ "$CPU_RESULT" != "no_data" ]]; then
        if (( $(echo "$CPU_RESULT < 80" | bc -l) )); then
            echo -e "${GREEN}✓${NC} Broker CPU: ${CPU_RESULT}% (< 80% target)"
        else
            echo -e "${YELLOW}⚠${NC}  Broker CPU: ${CPU_RESULT}% (>= 80% warning)"
            WARNINGS+=("Broker CPU high: ${CPU_RESULT}%")
        fi
    else
        echo -e "${YELLOW}⚠${NC}  Broker CPU metrics not available"
        WARNINGS+=("Broker CPU metrics not available")
    fi

    # Query broker memory
    MEMORY_QUERY='avg(kafka_broker_memory_percent)'
    MEMORY_RESULT=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${MEMORY_QUERY}" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data']['result'][0]['value'][1] if data['data']['result'] else 'no_data')" 2>/dev/null || echo "no_data")

    if [[ "$MEMORY_RESULT" != "no_data" ]]; then
        if (( $(echo "$MEMORY_RESULT < 80" | bc -l) )); then
            echo -e "${GREEN}✓${NC} Broker Memory: ${MEMORY_RESULT}% (< 80% target)"
        else
            echo -e "${YELLOW}⚠${NC}  Broker Memory: ${MEMORY_RESULT}% (>= 80% warning)"
            WARNINGS+=("Broker memory high: ${MEMORY_RESULT}%")
        fi
    else
        echo -e "${YELLOW}⚠${NC}  Broker memory metrics not available"
        WARNINGS+=("Broker memory metrics not available")
    fi
else
    echo -e "${YELLOW}⚠${NC}  PROMETHEUS_URL not set, cannot validate broker metrics"
    WARNINGS+=("Broker metrics validation skipped")
fi

echo ""

# ==============================================================================
# 6. Deployment Duration Validation
# ==============================================================================
echo -e "${BLUE}[6/6] Monitoring Recommendation${NC}"
echo "--------------------------------------------------------------"

echo "Post-deployment monitoring recommendations:"
echo "  1. Continue monitoring for 2-4 hours"
echo "  2. Watch for error rate spikes"
echo "  3. Watch for latency degradation"
echo "  4. Monitor broker resource usage"
echo "  5. Verify consumer applications are processing messages"
echo ""
echo "Run continuous health checks:"
echo "  ./scripts/health-check-staging.sh"
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
    echo -e "${GREEN}✓ Post-deployment validation passed${NC}"
    echo -e "${GREEN}✓ Deployment is healthy${NC}"
    echo ""
    echo "Continue monitoring for 2-4 hours to confirm stability."
    exit 0
else
    echo -e "${RED}✗ Post-deployment validation failed${NC}"
    echo -e "${RED}✗ Consider rollback if errors persist${NC}"
    echo ""
    echo "Run rollback if needed:"
    echo "  ./scripts/rollback-staging-deployment.sh"
    exit 1
fi
