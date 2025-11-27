#!/bin/bash
#
# Consumer Migration Automation Example - Task 21
#
# This script demonstrates how to use the automation tools created for
# validating and testing consumer migrations to consolidated Kafka topics.
#
# Prerequisites:
# - Kafka cluster running (for actual tests)
# - Consumer templates configured
# - Migration guide reviewed
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=================================================="
echo "Consumer Migration Automation Example"
echo "Task 21 - Phase 5 Week 2"
echo "=================================================="
echo ""

# Step 1: Validate Consumer Configuration
echo -e "${YELLOW}Step 1: Validate Consumer Configuration${NC}"
echo ""

# Create example Flink consumer config
cat > /tmp/flink-consumer-config.json << 'EOF'
{
  "consumer_type": "flink",
  "topics": ["cryptofeed.trades", "cryptofeed.orderbook"],
  "bootstrap_servers": ["kafka1:9092", "kafka2:9092", "kafka3:9092"],
  "consumer_group": "flink-analytics-v2",
  "offset_reset": "earliest",
  "enable_headers": true,
  "protobuf_enabled": true,
  "schema_registry_url": "http://schema-registry:8081"
}
EOF

echo "Validating Flink consumer config..."
python scripts/validate-consumer-config.py /tmp/flink-consumer-config.json

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Flink consumer config is valid${NC}"
else
    echo -e "${RED}✗ Flink consumer config validation failed${NC}"
    exit 1
fi

echo ""

# Create example Python async consumer config
cat > /tmp/python-consumer-config.json << 'EOF'
{
  "consumer_type": "python-async",
  "topics": ["cryptofeed.trades"],
  "bootstrap_servers": ["localhost:9092"],
  "consumer_group": "python-processor-v2",
  "offset_reset": "latest",
  "enable_auto_commit": false,
  "batch_size": 100,
  "batch_timeout_ms": 5000,
  "enable_headers": true
}
EOF

echo "Validating Python async consumer config..."
python scripts/validate-consumer-config.py /tmp/python-consumer-config.json

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python async consumer config is valid${NC}"
else
    echo -e "${RED}✗ Python async consumer config validation failed${NC}"
    exit 1
fi

echo ""
echo "=================================================="
echo ""

# Step 2: Test Consumer Migration
echo -e "${YELLOW}Step 2: Test Consumer Migration${NC}"
echo ""

# Test consumer startup
cat > /tmp/test-startup.json << 'EOF'
{
  "test_type": "startup",
  "consumer_type": "python-async",
  "topics": ["cryptofeed.trades"],
  "bootstrap_servers": ["localhost:9092"],
  "timeout_seconds": 30
}
EOF

echo "Testing consumer startup..."
python scripts/test-consumer-migration.py /tmp/test-startup.json

echo ""

# Test offset commit
cat > /tmp/test-offset-commit.json << 'EOF'
{
  "test_type": "offset_commit",
  "consumer_type": "python-async",
  "topics": ["cryptofeed.trades"],
  "message_count": 100,
  "commit_interval_ms": 5000
}
EOF

echo "Testing offset commit behavior..."
python scripts/test-consumer-migration.py /tmp/test-offset-commit.json

echo ""

# Test header extraction
cat > /tmp/test-headers.json << 'EOF'
{
  "test_type": "header_extraction",
  "consumer_type": "python-async",
  "topics": ["cryptofeed.trades"],
  "message_count": 10
}
EOF

echo "Testing header extraction..."
python scripts/test-consumer-migration.py /tmp/test-headers.json

echo ""
echo "=================================================="
echo ""

# Step 3: Run Health Checks
echo -e "${YELLOW}Step 3: Run Health Checks${NC}"
echo ""

# Check consumer lag
cat > /tmp/check-lag.json << 'EOF'
{
  "check_type": "lag",
  "consumer_group": "python-processor-v2",
  "topics": ["cryptofeed.trades"],
  "threshold_seconds": 5
}
EOF

echo "Checking consumer lag..."
python scripts/check-consumer-health.py /tmp/check-lag.json

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Consumer lag is healthy (< 5 seconds)${NC}"
else
    echo -e "${RED}✗ Consumer lag exceeds threshold${NC}"
fi

echo ""

# Check consumer heartbeat
cat > /tmp/check-heartbeat.json << 'EOF'
{
  "check_type": "heartbeat",
  "consumer_group": "python-processor-v2"
}
EOF

echo "Checking consumer heartbeat..."
python scripts/check-consumer-health.py /tmp/check-heartbeat.json

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Consumer heartbeat is healthy${NC}"
else
    echo -e "${RED}✗ Consumer heartbeat check failed${NC}"
fi

echo ""
echo "=================================================="
echo ""

# Clean up
rm -f /tmp/flink-consumer-config.json
rm -f /tmp/python-consumer-config.json
rm -f /tmp/test-startup.json
rm -f /tmp/test-offset-commit.json
rm -f /tmp/test-headers.json
rm -f /tmp/check-lag.json
rm -f /tmp/check-heartbeat.json

echo -e "${GREEN}Consumer Migration Automation Example Complete!${NC}"
echo ""
echo "Next Steps:"
echo "1. Review validation results"
echo "2. Deploy consumers to staging environment"
echo "3. Run continuous health checks"
echo "4. Validate lag < 5 seconds (Week 2 requirement)"
echo "5. Prepare for production cutover (Week 3)"
echo ""
echo "For more information, see:"
echo "- docs/CONSUMER_MIGRATION_AUTOMATION_README.md"
echo "- docs/consumer-migration-guide-week2.md"
echo "- .kiro/specs/market-data-kafka-producer/PHASE_5_EXECUTION_PLAN.md"
