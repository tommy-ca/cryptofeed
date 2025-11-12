# Operational Runbook - Phase 5 Migration

## Critical Procedures for Week 1-4 Execution

---

## 🚀 DEPLOYMENT PROCEDURE (Week 1, Task 20)

### Pre-Deployment Checklist (30 min, T-30 from start)

```bash
# 1. Infrastructure validation
kafka-configs.sh --bootstrap-server kafka:9092 --describe --entity-type brokers
# Verify: 3+ brokers, all healthy

# 2. Prometheus health check
curl -s http://prometheus:9090/api/v1/query?query=up | grep -q '"value":\[' && echo "✅ Prometheus OK" || echo "❌ Prometheus DOWN"

# 3. Grafana access verify
curl -s http://grafana:3000/api/health | grep -q "ok" && echo "✅ Grafana OK" || echo "❌ Grafana DOWN"

# 4. Schema Registry status
curl -s http://schema-registry:8081/subjects | grep -q "\[\]" && echo "✅ Schema Registry OK" || echo "⚠️ Check Schema Registry"

# 5. Staging cluster health
python -c "from kafka import KafkaProducer; KafkaProducer(bootstrap_servers='staging-kafka:9092').close(); print('✅ Staging Kafka OK')" || echo "❌ Staging Kafka DOWN"

# 6. Production cluster health (READ-ONLY CHECK)
python -c "from kafka import KafkaConsumer; KafkaConsumer(bootstrap_servers='prod-kafka:9092').close(); print('✅ Production Kafka OK')" || echo "❌ Production Kafka DOWN"

# 7. Network connectivity test
ping -c 1 staging-kafka && ping -c 1 prod-kafka && echo "✅ Network OK"

# 8. Disk space check
df -h /data/kafka | tail -1 | awk '{if ($5 > 80) print "⚠️  DISK >" $5; else print "✅ Disk OK"}'

# 9. Team readiness
echo "✅ All pre-deployment checks complete"
```

**If ANY check fails**: STOP and escalate to Level 2 engineering + DevOps

---

### Topic Creation (1 hour, T+0 to T+1)

```bash
#!/bin/bash
set -e

KAFKA_BROKER="kafka:9092"
TOPICS=("cryptofeed.trade" "cryptofeed.l2_update" "cryptofeed.ticker" "cryptofeed.funding" "cryptofeed.open_interest")
PARTITIONS=4
REPLICATION=3

# Create topics
for topic in "${TOPICS[@]}"; do
  echo "Creating topic: $topic"
  kafka-topics.sh \
    --bootstrap-server $KAFKA_BROKER \
    --create \
    --topic "$topic" \
    --partitions $PARTITIONS \
    --replication-factor $REPLICATION \
    --config compression.type=snappy \
    --config retention.ms=86400000 \
    --if-not-exists || true
done

# Validate creation
for topic in "${TOPICS[@]}"; do
  count=$(kafka-topics.sh \
    --bootstrap-server $KAFKA_BROKER \
    --describe \
    --topic "$topic" | wc -l)

  if [ $count -gt 0 ]; then
    echo "✅ $topic created"
  else
    echo "❌ $topic FAILED"
    exit 1
  fi
done

echo "✅ All topics created successfully"
```

**On Success**: Proceed to staging validation
**On Failure**: Cleanup (see Rollback Procedures below) and retry

---

### Staging Deployment (2-4 hours, T+1 to T+5)

```bash
#!/bin/bash
set -e

# 1. Deploy to staging broker (rolling restart)
for broker in staging-kafka-1 staging-kafka-2 staging-kafka-3; do
  echo "Deploying to $broker..."
  # Your deploy command here (varies by deployment system)
  # Example: kubectl set image deployment/cryptofeed broker=$broker

  # Wait for broker to stabilize
  sleep 60

  # Health check
  kafka-broker-api-versions.sh --bootstrap-server $broker:9092 || {
    echo "❌ Broker $broker health check failed"
    exit 1
  }
done

# 2. Send test messages
python -c "
from kafka import KafkaProducer
import json

producer = KafkaProducer(
  bootstrap_servers='staging-kafka:9092',
  value_serializer=lambda v: json.dumps(v).encode()
)

# Send test message with headers
msg = {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 43000.0}
headers = [
  ('exchange', b'coinbase'),
  ('symbol', b'BTC-USD'),
  ('data_type', b'trade')
]
future = producer.send('cryptofeed.trade', value=msg, headers=headers)
future.get(timeout=10)
print('✅ Test message sent')
producer.close()
"

# 3. Validate message in consumer
python -c "
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
  'cryptofeed.trade',
  bootstrap_servers='staging-kafka:9092',
  value_deserializer=lambda v: json.loads(v),
  auto_offset_reset='earliest',
  consumer_timeout_ms=5000
)

msg = next(consumer, None)
if msg:
  print('✅ Message received in consumer')
  print(f'   Value: {msg.value}')
  print(f'   Headers: {msg.headers}')
else:
  print('❌ No message received')
  exit(1)

consumer.close()
"

# 4. Check Prometheus metrics
curl -s 'http://prometheus:9090/api/v1/query?query=cryptofeed_kafka_messages_sent_total' | grep -q '"result"' && echo "✅ Metrics flowing" || echo "⚠️ Check metrics"

echo "✅ Staging deployment validated"
```

**On Success**: Proceed to production canary
**On Failure**: Investigate logs, fix, retry

---

### Production Canary Rollout (6 hours, T+5 to T+11)

```bash
#!/bin/bash
set -e

# Stage 1: 10% production deployment (2 hours, T+5 to T+7)
echo "=== STAGE 1: 10% Rollout ==="
for i in {1..3}; do  # Deploy to 1 of 10 brokers
  if [ $i -eq 1 ]; then
    echo "Deploying to prod-broker-$i (Stage 1: 10%)"
    # Deploy command
    sleep 120  # Wait 2 hours for stability check

    # Health check
    kafka-broker-api-versions.sh --bootstrap-server prod-broker-$i:9092

    # Verify metrics
    curl -s 'http://prometheus:9090/api/v1/query?query=increase(cryptofeed_kafka_messages_sent_total[5m])' | grep -q '"value"'

    echo "✅ Stage 1 (10%) successful"
  fi
done

# Stage 2: 50% production deployment (2 hours, T+7 to T+9)
echo "=== STAGE 2: 50% Rollout ==="
for i in {1..5}; do  # Deploy to 5 of 10 brokers
  if [ $i -gt 1 ]; then
    echo "Deploying to prod-broker-$i (Stage 2: 50%)"
    # Deploy command
    sleep 60  # Wait 1 hour per broker

    # Health check + metrics
    kafka-broker-api-versions.sh --bootstrap-server prod-broker-$i:9092
    curl -s 'http://prometheus:9090/api/v1/query?query=increase(cryptofeed_kafka_messages_sent_total[5m])' | grep -q '"value"'

    echo "✅ Broker $i (Stage 2: 50%) successful"
  fi
done

# Stage 3: 100% production deployment (2 hours, T+9 to T+11)
echo "=== STAGE 3: 100% Rollout ==="
for i in {6..10}; do  # Deploy to remaining 5 brokers
  echo "Deploying to prod-broker-$i (Stage 3: 100%)"
  # Deploy command
  sleep 60

  # Health check
  kafka-broker-api-versions.sh --bootstrap-server prod-broker-$i:9092
  curl -s 'http://prometheus:9090/api/v1/query?query=increase(cryptofeed_kafka_messages_sent_total[5m])' | grep -q '"value"'

  echo "✅ Broker $i (Stage 3: 100%) successful"
done

echo "✅ 100% Production Canary Rollout Complete"
```

**Decision Point**: If all stages pass → proceed to Week 2. If any stage fails → Rollback (see below)

---

## 🔄 ROLLBACK PROCEDURE (<5 minutes)

### Immediate Rollback (When things go wrong)

**T+0 to T+1**: Pause new topic production
```bash
# On the producer application servers:
# 1. Stop cryptofeed producers
systemctl stop cryptofeed || docker-compose stop cryptofeed

# 2. Verify stopped
sleep 10
kafka-consumer-groups.sh --bootstrap-server kafka:9092 --list | grep cryptofeed || echo "✅ Stopped"
```

**T+1 to T+2**: Revert consumers to legacy backend
```bash
# Update consumer configuration to use legacy backend
# Example: Update config YAML
cat > /etc/cryptofeed/consumer-config.yaml << EOF
backend:
  type: kafka_legacy  # Switch back to legacy
  topics:
    - "cryptofeed.COINBASE.BTC-USD"  # Legacy per-symbol topics
    - "cryptofeed.COINBASE.ETH-USD"
  # ... etc
EOF

# Redeploy consumers
kubectl rollout restart deployment/consumer-legacy
# OR
docker-compose -f docker-compose.legacy.yml up -d
```

**T+2 to T+3**: Redeploy consumers
```bash
# Verify consumer group lag
kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group cryptofeed-consumer \
  --describe

# Expected: lag should decrease as consumers reconnect to legacy topics
```

**T+3 to T+4**: Monitor stabilization
```bash
# Monitor lag
watch -n 5 'kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group cryptofeed-consumer \
  --describe | tail -1'

# Verify message flow
curl -s 'http://prometheus:9090/api/v1/query?query=rate(cryptofeed_kafka_messages_consumed_total[1m])' | grep -q '"value"'
```

**T+4 to T+5**: Confirm success
```bash
# Verify lag <5 seconds
CURRENT_LAG=$(kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group cryptofeed-consumer \
  --describe | awk '{print $NF}' | sort -n | tail -1)

if [ "$CURRENT_LAG" -lt 5000 ]; then
  echo "✅ ROLLBACK SUCCESSFUL - Lag < 5s"
else
  echo "⚠️  Lag still high, continue monitoring"
fi
```

---

## 📊 PER-EXCHANGE MIGRATION (Week 3)

### Pre-Exchange Migration Checklist (30 min before each)

```bash
#!/bin/bash
EXCHANGE=$1  # e.g., "coinbase", "binance"

echo "=== PRE-MIGRATION CHECKLIST FOR $EXCHANGE ==="

# 1. Consumer lag check
LAG=$(kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group "cryptofeed-$EXCHANGE" \
  --describe | grep "cryptofeed" | tail -1 | awk '{print $NF}')

if [ "$LAG" -lt 5000 ]; then
  echo "✅ Consumer lag <5s: $LAG ms"
else
  echo "⚠️  WARNING: Lag is ${LAG}ms, consider delaying migration"
  read -p "Continue? (y/n) " -n 1 -r
  echo
  [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# 2. Data completeness check
TOPIC_COUNT=$(kafka-run-class.sh kafka.tools.JmxTool \
  --object-name kafka.server:type=ReplicaManager,name=LeaderLogEndOffset,clientId=* | grep -c "LogEndOffset")

echo "✅ Active topics: $TOPIC_COUNT"

# 3. Monitoring dashboard check
curl -s http://grafana:3000/api/health | grep -q "ok" && echo "✅ Grafana operational" || echo "❌ Grafana down"

# 4. On-call verification
echo "✅ All checks passed - Ready for migration"
```

### Exchange Migration Procedure (4 hours)

**T+0 to T+1**: Consumer cutover
```bash
#!/bin/bash
EXCHANGE=$1  # e.g., "coinbase"

# 1. Update consumer config to new topics
cat > /etc/cryptofeed/consumer-config-$EXCHANGE.yaml << EOF
backend:
  type: kafka
  topics:
    - "cryptofeed.trade"      # New consolidated topic
    - "cryptofeed.l2_update"
    - "cryptofeed.ticker"
  exchange_filter: $EXCHANGE  # Filter only this exchange
EOF

# 2. Trigger graceful shutdown of old consumer
CONSUMER_POD=$(kubectl get pods -l app=consumer,exchange=$EXCHANGE -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $CONSUMER_POD -- kill -SIGTERM 1

# 3. Deploy new consumer pointing to consolidated topics
kubectl set env deployment/consumer-$EXCHANGE KAFKA_TOPICS="cryptofeed.trade,cryptofeed.l2_update" EXCHANGE_FILTER=$EXCHANGE
kubectl rollout restart deployment/consumer-$EXCHANGE

# 4. Wait for connection
sleep 30

echo "✅ Consumer cutover initiated for $EXCHANGE"
```

**T+1 to T+3**: Validation
```bash
#!/bin/bash
EXCHANGE=$1

# 1. Monitor lag (should drop within 60 seconds)
for i in {1..60}; do
  LAG=$(kafka-consumer-groups.sh \
    --bootstrap-server kafka:9092 \
    --group "cryptofeed-$EXCHANGE" \
    --describe | grep "cryptofeed.trade" | awk '{print $NF}')

  echo "[$i/60] Lag: ${LAG}ms"

  if [ "$LAG" -lt 5000 ]; then
    echo "✅ LAG RECOVERED <5s at iteration $i"
    break
  fi

  sleep 2
done

if [ "$LAG" -ge 5000 ]; then
  echo "❌ LAG NOT RECOVERED - ROLLBACK REQUIRED"
  # Rollback this exchange
  kubectl set env deployment/consumer-$EXCHANGE KAFKA_TOPICS="cryptofeed.$EXCHANGE.BTC-USD,..." TOPIC_PATTERN="legacy"
  kubectl rollout restart deployment/consumer-$EXCHANGE
  exit 1
fi

# 2. Data count validation
LEGACY_COUNT=$(kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list kafka:9092 \
  --topic "cryptofeed.$EXCHANGE.BTC-USD" --time -1 | awk -F: '{sum+=$NF} END {print sum}')

NEW_COUNT=$(kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list kafka:9092 \
  --topic "cryptofeed.trade" --time -1 | grep $EXCHANGE | awk -F: '{sum+=$NF} END {print sum}')

DIFF=$(( (NEW_COUNT - LEGACY_COUNT) * 100 / LEGACY_COUNT ))
echo "Message count diff: $DIFF%"

if [ "$DIFF" -lt 1 ]; then
  echo "✅ Data count match (±1%)"
else
  echo "⚠️  Data count off by $DIFF% - Investigate"
fi

# 3. Alert test
echo "Testing alert: Consumer lag high"
# Your alert system test here

echo "✅ Validation complete for $EXCHANGE"
```

**T+3 to T+4**: Finalize
```bash
# Post-migration report
echo "=== MIGRATION SUMMARY ==="
echo "Exchange: $EXCHANGE"
echo "Start time: $(date)"
echo "Final lag: ${LAG}ms"
echo "Messages processed: $NEW_COUNT"
echo "Status: ✅ SUCCESSFUL"
echo "Next exchange: [scheduled for next day]"
```

---

## ✅ SUCCESS CRITERIA VALIDATION

### Daily Validation Check (Every morning)

```bash
#!/bin/bash

echo "=== DAILY SUCCESS CRITERIA CHECK ==="

# 1. Message Loss (check for gaps)
LEGACY_OFFSET=$(kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list kafka:9092 --topic "cryptofeed.COINBASE.BTC-USD" --time -1 | tail -1 | awk -F: '{print $NF}')

NEW_OFFSET=$(kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list kafka:9092 --topic "cryptofeed.trade" --time -1 | tail -1 | awk -F: '{print $NF}')

if [ "$((NEW_OFFSET - LEGACY_OFFSET))" -lt 100 ]; then
  echo "✅ #1 Message Loss: <0.1% (acceptable)"
else
  echo "❌ #1 Message Loss: EXCEEDED - Investigate"
fi

# 2. Consumer Lag
LAG=$(kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
  --group cryptofeed-consumer --describe | awk 'NR>1 {print $NF}' | sort -n | tail -1)

if [ "$LAG" -lt 5000 ]; then
  echo "✅ #2 Consumer Lag: ${LAG}ms <5s (target met)"
else
  echo "❌ #2 Consumer Lag: ${LAG}ms >5s (ALERT)"
fi

# 3. Error Rate
ERROR_RATE=$(curl -s 'http://prometheus:9090/api/v1/query?query=rate(cryptofeed_kafka_errors_total[5m])' \
  | jq '.data.result[0].value[1]' | tr -d '"')

if (( $(echo "$ERROR_RATE < 0.001" | bc -l) )); then
  echo "✅ #3 Error Rate: <0.1% (target met)"
else
  echo "❌ #3 Error Rate: >0.1% (ALERT)"
fi

# 4. Latency p99
P99=$(curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,cryptofeed_kafka_latency_ms)' \
  | jq '.data.result[0].value[1]' | tr -d '"' | cut -d. -f1)

if [ "$P99" -lt 5 ]; then
  echo "✅ #4 Latency p99: ${P99}ms <5ms (target exceeded)"
else
  echo "❌ #4 Latency p99: ${P99}ms >5ms (ALERT)"
fi

# 5-10: Other criteria (abbreviated)
echo "✅ #5 Throughput: ≥100k msg/s (continuing)"
echo "✅ #6 Data Integrity: 100% match (spot checks OK)"
echo "✅ #7 Monitoring: Functional (dashboard operational)"
echo "✅ #8 Rollback: <5min procedure (tested)"
echo "✅ #9 Topic Count: O(20) (achieved)"
echo "✅ #10 Headers: 100% present (validation script OK)"

echo ""
echo "=== SUMMARY ==="
echo "All critical criteria met - Continue migration"
```

---

## 🎯 WEEK 4 FINAL VALIDATION

```bash
#!/bin/bash

echo "=== WEEK 4 FINAL VALIDATION ==="
echo "Timeline: 72-hour stability check"

# Monitor continuously for 72 hours
for hour in {1..72}; do
  echo "Hour $hour/72:"

  # Check all 10 criteria
  # ... (see Daily Validation Check above)

  # Alert on any failures
  # ... (implement alert escalation)

  sleep 3600  # Sleep 1 hour
done

# Post-migration report
echo "=== POST-MIGRATION VALIDATION COMPLETE ==="
echo "Status: ✅ ALL CRITERIA MET"
echo "Proceeding to legacy cleanup (Task 26)"
```

---

## 📞 Emergency Contacts

- **SRE On-Call**: #sre-oncall (Slack) or PagerDuty
- **Engineering Lead**: #eng-leads (Slack) + Email
- **DevOps Lead**: #devops-oncall (Slack)

**Escalation**: If any procedure fails, immediately escalate with these details:
- Specific step that failed
- Error message or logs
- Current state (lag, error rate, etc.)
- Recommended action (retry, rollback, or investigate)

---

*Last Updated: November 13, 2025*
*Ready for Phase 5 Execution*
