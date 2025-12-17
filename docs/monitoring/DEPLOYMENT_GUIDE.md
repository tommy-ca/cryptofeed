# Monitoring Deployment Guide - Week 2

**Status**: Production Ready
**Date**: November 26, 2025
**Scope**: Grafana dashboard and Prometheus alert deployment automation
**Owner**: Data Engineering Team

## Overview

This guide provides step-by-step instructions for deploying monitoring infrastructure for the market-data-kafka-producer. The monitoring stack includes:

- **Grafana Dashboard**: 9 visualization panels for producer metrics
- **Prometheus Alert Rules**: 6+ alert rules (critical, warning, info)
- **Health Checks**: Automated validation of monitoring infrastructure

## Prerequisites

Before deploying monitoring, ensure the following are available:

- [ ] Grafana 8.0+ running and accessible
- [ ] Prometheus 2.30+ running and accessible
- [ ] Grafana admin credentials (username/password or API key)
- [ ] Network access from deployment machine to Grafana/Prometheus
- [ ] `jq` command-line JSON processor installed
- [ ] `curl` command-line tool installed
- [ ] Python 3.8+ for validation scripts

## Quick Start

### 1. Deploy Grafana Dashboard

```bash
# Set Grafana credentials
export GRAFANA_URL="http://localhost:3000"
export GRAFANA_USER="admin"
export GRAFANA_PASSWORD="admin"

# Deploy dashboard (with validation)
./scripts/deploy-grafana-dashboard.sh

# Expected output:
# [INFO] Validating dashboard JSON...
# [INFO] Dashboard JSON is valid
# [INFO] Checking Grafana API...
# [INFO] Grafana API is accessible
# [INFO] Deploying dashboard to Grafana...
# [INFO] Dashboard deployed successfully
# [INFO] Dashboard URL: http://localhost:3000/d/kafka-producer-monitoring
```

### 2. Deploy Prometheus Alert Rules

```bash
# Set Prometheus configuration path
export PROMETHEUS_CONFIG_DIR="/etc/prometheus"

# Deploy alert rules (with validation)
./scripts/deploy-prometheus-alerts.sh

# Expected output:
# [INFO] Validating alert rules YAML...
# [INFO] Alert rules YAML is valid
# [INFO] Checking Prometheus API...
# [INFO] Prometheus API is accessible
# [INFO] Copying alert rules to /etc/prometheus/rules/
# [INFO] Reloading Prometheus configuration...
# [INFO] Alert rules deployed successfully
```

### 3. Validate Deployment

```bash
# Run health checks
./scripts/check-monitoring-health.sh

# Expected output:
# [INFO] Checking Grafana dashboard...
# [OK] Dashboard exists: kafka-producer-monitoring
# [INFO] Checking Prometheus alert rules...
# [OK] 8 alert rules loaded
# [INFO] Checking metrics availability...
# [OK] 9 metrics scraped in last 5 minutes
# [INFO] Monitoring health check: PASSED
```

## Detailed Deployment Steps

### Step 1: Validate Configuration Files

Before deployment, validate all configuration files:

```bash
# Validate Grafana dashboard JSON
python3 scripts/validate_dashboard_json.py docs/monitoring/grafana-dashboard.json

# Validate Prometheus alert rules YAML
python3 scripts/validate_alerts_yaml.py docs/monitoring/alert-rules.yaml

# Check PromQL syntax
python3 scripts/validate_promql_syntax.py docs/monitoring/alert-rules.yaml
```

**Expected Results**:
- Dashboard JSON: 9 panels, all with valid PromQL queries
- Alert rules YAML: 6+ alerts, all with severity labels
- PromQL syntax: All expressions valid (balanced parentheses, valid functions)

### Step 2: Deploy Grafana Dashboard

#### 2.1 Configure Grafana Connection

```bash
# Option 1: Environment variables
export GRAFANA_URL="http://grafana.example.com:3000"
export GRAFANA_API_KEY="your-api-key-here"

# Option 2: Credentials file
cat > ~/.grafana_creds << EOF
GRAFANA_URL=http://grafana.example.com:3000
GRAFANA_USER=admin
GRAFANA_PASSWORD=secret
EOF
source ~/.grafana_creds
```

#### 2.2 Deploy Dashboard

```bash
# Dry-run first (recommended)
./scripts/deploy-grafana-dashboard.sh --dry-run

# Actual deployment
./scripts/deploy-grafana-dashboard.sh

# Verify deployment
curl -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
  "$GRAFANA_URL/api/dashboards/uid/kafka-producer-monitoring" | jq '.dashboard.title'
```

**Expected Output**:
```json
"Kafka Producer Monitoring - market-data-kafka-producer"
```

#### 2.3 Configure Data Source

Ensure Prometheus data source is configured in Grafana:

```bash
# Create Prometheus data source via API
curl -X POST \
  -u "$GRAFANA_USER:$GRAFANA_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://localhost:9090",
    "access": "proxy",
    "isDefault": true
  }' \
  "$GRAFANA_URL/api/datasources"
```

### Step 3: Deploy Prometheus Alert Rules

#### 3.1 Configure Prometheus

Add alert rules file to Prometheus configuration:

```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 10s
  evaluation_interval: 10s

rule_files:
  - '/etc/prometheus/rules/alert-rules.yaml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

scrape_configs:
  - job_name: 'cryptofeed-kafka-producer'
    static_configs:
      - targets: ['localhost:8080']
        labels:
          service: 'kafka-producer'
```

#### 3.2 Deploy Alert Rules

```bash
# Copy alert rules to Prometheus rules directory
sudo cp docs/monitoring/alert-rules.yaml /etc/prometheus/rules/

# Validate Prometheus configuration
promtool check config /etc/prometheus/prometheus.yml

# Reload Prometheus (without restart)
curl -X POST http://localhost:9090/-/reload
```

#### 3.3 Verify Alert Rules Loaded

```bash
# Query Prometheus for loaded rules
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name == "kafka_producer_alerts") | .rules[].alert'

# Expected output (alert names):
# "KafkaProducerErrorRateHigh"
# "ConsumerLagHigh"
# "KafkaBrokerDown"
# "ProducerLatencyHigh"
# "DLQMessageRateHigh"
# "KafkaTopicPartitionUnbalanced"
```

### Step 4: Configure Alerting Channels

#### 4.1 Slack Integration

Create Slack webhook URL and configure Alertmanager:

```yaml
# /etc/alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  receiver: 'cryptofeed-team'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'cryptofeed-team'
    slack_configs:
      - channel: '#data-alerts'
        title: 'Cryptofeed Kafka Producer Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'
        send_resolved: true

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}'

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#data-warnings'
        title: 'Warning: {{ .GroupLabels.alertname }}'
```

#### 4.2 Test Alert Notifications

```bash
# Send test alert to Alertmanager
curl -H "Content-Type: application/json" -d '[{
  "labels": {
    "alertname": "TestAlert",
    "severity": "warning"
  },
  "annotations": {
    "summary": "Test alert from deployment script"
  }
}]' http://localhost:9093/api/v1/alerts

# Check Slack channel for notification
```

### Step 5: Validate Monitoring Infrastructure

Run comprehensive health checks:

```bash
# Full monitoring health check
./scripts/check-monitoring-health.sh

# Individual checks
./scripts/check-monitoring-health.sh --check-grafana
./scripts/check-monitoring-health.sh --check-prometheus
./scripts/check-monitoring-health.sh --check-alerts
./scripts/check-monitoring-health.sh --check-metrics
```

**Health Check Criteria**:

| Check | Pass Criteria | Failure Action |
|-------|---------------|----------------|
| Grafana Dashboard | Dashboard exists, 9 panels | Re-deploy dashboard |
| Prometheus Alerts | 6+ alert rules loaded | Re-deploy alert rules |
| Metrics Availability | All 9 metrics scraped <5min ago | Check producer /metrics endpoint |
| Alert Firing Test | Test alert reaches Slack | Check Alertmanager config |

## Rollback Procedures

### Rollback Grafana Dashboard

```bash
# Backup current dashboard before rollback
./scripts/rollback-grafana-dashboard.sh --backup

# Rollback to previous version
./scripts/rollback-grafana-dashboard.sh --version previous

# Rollback to specific version
./scripts/rollback-grafana-dashboard.sh --version 2
```

### Rollback Prometheus Alert Rules

```bash
# Backup current alert rules
sudo cp /etc/prometheus/rules/alert-rules.yaml /etc/prometheus/rules/alert-rules.yaml.backup

# Restore previous alert rules
sudo cp /etc/prometheus/rules/alert-rules.yaml.backup /etc/prometheus/rules/alert-rules.yaml

# Reload Prometheus
curl -X POST http://localhost:9090/-/reload
```

## Troubleshooting

### Dashboard Not Showing Data

**Symptoms**: Grafana dashboard panels show "No data"

**Diagnosis**:
```bash
# Check Prometheus is scraping metrics
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job == "cryptofeed-kafka-producer")'

# Check metrics endpoint is accessible
curl http://localhost:8080/metrics | grep cryptofeed_kafka_messages_sent_total

# Test PromQL query directly
curl 'http://localhost:9090/api/v1/query?query=rate(messages_produced_total[5m])'
```

**Resolution**:
1. Ensure KafkaCallback is running with metrics endpoint enabled
2. Verify Prometheus scrape job is configured correctly
3. Check network connectivity between Prometheus and producer

### Alert Rules Not Firing

**Symptoms**: Alerts expected to fire are not triggering

**Diagnosis**:
```bash
# Check alert rules are loaded
curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name == "kafka_producer_alerts")'

# Check alert evaluation
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.alertname == "KafkaProducerErrorRateHigh")'

# Simulate alert condition (test script)
python3 scripts/test_prometheus_alerts.py --alert KafkaProducerErrorRateHigh
```

**Resolution**:
1. Verify PromQL expressions are correct for current metrics
2. Check alert evaluation interval in prometheus.yml
3. Ensure Alertmanager is configured and reachable

### Grafana API Authentication Fails

**Symptoms**: `deploy-grafana-dashboard.sh` fails with 401 Unauthorized

**Diagnosis**:
```bash
# Test Grafana authentication
curl -u "$GRAFANA_USER:$GRAFANA_PASSWORD" http://localhost:3000/api/health

# Check API key (if using)
curl -H "Authorization: Bearer $GRAFANA_API_KEY" http://localhost:3000/api/health
```

**Resolution**:
1. Verify credentials are correct
2. Create new API key if password authentication is disabled
3. Check Grafana logs for authentication errors

## Success Criteria

Deployment is considered successful when all of the following are true:

- [ ] Grafana dashboard deployed with 9 panels visible
- [ ] All dashboard panels show data (no "No data" errors)
- [ ] Prometheus has 6+ alert rules loaded
- [ ] Test alert successfully reaches Slack/PagerDuty
- [ ] Health check script passes all validation
- [ ] Metrics scraped within last 5 minutes
- [ ] Rollback procedures tested and documented

## Metric Definitions

### Core Metrics (9 total)

1. **messages_produced_total** (Counter)
   - Description: Total messages successfully produced to Kafka
   - Labels: `exchange`, `symbol`, `data_type`
   - Query: `rate(messages_produced_total[5m])`

2. **produce_latency_seconds** (Histogram)
   - Description: Message production latency from callback to Kafka ACK
   - Labels: `exchange`, `data_type`
   - Query: `histogram_quantile(0.99, rate(produce_latency_seconds_bucket[5m]))`

3. **produce_errors_total** (Counter)
   - Description: Total message production errors
   - Labels: `exchange`, `data_type`, `error_type`
   - Query: `rate(produce_errors_total[5m])`

4. **kafka_broker_latency_seconds** (Histogram)
   - Description: Round-trip latency to Kafka broker
   - Labels: `broker_id`, `operation`
   - Query: `histogram_quantile(0.99, rate(kafka_broker_latency_seconds_bucket[5m]))`

5. **kafka_partition_lag_records** (Gauge)
   - Description: Number of records behind in partition (consumer lag)
   - Labels: `partition`
   - Query: `max(kafka_partition_lag_records)`

6. **kafka_buffer_utilization_percent** (Gauge)
   - Description: Percentage of producer buffer pool in use
   - Labels: `producer_id`
   - Query: `kafka_buffer_utilization_percent`

7. **message_size_bytes** (Histogram)
   - Description: Distribution of serialized message sizes
   - Labels: `data_type`, `compression_enabled`
   - Query: `histogram_quantile(0.95, rate(message_size_bytes_bucket[5m]))`

8. **serialization_latency_seconds** (Histogram)
   - Description: Time to serialize message to protobuf format
   - Labels: `data_type`
   - Query: `histogram_quantile(0.99, rate(serialization_latency_seconds_bucket[5m]))`

9. **cryptofeed_kafka_consumer_lag_messages** (Gauge)
   - Description: Consumer group lag in messages
   - Labels: `consumer_group`, `exchange`
   - Query: `cryptofeed_kafka_consumer_lag_messages`

## Alert Runbooks

### Critical Alerts

#### KafkaProducerErrorRateHigh
- **Threshold**: Error rate > 1% for 5 minutes
- **Impact**: Message loss, data quality issues
- **Runbook**: `/docs/kafka/troubleshooting.md#error-rate-high`
- **Actions**:
  1. Check producer logs for exception details
  2. Verify Kafka broker availability (disk space, network)
  3. Check message serialization (protobuf schema changes)
  4. Review consumer group coordination (rebalancing)

#### ConsumerLagHigh
- **Threshold**: Lag > 30 seconds (30 messages @ 1 msg/s avg)
- **Impact**: Stale data, delayed analytics
- **Runbook**: `/docs/kafka/troubleshooting.md#lag-high`
- **Actions**:
  1. Check consumer group status (rebalancing?)
  2. Scale up consumers (increase replicas)
  3. Check broker load (CPU, memory, disk)
  4. Reduce consumer batch size

#### KafkaBrokerDown
- **Threshold**: Broker state = 'down' for 1 minute
- **Impact**: Partition unavailability, message loss
- **Runbook**: `/docs/kafka/troubleshooting.md#broker-down`
- **Actions**:
  1. SSH to broker and check process status
  2. Check disk space and logs
  3. Verify network connectivity
  4. Restart broker if needed

### Warning Alerts

#### ProducerLatencyHigh
- **Threshold**: P99 latency > 10ms for 10 minutes
- **Impact**: Slower message processing
- **Runbook**: `/docs/kafka/troubleshooting.md#latency-high`

#### DLQMessageRateHigh
- **Threshold**: DLQ rate > 0.1% for 5 minutes
- **Impact**: Data quality issues
- **Runbook**: `/docs/kafka/troubleshooting.md#dlq-high`

## References

- **Prometheus Official Docs**: https://prometheus.io/docs/
- **Grafana Dashboards**: https://grafana.com/docs/grafana/latest/dashboards/
- **Kafka Monitoring**: https://kafka.apache.org/documentation/#monitoring
- **PromQL Tutorial**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Alertmanager Config**: https://prometheus.io/docs/alerting/latest/configuration/

## Changelog

- **2025-11-26**: Initial deployment guide created (Task 22)
- **Future**: Add automated deployment via CI/CD pipeline
