# Prometheus Metrics Setup Guide

**Status**: Production Ready
**Date**: November 11, 2025
**Scope**: Kafka producer monitoring and alerting
**Owner**: Observability Team

## Overview

This guide provides setup instructions for monitoring the market-data-kafka-producer using Prometheus and Grafana. The producer exposes 9 metrics across three categories:

| Category | Metrics | Purpose |
|----------|---------|---------|
| **Producer** | messages_produced_total, produce_latency_seconds, produce_errors_total, producer_buffer_usage_bytes | Measure production success, latency, errors |
| **Kafka** | kafka_broker_latency_seconds, kafka_partition_lag_records, kafka_buffer_utilization_percent | Monitor broker health and consumer lag |
| **Serialization** | message_size_bytes, serialization_latency_seconds | Track serialization performance |

---

## Prometheus Configuration

### 1. Add Kafka Producer Scrape Job

Edit your `prometheus.yml` configuration file and add the following scrape job:

```yaml
global:
  scrape_interval: 10s       # Default scrape interval
  evaluation_interval: 10s    # Default evaluation interval
  external_labels:
    cluster: 'cryptofeed'

scrape_configs:
  # Kafka Producer Metrics
  - job_name: 'cryptofeed-kafka-producer'
    static_configs:
      - targets: ['localhost:8080']  # KafkaCallback /metrics endpoint
        labels:
          environment: 'production'
          service: 'kafka-producer'
    scrape_interval: 10s
    scrape_timeout: 5s
    metrics_path: '/metrics'
```

### 2. Data Retention Configuration

For production deployments, configure appropriate retention:

```yaml
global:
  # ...
  external_labels:
    prometheus: 'cryptofeed'

# Remote storage (optional, for long-term retention)
remote_write:
  - url: "http://remote-prometheus:9090/api/v1/write"
    write_relabel_configs:
      - source_labels: [__name__]
        regex: 'cryptofeed.*'
        action: keep
```

**Retention Recommendations**:
- Development: 7 days (minimal storage)
- Staging: 14 days
- Production: 30 days (or use remote storage)

### 3. Alertmanager Configuration

Configure Alertmanager for routing alerts:

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  receiver: 'cryptofeed-team'
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

receivers:
  - name: 'cryptofeed-team'
    slack_configs:
      - channel: '#kafka-alerts'
        title: 'Cryptofeed Kafka Producer Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'
        send_resolved: true
```

---

## Metrics Reference

### Producer Metrics

#### 1. `messages_produced_total` (Counter)

**Type**: Counter
**Unit**: Count
**Labels**: `exchange`, `symbol`, `data_type`, `partition_strategy`
**Description**: Total number of messages successfully produced to Kafka brokers

**Query Examples**:
```promql
# Messages per exchange
sum by (exchange) (rate(messages_produced_total[5m]))

# Messages per data type
sum by (data_type) (rate(messages_produced_total[5m]))

# Total messages in last hour
increase(messages_produced_total[1h])
```

**Interpretation**:
- Increasing counter indicates healthy producer
- Flat line indicates producer stalled or disconnected
- Sudden drop suggests producer restart or Kafka connection loss

#### 2. `produce_latency_seconds` (Histogram)

**Type**: Histogram
**Unit**: Seconds
**Labels**: `exchange`, `data_type`
**Buckets**: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
**Description**: Message production latency from callback to broker acknowledgment

**Query Examples**:
```promql
# P99 latency (99th percentile)
histogram_quantile(0.99, rate(produce_latency_seconds_bucket[5m]))

# P95 latency
histogram_quantile(0.95, rate(produce_latency_seconds_bucket[5m]))

# Average latency per exchange
sum by (exchange) (rate(produce_latency_seconds_sum[5m])) /
sum by (exchange) (rate(produce_latency_seconds_count[5m]))

# P99 latency by data type
histogram_quantile(0.99, sum by (data_type, le) (rate(produce_latency_seconds_bucket[5m])))
```

**Interpretation**:
- Latency < 5ms: Excellent (target)
- Latency 5-10ms: Good
- Latency 10-50ms: Acceptable with warnings
- Latency > 50ms: Degraded, investigate

**Target SLA**: p99 < 15ms

#### 3. `produce_errors_total` (Counter)

**Type**: Counter
**Unit**: Count
**Labels**: `exchange`, `data_type`, `error_type`
**Description**: Total number of message production errors

**Error Types**:
- `serialization_error`: Protobuf serialization failed
- `kafka_error`: Kafka broker error (auth, connection, etc.)
- `network_error`: Network connectivity issue
- `buffer_full`: Producer buffer exhausted
- `timeout`: Request timeout

**Query Examples**:
```promql
# Error rate (%)
100 * (
  sum(rate(produce_errors_total[5m])) /
  (sum(rate(messages_produced_total[5m])) + sum(rate(produce_errors_total[5m])))
)

# Errors by type
sum by (error_type) (rate(produce_errors_total[5m]))

# Errors per exchange
sum by (exchange) (rate(produce_errors_total[5m]))
```

**Interpretation**:
- 0 errors: Healthy
- < 0.1% error rate: Acceptable
- 0.1-1% error rate: Monitor closely, investigate trends
- > 1% error rate: Alert immediately

**Target SLA**: < 0.1% error rate

#### 4. `producer_buffer_usage_bytes` (Gauge)

**Type**: Gauge
**Unit**: Bytes
**Labels**: `producer_id`
**Description**: Current bytes in producer buffer waiting for transmission

**Query Examples**:
```promql
# Current buffer usage (MB)
producer_buffer_usage_bytes / 1024 / 1024

# Buffer usage trend (4h window)
rate(producer_buffer_usage_bytes[4h])

# Alert if buffer grows
rate(producer_buffer_usage_bytes[10m]) > 0
```

**Interpretation**:
- 0-50%: Healthy
- 50-80%: Monitor, may indicate slow broker
- > 80%: Alert, buffer filling up
- > 95%: Critical, producer may drop messages

**Target SLA**: < 50% utilization

---

### Kafka Metrics

#### 5. `kafka_broker_latency_seconds` (Histogram)

**Type**: Histogram
**Unit**: Seconds
**Labels**: `broker_id`, `operation`
**Buckets**: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
**Description**: Round-trip latency to Kafka broker

**Operations**:
- `produce`: Produce request
- `fetch_metadata`: Metadata fetch request
- `list_offsets`: Offset list request

**Query Examples**:
```promql
# P99 broker latency
histogram_quantile(0.99, rate(kafka_broker_latency_seconds_bucket[5m]))

# Latency by broker
histogram_quantile(0.99, sum by (broker_id, le) (rate(kafka_broker_latency_seconds_bucket[5m])))

# Latency by operation
histogram_quantile(0.99, sum by (operation, le) (rate(kafka_broker_latency_seconds_bucket[5m])))
```

**Interpretation**:
- < 5ms: Healthy
- 5-20ms: Good, no action needed
- 20-50ms: Slow broker, investigate network/broker CPU
- > 50ms: Critical, broker performance degraded

#### 6. `kafka_partition_lag_records` (Gauge)

**Type**: Gauge
**Unit**: Records
**Labels**: `partition`
**Description**: Number of records behind in partition (consumer lag)

**Query Examples**:
```promql
# Total partition lag
sum(kafka_partition_lag_records)

# Max lag per partition
max(kafka_partition_lag_records)

# Lag growth rate
rate(kafka_partition_lag_records[5m])
```

**Interpretation**:
- 0-100: Healthy
- 100-1000: Monitor consumer group, may be slow
- > 1000: Alert, consumer group lagging significantly
- Growing continuously: Consumers unable to keep up

**Target SLA**: < 100 records lag

#### 7. `kafka_buffer_utilization_percent` (Gauge)

**Type**: Gauge
**Unit**: Percentage (0-100)
**Labels**: `producer_id`
**Description**: Percentage of producer buffer pool in use

**Query Examples**:
```promql
# Current utilization
kafka_buffer_utilization_percent

# Average utilization (5m window)
avg_over_time(kafka_buffer_utilization_percent[5m])

# Alert threshold
kafka_buffer_utilization_percent > 80
```

**Interpretation**:
- 0-50%: Healthy
- 50-80%: Good, monitor for trends
- 80-95%: Warning, may have slow brokers
- > 95%: Alert, critical buffer pressure

---

### Serialization Metrics

#### 8. `message_size_bytes` (Histogram)

**Type**: Histogram
**Unit**: Bytes
**Labels**: `data_type`, `compression_enabled`
**Buckets**: [100, 250, 500, 1000, 2500, 5000, 10000]
**Description**: Distribution of serialized message sizes

**Query Examples**:
```promql
# Average message size by data type
avg by (data_type) (message_size_bytes)

# P95 message size
histogram_quantile(0.95, rate(message_size_bytes_bucket[5m]))

# Compression impact (uncompressed vs compressed)
sum by (compression_enabled) (rate(message_size_bytes_sum[5m])) /
sum by (compression_enabled) (rate(message_size_bytes_count[5m]))
```

**Expected Sizes** (protobuf):
- Trade: 200-400 bytes
- Ticker: 150-300 bytes
- OrderBook: 1000-5000 bytes
- Candle: 300-600 bytes
- Liquidation: 200-400 bytes

**Interpretation**:
- Message sizes consistent: Normal
- Growing message sizes: May indicate larger payloads or schema changes
- > 10KB: Potential issue, investigate

#### 9. `serialization_latency_seconds` (Histogram)

**Type**: Histogram
**Unit**: Seconds
**Labels**: `data_type`
**Buckets**: [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
**Description**: Time to serialize message to protobuf format

**Query Examples**:
```promql
# P99 serialization latency
histogram_quantile(0.99, rate(serialization_latency_seconds_bucket[5m]))

# Average latency by data type
avg by (data_type) (rate(serialization_latency_seconds_sum[5m]) /
                    rate(serialization_latency_seconds_count[5m]))

# Latency growth over time
rate(serialization_latency_seconds_sum[5m]) /
rate(serialization_latency_seconds_count[5m])
```

**Expected Latencies** (protobuf):
- Trade: 20-50µs
- Ticker: 15-40µs
- OrderBook: 100-200µs
- Candle: 30-60µs

**Interpretation**:
- < 100µs: Excellent
- 100-500µs: Good
- 500µs - 1ms: Slow, investigate schema complexity
- > 1ms: Critical, likely regression

---

## Alert Rules

The following alert rules are defined in `alert-rules.yaml`:

### Critical Alerts (Page immediately)

1. **KafkaProducerErrorRateHigh**: Error rate > 1% for 5 minutes
   - Severity: critical
   - Action: Check producer logs, verify Kafka brokers online

2. **KafkaProducerLatencyHigh**: P99 latency > 50ms for 10 minutes
   - Severity: critical
   - Action: Check Kafka broker health, network latency

3. **KafkaProducerBufferCritical**: Buffer utilization > 95% for 5 minutes
   - Severity: critical
   - Action: Check Kafka broker responsiveness, may need to scale

### Warning Alerts (Notify team)

4. **KafkaProducerLatencyWarning**: P99 latency > 15ms for 10 minutes
   - Severity: warning
   - Action: Monitor trends, optimize producer config

5. **KafkaProducerBufferWarning**: Buffer utilization > 80% for 10 minutes
   - Severity: warning
   - Action: Investigate slow brokers, check network

6. **KafkaPartitionLagHigh**: Partition lag > 100 records for 15 minutes
   - Severity: warning
   - Action: Scale consumer group, check processing performance

7. **KafkaProducerErrorsDetected**: Any errors for 1 minute
   - Severity: warning
   - Action: Check producer logs for error patterns

### Info Alerts (Dashboard only)

8. **KafkaProducerLowThroughput**: Messages < 100/sec for 5 minutes
   - Severity: info
   - Action: Check if intentional (low market activity)

---

## Grafana Dashboard Setup

### 1. Import Dashboard JSON

The Grafana dashboard template is available in `docs/monitoring/grafana-dashboard.json`.

**Import steps**:
1. Open Grafana UI (http://localhost:3000)
2. Go to Dashboards → Import
3. Copy-paste content from `grafana-dashboard.json`
4. Select Prometheus data source
5. Click Import

### 2. Dashboard Panels

The dashboard includes the following panels:

**Row 1: Production Metrics**
- Messages Produced (rate graph, stacked by data_type)
- Error Rate (percentage, threshold line at 1%)
- P99 Latency (graph with threshold at 15ms)

**Row 2: Buffer & Throughput**
- Buffer Utilization (gauge, threshold at 80%)
- Message Throughput (rate, data types)
- Broker Latency (P99 by broker)

**Row 3: Serialization**
- Message Size Distribution (by data_type)
- Serialization Latency (P95/P99)
- Compression Impact (uncompressed vs compressed)

**Row 4: Kafka Health**
- Partition Lag (by partition)
- Broker Connection Status
- Topic Replication Health

### 3. Customization

To customize for your environment:

1. **Adjust thresholds**: Edit alert lines in each panel
2. **Add custom labels**: Modify queries to include cluster, environment
3. **Add dimensions**: Add panels for specific exchanges or data types
4. **Set refresh rate**: Dashboard → Refresh every 10s (default)

---

## Running Prometheus and Grafana Locally

### Docker Compose Setup

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert-rules.yaml:/etc/prometheus/alert-rules.yaml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--alert.rules-path=/etc/prometheus/alert-rules.yaml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'

volumes:
  prometheus_data:
  grafana_data:
```

**Start services**:
```bash
docker-compose up -d
```

**Access**:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Alertmanager: http://localhost:9093

---

## Troubleshooting

### Metrics Not Appearing

1. **Check KafkaCallback is exporting metrics**:
   ```bash
   curl http://localhost:8080/metrics | grep messages_produced_total
   ```

2. **Verify Prometheus scrape job**:
   - Prometheus UI → Status → Targets
   - Check if kafka-producer target is "UP"

3. **Check metrics in TSDB**:
   ```promql
   # In Prometheus query UI
   {job="cryptofeed-kafka-producer"}
   ```

### High Alert False Positives

1. **Adjust scrape/evaluation intervals** in prometheus.yml
2. **Increase alert duration thresholds** (e.g., 5m → 10m)
3. **Add environment-specific label filters**

### Missing Historical Data

1. **Check storage size**: `du -sh ./prometheus_data/`
2. **Adjust retention_days** in prometheus.yml
3. **Consider remote storage** for long-term retention

---

## Production Deployment Checklist

- [ ] Prometheus configured with 30-day retention
- [ ] Alertmanager configured with notification channels (Slack, PagerDuty, etc.)
- [ ] Alert rules reviewed and thresholds validated for environment
- [ ] Grafana dashboards imported and customized
- [ ] Dashboard refresh rate set to 10s
- [ ] Team trained on interpreting alerts
- [ ] Runbook linked from Grafana alerts
- [ ] Metrics retention backed up (optional)
- [ ] Alert notification channels tested
- [ ] On-call rotation configured in PagerDuty/similar

---

## References

- **Prometheus Official Docs**: https://prometheus.io/docs/
- **Alert Writing Guide**: https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/
- **Grafana Dashboards**: https://grafana.com/grafana/dashboards
- **Kafka Monitoring**: https://kafka.apache.org/documentation/#monitoring
- **PromQL Tutorial**: https://prometheus.io/docs/prometheus/latest/querying/basics/

