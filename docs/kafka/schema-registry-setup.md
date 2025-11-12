# Schema Registry Setup Guide

Task 18 - SchemaRegistry Integration Setup for Confluent and Buf Registries

## Table of Contents

1. [Overview](#overview)
2. [Confluent Schema Registry Setup](#confluent-schema-registry-setup)
3. [Buf Schema Registry Setup](#buf-schema-registry-setup)
4. [KafkaCallback Integration](#kafkacallback-integration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring](#monitoring)

---

## Overview

Cryptofeed integrates with schema registries to:
- **Register** protobuf schemas for all message types
- **Validate** schema compatibility before publishing
- **Embed** schema IDs in Kafka message headers
- **Cache** schemas for performance
- **Enforce** backward/forward compatibility

### Supported Registries

| Registry | Type | Protocol | Status | Authentication |
|----------|------|----------|--------|---|
| **Confluent Schema Registry** | HTTP | HTTP/HTTPS | Recommended | Username/Password or mTLS |
| **Buf Schema Registry** | gRPC | gRPC/gRPCS | Beta | Bearer Token |

---

## Confluent Schema Registry Setup

### Installation Options

#### Option 1: Docker (Recommended for Development)

```bash
# Start Confluent Schema Registry with Docker
docker run -d \
  --name schema-registry \
  --net kafka-network \
  -p 8081:8081 \
  -e SCHEMA_REGISTRY_HOST_NAME=schema-registry \
  -e SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS=kafka:9092 \
  -e SCHEMA_REGISTRY_LISTENERS=http://0.0.0.0:8081 \
  confluentinc/cp-schema-registry:7.6.0
```

#### Option 2: Docker Compose (Complete Stack)

Create `docker-compose-schema-registry.yml`:

```yaml
version: "3"

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_SYNC_LIMIT: 2
      ZOOKEEPER_INIT_LIMIT: 5

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"

  schema-registry:
    image: confluentinc/cp-schema-registry:7.6.0
    ports:
      - "8081:8081"
    depends_on:
      - kafka
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:9092
      SCHEMA_REGISTRY_LISTENERS: http://0.0.0.0:8081
      SCHEMA_REGISTRY_DEBUG: "true"
```

Start stack:
```bash
docker-compose -f docker-compose-schema-registry.yml up -d
```

Verify:
```bash
# Test Schema Registry connectivity
curl http://localhost:8081/subjects

# Should return: [] (empty list initially)
```

#### Option 3: Kubernetes (Production)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: schema-registry
  namespace: data-platform

spec:
  replicas: 3
  selector:
    matchLabels:
      app: schema-registry

  template:
    metadata:
      labels:
        app: schema-registry

    spec:
      containers:
      - name: schema-registry
        image: confluentinc/cp-schema-registry:7.6.0
        ports:
        - containerPort: 8081

        env:
        - name: SCHEMA_REGISTRY_HOST_NAME
          valueFrom:
            fieldRef:
              fieldPath: status.podIP

        - name: SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS
          value: "kafka-cluster:9092"

        - name: SCHEMA_REGISTRY_LISTENERS
          value: "http://0.0.0.0:8081"

        # Enable authentication
        - name: SCHEMA_REGISTRY_AUTHENTICATION_METHOD
          value: "BASIC"

        - name: SCHEMA_REGISTRY_AUTHENTICATION_ROLES
          value: "superuser"

        # Resource limits
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

        # Health checks
        livenessProbe:
          httpGet:
            path: /subjects
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /subjects
            port: 8081
          initialDelaySeconds: 20
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: schema-registry-service
  namespace: data-platform

spec:
  selector:
    app: schema-registry

  ports:
  - protocol: TCP
    port: 8081
    targetPort: 8081

  type: ClusterIP
```

Deploy:
```bash
kubectl apply -f schema-registry-deployment.yaml
```

### Configuration

Create `config/schema-registry.yaml`:

```yaml
schema_registry:
  # Connection
  type: confluent
  url: http://localhost:8081

  # Authentication (optional)
  username: ${SCHEMA_REGISTRY_USER}
  password: ${SCHEMA_REGISTRY_PASSWORD}

  # TLS/mTLS (optional)
  tls_enabled: false
  tls_ca_cert: /etc/ssl/certs/ca.pem
  tls_client_cert: /etc/ssl/certs/client.pem
  tls_client_key: /etc/ssl/private/client.key

  # Compatibility
  compatibility_mode: BACKWARD  # BACKWARD, FORWARD, FULL, TRANSITIVE

  # Caching
  cache_size: 1000
  cache_ttl_seconds: 3600

  # Timeouts
  connect_timeout_seconds: 10
  request_timeout_seconds: 30
```

### Testing Confluent Registry

```python
"""Test Confluent Schema Registry integration."""

from cryptofeed.backends.kafka_schema import (
    SchemaRegistryConfig,
    ConfluentSchemaRegistry,
    CompatibilityMode,
)

# Configuration
config = SchemaRegistryConfig(
    registry_type="confluent",
    url="http://localhost:8081",
    username="user",  # optional
    password="pass",  # optional
    compatibility_mode=CompatibilityMode.BACKWARD,
)

# Create registry client
registry = ConfluentSchemaRegistry(config)

# Register a schema
schema = """{
    "type": "record",
    "name": "Trade",
    "namespace": "cryptofeed.schema.v1",
    "fields": [
        {"name": "symbol", "type": "string"},
        {"name": "price", "type": "double"},
        {"name": "amount", "type": "double"},
        {"name": "timestamp", "type": "long"}
    ]
}"""

schema_id = registry.register_schema(
    subject="cryptofeed-trades",
    schema=schema,
    schema_type="AVRO"
)

print(f"Registered schema with ID: {schema_id}")

# Retrieve schema
schema_info = registry.get_schema_by_id(schema_id)
print(f"Schema: {schema_info}")

# Check compatibility
new_schema = """{
    "type": "record",
    "name": "Trade",
    "namespace": "cryptofeed.schema.v1",
    "fields": [
        {"name": "symbol", "type": "string"},
        {"name": "price", "type": "double"},
        {"name": "amount", "type": "double"},
        {"name": "timestamp", "type": "long"},
        {"name": "exchange", "type": ["null", "string"], "default": null}
    ]
}"""

is_compatible = registry.check_compatibility(
    subject="cryptofeed-trades",
    schema=new_schema,
)

if is_compatible:
    print("✅ New schema is compatible!")
else:
    print("❌ New schema is NOT compatible!")
```

---

## Buf Schema Registry Setup

### Installation

Buf Schema Registry is available as:
- SaaS (hosted by Buf)
- Self-hosted (gRPC service)

### Option 1: Buf SaaS (Recommended)

1. **Create Buf Account**
   ```bash
   # Register at https://buf.build
   # Create organization and repository
   ```

2. **Generate API Token**
   ```bash
   # At https://buf.build/settings/tokens
   # Create API token for cryptofeed-data
   ```

3. **Configure**
   ```yaml
   schema_registry:
     type: buf
     url: grpc://buf.build:443
     api_token: ${BUF_API_TOKEN}
   ```

### Option 2: Self-Hosted Buf

#### Docker Setup

```bash
# Buf Schema Registry requires running Buf locally
docker run -d \
  --name buf-registry \
  -p 5051:5051 \
  bufbuild/buf-alpha-registry:latest
```

#### Kubernetes Setup

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: buf-registry
  namespace: data-platform

spec:
  replicas: 2
  selector:
    matchLabels:
      app: buf-registry

  template:
    metadata:
      labels:
        app: buf-registry

    spec:
      containers:
      - name: buf-registry
        image: bufbuild/buf-alpha-registry:latest
        ports:
        - containerPort: 5051

        env:
        - name: BUF_REGISTRY_BIND_ADDRESS
          value: "0.0.0.0:5051"

        # Resource limits
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "250m"
```

### Testing Buf Registry

```python
"""Test Buf Schema Registry integration."""

from cryptofeed.backends.kafka_schema import (
    SchemaRegistryConfig,
    BufSchemaRegistry,
)

# Configuration for Buf (SaaS)
config = SchemaRegistryConfig(
    registry_type="buf",
    url="grpc://buf.build:443",
    api_token="YOUR_BUF_TOKEN",
)

# Or for self-hosted
config = SchemaRegistryConfig(
    registry_type="buf",
    url="grpc://localhost:5051",
    api_token="YOUR_TOKEN",
)

# Create registry client
registry = BufSchemaRegistry(config)

# Register schema (proto format)
schema = """
syntax = "proto3";

package cryptofeed.schema.v1;

message Trade {
  string symbol = 1;
  double price = 2;
  double amount = 3;
  int64 timestamp = 4;
}
"""

schema_id = registry.register_schema(
    subject="cryptofeed/schema/trades",
    schema=schema,
    schema_type="PROTOBUF"
)

print(f"Registered schema with ID: {schema_id}")
```

---

## KafkaCallback Integration

### Configuration

Update `config/kafka.yaml`:

```yaml
kafka:
  # Existing Kafka configuration
  bootstrap_servers:
    - kafka1:9092
    - kafka2:9092
    - kafka3:9092

  producer:
    acks: all
    compression.type: snappy

  # NEW: Schema Registry integration
  schema_registry:
    enabled: true

    # Choose one: confluent or buf
    type: confluent
    url: http://schema-registry:8081

    # Authentication (for Confluent)
    username: ${SCHEMA_REGISTRY_USER:-}
    password: ${SCHEMA_REGISTRY_PASSWORD:-}

    # Authentication (for Buf)
    api_token: ${BUF_API_TOKEN:-}

    # Compatibility checking
    compatibility_mode: BACKWARD

    # Performance
    cache_size: 1000
    cache_ttl_seconds: 3600

    # Embedding
    embed_schema_id: true
    schema_id_in_headers: true
```

### Python Integration

```python
"""Integrate schema registry with KafkaCallback."""

from cryptofeed.kafka_callback import KafkaCallback
from cryptofeed.backends.kafka_schema import (
    SchemaRegistry,
    SchemaRegistryConfig,
)
from cryptofeed.feed_handler import FeedHandler

# Create schema registry
schema_config = SchemaRegistryConfig(
    registry_type="confluent",
    url="http://localhost:8081",
    compatibility_mode="BACKWARD",
)
schema_registry = SchemaRegistry.create(schema_config)

# Create Kafka callback with schema registry
kafka_callback = KafkaCallback(
    bootstrap_servers=["localhost:9092"],
    schema_registry=schema_registry,
    schema_registry_enabled=True,
    embed_schema_id=True,
)

# Add to feed handler
feed_handler = FeedHandler()
feed_handler.add_callback(kafka_callback, ["trades", "orderbook"])

# Start ingestion
feed_handler.start()
```

---

## Verification

### Manual Verification

```bash
# 1. Check Schema Registry is running
curl -s http://localhost:8081/subjects | jq .

# 2. Register a test schema
curl -X POST \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  -d '{"schema": "{\"type\": \"record\", \"name\": \"test\"}"}' \
  http://localhost:8081/subjects/test-subject/versions

# 3. List all subjects
curl -s http://localhost:8081/subjects | jq .

# 4. Get schema by ID
curl -s http://localhost:8081/schemas/ids/1 | jq .
```

### Integration Tests

```bash
# Run schema registry integration tests
python -m pytest tests/unit/kafka/test_schema_registry.py -v

# Run with live Confluent registry
pytest tests/integration/test_schema_registry_confluent.py -v -s

# Run with live Buf registry
pytest tests/integration/test_schema_registry_buf.py -v -s
```

### Docker Compose Health Check

```bash
# Start stack
docker-compose -f docker-compose-schema-registry.yml up -d

# Wait for Schema Registry to be ready
docker-compose -f docker-compose-schema-registry.yml logs schema-registry

# Verify connectivity
docker-compose exec schema-registry \
  curl -s http://localhost:8081/subjects | jq .

# Run tests against Docker stack
pytest tests/integration/test_schema_registry_e2e.py -v
```

---

## Troubleshooting

### Connection Errors

**Problem:**
```
SchemaRegistryError: Failed to register schema: Connection refused
```

**Solution:**
1. Verify Schema Registry is running:
   ```bash
   docker ps | grep schema-registry
   ```

2. Check connectivity:
   ```bash
   curl http://localhost:8081/subjects
   ```

3. Check network (Docker):
   ```bash
   docker network ls
   docker network inspect kafka-network
   ```

### Authentication Errors

**Problem:**
```
SchemaRegistryError: 401 Unauthorized
```

**Solution:**
1. Verify credentials in config
2. Check Schema Registry has authentication enabled
3. Test with curl:
   ```bash
   curl -u user:pass http://localhost:8081/subjects
   ```

### Schema Registration Fails

**Problem:**
```
SchemaRegistrationError: 409 Conflict - Schema already exists
```

**Solution:**
1. Schema already registered for subject
2. Either use existing schema ID or update version
3. Check existing schema:
   ```bash
   curl http://localhost:8081/subjects/cryptofeed-trades/versions
   ```

### Performance Issues

**Problem:**
- High schema registration latency
- Schema retrieval slow

**Solution:**
1. Enable caching:
   ```yaml
   schema_registry:
     cache_size: 2000
     cache_ttl_seconds: 7200
   ```

2. Use local registry copy
3. Monitor network latency to registry

---

## Performance Tuning

### Caching Configuration

```python
config = SchemaRegistryConfig(
    registry_type="confluent",
    url="http://schema-registry:8081",
    cache_size=5000,           # Store up to 5000 schemas
    cache_ttl_seconds=86400,   # 24-hour cache TTL
)
```

### Connection Pooling

```yaml
schema_registry:
  url: http://schema-registry:8081

  # Connection pool size (for Confluent HTTP)
  pool_size: 20
  pool_timeout_seconds: 30
```

### Batch Schema Registration

```python
"""Register multiple schemas efficiently."""

schema_registry = SchemaRegistry.create(config)

# Batch registration with progress tracking
schemas = {
    "trades": trade_schema,
    "orderbook": orderbook_schema,
    "ticker": ticker_schema,
}

schema_ids = {}
for subject, schema in schemas.items():
    try:
        schema_ids[subject] = schema_registry.register_schema(
            subject=subject,
            schema=schema,
        )
        print(f"✅ Registered {subject}")
    except Exception as e:
        print(f"❌ Failed to register {subject}: {e}")

print(f"Registered {len(schema_ids)} schemas")
```

---

## Monitoring

### Prometheus Metrics

```python
"""Export schema registry metrics to Prometheus."""

from prometheus_client import Counter, Histogram

# Schema registration metrics
schema_registrations_total = Counter(
    'cryptofeed_schema_registrations_total',
    'Total schema registrations',
    ['subject', 'status']
)

schema_retrieval_duration_seconds = Histogram(
    'cryptofeed_schema_retrieval_duration_seconds',
    'Schema retrieval latency',
    ['subject']
)

# Example usage
with schema_retrieval_duration_seconds.labels(
    subject='trades'
).time():
    schema = registry.get_schema_by_id(42)

schema_registrations_total.labels(
    subject='trades',
    status='success'
).inc()
```

### Logging

```python
import logging

# Enable debug logging for schema registry
logging.getLogger('ConfluentSchemaRegistry').setLevel(logging.DEBUG)
logging.getLogger('BufSchemaRegistry').setLevel(logging.DEBUG)

# Example output
# [Schema Registry] Registered schema for subject=trades, schema_id=42
# [Schema Registry] Retrieved cached schema for schema_id=42
# [Schema Registry] Compatibility check for subject=trades: True
```

### Health Checks

```python
"""Health check endpoint for schema registry."""

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/health/schema-registry")
async def schema_registry_health():
    """Check schema registry connectivity and status."""
    try:
        # Test registry connectivity
        subjects = registry.get_all_subjects()

        return {
            "status": "healthy",
            "registry_type": config.registry_type,
            "url": config.url,
            "subjects_count": len(subjects),
            "cache_size": registry.cache_size,
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Schema registry unhealthy: {str(e)}"
        )
```

---

## References

- [Confluent Schema Registry Docs](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [Confluent Docker Images](https://hub.docker.com/u/confluentinc)
- [Buf Schema Registry](https://buf.build/docs/registry)
- [Protobuf Schema Evolution](https://developers.google.com/protocol-buffers/docs/overview#updating-a-message-type)

---

*Last Updated: 2025-11-12*
*Task: 18 - Schema Registry Integration*
