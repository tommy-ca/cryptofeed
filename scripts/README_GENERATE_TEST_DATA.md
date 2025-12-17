# Synthetic Test Data Generator

## Overview

The test data generator creates realistic market data for staging validation and load testing of the Kafka producer backend. Supports configurable volume profiles and spike scenarios.

## Features

- **Volume Profiles**: Pre-configured scenarios (low, medium, high, smoke)
- **Spike Scenario**: Ramp up → peak → ramp down pattern
- **Multiple Data Types**: Trades, tickers, order books, funding rates
- **Output Formats**: Kafka topics, file (JSON lines), or stdout
- **Deterministic Generation**: Seed support for reproducible tests

## Quick Start

### 1. Generate Smoke Test (Quick Validation)

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile smoke \
    --output stdout
```

### 2. Load Test - Low Volume (1K msg/s for 1 hour)

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile low \
    --output kafka \
    --kafka-brokers localhost:9092
```

### 3. Load Test - Medium Volume (50K msg/s for 1 hour)

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile medium \
    --output kafka \
    --kafka-brokers kafka1:9092,kafka2:9092,kafka3:9092
```

### 4. Load Test - High Volume (150K msg/s for 1 hour)

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile high \
    --output kafka \
    --kafka-brokers kafka1:9092,kafka2:9092,kafka3:9092
```

### 5. Spike Scenario (10K → 200K → 10K msg/s over 30 minutes)

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile spike \
    --output kafka \
    --kafka-brokers localhost:9092
```

## Configuration

### Profile Configuration (`test-data-config.yaml`)

```yaml
profiles:
  smoke:
    messages_per_second: 100
    duration_seconds: 60
    exchanges:
      - coinbase
    symbols:
      - BTC-USD
    data_types:
      - trade
    description: "Smoke test: 100 msg/s for 1 minute"

  spike:
    spike_scenario: true
    start_rate: 10000
    peak_rate: 200000
    end_rate: 10000
    ramp_up_seconds: 600
    peak_seconds: 600
    ramp_down_seconds: 600
    exchanges:
      - coinbase
      - binance
      - kraken
    symbols:
      - BTC-USD
      - ETH-USD
    data_types:
      - trade
      - ticker
    description: "Spike test"
```

## Volume Profiles

| Profile | Rate (msg/s) | Duration | Total Messages | Use Case |
|---------|-------------|----------|----------------|----------|
| **smoke** | 100 | 1 min | 6K | Quick validation |
| **low** | 1,000 | 1 hour | 3.6M | Basic load test |
| **medium** | 50,000 | 1 hour | 180M | Production simulation |
| **high** | 150,000 | 1 hour | 540M | Peak capacity test |
| **spike** | 10K→200K→10K | 30 min | 246M | Surge scenario |

## Output Options

### 1. Kafka Topics

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile medium \
    --output kafka \
    --kafka-brokers localhost:9092
```

Messages are published to topics:
- `cryptofeed.test.trade`
- `cryptofeed.test.ticker`
- `cryptofeed.test.orderbook`
- `cryptofeed.test.funding`

### 2. File Output (JSON Lines)

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile low \
    --output file \
    --file-output-dir ./test-data-output
```

Creates files:
- `./test-data-output/trade.jsonl`
- `./test-data-output/ticker.jsonl`
- etc.

### 3. Stdout (for inspection)

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile smoke \
    --output stdout
```

Prints first 10 messages to console.

## Reproducible Tests

Use `--seed` for deterministic generation:

```bash
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile low \
    --seed 42 \
    --output file
```

## Data Types

The generator supports:

- **trade**: Realistic price/amount with random walk
- **ticker**: Bid/ask spread (0.01-0.1%)
- **orderbook**: 10-level order book with descending bids/ascending asks
- **funding**: Funding rates (-0.1% to +0.1%)

## Performance

Generation performance (approximate):

- **Low volume (1K msg/s)**: ~0.1 CPU cores
- **Medium volume (50K msg/s)**: ~5 CPU cores
- **High volume (150K msg/s)**: ~15 CPU cores

Memory usage:
- Constant rate: ~50MB base overhead
- Spike scenario: ~100MB peak

## Validation

### Message Format

Each message includes:

```json
{
  "type": "trade",
  "exchange": "coinbase",
  "symbol": "BTC-USD",
  "timestamp": 1234567890.123456,
  "price": "50000.00",
  "amount": "0.123",
  "side": "buy"
}
```

### Data Quality Checks

1. **Price realism**: Random walk (±0.1% per step)
2. **Spread constraints**: Ask ≥ Bid
3. **Order book ordering**: Descending bids, ascending asks
4. **Timestamp consistency**: Sequential timestamps

## CLI Reference

```
usage: generate_test_data.py [-h] --config CONFIG --profile PROFILE
                              [--output {stdout,kafka,file}]
                              [--kafka-brokers KAFKA_BROKERS]
                              [--file-output-dir FILE_OUTPUT_DIR]
                              [--seed SEED] [--dry-run]

Generate synthetic test data for Kafka producer staging validation

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to test data configuration YAML file
  --profile PROFILE     Volume profile name (low, medium, high, spike, smoke)
  --output {stdout,kafka,file}
                        Output destination (default: stdout)
  --kafka-brokers KAFKA_BROKERS
                        Kafka broker addresses (comma-separated)
  --file-output-dir FILE_OUTPUT_DIR
                        Output directory for file output
  --seed SEED           Random seed for reproducibility
  --dry-run             Print configuration and exit without generating data
```

## Testing

Run unit tests:

```bash
pytest tests/unit/test_generate_test_data.py -v
```

All 26 tests should pass.

## Troubleshooting

### Issue: "Kafka brokers unreachable"

**Solution**: Verify Kafka cluster is running and brokers are accessible:

```bash
kafka-broker-api-versions.sh --bootstrap-server localhost:9092
```

### Issue: "Generation too slow"

**Solution**:
- Reduce message rate
- Use faster storage for file output
- Increase CPU cores

### Issue: "Out of memory"

**Solution**:
- Reduce spike scenario duration
- Use file output instead of in-memory buffering
- Increase available memory

## Integration with Staging Validation

### Week 1: Parallel Deployment

```bash
# Generate baseline load to new topics
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile medium \
    --output kafka \
    --kafka-brokers staging-kafka:9092
```

### Week 2: Consumer Validation

```bash
# Generate test data while monitoring consumer lag
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile high \
    --output kafka \
    --kafka-brokers staging-kafka:9092

# Monitor consumer lag (separate terminal)
kafka-consumer-groups.sh --bootstrap-server staging-kafka:9092 \
    --group cryptofeed-consumer \
    --describe
```

### Week 3: Spike Scenario Validation

```bash
# Simulate production spike
python scripts/generate_test_data.py \
    --config scripts/test-data-config.yaml \
    --profile spike \
    --output kafka \
    --kafka-brokers staging-kafka:9092
```

## Related Documentation

- [Kafka Producer Specification](../.kiro/specs/market-data-kafka-producer/design.md)
- [Data Integrity Validation](./validate_data_integrity.py)
- [Migration Guide](../.kiro/specs/market-data-kafka-producer/PHASE_5_EXECUTION_PLAN.md)
