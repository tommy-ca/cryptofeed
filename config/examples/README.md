# Cryptofeed Configuration Examples

This directory contains working configuration examples for common use cases. All examples are ready to use with minimal modifications.

## Available Examples

### binance-spot.yaml
Subscribes to Binance SPOT market data for BTC, ETH, and SOL.

**Channels:**
- Trades
- L2 Order Book
- Ticker

**Usage:**
```bash
docker-compose run cryptofeed --config /config/examples/binance-spot.yaml
```

### multi-exchange.yaml
Subscribes to multiple exchanges simultaneously (Binance, Coinbase, Kraken).

**Features:**
- Multiple exchange connections
- Different symbols per exchange
- Consolidated Kafka topics

**Usage:**
```bash
docker-compose run cryptofeed --config /config/examples/multi-exchange.yaml
```

### with-proxy.yaml
Demonstrates proxy configuration for geo-restricted exchanges.

**Features:**
- Global and per-exchange proxy settings
- HTTP and SOCKS5 proxy support
- Proxy validation and failover

**Usage:**
1. Copy proxy configuration to `config/proxy.yaml`
2. Update proxy URLs with your proxy service
3. Run with proxy configuration

```bash
docker-compose run cryptofeed --config /config/examples/with-proxy.yaml
```

## Customization

All examples can be customized by:

1. **Changing symbols:** Update the `symbols` list under each exchange
2. **Adding channels:** Add to `channels` list (trades, l2_book, ticker, funding, open_interest)
3. **Adding exchanges:** Add new exchange blocks following the same pattern
4. **Kafka settings:** Adjust `topic_strategy` (consolidated/per_symbol) and `partition_strategy`

## API Credentials

Set exchange API credentials via environment variables:

```bash
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
COINBASE_API_KEY=your_key_here
COINBASE_API_SECRET=your_secret_here
```

See `.env.example` in the project root for all supported exchanges.

## Quick Start

1. Copy an example to `config/config.yaml`:
   ```bash
   cp config/examples/binance-spot.yaml config/config.yaml
   ```

2. Set API credentials in `.env` or `docker-compose.yml`

3. Start the service:
   ```bash
   docker-compose up cryptofeed
   ```

## Documentation

- **Main config:** `config/config.yaml` - Minimal default configuration
- **Proxy config:** `config/proxy.yaml` - Proxy settings
- **Quick Start Guide:** `docs/docker/DOCKER_COMPOSE_QUICKSTART.md`
- **Proxy Guide:** `docs/proxy/README.md`
