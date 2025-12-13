### Minimal Kafka workflow ###################################################
# 1) Start broker:     make redpanda-up        (port defaults to 19092)
# 2) Run unit suite:   make test-kafka-unit
# 3) Run e2e suite:    make test-kafka-e2e     (requires broker)
# 4) Stop broker:      make redpanda-down
#############################################################################

.PHONY: docker-ps-19092 docker-stop-19092 redpanda-up redpanda-down redpanda-health
.PHONY: test-kafka-e2e test-kafka-binance test-kafka-binance-mullvad test-kafka-binance-futures test-kafka-binance-futures-mullvad test-kafka-unit test-kafka-perf test-kafka-all

# Advanced / scoped targets (legacy, keep for reference)
.PHONY: test-kafka-bisect test-kafka-files test-kafka-callback-integration

REDPANDA_COMPOSE_FILE := docker/infra/base.yml
REDPANDA_HOST_PORT ?= 19092
KAFKA_BOOTSTRAP_SERVERS ?= localhost:$(REDPANDA_HOST_PORT)

# Inspect containers using the configured Redpanda host port.
docker-ps-19092:
	docker ps --format '{{.ID}} {{.Names}} {{.Ports}}' | awk 'index($$0,"$(REDPANDA_HOST_PORT)")>0 {print $$0} END {if (NR==0) print "no container on port $(REDPANDA_HOST_PORT)"}'

# Stop containers currently bound to the configured Redpanda host port (use with care).
docker-stop-19092:
	docker ps --format '{{.ID}} {{.Names}} {{.Ports}}' | awk 'index($$0,"$(REDPANDA_HOST_PORT)")>0 {print $$1}' | xargs -r docker stop

# Start Redpanda on the configured host port using docker/infra/base.yml.
redpanda-up:
	REDPANDA_HOST_PORT=$(REDPANDA_HOST_PORT) docker compose -f $(REDPANDA_COMPOSE_FILE) up -d

# Stop the Redpanda docker compose stack.
redpanda-down:
	docker compose -f $(REDPANDA_COMPOSE_FILE) down

# Basic TCP health check against the Redpanda Kafka listener.
redpanda-health:
	nc -vz localhost $(REDPANDA_HOST_PORT)

# Kafka / Redpanda-related tests

test-kafka-e2e:
	KAFKA_BOOTSTRAP_SERVERS=$(KAFKA_BOOTSTRAP_SERVERS) python -m pytest tests/integration/kafka/test_kafka_protobuf_e2e.py -v

test-kafka-binance:
	CRYPTODATA_RUN_BINANCE_KAFKA_E2E=$(CRYPTODATA_RUN_BINANCE_KAFKA_E2E) KAFKA_BOOTSTRAP_SERVERS=$(KAFKA_BOOTSTRAP_SERVERS) python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

test-kafka-binance-mullvad:
	@echo "Using Mullvad SOCKS5 relays (EU/AP). Override envs to customize. See docs/e2e/PROXY_TESTING.md"
	CRYPTOFEED_PROXY_ENABLED=true \
	CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' \
	CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' \
	CRYPTODATA_RUN_BINANCE_KAFKA_E2E=$${CRYPTODATA_RUN_BINANCE_KAFKA_E2E:-true} \
	KAFKA_E2E_TOPIC_STRATEGY=$${KAFKA_E2E_TOPIC_STRATEGY:-consolidated} \
	KAFKA_BOOTSTRAP_SERVERS=$(KAFKA_BOOTSTRAP_SERVERS) python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v -s

test-kafka-binance-futures:
	CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=$${CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E:-true} \
	KAFKA_BOOTSTRAP_SERVERS=$(KAFKA_BOOTSTRAP_SERVERS) \
	python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v

test-kafka-binance-futures-mullvad:
	@echo "Using Mullvad SOCKS5 relays (EU/AP) for Binance Futures. Override envs to customize. See docs/e2e/PROXY_TESTING.md"
	CRYPTOFEED_PROXY_ENABLED=true \
	CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' \
	CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' \
	CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=$${CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E:-true} \
	KAFKA_E2E_TOPIC_STRATEGY=$${KAFKA_E2E_TOPIC_STRATEGY:-consolidated} \
	KAFKA_BOOTSTRAP_SERVERS=$(KAFKA_BOOTSTRAP_SERVERS) python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v -s

test-kafka-unit:
	KAFKA_BOOTSTRAP_SERVERS=$(KAFKA_BOOTSTRAP_SERVERS) python -m pytest tests/unit/kafka -v

test-kafka-perf:
	python -m pytest tests/performance/test_kafka_optimization.py -v

# Grouped Kafka unit tests

# Explicit target for slow Kafka callback integration tests

test-kafka-callback-integration:
	python -m pytest tests/unit/kafka/test_kafka_callback_pipeline_core.py -v
	python -m pytest tests/unit/kafka/test_kafka_callback_message_types.py -v
	python -m pytest tests/unit/kafka/test_kafka_callback_errors_and_config.py -v
	python -m pytest tests/unit/kafka/test_kafka_callback_partitioning_and_headers.py -v
	python -m pytest tests/unit/kafka/test_kafka_callback_backward_compat.py -v
	python -m pytest tests/unit/kafka/test_kafka_callback_performance.py -v

# Run each Kafka unit test file in isolation

test-kafka-files:
	@for f in tests/unit/kafka/test_*.py; do \
		if [ "$$f" = "tests/unit/kafka/test_kafka_callback_integration.py" ]; then \
			echo "=== Skipping slow file $$f (run via make test-kafka-callback-integration) ==="; \
			continue; \
		fi; \
		echo "=== Running $$f ==="; \
		python -m pytest $$f -q --durations=5 || exit 1; \
	done

test-kafka-all: test-kafka-e2e test-kafka-binance test-kafka-binance-futures test-kafka-unit test-kafka-perf

# Bisect Kafka unit tests to identify slow subsets

test-kafka-bisect:
	python tools/bisect_slow_tests.py tests/unit/kafka --max-depth=3 --min-size=50
