# Makefile for Cryptofeed Protobuf Schema Management

.PHONY: help install lint generate push clean test

# Variables
BUF_VERSION := 1.28.1
PROTO_DIR := proto
GEN_DIR := gen

# Help target
help:
	@echo "Available targets:"
	@echo "  install     - Install Buf CLI and dependencies"
	@echo "  lint        - Lint protobuf schemas" 
	@echo "  generate    - Generate code from protobuf schemas"
	@echo "  push        - Push schemas to Buf Schema Registry"
	@echo "  clean       - Clean generated code"
	@echo "  test        - Run adapter tests"
	@echo "  build       - Lint and generate code"
	@echo "  breaking    - Check for breaking changes"

# Install Buf CLI
install:
	@echo "Installing Buf CLI v$(BUF_VERSION)..."
	@curl -sSL "https://github.com/bufbuild/buf/releases/download/v$(BUF_VERSION)/buf-$$(uname -s)-$$(uname -m)" -o "/tmp/buf"
	@chmod +x "/tmp/buf"
	@sudo mv "/tmp/buf" "/usr/local/bin/buf"
	@buf --version
	@echo "Installing Python dependencies..."
	@pip install protobuf grpcio-tools

# Lint protobuf schemas
lint:
	@echo "Linting protobuf schemas..."
	@buf lint

# Generate code from schemas
generate:
	@echo "Generating code from protobuf schemas..."
	@buf generate
	@echo "Generated code in $(GEN_DIR)/"

# Push schemas to registry (requires tag)
push:
	@if [ -z "$(TAG)" ]; then echo "Usage: make push TAG=v1.0.0"; exit 1; fi
	@echo "Pushing schemas to Buf Schema Registry with tag $(TAG)..."
	@buf push --tag $(TAG)

# Check for breaking changes
breaking:
	@echo "Checking for breaking changes..."
	@buf breaking --against '.git#branch=main'

# Clean generated code
clean:
	@echo "Cleaning generated code..."
	@rm -rf $(GEN_DIR)/

# Run tests
test:
	@echo "Running adapter tests..."
	@python -m pytest tests/adapters/ -v

# Full build process
build: lint generate
	@echo "Build complete!"

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	@pip install -e .
	@pip install pytest pytest-cov
	@echo "Development setup complete!"

# Format protobuf files (if buf format is available)
format:
	@echo "Formatting protobuf files..."
	@buf format -w

# Validate schemas
validate: lint
	@echo "Validating schemas..."
	@buf build

# Package for distribution
package: clean generate
	@echo "Packaging generated code..."
	@cd $(GEN_DIR)/python && python setup.py sdist bdist_wheel
	@echo "Python package created in $(GEN_DIR)/python/dist/"

# Publish to package registries
publish-python:
	@echo "Publishing Python package..."
	@cd $(GEN_DIR)/python && twine upload dist/*

# Docker targets for consistent environment
docker-build:
	@docker build -t cryptofeed-schema-builder .

docker-generate:
	@docker run --rm -v $(PWD):/workspace cryptofeed-schema-builder make generate

# CI/CD targets
ci-lint: lint validate

ci-test: test

ci-build: ci-lint generate ci-test

# Show schema stats
stats:
	@echo "Schema statistics:"
	@find $(PROTO_DIR) -name "*.proto" -exec wc -l {} + | tail -1
	@echo "Generated code size:"
	@du -sh $(GEN_DIR)/ 2>/dev/null || echo "No generated code found"
