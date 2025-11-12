"""Tests for Kafka schema registry integration (Task 18).

Tests cover:
- SchemaRegistry client for Confluent and Buf registries
- Schema ID embedding in Kafka message headers
- Backward/forward compatibility validation
- Schema caching for performance
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any
import http.client
from requests.exceptions import ConnectionError, Timeout
from requests.auth import HTTPBasicAuth

from cryptofeed.backends.kafka_schema import (
    SchemaRegistry,
    ConfluentSchemaRegistry,
    BufSchemaRegistry,
    SchemaRegistryConfig,
    CompatibilityMode,
    SchemaRegistrationError,
    SchemaNotFoundError,
    CompatibilityCheckError,
)


class TestSchemaRegistryConfig:
    """Tests for SchemaRegistryConfig Pydantic model."""

    def test_confluent_registry_config(self):
        """Test Confluent registry configuration."""
        config = SchemaRegistryConfig(
            registry_type="confluent",
            url="http://schema-registry:8081",
            username="user",
            password="pass",
            compatibility_mode=CompatibilityMode.BACKWARD,
        )
        assert config.registry_type == "confluent"
        assert config.url == "http://schema-registry:8081"
        assert config.username == "user"
        assert config.password == "pass"
        assert config.compatibility_mode == CompatibilityMode.BACKWARD

    def test_buf_registry_config(self):
        """Test Buf registry configuration."""
        config = SchemaRegistryConfig(
            registry_type="buf",
            url="grpc://buf.example.com:5051",
            api_token="token123",
        )
        assert config.registry_type == "buf"
        assert config.url == "grpc://buf.example.com:5051"
        assert config.api_token == "token123"

    def test_invalid_registry_type(self):
        """Test invalid registry type raises error."""
        with pytest.raises(ValueError, match="Invalid registry_type"):
            SchemaRegistryConfig(
                registry_type="invalid",
                url="http://localhost:8081",
            )

    def test_compatibility_mode_validation(self):
        """Test compatibility mode validation."""
        # Valid modes
        for mode in ["BACKWARD", "FORWARD", "FULL", "TRANSITIVE"]:
            config = SchemaRegistryConfig(
                registry_type="confluent",
                url="http://localhost:8081",
                compatibility_mode=mode,
            )
            assert config.compatibility_mode == mode

    def test_default_values(self):
        """Test configuration defaults."""
        config = SchemaRegistryConfig(
            registry_type="confluent",
            url="http://localhost:8081",
        )
        assert config.compatibility_mode == CompatibilityMode.BACKWARD
        assert config.cache_size == 1000
        assert config.cache_ttl_seconds == 3600


class TestConfluentSchemaRegistry:
    """Tests for Confluent Schema Registry client."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SchemaRegistryConfig(
            registry_type="confluent",
            url="http://localhost:8081",
            username="user",
            password="pass",
        )

    @pytest.fixture
    def registry(self, config):
        """Create a registry instance."""
        return ConfluentSchemaRegistry(config)

    def test_initialization(self, registry, config):
        """Test registry initialization."""
        assert registry.config == config
        assert registry.cache_size == 1000
        assert registry.cache_ttl_seconds == 3600

    @patch("requests.post")
    def test_register_schema(self, mock_post, registry):
        """Test schema registration."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 42,
            "version": 1,
            "subject": "trades",
            "schema": '{"type":"record"}',
        }
        mock_post.return_value = mock_response

        schema_id = registry.register_schema(
            subject="trades",
            schema='{"type":"record","name":"Trade"}',
            schema_type="PROTOBUF",
        )

        assert schema_id == 42
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "trades" in call_args[0][0]
        # Check that HTTPBasicAuth was used
        assert isinstance(call_args[1]["auth"], HTTPBasicAuth)

    @patch("requests.post")
    def test_register_schema_already_exists(self, mock_post, registry):
        """Test schema registration when schema already exists."""
        mock_response = Mock()
        mock_response.status_code = 409  # Conflict
        mock_post.return_value = mock_response

        with pytest.raises(SchemaRegistrationError):
            registry.register_schema(
                subject="trades",
                schema='{"type":"record"}',
            )

    @patch("requests.post")
    def test_register_schema_network_error(self, mock_post, registry):
        """Test schema registration with network error."""
        mock_post.side_effect = ConnectionError("Connection failed")

        with pytest.raises(SchemaRegistrationError, match="Connection failed"):
            registry.register_schema(
                subject="trades",
                schema='{"type":"record"}',
            )

    @patch("requests.get")
    def test_get_schema_by_id(self, mock_get, registry):
        """Test retrieving schema by ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 42,
            "schema": '{"type":"record","name":"Trade"}',
            "schemaType": "PROTOBUF",
        }
        mock_get.return_value = mock_response

        schema = registry.get_schema_by_id(42)

        assert schema["id"] == 42
        assert "Trade" in schema["schema"]
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_get_schema_by_id_not_found(self, mock_get, registry):
        """Test get schema when not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(SchemaNotFoundError):
            registry.get_schema_by_id(999)

    @patch("requests.get")
    def test_get_schema_by_id_caching(self, mock_get, registry):
        """Test schema caching for performance."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 42,
            "schema": '{"type":"record","name":"Trade"}',
            "schemaType": "PROTOBUF",
        }
        mock_get.return_value = mock_response

        # First call hits registry
        schema1 = registry.get_schema_by_id(42)
        # Second call uses cache (no HTTP call)
        schema2 = registry.get_schema_by_id(42)

        assert schema1 == schema2
        # Should only call once due to caching (second call hits cache)
        assert mock_get.call_count == 1

    @patch("requests.post")
    def test_check_compatibility(self, mock_post, registry):
        """Test schema compatibility checking."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_compatible": True}
        mock_post.return_value = mock_response

        is_compatible = registry.check_compatibility(
            subject="trades",
            schema='{"type":"record","name":"Trade","fields":[{"name":"price"}]}',
            version=1,
        )

        assert is_compatible is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_check_compatibility_backward_incompatible(self, mock_post, registry):
        """Test backward incompatible schema."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_compatible": False}
        mock_post.return_value = mock_response

        is_compatible = registry.check_compatibility(
            subject="trades",
            schema='{"type":"record","name":"Trade","fields":[]}',
            version=1,
        )

        assert is_compatible is False

    @patch("requests.get")
    def test_get_schema_by_version(self, mock_get, registry):
        """Test retrieving schema by subject and version."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 42,
            "version": 1,
            "subject": "trades",
            "schema": '{"type":"record","name":"Trade"}',
        }
        mock_get.return_value = mock_response

        schema = registry.get_schema_by_version(subject="trades", version=1)

        assert schema["id"] == 42
        assert schema["version"] == 1

    @patch("requests.put")
    def test_set_compatibility_mode(self, mock_put, registry):
        """Test setting compatibility mode."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"compatibility": "BACKWARD"}
        mock_put.return_value = mock_response

        registry.set_compatibility_mode(subject="trades", mode="BACKWARD")

        mock_put.assert_called_once()
        call_args = mock_put.call_args
        assert "compatibility" in str(call_args)

    def test_schema_id_embed_format(self, registry):
        """Test schema ID embedding format (Confluent 4-byte prefix)."""
        # Confluent format: magic byte (0x0) + 4-byte big-endian schema ID
        schema_id = 42
        embedded = registry.embed_schema_id_in_message(b"message_data", schema_id)

        assert isinstance(embedded, bytes)
        assert len(embedded) >= 5  # At least magic byte + 4-byte ID
        assert embedded[0] == 0x00  # Magic byte for Confluent


class TestBufSchemaRegistry:
    """Tests for Buf Schema Registry client."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SchemaRegistryConfig(
            registry_type="buf",
            url="grpc://localhost:5051",
            api_token="test-token",
        )

    @pytest.fixture
    def registry(self, config):
        """Create a registry instance."""
        return BufSchemaRegistry(config)

    def test_initialization(self, registry, config):
        """Test Buf registry initialization."""
        assert registry.config == config
        assert registry.cache_size == 1000

    @patch("grpc.aio.secure_channel")
    def test_buf_register_schema(self, mock_channel, registry):
        """Test schema registration with Buf."""
        # This test mocks gRPC channel
        # In real implementation, would use buf.registry.v1.RegistryService
        with patch.object(registry, "_register_schema_grpc") as mock_register:
            mock_register.return_value = 42

            schema_id = registry.register_schema(
                subject="trades",
                schema="message Trade { ... }",
                schema_type="PROTOBUF",
            )

            assert schema_id == 42

    @patch.object(BufSchemaRegistry, "_get_schema_grpc")
    def test_buf_get_schema_by_id(self, mock_get, registry):
        """Test retrieving schema by ID from Buf."""
        mock_get.return_value = {
            "id": 42,
            "schema": "message Trade { ... }",
            "type": "PROTOBUF",
        }

        schema = registry.get_schema_by_id(42)

        assert schema["id"] == 42
        assert "Trade" in schema["schema"]

    def test_buf_schema_id_embed_format(self, registry):
        """Test Buf schema ID embedding format (different from Confluent)."""
        schema_id = 42
        embedded = registry.embed_schema_id_in_message(b"message_data", schema_id)

        assert isinstance(embedded, bytes)
        # Buf uses different format than Confluent
        assert len(embedded) > len(b"message_data")


class TestSchemaRegistryIntegration:
    """Integration tests for schema registry with KafkaCallback."""

    @pytest.fixture
    def confluent_config(self):
        """Create Confluent config."""
        return SchemaRegistryConfig(
            registry_type="confluent",
            url="http://localhost:8081",
        )

    @patch("requests.post")
    @patch("requests.get")
    def test_register_and_retrieve_schema(self, mock_get, mock_post, confluent_config):
        """Test registering and retrieving schema."""
        registry = ConfluentSchemaRegistry(confluent_config)

        # Register schema
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "id": 100,
            "version": 1,
            "subject": "trades",
            "schema": '{"type":"record","name":"Trade"}',
        }

        schema_id = registry.register_schema(
            subject="trades",
            schema='{"type":"record","name":"Trade"}',
        )
        assert schema_id == 100

        # Retrieve schema
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "id": 100,
            "schema": '{"type":"record","name":"Trade"}',
            "schemaType": "PROTOBUF",
        }

        schema = registry.get_schema_by_id(100)
        assert schema["id"] == 100

    @patch("requests.post")
    def test_compatibility_check_before_register(self, mock_post, confluent_config):
        """Test checking compatibility before registering new schema."""
        registry = ConfluentSchemaRegistry(confluent_config)

        # Check compatibility first
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"is_compatible": True}

        is_compatible = registry.check_compatibility(
            subject="trades",
            schema='{"type":"record","name":"Trade"}',
            version=1,
        )
        assert is_compatible is True

    def test_schema_registry_factory(self):
        """Test schema registry factory pattern."""
        confluent_config = SchemaRegistryConfig(
            registry_type="confluent",
            url="http://localhost:8081",
        )
        registry = SchemaRegistry.create(confluent_config)
        assert isinstance(registry, ConfluentSchemaRegistry)

        buf_config = SchemaRegistryConfig(
            registry_type="buf",
            url="grpc://localhost:5051",
        )
        registry = SchemaRegistry.create(buf_config)
        assert isinstance(registry, BufSchemaRegistry)

    @patch("requests.get")
    def test_cache_performance(self, mock_get, confluent_config):
        """Test schema caching improves performance."""
        registry = ConfluentSchemaRegistry(confluent_config)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 42,
            "schema": '{"type":"record"}',
            "schemaType": "PROTOBUF",
        }
        mock_get.return_value = mock_response

        # Multiple requests should hit cache
        for _ in range(10):
            registry.get_schema_by_id(42)

        # Should only call once due to caching (rest hit cache)
        assert mock_get.call_count == 1

    @patch("requests.post")
    def test_registration_timeout_handling(self, mock_post, confluent_config):
        """Test handling of timeout during registration."""
        registry = ConfluentSchemaRegistry(confluent_config)
        mock_post.side_effect = Timeout("Registration timeout")

        with pytest.raises(SchemaRegistrationError, match="timeout"):
            registry.register_schema(
                subject="trades",
                schema='{"type":"record"}',
            )

    def test_multiple_compatibility_modes(self, confluent_config):
        """Test all supported compatibility modes."""
        registry = ConfluentSchemaRegistry(confluent_config)

        modes = [
            CompatibilityMode.BACKWARD,
            CompatibilityMode.FORWARD,
            CompatibilityMode.FULL,
            CompatibilityMode.TRANSITIVE,
        ]

        for mode in modes:
            config = SchemaRegistryConfig(
                registry_type="confluent",
                url="http://localhost:8081",
                compatibility_mode=mode,
            )
            registry = ConfluentSchemaRegistry(config)
            assert registry.config.compatibility_mode == mode


class TestSchemaEmbeddingInKafkaMessages:
    """Tests for embedding schema IDs in Kafka messages."""

    @pytest.fixture
    def confluent_registry(self):
        """Create Confluent registry."""
        config = SchemaRegistryConfig(
            registry_type="confluent",
            url="http://localhost:8081",
        )
        return ConfluentSchemaRegistry(config)

    def test_embed_schema_id_confluent_format(self, confluent_registry):
        """Test embedding schema ID in Confluent format."""
        message_data = b"test_message"
        schema_id = 42

        embedded = confluent_registry.embed_schema_id_in_message(
            message_data, schema_id
        )

        # Confluent format: magic byte (0x0) + 4-byte big-endian schema ID
        assert embedded[0] == 0x00  # Magic byte
        # Extract schema ID from bytes
        extracted_id = int.from_bytes(embedded[1:5], byteorder="big")
        assert extracted_id == 42
        assert embedded[5:] == message_data

    def test_embed_schema_id_in_headers(self, confluent_registry):
        """Test embedding schema ID in message headers."""
        schema_id = 42

        header_value = confluent_registry.get_schema_id_header(schema_id)

        assert isinstance(header_value, bytes)
        assert len(header_value) == 4
        extracted_id = int.from_bytes(header_value, byteorder="big")
        assert extracted_id == 42

    def test_multiple_schema_ids(self, confluent_registry):
        """Test embedding multiple different schema IDs."""
        schema_ids = [1, 42, 100, 65535, 2147483647]

        for schema_id in schema_ids:
            header_value = confluent_registry.get_schema_id_header(schema_id)
            extracted_id = int.from_bytes(header_value, byteorder="big")
            assert extracted_id == schema_id


class TestErrorHandling:
    """Tests for error handling in schema registry."""

    @pytest.fixture
    def registry(self):
        """Create registry for error tests."""
        config = SchemaRegistryConfig(
            registry_type="confluent",
            url="http://localhost:8081",
        )
        return ConfluentSchemaRegistry(config)

    def test_schema_registration_error(self):
        """Test SchemaRegistrationError is raised."""
        with pytest.raises(SchemaRegistrationError):
            raise SchemaRegistrationError("Registration failed")

    def test_schema_not_found_error(self):
        """Test SchemaNotFoundError is raised."""
        with pytest.raises(SchemaNotFoundError):
            raise SchemaNotFoundError("Schema ID 999 not found")

    def test_compatibility_check_error(self):
        """Test CompatibilityCheckError is raised."""
        with pytest.raises(CompatibilityCheckError):
            raise CompatibilityCheckError("Schema incompatible")

    @patch("requests.post")
    def test_invalid_response_format(self, mock_post, registry):
        """Test handling of invalid response format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError(
            "Invalid JSON", "", 0
        )
        mock_post.return_value = mock_response

        with pytest.raises(SchemaRegistrationError):
            registry.register_schema(
                subject="trades",
                schema='{"type":"record"}',
            )

    @patch("requests.post")
    def test_http_500_error(self, mock_post, registry):
        """Test handling of HTTP 500 error."""
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal server error"

        with pytest.raises(SchemaRegistrationError):
            registry.register_schema(
                subject="trades",
                schema='{"type":"record"}',
            )
