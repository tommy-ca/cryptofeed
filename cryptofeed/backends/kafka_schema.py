"""Kafka schema registry integration for protobuf schema management.

This module provides:
- SchemaRegistry client for Confluent and Buf registries
- Schema registration and retrieval with caching
- Schema ID embedding in Kafka message headers
- Backward/forward compatibility validation

Task 18: Schema Registry Integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Any, Tuple
from urllib.parse import urljoin

import grpc
import requests
from aiohttp import ClientSession, ClientTimeout, BasicAuth
from aiohttp_socks import ProxyConnector
from pydantic import BaseModel, Field, field_validator, ConfigDict
from requests.auth import HTTPBasicAuth
from requests.exceptions import ConnectionError, Timeout, RequestException

from cryptofeed.proxy import get_proxy_injector


LOG = logging.getLogger("cryptofeed.schema")


# ============================================================================
# Enums
# ============================================================================


class CompatibilityMode(str, Enum):
    """Schema compatibility modes supported by Confluent Schema Registry."""

    BACKWARD = "BACKWARD"  # New schema can read old data
    FORWARD = "FORWARD"  # Old schema can read new data
    FULL = "FULL"  # Both directions compatible
    TRANSITIVE = "TRANSITIVE"  # Transitive compatibility across versions


# ============================================================================
# Configuration Models
# ============================================================================


class SchemaRegistryConfig(BaseModel):
    """Configuration for Kafka schema registry integration.

    Attributes:
        registry_type: Type of registry ('confluent' or 'buf')
        url: Registry URL (HTTP for Confluent, gRPC for Buf)
        username: Username for Confluent registry (optional)
        password: Password for Confluent registry (optional)
        api_token: API token for Buf registry (optional)
        compatibility_mode: Compatibility check mode (default: BACKWARD)
        cache_size: Number of schemas to cache (default: 1000)
        cache_ttl_seconds: Cache TTL in seconds (default: 3600)

    Examples:
        >>> # Confluent configuration
        >>> config = SchemaRegistryConfig(
        ...     registry_type="confluent",
        ...     url="http://localhost:8081",
        ...     username="user",
        ...     password="password"
        ... )

        >>> # Buf configuration
        >>> config = SchemaRegistryConfig(
        ...     registry_type="buf",
        ...     url="grpc://buf.example.com:5051",
        ...     api_token="token123"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    registry_type: str = Field(
        default="confluent",
        description="Registry type: 'confluent' or 'buf'",
    )
    url: str = Field(description="Registry URL")
    username: Optional[str] = Field(default=None, description="Confluent username")
    password: Optional[str] = Field(default=None, description="Confluent password")
    api_token: Optional[str] = Field(default=None, description="Buf API token")
    compatibility_mode: CompatibilityMode = Field(
        default=CompatibilityMode.BACKWARD,
        description="Schema compatibility mode",
    )
    cache_size: int = Field(default=1000, description="Schema cache size")
    cache_ttl_seconds: int = Field(
        default=3600, description="Schema cache TTL in seconds"
    )

    @field_validator("registry_type")
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        """Validate registry type is supported."""
        if v not in {"confluent", "buf"}:
            raise ValueError(
                f"Invalid registry_type: {v}. Must be 'confluent' or 'buf'"
            )
        return v.lower()

    @field_validator("compatibility_mode", mode="before")
    @classmethod
    def validate_compatibility(cls, v: str) -> str:
        """Validate compatibility mode."""
        if isinstance(v, str):
            if v not in {mode.value for mode in CompatibilityMode}:
                raise ValueError(
                    f"Invalid compatibility_mode: {v}. "
                    f"Must be one of: {', '.join(m.value for m in CompatibilityMode)}"
                )
        return v


def _schema_registry_http_settings():
    timeout = float(
        os.getenv("CRYPTOFEED_SCHEMA_REGISTRY_TIMEOUT")
        or os.getenv("CF_SCHEMA_REGISTRY_TIMEOUT")
        or 10
    )
    proxy = os.getenv("CRYPTOFEED_SCHEMA_REGISTRY_PROXY") or os.getenv(
        "CF_SCHEMA_REGISTRY_PROXY"
    )
    proxies = {"http": proxy, "https": proxy} if proxy else None
    return timeout, proxies


# ============================================================================
# Exception Classes
# ============================================================================


class SchemaRegistryError(Exception):
    """Base exception for schema registry errors."""

    pass


class SchemaRegistrationError(SchemaRegistryError):
    """Raised when schema registration fails."""

    pass


class SchemaNotFoundError(SchemaRegistryError):
    """Raised when schema is not found in registry."""

    pass


class CompatibilityCheckError(SchemaRegistryError):
    """Raised when schema compatibility check fails."""

    pass


# ============================================================================
# Schema Registry Abstract Base Class
# ============================================================================


class SchemaRegistry(ABC):
    """Abstract base class for schema registry implementations.

    Defines interface for schema registration, retrieval, and validation
    across different schema registry backends.
    """

    def __init__(self, config: SchemaRegistryConfig):
        """Initialize schema registry.

        Args:
            config: SchemaRegistryConfig with registry connection details
        """
        self.config = config
        self._http_timeout, self._http_proxies = _schema_registry_http_settings()
        self.cache_size = config.cache_size
        self.cache_ttl_seconds = config.cache_ttl_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def create(config: SchemaRegistryConfig) -> SchemaRegistry:
        """Factory method to create appropriate registry instance.

        Args:
            config: SchemaRegistryConfig with registry type

        Returns:
            ConfluentSchemaRegistry or BufSchemaRegistry instance

        Raises:
            ValueError: If registry type is not supported
        """
        if config.registry_type == "confluent":
            return ConfluentSchemaRegistry(config)
        elif config.registry_type == "buf":
            return BufSchemaRegistry(config)
        else:
            raise ValueError(f"Unknown registry type: {config.registry_type}")

    @abstractmethod
    def register_schema(
        self,
        subject: str,
        schema: str,
        schema_type: str = "PROTOBUF",
    ) -> int:
        """Register a schema with the registry.

        Args:
            subject: Subject name (e.g., 'trades', 'orderbook')
            schema: Schema definition (protobuf, Avro, JSON Schema)
            schema_type: Type of schema (default: PROTOBUF)

        Returns:
            Schema ID assigned by registry

        Raises:
            SchemaRegistrationError: If registration fails
        """
        pass

    @abstractmethod
    def get_schema_by_id(self, schema_id: int) -> Dict[str, Any]:
        """Retrieve schema by its ID.

        Args:
            schema_id: Schema ID from registry

        Returns:
            Dict with keys: 'id', 'schema', 'schemaType', etc.

        Raises:
            SchemaNotFoundError: If schema not found
        """
        pass

    @abstractmethod
    def get_schema_by_version(
        self, subject: str, version: int
    ) -> Dict[str, Any]:
        """Retrieve schema by subject and version.

        Args:
            subject: Subject name
            version: Version number

        Returns:
            Dict with schema details
        """
        pass

    @abstractmethod
    def check_compatibility(
        self,
        subject: str,
        schema: str,
        version: Optional[int] = None,
    ) -> bool:
        """Check if new schema is compatible with existing schema.

        Args:
            subject: Subject name
            schema: New schema to validate
            version: Version to check against (default: latest)

        Returns:
            True if compatible, False otherwise

        Raises:
            CompatibilityCheckError: If check fails
        """
        pass

    @abstractmethod
    def set_compatibility_mode(self, subject: str, mode: str) -> None:
        """Set compatibility mode for a subject.

        Args:
            subject: Subject name
            mode: Compatibility mode (BACKWARD, FORWARD, FULL, TRANSITIVE)

        Raises:
            SchemaRegistryError: If setting fails
        """
        pass

    def embed_schema_id_in_message(
        self, message_data: bytes, schema_id: int
    ) -> bytes:
        """Embed schema ID in message value (Confluent format).

        Format: magic byte (0x0) + 4-byte big-endian schema ID + message data

        Args:
            message_data: Original protobuf message bytes
            schema_id: Schema ID to embed

        Returns:
            Message with embedded schema ID
        """
        # Confluent wire format: magic byte (1) + schema ID (4) + message
        magic_byte = b"\x00"
        schema_id_bytes = struct.pack(">I", schema_id)  # Big-endian 4-byte int
        return magic_byte + schema_id_bytes + message_data

    def get_schema_id_header(self, schema_id: int) -> bytes:
        """Get schema ID as bytes for message header.

        Args:
            schema_id: Schema ID to encode

        Returns:
            4-byte big-endian encoded schema ID
        """
        return struct.pack(">I", schema_id)

    def extract_schema_id_from_message(self, message_data: bytes) -> Tuple[int, bytes]:
        """Extract schema ID from message value (Confluent format).

        Args:
            message_data: Message with embedded schema ID

        Returns:
            Tuple of (schema_id, original_message_data)

        Raises:
            ValueError: If message format is invalid
        """
        if len(message_data) < 5:
            raise ValueError("Message too short to contain schema ID")

        if message_data[0] != 0x00:
            raise ValueError("Invalid magic byte in message")

        schema_id = struct.unpack(">I", message_data[1:5])[0]
        original_data = message_data[5:]

        return schema_id, original_data


# ============================================================================
# Confluent Schema Registry Implementation
# ============================================================================


class ConfluentSchemaRegistry(SchemaRegistry):
    """Client for Confluent Schema Registry.

    Supports HTTP-based schema management with authentication,
    caching, and compatibility validation.
    """

    def __init__(self, config: SchemaRegistryConfig):
        """Initialize Confluent registry client."""
        super().__init__(config)
        self._auth = None
        if config.username and config.password:
            self._auth = HTTPBasicAuth(config.username, config.password)
        # Schema cache: {schema_id: schema_dict}
        self._schema_cache: Dict[int, Dict[str, Any]] = {}

    async def _http_request_async(
        self,
        url: str,
        method: str = "GET",
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generic async HTTP request helper with ProxyInjector integration.

        Args:
            url: Full URL to request
            method: HTTP method (GET, POST, PUT)
            json_data: JSON payload for POST/PUT requests

        Returns:
            Response JSON as dict

        Raises:
            SchemaRegistryError: On request failure
        """
        proxy_url = None
        connector = None

        # Lease proxy if configured
        injector = get_proxy_injector()
        if injector:
            proxy_url = injector.get_http_proxy_url("schema_registry")
            if proxy_url and proxy_url.startswith("socks"):
                connector = ProxyConnector.from_url(proxy_url)

        # Build auth
        auth = None
        if self.config.username and self.config.password:
            auth = BasicAuth(self.config.username, self.config.password)

        async with ClientSession(
            connector=connector,
            timeout=ClientTimeout(total=self._http_timeout),
            auth=auth
        ) as session:
            kwargs = {
                "proxy": proxy_url if not connector else None,
            }
            if json_data:
                kwargs["json"] = json_data

            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

    async def register_schema_async(
        self,
        subject: str,
        schema: str,
        schema_type: str = "PROTOBUF",
    ) -> int:
        """Register a schema with Confluent registry (async).

        Args:
            subject: Subject name
            schema: Schema definition
            schema_type: Type of schema

        Returns:
            Schema ID

        Raises:
            SchemaRegistrationError: If registration fails
        """
        url = urljoin(self.config.url, f"/subjects/{subject}/versions")
        payload = {
            "schema": schema,
            "schemaType": schema_type,
        }

        try:
            data = await self._http_request_async(url, method="POST", json_data=payload)
            schema_id = data.get("id")
            if schema_id is None:
                raise SchemaRegistrationError(
                    f"Registration succeeded but no schema ID returned: {data}"
                )
            self.logger.info(
                f"Registered schema for subject={subject}, schema_id={schema_id}"
            )
            return schema_id
        except Exception as e:
            if "409" in str(e):
                raise SchemaRegistrationError(
                    f"Schema already exists for subject {subject}"
                ) from e
            raise SchemaRegistrationError(
                f"Failed to register schema: {str(e)}"
            ) from e

    async def get_schema_by_id_async(self, schema_id: int) -> Dict[str, Any]:
        """Retrieve schema by ID from Confluent registry (async).

        Args:
            schema_id: Schema ID

        Returns:
            Dict with schema details

        Raises:
            SchemaNotFoundError: If schema not found
        """
        # Check cache first
        if schema_id in self._schema_cache:
            self.logger.debug(f"Retrieved cached schema for schema_id={schema_id}")
            return self._schema_cache[schema_id]

        url = urljoin(self.config.url, f"/schemas/ids/{schema_id}")

        try:
            data = await self._http_request_async(url, method="GET")
            # Cache the schema
            self._schema_cache[schema_id] = data
            self.logger.debug(f"Retrieved schema for schema_id={schema_id}")
            return data
        except Exception as e:
            if "404" in str(e):
                raise SchemaNotFoundError(f"Schema ID {schema_id} not found") from e
            raise SchemaNotFoundError(
                f"Failed to retrieve schema: {str(e)}"
            ) from e

    async def get_schema_by_version_async(
        self, subject: str, version: int
    ) -> Dict[str, Any]:
        """Retrieve schema by subject and version (async).

        Args:
            subject: Subject name
            version: Version number

        Returns:
            Dict with schema details

        Raises:
            SchemaNotFoundError: If schema not found
        """
        url = urljoin(
            self.config.url, f"/subjects/{subject}/versions/{version}"
        )

        try:
            return await self._http_request_async(url, method="GET")
        except Exception as e:
            raise SchemaNotFoundError(
                f"Schema not found for subject={subject}, version={version}: {str(e)}"
            ) from e

    async def check_compatibility_async(
        self,
        subject: str,
        schema: str,
        version: Optional[int] = None,
    ) -> bool:
        """Check if new schema is compatible with existing schema (async).

        Args:
            subject: Subject name
            schema: New schema to validate
            version: Version to check against

        Returns:
            True if compatible, False otherwise

        Raises:
            CompatibilityCheckError: If check fails
        """
        if version is None:
            url = urljoin(self.config.url, f"/compatibility/subjects/{subject}/versions/latest")
        else:
            url = urljoin(
                self.config.url,
                f"/compatibility/subjects/{subject}/versions/{version}"
            )

        payload = {"schema": schema}

        try:
            data = await self._http_request_async(url, method="POST", json_data=payload)
            is_compatible = data.get("is_compatible", False)
            self.logger.debug(
                f"Compatibility check for subject={subject}: {is_compatible}"
            )
            return is_compatible
        except Exception as e:
            raise CompatibilityCheckError(
                f"Compatibility check failed: {str(e)}"
            ) from e

    async def set_compatibility_mode_async(self, subject: str, mode: str) -> None:
        """Set compatibility mode for a subject (async).

        Args:
            subject: Subject name
            mode: Compatibility mode

        Raises:
            SchemaRegistryError: If setting fails
        """
        url = urljoin(self.config.url, f"/config/{subject}")
        payload = {"compatibility": mode}

        try:
            await self._http_request_async(url, method="PUT", json_data=payload)
            self.logger.info(
                f"Set compatibility mode for subject={subject} to {mode}"
            )
        except Exception as e:
            raise SchemaRegistryError(
                f"Failed to set compatibility mode: {str(e)}"
            ) from e

    def register_schema(
        self,
        subject: str,
        schema: str,
        schema_type: str = "PROTOBUF",
    ) -> int:
        """Register a schema with Confluent registry.

        Args:
            subject: Subject name
            schema: Schema definition
            schema_type: Type of schema

        Returns:
            Schema ID

        Raises:
            SchemaRegistrationError: If registration fails
        """
        url = urljoin(self.config.url, f"/subjects/{subject}/versions")

        payload = {
            "schema": schema,
            "schemaType": schema_type,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                auth=self._auth,
                timeout=self._http_timeout,
                proxies=self._http_proxies,
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    raise SchemaRegistrationError(
                        f"Invalid JSON response: {str(e)}"
                    ) from e
                schema_id = data.get("id")
                if schema_id is None:
                    raise SchemaRegistrationError(
                        f"Registration succeeded but no schema ID returned: {data}"
                    )
                self.logger.info(
                    f"Registered schema for subject={subject}, schema_id={schema_id}"
                )
                return schema_id

            elif response.status_code == 409:
                raise SchemaRegistrationError(
                    f"Schema already exists for subject {subject}"
                )

            else:
                raise SchemaRegistrationError(
                    f"Registration failed with status {response.status_code}: "
                    f"{response.text}"
                )

        except (ConnectionError, Timeout) as e:
            raise SchemaRegistrationError(
                f"Failed to register schema: {str(e)}"
            ) from e
        except RequestException as e:
            raise SchemaRegistrationError(
                f"Request error during registration: {str(e)}"
            ) from e

    def get_schema_by_id(self, schema_id: int) -> Dict[str, Any]:
        """Retrieve schema by ID from Confluent registry.

        Args:
            schema_id: Schema ID

        Returns:
            Dict with schema details

        Raises:
            SchemaNotFoundError: If schema not found
        """
        # Check cache first
        if schema_id in self._schema_cache:
            self.logger.debug(f"Retrieved cached schema for schema_id={schema_id}")
            return self._schema_cache[schema_id]

        url = urljoin(self.config.url, f"/schemas/ids/{schema_id}")

        try:
            response = requests.get(
                url,
                auth=self._auth,
                timeout=self._http_timeout,
                proxies=self._http_proxies,
            )

            if response.status_code == 200:
                data = response.json()
                # Cache the schema
                self._schema_cache[schema_id] = data
                self.logger.debug(f"Retrieved schema for schema_id={schema_id}")
                return data

            elif response.status_code == 404:
                raise SchemaNotFoundError(f"Schema ID {schema_id} not found")

            else:
                raise SchemaNotFoundError(
                    f"Failed to retrieve schema: {response.text}"
                )

        except RequestException as e:
            raise SchemaNotFoundError(
                f"Request error retrieving schema: {str(e)}"
            ) from e

    def get_schema_by_version(
        self, subject: str, version: int
    ) -> Dict[str, Any]:
        """Retrieve schema by subject and version.

        Args:
            subject: Subject name
            version: Version number

        Returns:
            Dict with schema details
        """
        url = urljoin(
            self.config.url, f"/subjects/{subject}/versions/{version}"
        )

        try:
            response = requests.get(
                url,
                auth=self._auth,
                timeout=self._http_timeout,
                proxies=self._http_proxies,
            )

            if response.status_code == 200:
                return response.json()

            else:
                raise SchemaNotFoundError(
                    f"Schema not found for subject={subject}, version={version}"
                )

        except RequestException as e:
            raise SchemaNotFoundError(
                f"Request error: {str(e)}"
            ) from e

    def check_compatibility(
        self,
        subject: str,
        schema: str,
        version: Optional[int] = None,
    ) -> bool:
        """Check if new schema is compatible with existing schema.

        Args:
            subject: Subject name
            schema: New schema to validate
            version: Version to check against

        Returns:
            True if compatible, False otherwise

        Raises:
            CompatibilityCheckError: If check fails
        """
        if version is None:
            url = urljoin(self.config.url, f"/compatibility/subjects/{subject}/versions/latest")
        else:
            url = urljoin(
                self.config.url,
                f"/compatibility/subjects/{subject}/versions/{version}"
            )

        payload = {"schema": schema}

        try:
            response = requests.post(
                url,
                json=payload,
                auth=self._auth,
                timeout=self._http_timeout,
                proxies=self._http_proxies,
            )

            if response.status_code == 200:
                data = response.json()
                is_compatible = data.get("is_compatible", False)
                self.logger.debug(
                    f"Compatibility check for subject={subject}: {is_compatible}"
                )
                return is_compatible

            else:
                raise CompatibilityCheckError(
                    f"Compatibility check failed: {response.text}"
                )

        except RequestException as e:
            raise CompatibilityCheckError(
                f"Request error: {str(e)}"
            ) from e

    def set_compatibility_mode(self, subject: str, mode: str) -> None:
        """Set compatibility mode for a subject.

        Args:
            subject: Subject name
            mode: Compatibility mode

        Raises:
            SchemaRegistryError: If setting fails
        """
        url = urljoin(self.config.url, f"/config/{subject}")

        payload = {"compatibility": mode}

        try:
            response = requests.put(
                url,
                json=payload,
                auth=self._auth,
                timeout=self._http_timeout,
                proxies=self._http_proxies,
            )

            if response.status_code == 200:
                self.logger.info(
                    f"Set compatibility mode for subject={subject} to {mode}"
                )
            else:
                raise SchemaRegistryError(
                    f"Failed to set compatibility mode: {response.text}"
                )

        except RequestException as e:
            raise SchemaRegistryError(
                f"Request error: {str(e)}"
            ) from e


# ============================================================================
# Buf Schema Registry Implementation
# ============================================================================


class BufSchemaRegistry(SchemaRegistry):
    """Client for Buf Schema Registry.

    Supports gRPC-based schema management with authentication
    and caching.
    """

    def __init__(self, config: SchemaRegistryConfig):
        """Initialize Buf registry client."""
        super().__init__(config)
        self._channel = None
        self._metadata = []
        if config.api_token:
            self._metadata = [("authorization", f"Bearer {config.api_token}")]

    def register_schema(
        self,
        subject: str,
        schema: str,
        schema_type: str = "PROTOBUF",
    ) -> int:
        """Register a schema with Buf registry.

        Args:
            subject: Subject name
            schema: Schema definition
            schema_type: Type of schema

        Returns:
            Schema ID

        Raises:
            SchemaRegistrationError: If registration fails
        """
        try:
            schema_id = self._register_schema_grpc(subject, schema, schema_type)
            self.logger.info(
                f"Registered schema for subject={subject}, schema_id={schema_id}"
            )
            return schema_id

        except grpc.RpcError as e:
            raise SchemaRegistrationError(
                f"Failed to register schema: {e.details()}"
            ) from e

    def _register_schema_grpc(
        self, subject: str, schema: str, schema_type: str
    ) -> int:
        """Internal gRPC schema registration."""
        # Placeholder for actual gRPC implementation
        # Would use buf.registry.v1.RegistryService.CreateRepository
        raise NotImplementedError("Buf registry integration pending gRPC setup")

    def get_schema_by_id(self, schema_id: int) -> Dict[str, Any]:
        """Retrieve schema by ID from Buf registry.

        Args:
            schema_id: Schema ID

        Returns:
            Dict with schema details

        Raises:
            SchemaNotFoundError: If schema not found
        """
        try:
            schema = self._get_schema_grpc(schema_id)
            self.logger.debug(f"Retrieved schema for schema_id={schema_id}")
            return schema

        except grpc.RpcError as e:
            raise SchemaNotFoundError(
                f"Schema ID {schema_id} not found: {e.details()}"
            ) from e

    def _get_schema_grpc(self, schema_id: int) -> Dict[str, Any]:
        """Internal gRPC schema retrieval."""
        # Placeholder for actual gRPC implementation
        raise NotImplementedError("Buf registry integration pending gRPC setup")

    def get_schema_by_version(
        self, subject: str, version: int
    ) -> Dict[str, Any]:
        """Retrieve schema by subject and version.

        Args:
            subject: Subject name
            version: Version number

        Returns:
            Dict with schema details
        """
        # Buf uses different versioning model
        raise NotImplementedError("Buf version-based retrieval not yet implemented")

    def check_compatibility(
        self,
        subject: str,
        schema: str,
        version: Optional[int] = None,
    ) -> bool:
        """Check if new schema is compatible.

        Args:
            subject: Subject name
            schema: New schema to validate
            version: Version to check against

        Returns:
            True if compatible, False otherwise
        """
        # Buf registry has different compatibility model
        raise NotImplementedError("Buf compatibility checking not yet implemented")

    def set_compatibility_mode(self, subject: str, mode: str) -> None:
        """Set compatibility mode for a subject.

        Args:
            subject: Subject name
            mode: Compatibility mode
        """
        raise NotImplementedError("Buf compatibility mode not yet implemented")
