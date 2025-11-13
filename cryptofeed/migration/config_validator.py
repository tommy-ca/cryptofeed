"""Config validator for Phase 2 Kafka configurations.

This module provides validation of Phase 2 KafkaCallback configurations,
including schema validation and optional Kafka connectivity testing.

Key Features:
- Validate configuration schema and values
- Detect common configuration errors
- Optional Kafka connectivity testing
- Human-readable validation reports
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from pydantic import ValidationError


class ValidationError(Exception):
    """Validation error raised during config validation."""
    pass


class ValidationResult:
    """Result of configuration validation."""

    def __init__(
        self,
        is_valid: bool,
        config: Optional[Any] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        """Initialize validation result.

        Args:
            is_valid: Whether validation passed
            config: Validated config object if valid
            errors: List of error messages
            warnings: List of warning messages
        """
        self.is_valid = is_valid
        self.config = config
        self.errors = errors or []
        self.warnings = warnings or []

    def format_report(self) -> str:
        """Format validation result as human-readable report."""
        lines = []

        if self.is_valid:
            lines.append("✓ Configuration is valid")
        else:
            lines.append("✗ Configuration has errors:")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """Get validation summary as dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ConfigValidator:
    """Validate Phase 2 Kafka configurations.

    This class provides validation of Phase 2 KafkaCallback configurations,
    checking both schema validity and runtime compatibility.

    Example:
        >>> validator = ConfigValidator()
        >>> config = {'bootstrap_servers': ['kafka:9092']}
        >>> result = validator.validate(config)
        >>> if result.is_valid:
        ...     print("Config is valid")
        ... else:
        ...     print(result.format_report())
    """

    def validate(
        self,
        config_dict: Dict[str, Any],
        test_kafka: bool = False,
        test_topic_creation: bool = False,
    ) -> ValidationResult:
        """Validate Phase 2 configuration dictionary.

        Args:
            config_dict: Configuration dictionary to validate
            test_kafka: Whether to test Kafka connectivity
            test_topic_creation: Whether to test topic creation capability

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        config_obj = None

        # Validate bootstrap_servers
        if "bootstrap_servers" not in config_dict:
            errors.append("Required field 'bootstrap_servers' is missing")
            return ValidationResult(False, errors=errors)

        if not isinstance(config_dict["bootstrap_servers"], list):
            errors.append("'bootstrap_servers' must be a list")
            return ValidationResult(False, errors=errors)

        if not config_dict["bootstrap_servers"]:
            errors.append("'bootstrap_servers' cannot be empty")
            return ValidationResult(False, errors=errors)

        # Validate each broker address format
        for server in config_dict["bootstrap_servers"]:
            if not self._is_valid_broker_address(server):
                errors.append(
                    f"Invalid broker address format: '{server}'. "
                    f"Use 'host:port' format (e.g., 'kafka:9092')"
                )

        if errors:
            return ValidationResult(False, errors=errors)

        # Validate topic strategy if present
        if "topic" in config_dict:
            topic_config = config_dict["topic"]
            if isinstance(topic_config, dict) and "strategy" in topic_config:
                valid_strategies = {"consolidated", "per_symbol"}
                if topic_config["strategy"] not in valid_strategies:
                    errors.append(
                        f"Invalid topic strategy '{topic_config['strategy']}'. "
                        f"Must be one of: {', '.join(sorted(valid_strategies))}"
                    )

                if topic_config["strategy"] == "per_symbol":
                    warnings.append(
                        "Topic strategy is 'per_symbol' (legacy-compatible mode). "
                        "Consider migrating to 'consolidated' for better scalability."
                    )

        # Validate partition strategy if present
        if "partition" in config_dict:
            partition_config = config_dict["partition"]
            if isinstance(partition_config, dict) and "strategy" in partition_config:
                valid_strategies = {"composite", "symbol", "exchange", "round_robin"}
                if partition_config["strategy"] not in valid_strategies:
                    errors.append(
                        f"Invalid partition strategy '{partition_config['strategy']}'. "
                        f"Must be one of: {', '.join(sorted(valid_strategies))}"
                    )

        # Validate acks if present
        if "acks" in config_dict:
            if config_dict["acks"] not in {"0", "1", "all"}:
                errors.append(f"acks must be '0', '1', or 'all', got {config_dict['acks']}")

        # Validate compression_type if present
        if "compression_type" in config_dict:
            valid_compression = {"none", "gzip", "snappy", "lz4", "zstd"}
            if config_dict["compression_type"] not in valid_compression:
                errors.append(
                    f"compression_type must be one of {valid_compression}, "
                    f"got {config_dict['compression_type']}"
                )

        # Validate numeric fields
        numeric_fields = {
            "retries": (lambda v: v >= 0, "must be >= 0"),
            "retry_backoff_ms": (lambda v: v >= 0, "must be >= 0"),
            "batch_size": (lambda v: v > 0, "must be > 0"),
            "linger_ms": (lambda v: v >= 0, "must be >= 0"),
        }

        for field_name, (validator, msg) in numeric_fields.items():
            if field_name in config_dict:
                value = config_dict[field_name]
                if not isinstance(value, int):
                    errors.append(f"'{field_name}' must be an integer, got {type(value).__name__}")
                elif not validator(value):
                    errors.append(f"'{field_name}' {msg}")

        # Try to instantiate config object if no errors
        if not errors:
            try:
                config_obj = self._create_config_object(config_dict)
            except Exception as e:
                errors.append(f"Failed to create config object: {e}")

        # Test Kafka connectivity if requested
        if not errors and test_kafka:
            kafka_result = self.test_kafka_connectivity(config_dict)
            if not kafka_result.is_valid:
                errors.extend(kafka_result.errors)
            warnings.extend(kafka_result.warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            config=config_obj,
            errors=errors,
            warnings=warnings,
        )

    def validate_yaml_file(self, yaml_path: str | Path) -> ValidationResult:
        """Validate Phase 2 configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ValidationResult with validation status
        """
        yaml_path = Path(yaml_path)

        # Check if file exists
        if not yaml_path.exists():
            return ValidationResult(False, errors=[f"File not found: {yaml_path}"])

        # Load YAML
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return ValidationResult(False, errors=[f"Invalid YAML syntax: {e}"])

        if config_dict is None:
            return ValidationResult(
                False,
                errors=["Configuration file is empty or contains only comments"]
            )

        # Validate
        return self.validate(config_dict)

    def test_kafka_connectivity(
        self,
        config_dict: Dict[str, Any],
        timeout_seconds: int = 5,
    ) -> ValidationResult:
        """Test Kafka connectivity with given configuration.

        Args:
            config_dict: Configuration dictionary with bootstrap_servers
            timeout_seconds: Connection timeout in seconds

        Returns:
            ValidationResult with connectivity test results
        """
        # This is a placeholder for Kafka connectivity testing
        # Actual implementation would use aiokafka or kafka-python
        # For now, we just validate that the servers are specified

        warnings = []

        bootstrap_servers = config_dict.get("bootstrap_servers", [])

        if not bootstrap_servers:
            return ValidationResult(False, errors=["No bootstrap servers specified"])

        # Validate server format
        for server in bootstrap_servers:
            if not self._is_valid_broker_address(server):
                return ValidationResult(
                    False,
                    errors=[f"Invalid broker address format: {server}. Use 'host:port'"]
                )

        # In a real implementation, we would test connectivity here
        # For now, just warn if we can't test
        warnings.append(
            "Kafka connectivity test requires a running Kafka cluster. "
            "Configuration syntax is valid; actual connection testing skipped."
        )

        return ValidationResult(True, errors=[], warnings=warnings)

    def _create_config_object(self, config_dict: Dict[str, Any]) -> Any:
        """Create Phase 2 KafkaConfig object from dictionary.

        This method imports KafkaConfig to avoid circular imports during
        module initialization.

        Args:
            config_dict: Configuration dictionary

        Returns:
            KafkaConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            from cryptofeed.kafka_callback import KafkaConfig
            return KafkaConfig.from_dict(config_dict)
        except ImportError:
            # If kafka_callback not available, use simplified validation
            return None
        except Exception as e:
            raise ValueError(f"Failed to create KafkaConfig: {e}")

    def _is_valid_broker_address(self, address: str) -> bool:
        """Validate broker address format.

        Args:
            address: Broker address to validate (e.g., 'kafka:9092')

        Returns:
            True if address is valid, False otherwise
        """
        if not isinstance(address, str):
            return False

        # Check for host:port format
        parts = address.rsplit(":", 1)
        if len(parts) != 2:
            return False

        host, port_str = parts

        # Host can't be empty
        if not host:
            return False

        # Port must be numeric and in valid range
        try:
            port = int(port_str)
            return 1 <= port <= 65535
        except ValueError:
            return False
