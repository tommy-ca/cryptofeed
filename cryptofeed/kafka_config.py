"""Kafka configuration models for Task 4.

This module re-exports the Pydantic configuration models from kafka_callback.py
for clean imports and API consistency.

Classes:
    - KafkaTopicConfig: Topic management configuration
    - KafkaPartitionConfig: Partition strategy configuration
    - KafkaProducerConfig: Producer client configuration
    - KafkaConfig: Top-level composite configuration

Example:
    >>> from cryptofeed.kafka_config import KafkaConfig
    >>> config = KafkaConfig.from_yaml('config/kafka.yaml')
    >>> config = KafkaConfig(bootstrap_servers=['kafka:9092'])
"""

from cryptofeed.kafka_callback import (
    KafkaTopicConfig,
    KafkaPartitionConfig,
    KafkaProducerConfig,
    KafkaConfig,
)

__all__ = [
    "KafkaTopicConfig",
    "KafkaPartitionConfig",
    "KafkaProducerConfig",
    "KafkaConfig",
]
