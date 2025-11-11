"""Performance benchmarking tests for market-data-kafka-producer specification.

This package contains performance validation tests for Phase 4:
- Task 10: End-to-end latency benchmarking
- Task 10.1: Throughput testing
- Task 10.2: Memory profiling under load
- Task 10.3: CPU usage analysis

Tests establish baseline metrics and verify performance targets without
requiring a live Kafka cluster (using test producer stubs).
"""
