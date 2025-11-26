#!/usr/bin/env python3
"""
Data Integrity Validation Script for Kafka Migration

Validates data integrity between legacy JSON messages and new protobuf messages
during the Blue-Green migration. Implements hash-based comparison with float
normalization and offset-based message counting.

Usage:
    python scripts/validate-data-integrity.py \
        --legacy-topic cryptofeed.trades.coinbase.btc-usd \
        --new-topic cryptofeed.trades \
        --exchange coinbase \
        --symbol BTC-USD \
        --sample-size 1000

Features:
- Normalize messages (JSON and protobuf) with 8-decimal float precision
- Hash-based comparison for data integrity validation
- O(1) offset-based message counting (no wc -l)
- Per-exchange/symbol filtering for granular validation
- Detailed comparison reports with mismatch details

Requirements:
- kafka-python or confluent-kafka
- google-protobuf
- cryptofeed protobuf bindings
"""
import sys
import json
import hashlib
import argparse
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Tuple, Optional
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message


# Metadata fields to remove before hashing (not part of data content)
METADATA_FIELDS = {
    'headers', 'partition', 'offset', 'producer_version',
    'timestamp_generated', 'raw', 'raw_id'
}

# Fields requiring float normalization (8 decimals)
FLOAT_FIELDS = {'price', 'amount', 'bid', 'ask', 'quantity', 'timestamp'}


def normalize_float(value: Any, decimals: int = 8) -> str:
    """
    Normalize float value to fixed decimal precision.

    Args:
        value: Float, Decimal, or string representation
        decimals: Number of decimal places (default: 8)

    Returns:
        String representation with normalized precision
    """
    if value is None:
        return None

    # Convert to Decimal for precise arithmetic
    if isinstance(value, str):
        dec = Decimal(value)
    elif isinstance(value, (int, float)):
        dec = Decimal(str(value))
    elif isinstance(value, Decimal):
        dec = value
    else:
        raise TypeError(f"Unsupported type for normalization: {type(value)}")

    # Quantize to specified precision
    quantize_pattern = Decimal(10) ** -decimals
    normalized = dec.quantize(quantize_pattern, rounding=ROUND_HALF_UP)

    return str(normalized)


def normalize_message(msg: Any, format_type: str) -> Dict[str, Any]:
    """
    Normalize message to canonical form for hashing.

    Handles both JSON dict and protobuf Message types. Removes metadata fields,
    normalizes float precision to 8 decimals, and returns canonical dict.

    Args:
        msg: Message object (dict for JSON, protobuf Message for protobuf)
        format_type: 'json' or 'protobuf'

    Returns:
        Normalized dict with canonical field order and precision
    """
    if format_type == 'json':
        # Already a dict, make a copy
        normalized = dict(msg)
    elif format_type == 'protobuf':
        # Convert protobuf to dict
        if isinstance(msg, Message):
            normalized = MessageToDict(
                msg,
                including_default_value_fields=False,
                preserving_proto_field_name=True
            )
        else:
            raise TypeError(f"Expected protobuf Message, got {type(msg)}")
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")

    # Remove metadata fields
    for field in METADATA_FIELDS:
        normalized.pop(field, None)

    # Normalize float precision for comparison fields
    for field, value in list(normalized.items()):
        if field in FLOAT_FIELDS and value is not None:
            # Special handling for timestamp: convert microseconds to seconds if needed
            if field == 'timestamp' and isinstance(value, int) and value > 1e12:
                # Likely microseconds, convert to seconds
                value = value / 1e6
            normalized[field] = normalize_float(value, decimals=8)

    # Remove None values (optional fields not present)
    normalized = {k: v for k, v in normalized.items() if v is not None}

    return normalized


def hash_message(msg: Any, format_type: str) -> str:
    """
    Generate SHA256 hash of normalized message.

    Args:
        msg: Message object (dict or protobuf)
        format_type: 'json' or 'protobuf'

    Returns:
        SHA256 hex digest (64 characters)
    """
    # Normalize message first
    normalized = normalize_message(msg, format_type)

    # Sort keys for canonical ordering
    canonical_json = json.dumps(normalized, sort_keys=True)

    # Generate SHA256 hash
    hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_obj.hexdigest()


def validate_message_count(
    consumer: Any,
    topic: str,
    partitions: List[int]
) -> int:
    """
    Count messages using Kafka offsets (O(1) operation).

    Avoids slow O(N) counting via wc -l or iterating all messages.

    Args:
        consumer: Kafka consumer instance (confluent-kafka or kafka-python)
        topic: Topic name
        partitions: List of partition IDs

    Returns:
        Total message count across all partitions
    """
    total_count = 0

    # Build topic-partition tuples
    topic_partitions = [(topic, p) for p in partitions]

    # Get beginning and end offsets
    beginning_offsets = consumer.beginning_offsets(topic_partitions)
    end_offsets = consumer.end_offsets(topic_partitions)

    # Calculate count per partition
    for tp in topic_partitions:
        begin = beginning_offsets.get(tp, 0)
        end = end_offsets.get(tp, 0)
        partition_count = end - begin
        total_count += partition_count

    return total_count


def compare_messages(
    legacy_msgs: List[Dict],
    new_msgs: List[Dict],
    format_types: Tuple[str, str] = ('json', 'protobuf')
) -> Dict[str, Any]:
    """
    Compare two message lists using hash-based validation.

    Args:
        legacy_msgs: List of legacy JSON messages
        new_msgs: List of new protobuf messages
        format_types: Tuple of (legacy_format, new_format)

    Returns:
        Comparison result dict with match statistics and mismatch details
    """
    legacy_format, new_format = format_types

    # Compare min length to handle length mismatches
    min_length = min(len(legacy_msgs), len(new_msgs))
    length_mismatch = len(legacy_msgs) != len(new_msgs)

    matches = 0
    mismatches = 0
    mismatch_details = []

    for i in range(min_length):
        legacy_hash = hash_message(legacy_msgs[i], legacy_format)
        new_hash = hash_message(new_msgs[i], new_format)

        if legacy_hash == new_hash:
            matches += 1
        else:
            mismatches += 1
            mismatch_details.append({
                'index': i,
                'legacy_hash': legacy_hash,
                'new_hash': new_hash,
                'legacy_msg': normalize_message(legacy_msgs[i], legacy_format),
                'new_msg': normalize_message(new_msgs[i], new_format)
            })

    return {
        'total_compared': min_length,
        'matches': matches,
        'mismatches': mismatches,
        'mismatch_details': mismatch_details,
        'legacy_count': len(legacy_msgs),
        'new_count': len(new_msgs),
        'length_mismatch': length_mismatch
    }


def generate_comparison_report(
    comparison_result: Dict[str, Any],
    exchange: str,
    symbol: str
) -> Dict[str, Any]:
    """
    Generate human-readable comparison report.

    Args:
        comparison_result: Result from compare_messages()
        exchange: Exchange identifier
        symbol: Symbol identifier

    Returns:
        Report dict with status, match rate, and mismatch samples
    """
    total = comparison_result['total_compared']
    matches = comparison_result['matches']
    mismatches = comparison_result['mismatches']

    match_rate = matches / total if total > 0 else 0.0

    # Determine status
    if comparison_result['length_mismatch']:
        status = 'WARNING'
    elif mismatches == 0:
        status = 'PASS'
    else:
        status = 'FAIL'

    report = {
        'exchange': exchange,
        'symbol': symbol,
        'status': status,
        'total_compared': total,
        'matches': matches,
        'mismatches': mismatches,
        'match_rate': match_rate,
        'length_mismatch': comparison_result['length_mismatch'],
        'legacy_count': comparison_result['legacy_count'],
        'new_count': comparison_result['new_count'],
        'mismatch_samples': comparison_result['mismatch_details'][:10]  # First 10 mismatches
    }

    return report


def filter_messages(
    messages: List[Dict],
    exchange: Optional[str] = None,
    symbol: Optional[str] = None
) -> List[Dict]:
    """
    Filter messages by exchange and/or symbol.

    Args:
        messages: List of message dicts
        exchange: Optional exchange filter
        symbol: Optional symbol filter

    Returns:
        Filtered message list
    """
    filtered = messages

    if exchange is not None:
        filtered = [msg for msg in filtered if msg.get('exchange') == exchange]

    if symbol is not None:
        filtered = [msg for msg in filtered if msg.get('symbol') == symbol]

    return filtered


def main():
    """
    Main CLI entry point for data integrity validation.
    """
    parser = argparse.ArgumentParser(
        description='Validate data integrity between legacy and new Kafka topics'
    )
    parser.add_argument(
        '--legacy-topic',
        required=True,
        help='Legacy topic name (per-symbol format)'
    )
    parser.add_argument(
        '--new-topic',
        required=True,
        help='New topic name (consolidated format)'
    )
    parser.add_argument(
        '--exchange',
        help='Filter by exchange (e.g., coinbase)'
    )
    parser.add_argument(
        '--symbol',
        help='Filter by symbol (e.g., BTC-USD)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of messages to compare (default: 1000)'
    )
    parser.add_argument(
        '--output',
        default='comparison-report.json',
        help='Output report file (default: comparison-report.json)'
    )

    args = parser.parse_args()

    print(f"Data Integrity Validation")
    print(f"Legacy Topic: {args.legacy_topic}")
    print(f"New Topic: {args.new_topic}")
    print(f"Exchange: {args.exchange or 'ALL'}")
    print(f"Symbol: {args.symbol or 'ALL'}")
    print(f"Sample Size: {args.sample_size}")
    print()

    # TODO: Implement Kafka consumer integration
    # This would involve:
    # 1. Connect to Kafka cluster
    # 2. Consume messages from legacy and new topics
    # 3. Apply filters (exchange, symbol)
    # 4. Run comparison
    # 5. Generate and save report

    print("Note: Full Kafka integration pending. Use as library for now.")
    print("Example:")
    print("  from validate_data_integrity import compare_messages, generate_comparison_report")
    print("  result = compare_messages(legacy_msgs, new_msgs)")
    print("  report = generate_comparison_report(result, 'coinbase', 'BTC-USD')")

    return 0


if __name__ == '__main__':
    sys.exit(main())
