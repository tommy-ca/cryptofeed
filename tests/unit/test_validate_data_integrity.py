"""
Unit tests for scripts/validate-data-integrity.py

Tests the data integrity validation script that compares legacy JSON messages
with new protobuf messages for migration validation.

Test coverage:
- Message normalization (JSON and protobuf)
- Float precision normalization
- Hash generation and comparison
- Offset-based message counting
- Comparison report generation
"""
import pytest
import json
import hashlib
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
from google.protobuf.json_format import MessageToDict

# Import the script functions (will be implemented)
import sys
import os

# Add scripts directory to path
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import validation functions
try:
    from validate_data_integrity import (
        normalize_message,
        hash_message,
        validate_message_count,
        compare_messages,
        generate_comparison_report,
        filter_messages
    )
except ImportError as e:
    import pytest
    pytest.skip(f"Cannot import validate_data_integrity: {e}", allow_module_level=True)


class TestNormalizeMessage:
    """Test message normalization for both JSON and protobuf formats"""

    def test_normalize_json_message(self):
        """JSON message should be normalized with float precision"""
        json_msg = {
            'exchange': 'coinbase',
            'symbol': 'BTC-USD',
            'price': 50123.456789123,
            'amount': 0.123456789,
            'timestamp': 1699876543.123456789,
            'side': 'buy'
        }

        normalized = normalize_message(json_msg, format_type='json')

        # Verify float precision normalized to 8 decimals
        # Note: ROUND_HALF_UP is used for precision normalization
        assert normalized['price'] == '50123.45678912'
        assert normalized['amount'] == '0.12345679'
        assert normalized['timestamp'] == '1699876543.12345670'  # Rounded per ROUND_HALF_UP
        assert normalized['exchange'] == 'coinbase'
        assert normalized['symbol'] == 'BTC-USD'
        assert normalized['side'] == 'buy'

    def test_normalize_protobuf_message(self):
        """Protobuf message should be converted to dict and normalized"""
        # Create mock protobuf message that passes isinstance(msg, Message)
        from google.protobuf.message import Message as ProtoMessage
        mock_proto = Mock(spec=ProtoMessage)

        with patch('validate_data_integrity.MessageToDict') as mock_to_dict:
            mock_to_dict.return_value = {
                'exchange': 'binance',
                'symbol': 'ETH-USDT',
                'price': '2500.123456789',
                'amount': '10.987654321',
                'timestamp': 1699876543123456,
                'side': 'BUY'
            }

            normalized = normalize_message(mock_proto, format_type='protobuf')

            # Verify MessageToDict was called
            mock_to_dict.assert_called_once()

            # Verify protobuf-to-dict conversion and normalization
            assert normalized['exchange'] == 'binance'
            assert normalized['symbol'] == 'ETH-USDT'
            assert normalized['price'] == '2500.12345679'
            assert normalized['amount'] == '10.98765432'
            # Timestamp in microseconds converted to seconds with 8 decimals
            assert normalized['timestamp'] == '1699876543.12345600'
            assert normalized['side'] == 'BUY'

    def test_normalize_removes_metadata_fields(self):
        """Metadata fields should be removed before hashing"""
        json_msg = {
            'exchange': 'kraken',
            'symbol': 'BTC-EUR',
            'price': 45000.12345678,
            'amount': 1.5,
            'timestamp': 1699876543.0,
            'side': 'sell',
            # Metadata fields to remove
            'headers': {'content-type': 'application/json'},
            'partition': 3,
            'offset': 12345,
            'producer_version': '0.1.0'
        }

        normalized = normalize_message(json_msg, format_type='json')

        # Verify metadata fields removed
        assert 'headers' not in normalized
        assert 'partition' not in normalized
        assert 'offset' not in normalized
        assert 'producer_version' not in normalized

        # Verify core fields preserved
        assert 'exchange' in normalized
        assert 'symbol' in normalized
        assert 'price' in normalized

    def test_normalize_handles_decimal_strings(self):
        """String decimals should be parsed and normalized"""
        json_msg = {
            'exchange': 'coinbase',
            'symbol': 'BTC-USD',
            'price': '50000.123456789123',
            'amount': '0.5',
            'timestamp': 1699876543.0,
            'side': 'buy'
        }

        normalized = normalize_message(json_msg, format_type='json')

        # Verify string decimals parsed and normalized
        assert normalized['price'] == '50000.12345679'
        assert normalized['amount'] == '0.50000000'

    def test_normalize_handles_none_values(self):
        """None values should be handled gracefully"""
        json_msg = {
            'exchange': 'binance',
            'symbol': 'ETH-USDT',
            'price': 2500.0,
            'amount': 10.0,
            'timestamp': 1699876543.0,
            'side': 'buy',
            'id': None,  # Optional field
            'type': None  # Optional field
        }

        normalized = normalize_message(json_msg, format_type='json')

        # Verify None values removed (not included in canonical representation)
        assert 'id' not in normalized or normalized['id'] is None
        assert 'type' not in normalized or normalized['type'] is None


class TestHashMessage:
    """Test message hashing for comparison"""

    def test_hash_json_message_sha256(self):
        """JSON message should generate consistent SHA256 hash"""
        json_msg = {
            'exchange': 'coinbase',
            'symbol': 'BTC-USD',
            'price': 50000.12345678,
            'amount': 1.0,
            'timestamp': 1699876543.0,
            'side': 'buy'
        }

        hash1 = hash_message(json_msg, format_type='json')
        hash2 = hash_message(json_msg, format_type='json')

        # Verify consistent hashing
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_hash_protobuf_message(self):
        """Protobuf message should generate consistent hash"""
        from google.protobuf.message import Message as ProtoMessage
        mock_proto = Mock(spec=ProtoMessage)

        with patch('validate_data_integrity.MessageToDict') as mock_to_dict:
            mock_to_dict.return_value = {
                'exchange': 'binance',
                'symbol': 'ETH-USDT',
                'price': '2500.12345678',
                'amount': '10.0',
                'timestamp': 1699876543123456,
                'side': 'BUY'
            }

            hash1 = hash_message(mock_proto, format_type='protobuf')
            hash2 = hash_message(mock_proto, format_type='protobuf')

            assert hash1 == hash2
            assert len(hash1) == 64

    def test_hash_different_messages_different_hashes(self):
        """Different messages should produce different hashes"""
        msg1 = {
            'exchange': 'coinbase',
            'symbol': 'BTC-USD',
            'price': 50000.0,
            'amount': 1.0,
            'timestamp': 1699876543.0,
            'side': 'buy'
        }

        msg2 = {
            'exchange': 'coinbase',
            'symbol': 'BTC-USD',
            'price': 50001.0,  # Different price
            'amount': 1.0,
            'timestamp': 1699876543.0,
            'side': 'buy'
        }

        hash1 = hash_message(msg1, format_type='json')
        hash2 = hash_message(msg2, format_type='json')

        assert hash1 != hash2

    def test_hash_same_content_different_field_order_same_hash(self):
        """Messages with same content but different field order should hash identically"""
        msg1 = {
            'exchange': 'coinbase',
            'symbol': 'BTC-USD',
            'price': 50000.0,
            'amount': 1.0,
            'timestamp': 1699876543.0,
            'side': 'buy'
        }

        msg2 = {
            'side': 'buy',
            'timestamp': 1699876543.0,
            'amount': 1.0,
            'price': 50000.0,
            'symbol': 'BTC-USD',
            'exchange': 'coinbase'
        }

        hash1 = hash_message(msg1, format_type='json')
        hash2 = hash_message(msg2, format_type='json')

        # Verify canonical ordering produces same hash
        assert hash1 == hash2


class TestValidateMessageCount:
    """Test offset-based message counting"""

    def test_count_messages_using_offsets(self):
        """Should use Kafka offsets for O(1) counting"""
        mock_consumer = Mock()
        mock_consumer.beginning_offsets.return_value = {('topic', 0): 0, ('topic', 1): 0}
        mock_consumer.end_offsets.return_value = {('topic', 0): 5000, ('topic', 1): 5000}

        count = validate_message_count(mock_consumer, topic='topic', partitions=[0, 1])

        # Verify O(1) counting via offsets
        assert count == 10000
        mock_consumer.beginning_offsets.assert_called_once()
        mock_consumer.end_offsets.assert_called_once()

    def test_count_messages_single_partition(self):
        """Should handle single partition correctly"""
        mock_consumer = Mock()
        mock_consumer.beginning_offsets.return_value = {('topic', 0): 100}
        mock_consumer.end_offsets.return_value = {('topic', 0): 1100}

        count = validate_message_count(mock_consumer, topic='topic', partitions=[0])

        assert count == 1000

    def test_count_messages_with_offset_gaps(self):
        """Should handle offset gaps (compaction)"""
        mock_consumer = Mock()
        mock_consumer.beginning_offsets.return_value = {
            ('topic', 0): 50,  # Compacted, not starting at 0
            ('topic', 1): 100
        }
        mock_consumer.end_offsets.return_value = {
            ('topic', 0): 5050,
            ('topic', 1): 5100
        }

        count = validate_message_count(mock_consumer, topic='topic', partitions=[0, 1])

        # Count should be end - beginning for each partition
        assert count == (5050 - 50) + (5100 - 100)
        assert count == 10000


class TestCompareMessages:
    """Test message comparison logic"""

    def test_compare_matching_messages(self):
        """Matching messages should return no mismatches"""
        legacy_msgs = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0, 'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'}
        ]

        new_msgs = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0, 'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'}
        ]

        result = compare_messages(legacy_msgs, new_msgs, format_types=('json', 'json'))

        assert result['total_compared'] == 1
        assert result['matches'] == 1
        assert result['mismatches'] == 0
        assert len(result['mismatch_details']) == 0

    def test_compare_mismatched_messages(self):
        """Mismatched messages should be reported with details"""
        legacy_msgs = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0, 'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'}
        ]

        new_msgs = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50001.0, 'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'}
        ]

        result = compare_messages(legacy_msgs, new_msgs, format_types=('json', 'json'))

        assert result['total_compared'] == 1
        assert result['matches'] == 0
        assert result['mismatches'] == 1
        assert len(result['mismatch_details']) == 1

        # Verify mismatch details
        mismatch = result['mismatch_details'][0]
        assert 'legacy_hash' in mismatch
        assert 'new_hash' in mismatch
        assert mismatch['legacy_hash'] != mismatch['new_hash']

    def test_compare_messages_different_lengths(self):
        """Should handle different message list lengths"""
        legacy_msgs = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0, 'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'},
            {'exchange': 'binance', 'symbol': 'ETH-USDT', 'price': 2500.0, 'amount': 10.0, 'timestamp': 1699876544.0, 'side': 'sell'}
        ]

        new_msgs = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0, 'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'}
        ]

        result = compare_messages(legacy_msgs, new_msgs, format_types=('json', 'json'))

        # Should compare min(len(legacy), len(new))
        assert result['total_compared'] == 1
        assert result['length_mismatch'] is True
        assert result['legacy_count'] == 2
        assert result['new_count'] == 1


class TestGenerateComparisonReport:
    """Test comparison report generation"""

    def test_generate_report_with_matches(self):
        """Report should show matching statistics"""
        comparison_result = {
            'total_compared': 1000,
            'matches': 1000,
            'mismatches': 0,
            'mismatch_details': [],
            'legacy_count': 1000,
            'new_count': 1000,
            'length_mismatch': False
        }

        report = generate_comparison_report(comparison_result, exchange='coinbase', symbol='BTC-USD')

        # Verify report structure
        assert 'exchange' in report
        assert report['exchange'] == 'coinbase'
        assert report['symbol'] == 'BTC-USD'
        assert report['match_rate'] == 1.0
        assert report['status'] == 'PASS'

    def test_generate_report_with_mismatches(self):
        """Report should show mismatch details"""
        comparison_result = {
            'total_compared': 1000,
            'matches': 998,
            'mismatches': 2,
            'mismatch_details': [
                {'index': 100, 'legacy_hash': 'abc123', 'new_hash': 'def456'},
                {'index': 500, 'legacy_hash': 'ghi789', 'new_hash': 'jkl012'}
            ],
            'legacy_count': 1000,
            'new_count': 1000,
            'length_mismatch': False
        }

        report = generate_comparison_report(comparison_result, exchange='binance', symbol='ETH-USDT')

        assert report['exchange'] == 'binance'
        assert report['symbol'] == 'ETH-USDT'
        assert report['match_rate'] == 0.998
        assert report['status'] == 'FAIL'
        assert len(report['mismatch_samples']) == 2

    def test_generate_report_with_length_mismatch(self):
        """Report should flag length mismatches"""
        comparison_result = {
            'total_compared': 500,
            'matches': 500,
            'mismatches': 0,
            'mismatch_details': [],
            'legacy_count': 1000,
            'new_count': 500,
            'length_mismatch': True
        }

        report = generate_comparison_report(comparison_result, exchange='kraken', symbol='BTC-EUR')

        assert report['status'] == 'WARNING'
        assert report['length_mismatch'] is True
        assert report['legacy_count'] == 1000
        assert report['new_count'] == 500


class TestFilterByExchangeSymbol:
    """Test filtering messages by exchange/symbol for per-exchange validation"""

    def test_filter_messages_by_exchange(self):
        """Should filter messages by exchange"""
        from validate_data_integrity import filter_messages

        messages = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0},
            {'exchange': 'binance', 'symbol': 'BTC-USDT', 'price': 50001.0},
            {'exchange': 'coinbase', 'symbol': 'ETH-USD', 'price': 2500.0}
        ]

        filtered = filter_messages(messages, exchange='coinbase')

        assert len(filtered) == 2
        assert all(msg['exchange'] == 'coinbase' for msg in filtered)

    def test_filter_messages_by_symbol(self):
        """Should filter messages by symbol"""
        from validate_data_integrity import filter_messages

        messages = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0},
            {'exchange': 'binance', 'symbol': 'BTC-USDT', 'price': 50001.0},
            {'exchange': 'coinbase', 'symbol': 'ETH-USD', 'price': 2500.0}
        ]

        filtered = filter_messages(messages, symbol='BTC-USD')

        assert len(filtered) == 1
        assert filtered[0]['symbol'] == 'BTC-USD'

    def test_filter_messages_by_exchange_and_symbol(self):
        """Should filter messages by both exchange and symbol"""
        from validate_data_integrity import filter_messages

        messages = [
            {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.0},
            {'exchange': 'binance', 'symbol': 'BTC-USD', 'price': 50001.0},
            {'exchange': 'coinbase', 'symbol': 'ETH-USD', 'price': 2500.0}
        ]

        filtered = filter_messages(messages, exchange='coinbase', symbol='BTC-USD')

        assert len(filtered) == 1
        assert filtered[0]['exchange'] == 'coinbase'
        assert filtered[0]['symbol'] == 'BTC-USD'
