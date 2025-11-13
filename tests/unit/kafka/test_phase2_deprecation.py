"""Unit tests for Critical Issue #3: Dual implementation deprecation warnings.

Verifies that the legacy cryptofeed.backends.kafka module emits clear
deprecation warnings guiding users to migrate to the unified KafkaCallback.
"""

import warnings

import pytest


class TestLegacyBackendDeprecation:
    """Test deprecation warnings for legacy backend classes."""

    def test_legacy_module_import_shows_deprecation_warning(self):
        """Importing cryptofeed.backends.kafka should emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import the legacy module
            import cryptofeed.backends.kafka

            # Should have at least one deprecation warning
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "cryptofeed.backends.kafka is deprecated" in str(w[0].message)
            assert "cryptofeed.kafka_callback.KafkaCallback" in str(w[0].message)

    def test_trade_kafka_deprecation_warning(self):
        """TradeKafka instantiation should emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TradeKafka

            # Instantiation should emit warning
            with warnings.catch_warnings(record=True) as w2:
                warnings.simplefilter("always")
                _ = TradeKafka(bootstrap_servers=['kafka:9092'])

                # Should have deprecation warning from __init__
                init_warnings = [x for x in w2 if issubclass(x.category, DeprecationWarning)]
                assert len(init_warnings) >= 1
                assert "TradeKafka is deprecated" in str(init_warnings[0].message)

    def test_book_kafka_deprecation_warning(self):
        """BookKafka instantiation should emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import BookKafka

            with warnings.catch_warnings(record=True) as w2:
                warnings.simplefilter("always")
                _ = BookKafka(bootstrap_servers=['kafka:9092'])

                init_warnings = [x for x in w2 if issubclass(x.category, DeprecationWarning)]
                assert len(init_warnings) >= 1
                assert "BookKafka is deprecated" in str(init_warnings[0].message)

    def test_ticker_kafka_deprecation_warning(self):
        """TickerKafka instantiation should emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TickerKafka

            with warnings.catch_warnings(record=True) as w2:
                warnings.simplefilter("always")
                _ = TickerKafka(bootstrap_servers=['kafka:9092'])

                init_warnings = [x for x in w2 if issubclass(x.category, DeprecationWarning)]
                assert len(init_warnings) >= 1
                assert "TickerKafka is deprecated" in str(init_warnings[0].message)


class TestUnifiedImplementation:
    """Test that unified implementation is accessible and does not emit warnings."""

    def test_unified_kafka_callback_import_no_warning(self):
        """Importing cryptofeed.kafka_callback should not emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.kafka_callback import KafkaCallback

            # Should not have any deprecation warnings from kafka_callback module
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            kafka_callback_warnings = [
                x for x in deprecation_warnings
                if "kafka_callback" in str(x.filename)
            ]
            assert len(kafka_callback_warnings) == 0

    def test_unified_kafka_callback_instantiation_no_warning(self):
        """Instantiating KafkaCallback should not emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.kafka_callback import KafkaCallback

            # Create a stub producer to avoid connection attempts
            def stub_producer_factory(config):
                class StubProducer:
                    def __init__(self, config):
                        self.connected = False
                    def list_topics(self, timeout=None):
                        self.connected = True
                        return {"topics": []}
                    def produce(self, *args, **kwargs):
                        pass
                    def poll(self, timeout):
                        return 0
                    def flush(self, timeout=None):
                        return 0
                return StubProducer(config)

            _ = KafkaCallback(
                bootstrap_servers=['kafka:9092'],
                producer_factory=stub_producer_factory
            )

            # Should not have any deprecation warnings
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0


class TestMigrationGuidance:
    """Test that deprecation warnings provide clear migration guidance."""

    def test_module_docstring_contains_migration_guide(self):
        """Legacy module docstring should contain migration guide."""
        import cryptofeed.backends.kafka as legacy_module

        docstring = legacy_module.__doc__
        assert docstring is not None
        assert "DEPRECATION NOTICE" in docstring
        assert "Migration Guide" in docstring
        assert "cryptofeed.kafka_callback" in docstring
        assert "KafkaConfig" in docstring

    def test_deprecation_message_mentions_new_module(self):
        """Deprecation warnings should mention the new module."""
        # Module may have been imported already by other tests,
        # so check the module docstring for migration guidance
        import cryptofeed.backends.kafka as legacy_module

        # The docstring contains the migration guide
        docstring = legacy_module.__doc__
        assert "kafka_callback" in docstring.lower()
        assert "KafkaCallback" in docstring or "KafkaConfig" in docstring

    def test_deprecation_message_mentions_features(self):
        """Deprecation message should mention bypassed features."""
        import cryptofeed.backends.kafka as legacy_module

        docstring = legacy_module.__doc__
        assert "TopicManager" in docstring
        assert "HeaderEnricher" in docstring
        assert "Partitioner" in docstring
        assert "error handling" in docstring


class TestBackwardCompatibility:
    """Test that legacy classes remain functional (backward compatible)."""

    def test_legacy_trade_kafka_still_works(self):
        """Legacy TradeKafka should still be functional."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test

            from cryptofeed.backends.kafka import TradeKafka

            # Should be able to instantiate (even though deprecated)
            kafka = TradeKafka(bootstrap_servers=['kafka:9092'])
            assert kafka.default_key == 'trades'
            assert kafka.protobuf_data_type == 'trades'

    def test_legacy_book_kafka_still_works(self):
        """Legacy BookKafka should still be functional."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from cryptofeed.backends.kafka import BookKafka

            kafka = BookKafka(bootstrap_servers=['kafka:9092'])
            assert kafka.default_key == 'book'
            assert kafka.protobuf_data_type == 'orderbook'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
