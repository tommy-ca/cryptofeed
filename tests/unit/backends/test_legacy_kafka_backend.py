import importlib
import sys
import warnings


def test_legacy_kafka_import_emits_no_deprecation_warning():
    """
    Ensure the legacy backend stays maintained without raising DeprecationWarnings.
    """

    module_name = "cryptofeed.backends.kafka"

    # Force re-import so warnings triggered at import time would surface.
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.import_module(module_name)
