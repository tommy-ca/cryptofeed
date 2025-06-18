"""
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

from pathlib import Path

from cryptofeed.exchanges import EXCHANGE_MAP


def test_exchanges_fh():
    """
    Ensure all exchanges are in feedhandler's string to class mapping
    """
    path = Path(__file__).resolve().parent
    files = list(path.glob("../../cryptofeed/exchanges/*.py"))
    # Convert Path objects to filenames and filter
    files = [f.name for f in files if "__" not in f.name and "mixins" not in f.name]
    files = [f.replace("cryptodotcom", "CRYPTO.COM") for f in files]
    files = [f.replace("bitdotcom", "BIT.COM") for f in files]
    files = [f[:-3].upper() for f in files]  # Drop extension .py and uppercase

    # Exclude known disabled exchanges (those with import or other issues)
    disabled_exchanges = {"OKCOIN"}  # Add other disabled exchanges here as needed
    files = [f for f in files if f not in disabled_exchanges]

    assert sorted(files) == sorted(EXCHANGE_MAP.keys())
