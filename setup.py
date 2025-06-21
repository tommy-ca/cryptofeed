"""Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com.

Please see the LICENSE file for the terms and conditions
associated with this software.

DEPRECATED: This project now uses hatch as the build backend.
All configuration is in pyproject.toml. This setup.py is kept for
backward compatibility only and should not be used for new development.

Use 'hatch build' or 'python -m build' instead of 'python setup.py build'.
"""

# For backward compatibility, redirect to build backend
try:
    from setuptools import setup

    # Empty setup() call - all configuration is now in pyproject.toml
    setup()
except ImportError:
    # If setuptools is not available, provide helpful error
    raise ImportError(
        "This project uses hatch as the build backend. Please use:\n"
        "  pip install hatch\n"
        "  hatch build\n"
        "or\n"
        "  pip install build\n"
        "  python -m build\n"
        "All configuration is in pyproject.toml."
    )
