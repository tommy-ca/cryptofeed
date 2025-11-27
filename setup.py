'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
import os
import sys

from setuptools import Extension, setup
from setuptools import find_packages
from setuptools.command.test import test as TestCommand
from Cython.Build import cythonize

_BASE_REQUIREMENTS = [
    "aiodns>=1.1",
    "aiofile>=2.0.0",
    "aiohttp>=3.9.5",
    "charset-normalizer>=3.3.0",
    "cython",
    "order_book>=0.6.1",
    "pyyaml",
    "requests>=2.18.4",
    "websockets>=14.1",
    "orjson>=3.10.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]


def get_long_description():
    """Read the contents of README.md, INSTALL.md and CHANGES.md files."""
    from os import path

    repo_dir = path.abspath(path.dirname(__file__))
    markdown = []
    for filename in ["README.md", "INSTALL.md", "CHANGES.md"]:
        with open(path.join(repo_dir, filename), encoding="utf-8") as markdown_file:
            markdown.append(markdown_file.read())
    return "\n\n----\n\n".join(markdown)


def load_requirements(filename: str = "requirements.txt") -> list[str]:
    repo_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(repo_dir, filename)
    requirements: list[str] = []
    try:
        with open(path, encoding="utf-8") as req_file:
            for raw_line in req_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                if line:
                    requirements.append(line)
    except FileNotFoundError:
        return list(_BASE_REQUIREMENTS)

    return requirements or list(_BASE_REQUIREMENTS)


class Test(TestCommand):
    def run_tests(self):
        import pytest
        errno = pytest.main(['tests/'])
        sys.exit(errno)


extra_compile_args = ["/O2" if os.name == "nt" else "-O3"]
define_macros = []

# comment out line to compile with type check assertions
# verify value at runtime with cryptofeed.types.COMPILED_WITH_ASSERTIONS
define_macros.append(('CYTHON_WITHOUT_ASSERTIONS', None))

extension = Extension("cryptofeed.types", ["cryptofeed/types.pyx"],
                      extra_compile_args=extra_compile_args,
                      define_macros=define_macros)

setup(
    name="cryptofeed",
    ext_modules=cythonize([extension], language_level=3, force=True),
    version="2.4.1",
    author="Bryant Moscon",
    author_email="bmoscon@gmail.com",
    description="Cryptocurrency Exchange Websocket Data Feed Handler",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    license="XFree86",
    keywords=["cryptocurrency", "bitcoin", "btc", "feed handler", "market feed", "market data", "crypto assets",
              "Trades", "Tickers", "BBO", "Funding", "Open Interest", "Liquidation", "Order book", "Bid", "Ask",
              "fmfw.io", "Bitfinex", "bitFlyer", "AscendEX", "Bitstamp", "Blockchain.com", "Bybit",
              "Binance", "Binance Delivery", "Binance Futures", "Binance US", "BitMEX", "Coinbase", "Deribit", "EXX",
              "Gate.io", "Gemini", "HitBTC", "Huobi", "Huobi DM", "Huobi Swap", "Kraken",
              "Kraken Futures", "OKCoin", "OKX", "Poloniex", "ProBit", "Upbit"],
    url="https://github.com/bmoscon/cryptofeed",
    packages=find_packages(exclude=['tests*']),
    cmdclass={'test': Test},
    python_requires='>=3.9',
    classifiers=[
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: AsyncIO",
    ],
    tests_require=["pytest"],
    install_requires=list(_BASE_REQUIREMENTS),
    extras_require={
        "arctic": ["arctic", "pandas"],
        "backpack": ["PyNaCl>=1.5"],
        "ccxt": ["ccxt>=4.5.9", "pydantic>=2.0.0", "pydantic-settings>=2.0.0"],
        "gcp_pubsub": ["google_cloud_pubsub>=2.4.1", "gcloud_aio_pubsub"],
        "kafka": ["aiokafka>=0.7.0"],
        "mongo": ["motor"],
        "postgres": ["asyncpg"],
        "proxy": ["aiohttp-socks>=0.9.2", "python-socks>=2.4.3"],
        "quality": ["pyscn>=0.6.0"],
        "quasardb": ["quasardb", "numpy"],
        "rabbit": ["aio_pika", "pika"],
        "redis": ["hiredis", "redis>=4.5.1"],
        "uvloop": ['uvloop; platform_system!="Windows"'],
        "zmq": ["pyzmq"],
        "socks": ["python-socks>=2.4.3"],
        "all": [
            "arctic",
            "pandas",
            "google_cloud_pubsub>=2.4.1",
            "gcloud_aio_pubsub",
            "aiokafka>=0.7.0",
            "motor",
            "asyncpg",
            "aio_pika",
            "pika",
            "hiredis",
            "redis>=4.5.1",
            "pyzmq",
            "aiohttp-socks>=0.9.2",
            "python-socks>=2.4.3",
            "pyscn>=0.6.0",
            "quasardb",
            "numpy",
            "PyNaCl>=1.5",
            "ccxt>=4.5.9",
            "pydantic>=2.0.0",
            "pydantic-settings>=2.0.0",
            'uvloop; platform_system!="Windows"',
        ],
    },
)
