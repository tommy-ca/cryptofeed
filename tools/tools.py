"""Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com.

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

from urllib.request import urlopen

import requests
from yapic import json


"""
Just random functions I used while developing the library.
They may come in handy again . . .
"""


def poloniex_get_ticker_map():
    """Mappings between pair strings and pair IDs are not documented
    so we can use their ticker endpoint which has the mappings embedded.
    """
    with urlopen("https://poloniex.com/public?command=returnTicker") as url:
        data = json.loads(url.read().decode())
        for _key in data:
            pass

        for _key in data:
            pass


def bittrex_get_trading_pairs():
    with urlopen("https://bittrex.com/api/v1.1/public/getmarkets") as url:
        data = json.loads(url.read().decode())
        for _market in data["result"]:
            pass


def coinbase_get_trading_pairs():
    with urlopen("https://api.pro.coinbase.com/products") as url:
        data = json.loads(url.read().decode())
        for _pair in data:
            pass


def hitbtc_get_trading_pairs():
    with urlopen("https://api.hitbtc.com/api/2/public/symbol") as url:
        data = json.loads(url.read().decode())
        for _pair in data:
            pass


def cex_get_trading_pairs():
    r = requests.get("https://cex.io/api/currency_limits")
    for _data in r.json()["data"]["pairs"]:
        pass


def exx_get_trading_pairs():
    r = requests.get("https://api.exx.com/data/v1/tickers")
    for _key in r.json():
        pass


def bitmex_instruments():
    r = requests.get("https://www.bitmex.com/api/v1/instrument/active")
    data = r.json()
    for _d in data:
        pass


if __name__ == "__main__":
    bitmex_instruments()
