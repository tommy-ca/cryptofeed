import os
from decimal import Decimal

import pytest

from cryptofeed.exchanges import Binance
from cryptofeed.json_utils import json as json_parser


if not os.getenv("CF_LIVE_REST"):
    pytest.skip("CF_LIVE_REST not set; skipping live REST integration tests", allow_module_level=True)


def temp_f(r, address, json=False, text=False, uuid=None):
        if r.status_code == 451:
            return {'symbols': []}
        r.raise_for_status()
        if json:
            return json_parser.loads(r.text, parse_float=Decimal)
        if text:
            return r.text
        return r 

Binance.http_sync.process_response = temp_f
