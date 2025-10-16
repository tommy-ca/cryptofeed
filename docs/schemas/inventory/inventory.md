# Schema Inventory

Generated at: 2025-10-16T19:28:48.528241+00:00

## cryptofeed – balances

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `balance` | Decimal | 1e0 | complete |  |
| `currency` | str |  | complete |  |
| `exchange` | str |  | complete |  |
| `reserved` | Decimal | 1e-1 | complete |  |

## cryptofeed – funding

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `exchange` | str |  | complete |  |
| `mark_price` | Decimal | 1e-2 | complete |  |
| `next_funding_time` | unknown |  | complete |  |
| `predicted_rate` | unknown |  | complete |  |
| `rate` | Decimal | 1e-4 | complete |  |
| `symbol` | str |  | complete |  |
| `timestamp` | float |  | complete |  |

## cryptofeed – open_interest

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `exchange` | str |  | complete |  |
| `open_interest` | Decimal | 1e0 | complete |  |
| `symbol` | str |  | complete |  |
| `timestamp` | float |  | complete |  |

## cryptofeed – order_book

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `book` | dict |  | complete |  |
| `delta` | unknown |  | complete |  |
| `exchange` | str |  | complete |  |
| `symbol` | str |  | complete |  |
| `timestamp` | unknown |  | complete |  |

## cryptofeed – ticker

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `ask` | Decimal | 1e-2 | complete |  |
| `bid` | Decimal | 1e-2 | complete |  |
| `exchange` | str |  | complete |  |
| `symbol` | str |  | complete |  |
| `timestamp` | float |  | complete |  |

## cryptofeed – trade

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `amount` | Decimal | 1e-1 | complete |  |
| `exchange` | str |  | complete |  |
| `id` | str |  | complete |  |
| `price` | Decimal | 1e-2 | complete |  |
| `side` | str |  | complete |  |
| `symbol` | str |  | complete |  |
| `timestamp` | float |  | complete |  |
| `type` | unknown |  | complete |  |

## dbn – *

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `*` | unknown |  | missing | No DBN layout directory provided. |

## tardis-node – *

| Field | Type | Precision | Status | Notes |
| --- | --- | --- | --- | --- |
| `*` | unknown |  | missing | No tardis-node schema path provided. |
