"""Probe Mullvad SOCKS5 relays for Binance REST/WS reachability.

- Fetches a list of SOCKS5 relays (Mullvad relay artifacts by default)
- Tests Binance REST ping and a public trade WebSocket through each proxy
- Emits per-proxy results with basic latency metrics

Usage:
    python tools/binance_proxy_probe.py --regions eu ap --limit 3

Requirements:
    pip install python-socks websockets aiohttp

Notes:
    - This script is best-effort and intended for manual selection of healthy
      relays before running opt-in Binance E2E tests.
    - Binance may geofence certain regions (HTTP 451/403); those are marked
      as GEOBLOCK.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

import aiohttp
from cryptofeed.connection import HTTPAsyncConn
from cryptofeed.proxy import (
    ConnectionProxies,
    ProxyConfig,
    ProxySettings,
    get_proxy_injector,
    init_proxy_system,
)

PROXY_LIST_URL = "https://raw.githubusercontent.com/mullvad/mulvad-relay-list/refs/heads/proxy-artifacts/relays.txt"

# Require list integrity by default. Update this hash when the list source changes.
# To refresh: curl -s PROXY_LIST_URL | sha256sum
DEFAULT_LIST_SHA256: str | None = "d5558cd419c8d46bdc958064cb97f963d1ea793866414c025906ec15033512ed"
BINANCE_REST_PING = "https://api.binance.com/api/v3/ping"
BINANCE_WS_TRADES = "wss://stream.binance.com:9443/ws/btcusdt@trade"

REGION_PREFIXES = {
    "eu": (
        "al-",
        "at-",
        "be-",
        "bg-",
        "ch-",
        "cy-",
        "cz-",
        "de-",
        "dk-",
        "ee-",
        "es-",
        "fi-",
        "fr-",
        "gb-",
        "gr-",
        "hr-",
        "hu-",
        "ie-",
        "it-",
        "lt-",
        "lu-",
        "lv-",
        "nl-",
        "no-",
        "pl-",
        "pt-",
        "ro-",
        "rs-",
        "se-",
        "si-",
        "sk-",
        "tr-",
        "ua-",
    ),
    "ap": (
        "au-",
        "hk-",
        "id-",
        "jp-",
        "my-",
        "nz-",
        "ph-",
        "sg-",
        "th-",
    ),
}


def _extract_country(proxy_url: str) -> str:
    host = urlparse(proxy_url).hostname or ""
    return host.split("-")[0] if host else ""


def _verify_sha256(text: str, expected: str | None) -> None:
    """Raise ValueError if expected hash is provided and does not match."""

    if not expected:
        return
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != expected.lower():
        raise ValueError(
            f"Proxy list checksum mismatch: expected {expected}, got {digest}"
        )


def _validate_list_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Proxy list URL must use https")
    if parsed.netloc not in {"raw.githubusercontent.com"}:
        raise ValueError(f"Proxy list host not allowed: {parsed.netloc}")


async def _fetch_proxy_list(url: str, expected_sha256: str | None = None) -> List[str]:
    _validate_list_url(url)
    if not expected_sha256:
        raise ValueError("Proxy list checksum is required; provide --list-sha256")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            text = await resp.text()
            _verify_sha256(text, expected_sha256)
            return [line.strip() for line in text.splitlines() if line.strip()]


def _filter_regions(
    proxies: Iterable[str], regions: List[str], limit: int, per_country: bool
) -> List[str]:
    selected: List[str] = []
    per_region = max(1, limit)
    region_sets = {r: [] for r in regions}
    seen_country_per_region: dict[tuple[str, str], bool] = {}
    for proxy in proxies:
        host = urlparse(proxy).hostname or ""
        country = _extract_country(proxy)
        for region in regions:
            prefixes = REGION_PREFIXES.get(region, ())
            if host.startswith(prefixes):
                key = (region, country)
                if per_country and seen_country_per_region.get(key):
                    break
                if len(region_sets[region]) < per_region:
                    region_sets[region].append(proxy)
                    if per_country:
                        seen_country_per_region[key] = True
                break
    for region in regions:
        selected.extend(region_sets[region][:per_region])
    return selected


async def _check_rest(proxy_url: str, timeout: float) -> Tuple[str, float]:
    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(http=ProxyConfig(url=proxy_url)),
    )
    init_proxy_system(settings)
    conn = HTTPAsyncConn("binance-probe-rest", exchange_id="binance")
    start = time.perf_counter()
    try:
        try:
            payload = await asyncio.wait_for(
                conn.read(BINANCE_REST_PING), timeout=timeout
            )
            if payload.strip() != "{}":
                return "BAD_PAYLOAD", time.perf_counter() - start
            return "OK", time.perf_counter() - start
        except asyncio.TimeoutError:
            return "TIMEOUT", time.perf_counter() - start
        except aiohttp.ClientResponseError as exc:
            if exc.status in (403, 451):
                return "GEOBLOCK", time.perf_counter() - start
            return f"HTTP_{exc.status}", time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            return f"ERROR:{type(exc).__name__}", time.perf_counter() - start
    finally:
        await conn.close()
        init_proxy_system(ProxySettings(enabled=False))


async def _check_ws(proxy_url: str, timeout: float) -> Tuple[str, float]:
    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(websocket=ProxyConfig(url=proxy_url)),
    )
    init_proxy_system(settings)
    injector = get_proxy_injector()
    start = time.perf_counter()
    try:
        websocket = await asyncio.wait_for(
            injector.create_websocket_connection(BINANCE_WS_TRADES, "binance"),
            timeout=timeout,
        )
        try:
            _ = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            return "OK", time.perf_counter() - start
        except asyncio.TimeoutError:
            return "TIMEOUT_MSG", time.perf_counter() - start
        finally:
            await websocket.close()
    except asyncio.TimeoutError:
        return "TIMEOUT_CONNECT", time.perf_counter() - start
    except ImportError:
        return "MISSING_PYTHON_SOCKS", time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        if status in (403, 451):
            return "GEOBLOCK", time.perf_counter() - start
        if isinstance(exc, OSError):
            return "OSERROR", time.perf_counter() - start
        return f"ERROR:{type(exc).__name__}", time.perf_counter() - start
    finally:
        init_proxy_system(ProxySettings(enabled=False))


async def probe_proxy(
    proxy_url: str, rest_timeout: float, ws_timeout: float
) -> Tuple[str, str, float, str, float]:
    rest_status, rest_latency = await _check_rest(proxy_url, rest_timeout)
    ws_status, ws_latency = await _check_ws(proxy_url, ws_timeout)
    return proxy_url, rest_status, rest_latency, ws_status, ws_latency


def _format_row(
    proxy: str, rest_status: str, rest_latency: float, ws_status: str, ws_latency: float
) -> str:
    country = _extract_country(proxy)
    return (
        f"{country:3} {proxy:56} REST={rest_status:<12} {rest_latency * 1000:7.1f}ms "
        f"WS={ws_status:<14} {ws_latency * 1000:7.1f}ms"
    )


async def main(args: argparse.Namespace) -> None:
    proxies = await _fetch_proxy_list(args.list_url, expected_sha256=args.list_sha256)
    targets = _filter_regions(proxies, args.regions, args.limit, args.per_country)
    if not targets:
        raise SystemExit("No proxies selected; adjust --regions or --limit")

    print(
        f"Testing {len(targets)} proxies (regions={','.join(args.regions)}, limit={args.limit})"
    )
    for proxy in targets:
        result = await probe_proxy(proxy, args.rest_timeout, args.ws_timeout)
        print(_format_row(*result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe Binance REST/WS over Mullvad SOCKS5 proxies"
    )
    parser.add_argument(
        "--list-url",
        default=PROXY_LIST_URL,
        help="URL of proxy list (one socks5:// per line, https/raw.githubusercontent.com only)",
    )
    parser.add_argument(
        "--list-sha256",
        default=DEFAULT_LIST_SHA256,
        help="Required SHA256 checksum for the proxy list; mismatch aborts",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["eu", "ap"],
        choices=list(REGION_PREFIXES.keys()),
        help="Region filters",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Proxies per region to test (per-region, not global)",
    )
    parser.add_argument(
        "--per-country",
        action="store_true",
        help="Select at most one proxy per country (after region filter)",
    )
    parser.add_argument(
        "--rest-timeout", type=float, default=5.0, help="REST timeout seconds"
    )
    parser.add_argument(
        "--ws-timeout",
        type=float,
        default=8.0,
        help="WebSocket connect+recv timeout seconds",
    )
    asyncio.run(main(parser.parse_args()))
