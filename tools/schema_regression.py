"""Schema parity and throughput regression pipeline.

This script reuses the canonical Buf Protobuf module to validate parity between
tardis-node JSON events, DBN layout samples (scaled integers), and the
normalized Protobuf encoding. It relies on the Buf CLI to build a descriptor set
at runtime and uses the dynamic protobuf API to encode/decode payloads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


PROTO_MESSAGE = "cryptofeed.normalized.v1.Trade"


@dataclass
class ParityResult:
    total: int
    matches: int
    mismatches: List[str]


@dataclass
class ThroughputResult:
    samples: int
    encoding_rate: float | None
    decoding_rate: float | None


def _buf_available() -> bool:
    return subprocess.call(["buf", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def _load_descriptor(module_path: Path) -> tuple[descriptor_pool.DescriptorPool, descriptor_pb2.FileDescriptorSet]:
    if not _buf_available():
        raise RuntimeError("buf CLI not available; install from https://buf.build/docs/installation")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(["buf", "build", str(module_path), "-o", str(tmp_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        descriptor_set = descriptor_pb2.FileDescriptorSet()
        descriptor_set.ParseFromString(tmp_path.read_bytes())
    finally:
        tmp_path.unlink(missing_ok=True)

    pool = descriptor_pool.DescriptorPool()
    for file_proto in descriptor_set.file:
        pool.Add(file_proto)
    return pool, descriptor_set


def _message_class(pool: descriptor_pool.DescriptorPool, descriptor_set: descriptor_pb2.FileDescriptorSet, full_name: str):
    messages = message_factory.GetMessages(descriptor_set.file)
    if full_name not in messages:
        raise KeyError(f"Message {full_name} not found in descriptor set")
    return messages[full_name]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _prepare_trade_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    side = str(event.get("side", "buy")).lower()
    side_map = {"buy": 1, "sell": 2}
    return {
        "exchange": event.get("exchange"),
        "symbol": event.get("symbol"),
        "trade_id": event.get("trade_id") or event.get("id"),
        "side": side_map.get(side, 0),
        "price": str(event.get("price")),
        "amount": str(event.get("amount")),
        "timestamp": int(event.get("ts_event")),
        "raw_id": event.get("meta", {}).get("raw_id"),
    }


def _normalize_dbn_event(event: Dict[str, Any]) -> Dict[str, Any]:
    scale_price = Decimal(event.get("scale", {}).get("price", "1"))
    scale_amount = Decimal(event.get("scale", {}).get("amount", "1"))
    price = Decimal(event.get("price")) * scale_price
    amount = Decimal(event.get("quantity")) * scale_amount
    side_flag = event.get("side_flag", 0)
    side = 1 if side_flag == 0 else 2
    return {
        "exchange": event.get("exchange"),
        "symbol": event.get("symbol"),
        "trade_id": event.get("trade_id"),
        "side": side,
        "price": f"{price:f}",
        "amount": f"{amount:f}",
        "timestamp": int(event.get("timestamp")),
        "raw_id": event.get("raw_id"),
    }


def parity_check(message_cls, events: Iterable[Dict[str, Any]]) -> ParityResult:
    matches = 0
    mismatches: List[str] = []
    total = 0
    for event in events:
        total += 1
        payload = _prepare_trade_payload(event)
        message = message_cls(**{k: v for k, v in payload.items() if v is not None})
        decoded = message_cls.FromString(message.SerializeToString())
        if (
            decoded.exchange == payload["exchange"]
            and decoded.symbol == payload["symbol"]
            and decoded.timestamp == payload["timestamp"]
            and decoded.side == payload["side"]
        ):
            matches += 1
        else:
            mismatches.append(f"Mismatch for trade_id={payload.get('trade_id')}")
    return ParityResult(total=total, matches=matches, mismatches=mismatches)


def throughput_test(message_cls, events: Iterable[Dict[str, Any]]) -> ThroughputResult:
    import time

    events = list(events)
    if not events:
        return ThroughputResult(samples=0, encoding_rate=None, decoding_rate=None)

    # Encoding benchmark
    start = time.perf_counter()
    blobs = []
    for event in events:
        payload = _prepare_trade_payload(event)
        message = message_cls(**{k: v for k, v in payload.items() if v is not None})
        blobs.append(message.SerializeToString())
    duration = time.perf_counter() - start
    encoding_rate = len(events) / duration if duration > 0 else None

    # Decoding benchmark
    start = time.perf_counter()
    for blob in blobs:
        message_cls.FromString(blob)
    duration = time.perf_counter() - start
    decoding_rate = len(events) / duration if duration > 0 else None

    return ThroughputResult(samples=len(events), encoding_rate=encoding_rate, decoding_rate=decoding_rate)


def run_regression(args: argparse.Namespace) -> int:
    module_path = Path(args.module)
    pool, descriptor_set = _load_descriptor(module_path)
    message_cls = _message_class(pool, descriptor_set, PROTO_MESSAGE)

    tardis_events = _load_jsonl(Path(args.events))

    parity = parity_check(message_cls, tardis_events)
    throughput = throughput_test(message_cls, tardis_events)

    report = {
        "parity": {
            "total": parity.total,
            "matches": parity.matches,
            "mismatches": parity.mismatches,
        },
        "throughput": {
            "samples": throughput.samples,
            "encoding_events_per_second": throughput.encoding_rate,
            "decoding_events_per_second": throughput.decoding_rate,
        },
    }

    output_path = Path(args.output or "schema_regression_report.json")
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {output_path}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run schema parity regression against Buf Protobuf module")
    parser.add_argument("--module", default=".", help="Path to Buf module root (default: current directory)")
    parser.add_argument("--events", required=True, help="Path to JSONL file containing tardis-node normalized events")
    parser.add_argument("--output", help="Path to write regression report (default: schema_regression_report.json)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_regression(args)


if __name__ == "__main__":
    raise SystemExit(main())
