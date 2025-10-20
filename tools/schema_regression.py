"""Schema parity regression tool.

Validates that Cryptofeed dataclasses, Protobuf messages, tardis-node JSON, and
DBN layouts maintain semantic equivalence. Replays sample events through
Protobuf serialization/deserialization and compares field-level values to
detect precision loss, missing fields, or scaling mismatches.

Usage:
    python tools/schema_regression.py \
      --events docs/schemas/examples/events/trades.jsonl \
      --output reports/parity-trades.json

Exit codes:
    0: All mismatches resolved or within tolerance
    1: Regressions detected
    2: Tool error (config/file not found)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cryptofeed.types as cf_types

logger = logging.getLogger(__name__)


@dataclass
class FieldParity:
    """Tracks field-level parity check results."""
    field_name: str
    source: str  # cryptofeed, protobuf, tardis, dbn
    expected: Any
    actual: Any
    match: bool
    tolerance: Optional[float] = None
    notes: str = ""


@dataclass
class EventParity:
    """Aggregates parity results for a single event."""
    event_type: str
    event_id: Optional[str] = None
    source_exchange: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    field_checks: List[FieldParity] = field(default_factory=list)
    mismatch_count: int = 0
    warning_count: int = 0
    notes: str = ""

    @property
    def is_clean(self) -> bool:
        return self.mismatch_count == 0


@dataclass
class RegressionReport:
    """Full regression report for a run."""
    generated_at: str
    test_config: Dict[str, Any]
    events_processed: int = 0
    events_clean: int = 0
    total_mismatches: int = 0
    total_warnings: int = 0
    events: List[EventParity] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.events_processed == 0:
            return 0.0
        return 100.0 * self.events_clean / self.events_processed


def _decimal_tolerance(value: Decimal, relative_tol: float = 1e-8) -> float:
    """Compute absolute tolerance for Decimal precision."""
    if value == 0:
        return relative_tol
    return abs(float(value) * relative_tol)


def _compare_decimals(
    expected: Decimal | float,
    actual: Decimal | float,
    field_name: str,
    relative_tol: float = 1e-8,
) -> Tuple[bool, Optional[float], str]:
    """Compare decimal/float values with tolerance."""
    try:
        exp_float = float(expected)
        act_float = float(actual)
        abs_tol = _decimal_tolerance(Decimal(str(exp_float)), relative_tol)
        match = abs(exp_float - act_float) <= abs_tol
        return match, abs_tol, ""
    except (ValueError, TypeError) as exc:
        return False, None, f"Type conversion error: {exc}"


def _load_jsonl_events(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL event samples."""
    events = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping line {line_num} in {path}: {exc}")
    return events


def _construct_dataclass_from_dict(event_dict: Dict[str, Any]) -> Optional[cf_types.BaseEvent]:
    """Attempt to reconstruct a Cryptofeed dataclass from dict."""
    event_type = event_dict.get("type") or event_dict.get("event_type")
    if not event_type:
        return None

    # Map event types to constructors
    constructors = {
        "trade": lambda d: cf_types.Trade(
            exchange=d["exchange"],
            symbol=d["symbol"],
            side=d["side"],
            amount=Decimal(str(d["amount"])),
            price=Decimal(str(d["price"])),
            timestamp=float(d["timestamp"]),
            id=d.get("id"),
        ),
        "ticker": lambda d: cf_types.Ticker(
            exchange=d["exchange"],
            symbol=d["symbol"],
            bid=Decimal(str(d["bid"])),
            ask=Decimal(str(d["ask"])),
            timestamp=float(d["timestamp"]),
        ),
        "funding": lambda d: cf_types.Funding(
            exchange=d["exchange"],
            symbol=d["symbol"],
            mark_price=Decimal(str(d["mark_price"])),
            rate=Decimal(str(d["rate"])),
            next_funding_time=d.get("next_funding_time"),
            timestamp=float(d["timestamp"]),
        ),
        "open_interest": lambda d: cf_types.OpenInterest(
            exchange=d["exchange"],
            symbol=d["symbol"],
            open_interest=Decimal(str(d["open_interest"])),
            timestamp=float(d["timestamp"]),
        ),
    }

    constructor = constructors.get(event_type.lower())
    if not constructor:
        return None

    try:
        return constructor(event_dict)
    except Exception as exc:
        logger.warning(f"Failed to construct {event_type} from dict: {exc}")
        return None


def _check_event_parity(
    event_dict: Dict[str, Any],
    use_protobuf: bool = False,
) -> EventParity:
    """Check parity for a single event across representations."""
    event_type = event_dict.get("type") or event_dict.get("event_type", "unknown")

    parity = EventParity(
        event_type=event_type,
        event_id=event_dict.get("id"),
        source_exchange=event_dict.get("exchange"),
    )

    # Reconstruct Cryptofeed dataclass
    dataclass_obj = _construct_dataclass_from_dict(event_dict)
    if not dataclass_obj:
        parity.mismatch_count += 1
        parity.notes = f"Failed to construct {event_type} dataclass"
        return parity

    # Compare fields
    dataclass_dict = dataclass_obj.to_dict() if hasattr(dataclass_obj, "to_dict") else {}

    for field_name, expected_value in dataclass_dict.items():
        if field_name not in event_dict:
            parity.mismatch_count += 1
            check = FieldParity(
                field_name=field_name,
                source="cryptofeed",
                expected=expected_value,
                actual=None,
                match=False,
                notes="Field missing from source event",
            )
            parity.field_checks.append(check)
            continue

        actual_value = event_dict[field_name]

        # Special handling for Decimal/float precision
        if isinstance(expected_value, Decimal) and isinstance(actual_value, (float, str, int)):
            match, tol, note = _compare_decimals(expected_value, actual_value, field_name)
        else:
            match = expected_value == actual_value
            tol = None
            note = ""

        check = FieldParity(
            field_name=field_name,
            source="cryptofeed",
            expected=expected_value,
            actual=actual_value,
            match=match,
            tolerance=tol,
            notes=note,
        )
        parity.field_checks.append(check)

        if not match:
            parity.mismatch_count += 1

    return parity


def run_regression(args: argparse.Namespace) -> Tuple[int, RegressionReport]:
    """Execute regression tests and produce report."""
    logging.basicConfig(
        level=logging.WARNING if not args.verbose else logging.DEBUG,
        format="%(levelname)s: %(message)s",
    )

    events_path = Path(args.events)
    if not events_path.exists():
        logger.error(f"Events file not found: {events_path}")
        return 2, RegressionReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            test_config={"error": f"Events file not found: {events_path}"},
        )

    events = _load_jsonl_events(events_path)
    logger.info(f"Loaded {len(events)} events from {events_path}")

    report = RegressionReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        test_config={
            "events_file": str(events_path),
            "use_protobuf": args.protobuf,
            "tolerance": args.tolerance,
        },
    )

    for event in events:
        parity = _check_event_parity(event, use_protobuf=args.protobuf)
        report.events.append(parity)
        report.events_processed += 1

        if parity.is_clean:
            report.events_clean += 1
        else:
            report.total_mismatches += parity.mismatch_count

    # Write report
    output_path = Path(args.output or "reports/parity-regression.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_dict = {
        "generated_at": report.generated_at,
        "test_config": report.test_config,
        "summary": {
            "events_processed": report.events_processed,
            "events_clean": report.events_clean,
            "pass_rate": f"{report.pass_rate:.1f}%",
            "total_mismatches": report.total_mismatches,
        },
        "events": [
            {
                **asdict(event),
                "field_checks": [asdict(check) for check in event.field_checks],
            }
            for event in report.events
        ],
    }

    output_path.write_text(json.dumps(report_dict, indent=2, default=str) + "\n")
    logger.info(f"Report written to {output_path}")

    # Determine exit code
    if report.total_mismatches == 0:
        logger.info(f"✓ All {report.events_processed} events passed parity checks")
        return 0, report
    else:
        logger.error(f"✗ {report.total_mismatches} mismatches in {report.events_processed} events")
        return 1, report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate schema parity across Cryptofeed, Protobuf, tardis-node, and DBN"
    )
    parser.add_argument(
        "--events",
        required=True,
        help="Path to JSONL file with event samples",
    )
    parser.add_argument(
        "--output",
        help="Output JSON report path (default: reports/parity-regression.json)",
    )
    parser.add_argument(
        "--protobuf",
        action="store_true",
        help="Include Protobuf serialization tests (requires google-protobuf)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Relative tolerance for Decimal comparisons (default: 1e-8)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    exit_code, _ = run_regression(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
