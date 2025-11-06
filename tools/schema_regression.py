"""Schema parity regression tool.

Validates that Cryptofeed dataclasses and reference JSON payloads maintain
semantic equivalence. Replays sample events and compares field-level values to
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
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cryptofeed.types as cf_types
from cryptofeed.json_utils import loads as json_loads

logger = logging.getLogger(__name__)


class FixtureLoadError(Exception):
    def __init__(self, path: Path, line: int, original: Exception) -> None:
        super().__init__(f"Failed to load fixture line {line} in {path}: {original}")
        self.path = path
        self.line = line
        self.original = original


class FactoryNotFound(Exception):
    def __init__(self, event_type: str) -> None:
        super().__init__(f"No dataclass factory registered for event type '{event_type}'")
        self.event_type = event_type


class DataclassBuildError(Exception):
    def __init__(self, event_type: str, original: Exception) -> None:
        message = f"Failed to build dataclass for event '{event_type}': {original}"
        super().__init__(message)
        self.event_type = event_type
        self.original = original


class DecimalLoader:
    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict

    def load(self, path: Path) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json_loads(stripped, parse_float=Decimal, parse_int=Decimal))
                except json.JSONDecodeError as exc:
                    if self.strict:
                        raise FixtureLoadError(path, line_num, exc) from exc
                    logger.warning("Skipping malformed JSON on line %s in %s: %s", line_num, path, exc)
        return events


def _candidate_keys(name: str) -> Iterable[str]:
    snake = re.sub("(?<!^)(?=[A-Z])", "_", name).lower()
    squashed = snake.replace("_", "")
    return {snake, squashed, name.lower()}


@lru_cache(maxsize=1)
def _build_type_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    for attr in dir(cf_types):
        cls = getattr(cf_types, attr)
        # Skip private attributes and non-callable objects
        if attr.startswith("_"):
            continue
        if not hasattr(cls, "to_dict"):
            continue
        for key in _candidate_keys(attr):
            registry.setdefault(key, cls)
    return registry


class DataclassFactoryAdapter:
    def __init__(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        self.overrides = overrides or {}

    def build(self, event_dict: Dict[str, Any]) -> cf_types.BaseEvent:
        event_type = event_dict.get("type") or event_dict.get("event_type")
        if not event_type:
            raise FactoryNotFound("<missing>")

        normalized = event_type.lower()
        candidate_keys = {normalized, normalized.replace("_", ""), normalized.replace("-", "")}

        registry = _build_type_registry()

        cls = None
        for key in candidate_keys:
            if key in self.overrides:
                cls = self.overrides[key]
                break
            if key in registry:
                cls = registry[key]
                break

        if cls is None:
            raise FactoryNotFound(event_type)

        try:
            if hasattr(cls, "from_dict"):
                return cls.from_dict(event_dict)
            return cls(**event_dict)  # type: ignore[arg-type]
        except Exception as exc:
            raise DataclassBuildError(event_type, exc) from exc


@dataclass
class FieldParity:
    """Tracks field-level parity check results."""
    field_name: str
    source: str  # cryptofeed, protobuf, tardis, dbn
    expected: Any
    actual: Any
    match: bool
    tolerance: Optional[str] = None
    difference: Optional[str] = None
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
    factory_status: str = "ok"

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


def _decimal_tolerance(value: Decimal, relative_tol: Decimal = Decimal("1e-8")) -> Decimal:
    if value == 0:
        return relative_tol
    return abs(value * relative_tol)


def _compare_decimals(
    expected: Decimal,
    actual: Decimal,
    field_name: str,
    relative_tol: Decimal = Decimal("1e-8"),
) -> Tuple[bool, Decimal, str]:
    try:
        diff = abs(expected - actual)
        tolerance = _decimal_tolerance(expected, relative_tol)
        return diff <= tolerance, tolerance, ""
    except (InvalidOperation, TypeError) as exc:
        return False, Decimal(0), f"Decimal comparison error for {field_name}: {exc}"


def _load_jsonl_events(path: Path, *, strict: bool = True) -> List[Dict[str, Any]]:
    loader = DecimalLoader(strict=strict)
    return loader.load(path)


def _construct_dataclass_from_dict(adapter: DataclassFactoryAdapter, event_dict: Dict[str, Any]) -> Optional[cf_types.BaseEvent]:
    try:
        return adapter.build(event_dict)
    except FactoryNotFound as exc:
        logger.warning("%s", exc)
    except DataclassBuildError as exc:
        logger.warning("%s", exc)
    return None


def _check_event_parity(
    adapter: DataclassFactoryAdapter,
    event_dict: Dict[str, Any],
    relative_tol: Decimal = Decimal("1e-8"),
) -> EventParity:
    """Check parity for a single event across representations."""
    event_type = event_dict.get("type") or event_dict.get("event_type", "unknown")

    parity = EventParity(
        event_type=event_type,
        event_id=event_dict.get("id"),
        source_exchange=event_dict.get("exchange"),
    )

    # Reconstruct Cryptofeed dataclass
    dataclass_obj = _construct_dataclass_from_dict(adapter, event_dict)
    if not dataclass_obj:
        parity.mismatch_count += 1
        parity.warning_count += 1
        parity.notes = f"Failed to construct {event_type} dataclass"
        parity.factory_status = "missing"
        return parity

    # Compare fields
    dataclass_dict = dataclass_obj.to_dict() if hasattr(dataclass_obj, "to_dict") else {}

    for field_name, expected_value in dataclass_dict.items():
        # Ignore the synthetic 'type' field for parity; event_type is validated separately
        if field_name == "type":
            continue
        if field_name not in event_dict:
            # Treat absent optional fields (None) as acceptable
            if expected_value is None:
                continue
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

        match = expected_value == actual_value
        tolerance_str = None
        difference_str = None
        note = ""

        expected_decimal = isinstance(expected_value, Decimal)
        actual_decimal = isinstance(actual_value, Decimal)

        if expected_decimal and actual_decimal:
            match, tol, note = _compare_decimals(expected_value, actual_value, field_name, relative_tol)
            tolerance_str = str(tol)
            difference_str = str(abs(expected_value - actual_value))
        elif expected_decimal and isinstance(actual_value, str):
            try:
                actual_dec = Decimal(actual_value)
                match, tol, note = _compare_decimals(expected_value, actual_dec, field_name, relative_tol)
                tolerance_str = str(tol)
                difference_str = str(abs(expected_value - actual_dec))
                actual_value = actual_dec
            except InvalidOperation as exc:
                match = False
                note = f"Invalid decimal string: {exc}"
        elif isinstance(expected_value, (int, float)) and isinstance(actual_value, Decimal):
            expected_dec = Decimal(str(expected_value))
            match, tol, note = _compare_decimals(expected_dec, actual_value, field_name, relative_tol)
            tolerance_str = str(tol)
            difference_str = str(abs(expected_dec - actual_value))

        check = FieldParity(
            field_name=field_name,
            source="cryptofeed",
            expected=expected_value,
            actual=actual_value,
            match=match,
            tolerance=tolerance_str,
            difference=difference_str,
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

    strict_mode = getattr(args, "strict", True)

    try:
        events = _load_jsonl_events(events_path, strict=strict_mode)
    except FixtureLoadError as exc:
        logger.error("%s", exc)
        return 2, RegressionReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            test_config={"error": str(exc)},
        )
    logger.info(f"Loaded {len(events)} events from {events_path}")

    adapter = DataclassFactoryAdapter()
    tolerance = Decimal(str(args.tolerance))

    report = RegressionReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        test_config={
            "events_file": str(events_path),
            "tolerance": args.tolerance,
            "strict": strict_mode,
        },
    )

    for event in events:
        parity = _check_event_parity(adapter, event, relative_tol=tolerance)
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
        description="Validate schema parity between Cryptofeed dataclasses and reference JSON payloads"
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
        "--tolerance",
        type=float,
        default=1e-8,
        help="Relative tolerance for Decimal comparisons (default: 1e-8)",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop on fixture parse errors (default: True)",
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
