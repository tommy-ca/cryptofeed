"""Schema inventory CLI.

Generates consolidated field metadata across Cryptofeed dataclasses and, when
provided, tardis-node JSON schemas and DBN layout descriptions. The output is a
JSON + Markdown matrix surfacing coverage status, conflict markers, and basic
freshness alerts so teams can reason about schema gaps before generating Buf
modules.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cryptofeed.types as cf_types


INVENTORY_DIR_DEFAULT = Path("docs/schemas/inventory")
FRESHNESS_THRESHOLD_DAYS_DEFAULT = 7


@dataclass
class FieldRecord:
    source: str
    event_type: str
    field: str
    data_type: str
    precision: Optional[str] = None
    unit: Optional[str] = None
    status: str = "complete"  # complete / partial / missing / conflict
    notes: Optional[str] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _decimal_precision(value: Decimal) -> str:
    if not isinstance(value, Decimal):
        return ""
    tup = value.as_tuple()
    if tup.exponent >= 0:
        return f"1e{tup.exponent}"
    return f"1e{tup.exponent}"


def _collect_cryptofeed_records() -> List[FieldRecord]:
    sample_builders = {
        "trade": lambda: cf_types.Trade(
            "SAMPLE",
            "BTC-USD",
            "buy",
            Decimal("0.1"),
            Decimal("123.45"),
            1_690_000_000.0,
            id="sample",
        ),
        "ticker": lambda: cf_types.Ticker(
            "SAMPLE",
            "BTC-USD",
            Decimal("123.45"),
            Decimal("123.55"),
            1_690_000_000.0,
        ),
        "order_book": lambda: cf_types.OrderBook("SAMPLE", "BTC-USD"),
        "funding": lambda: cf_types.Funding(
            "SAMPLE",
            "BTC-USD",
            Decimal("123.45"),
            Decimal("0.0001"),
            None,
            1_690_000_000.0,
        ),
        "open_interest": lambda: cf_types.OpenInterest(
            "SAMPLE",
            "BTC-USD",
            Decimal("1000"),
            1_690_000_000.0,
        ),
        "balances": lambda: cf_types.Balance(
            "SAMPLE",
            "BTC",
            Decimal("1"),
            Decimal("0.5"),
        ),
    }

    records: List[FieldRecord] = []

    for event, builder in sample_builders.items():
        try:
            sample = builder()
        except Exception as exc:  # pylint: disable=broad-except
            records.append(
                FieldRecord(
                    source="cryptofeed",
                    event_type=event,
                    field="*",
                    data_type="unknown",
                    status="missing",
                    notes=f"Failed to build sample: {exc}",
                )
            )
            continue

        payload = sample.to_dict() if hasattr(sample, "to_dict") else {}
        for field, value in payload.items():
            dtype = type(value).__name__ if value is not None else "unknown"
            precision = _decimal_precision(value) if isinstance(value, Decimal) else None
            records.append(
                FieldRecord(
                    source="cryptofeed",
                    event_type=event,
                    field=field,
                    data_type=dtype,
                    precision=precision,
                )
            )

    return records


def _load_json_schema_fields(path: Path) -> List[FieldRecord]:
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)

    fields: List[FieldRecord] = []

    def walk(schema: Dict[str, Any], prefix: str = "") -> Iterable[FieldRecord]:
        properties = schema.get("properties") or {}
        for key, meta in properties.items():
            full_name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            dtype = meta.get("type", "unknown")
            yield FieldRecord(
                source="tardis-node",
                event_type=schema.get("title", "unknown").lower(),
                field=full_name,
                data_type=dtype,
                notes=meta.get("description"),
            )
            if meta.get("type") == "object":
                yield from walk(meta, full_name)

    fields.extend(list(walk(doc)))
    return fields


def _load_directory_json_schemas(directory: Path) -> List[FieldRecord]:
    records: List[FieldRecord] = []
    for json_file in directory.rglob("*.json"):
        try:
            records.extend(_load_json_schema_fields(json_file))
        except Exception as exc:  # pylint: disable=broad-except
            records.append(
                FieldRecord(
                    source="tardis-node",
                    event_type=json_file.stem,
                    field="*",
                    data_type="unknown",
                    status="missing",
                    notes=f"Failed to parse {json_file}: {exc}",
                )
            )
    return records


def _load_dbn_layout(directory: Path) -> List[FieldRecord]:
    records: List[FieldRecord] = []
    for layout in directory.rglob("*.yaml"):
        try:
            import yaml  # Lazy import for optional dependency

            with layout.open(encoding="utf-8") as handle:
                doc = yaml.safe_load(handle)
        except Exception as exc:  # pylint: disable=broad-except
            records.append(
                FieldRecord(
                    source="dbn",
                    event_type=layout.stem,
                    field="*",
                    data_type="unknown",
                    status="missing",
                    notes=f"Failed to parse {layout}: {exc}",
                )
            )
            continue

        fields = doc.get("fields", []) if isinstance(doc, dict) else []
        for field in fields:
            records.append(
                FieldRecord(
                    source="dbn",
                    event_type=str(doc.get("event_type", layout.stem)),
                    field=str(field.get("name")),
                    data_type=field.get("type", "unknown"),
                    precision=str(field.get("scale")) if field.get("scale") else None,
                    notes=f"offset={field.get('offset')} bytes",
                )
            )
    return records


def _group_by_event(records: Iterable[FieldRecord]) -> Dict[str, List[FieldRecord]]:
    grouped: Dict[str, List[FieldRecord]] = defaultdict(list)
    for record in records:
        key = f"{record.source}:{record.event_type}"
        grouped[key].append(record)
    return grouped


def _write_outputs(records: List[FieldRecord], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = _now().isoformat()
    data = {
        "generated_at": generated_at,
        "records": [asdict(rec) for rec in records],
    }

    json_path = output_dir / "inventory.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    grouped = _group_by_event(records)
    lines = ["# Schema Inventory", "", f"Generated at: {generated_at}", ""]
    for group, group_records in sorted(grouped.items()):
        source, event = group.split(":", maxsplit=1)
        lines.append(f"## {source} – {event}")
        lines.append("")
        lines.append("| Field | Type | Precision | Status | Notes |")
        lines.append("| --- | --- | --- | --- | --- |")
        for rec in sorted(group_records, key=lambda r: r.field):
            lines.append(
                f"| `{rec.field}` | {rec.data_type} | {rec.precision or ''} | {rec.status} | {rec.notes or ''} |"
            )
        lines.append("")

    md_path = output_dir / "inventory.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return data


def _freshness_alerts(data: Dict[str, Any], threshold_days: int) -> List[str]:
    alerts: List[str] = []
    generated_at = datetime.fromisoformat(data["generated_at"])
    age = _now() - generated_at
    if age > timedelta(days=threshold_days):
        alerts.append(
            f"Inventory is stale ({age.days} days old); threshold is {threshold_days} days."
        )

    missing = [rec for rec in data["records"] if rec["status"] in {"missing", "conflict"}]
    for rec in missing:
        alerts.append(
            f"{rec['source']}:{rec['event_type']} field {rec['field']} marked {rec['status']} ({rec.get('notes','no notes')})."
        )
    return alerts


def generate_inventory(args: argparse.Namespace) -> int:
    records: List[FieldRecord] = []

    # Cryptofeed dataclasses
    records.extend(_collect_cryptofeed_records())

    # Optional tardis-node schemas
    if args.tardis:
        tardis_path = Path(args.tardis)
        if tardis_path.is_file():
            records.extend(_load_json_schema_fields(tardis_path))
        elif tardis_path.is_dir():
            records.extend(_load_directory_json_schemas(tardis_path))
        else:
            records.append(
                FieldRecord(
                    source="tardis-node",
                    event_type="*",
                    field="*",
                    data_type="unknown",
                    status="missing",
                    notes=f"Path not found: {tardis_path}",
                )
            )
    else:
        records.append(
            FieldRecord(
                source="tardis-node",
                event_type="*",
                field="*",
                data_type="unknown",
                status="missing",
                notes="No tardis-node schema path provided.",
            )
        )

    # Optional DBN layouts
    if args.dbn:
        dbn_path = Path(args.dbn)
        if dbn_path.is_dir():
            records.extend(_load_dbn_layout(dbn_path))
        else:
            records.append(
                FieldRecord(
                    source="dbn",
                    event_type="*",
                    field="*",
                    data_type="unknown",
                    status="missing",
                    notes=f"DBN layout directory not found: {dbn_path}",
                )
            )
    else:
        records.append(
            FieldRecord(
                source="dbn",
                event_type="*",
                field="*",
                data_type="unknown",
                status="missing",
                notes="No DBN layout directory provided.",
            )
        )

    output_dir = Path(args.output or INVENTORY_DIR_DEFAULT)
    data = _write_outputs(records, output_dir)

    alerts = _freshness_alerts(data, args.threshold)
    if alerts:
        print("Inventory alerts:")
        for alert in alerts:
            print(f" - {alert}")
    else:
        print("Inventory generated with no alerts.")

    print(f"Outputs written to {output_dir}")
    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate schema inventory matrix")
    parser.add_argument(
        "--tardis",
        help="Path to tardis-node JSON schema file or directory (optional)",
    )
    parser.add_argument(
        "--dbn",
        help="Path to DBN layout directory containing YAML files (optional)",
    )
    parser.add_argument(
        "--output",
        help=f"Output directory (default: {INVENTORY_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=FRESHNESS_THRESHOLD_DAYS_DEFAULT,
        help="Freshness threshold in days",
    )
    parser.add_argument(
        "--mode",
        default="generate",
        choices=["generate"],
        help="Operation mode",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "generate":
        return generate_inventory(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
