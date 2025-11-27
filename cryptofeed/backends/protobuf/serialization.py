"""Protobuf serialization helpers and converter registry."""

import logging

from typing import Any, Callable, Dict, Tuple

from google.protobuf.message import Message

from cryptofeed.exceptions import ProtobufEncodeError, SerializationError

from .bindings import (
    trade_pb2,
    ticker_pb2,
    candle_pb2,
    funding_pb2,
    order_book_pb2,
    liquidation_pb2,
    open_interest_pb2,
    index_price_pb2,
    balance_pb2,
    position_pb2,
    fill_pb2,
    order_info_pb2,
    order_pb2,
    transaction_pb2,
    REQUIRED_MODULES,
)

from . import converters as _converters
from .validation import SchemaValidator

logger = logging.getLogger(__name__)

PROTO_MODULES: Dict[str, Any] = {
    "trade_pb2": trade_pb2,
    "ticker_pb2": ticker_pb2,
    "candle_pb2": candle_pb2,
    "funding_pb2": funding_pb2,
    "order_book_pb2": order_book_pb2,
    "liquidation_pb2": liquidation_pb2,
    "open_interest_pb2": open_interest_pb2,
    "index_price_pb2": index_price_pb2,
    "balance_pb2": balance_pb2,
    "position_pb2": position_pb2,
    "fill_pb2": fill_pb2,
    "order_info_pb2": order_info_pb2,
    "order_pb2": order_pb2,
    "transaction_pb2": transaction_pb2,
}

TYPE_NAME_OVERRIDES = {
    "Orderbook": "OrderBook",
    "Openinterest": "OpenInterest",
    "Orderinfo": "OrderInfo",
    "Fundingrate": "Funding",
}

SCHEMA_OVERRIDES = {
    "OrderBook": ("order_book_pb2", "Level2Book"),
    "Index": ("index_price_pb2", "IndexPrice"),
}

OPTIONAL_SCHEMAS = {
    "TradeSide",
    "PriceLevel",
    "Level2Delta",
    "Nbbo",
    "TopOfBook",
    "Events",
}


def _canonical_type_name(slug: str) -> str:
    parts = [segment for segment in slug.split("_") if segment]
    candidate = "".join(part.capitalize() for part in parts)
    return TYPE_NAME_OVERRIDES.get(candidate, candidate)


def _resolve_schema_class(type_name: str):
    if type_name in SCHEMA_OVERRIDES:
        module_name, attr_name = SCHEMA_OVERRIDES[type_name]
        module = PROTO_MODULES[module_name]
        return getattr(module, attr_name)
    for module in PROTO_MODULES.values():
        candidate = getattr(module, type_name, None)
        if candidate is not None:
            return candidate
    raise KeyError(f"No protobuf schema found for data type '{type_name}'")


def _type_from_message(message_name: str) -> str:
    overrides = {
        "Level2Book": "OrderBook",
        "IndexPrice": "Index",
        "FundingRate": "Funding",
    }
    if message_name in overrides:
        return overrides[message_name]
    return TYPE_NAME_OVERRIDES.get(message_name, message_name)


def _build_converter_registry() -> Tuple[
    Dict[str, Callable[[Any], Message]], Dict[str, Any]
]:
    converters: Dict[str, Callable[[Any], Message]] = {}
    schema_classes: Dict[str, Any] = {}
    for name, value in vars(_converters).items():
        if not name.endswith("_to_proto"):
            continue
        if not callable(value):
            continue
        slug = name[: -len("_to_proto")]
        type_name = _canonical_type_name(slug)
        try:
            schema_class = _resolve_schema_class(type_name)
        except KeyError:
            logger.debug("Skipping converter '%s' with unresolved schema", name)
            continue
        converters[type_name] = value
        schema_classes[type_name] = schema_class

    _validate_registry(converters)
    return converters, schema_classes


def _validate_registry(converters: Dict[str, Callable[[Any], Message]]) -> None:
    missing = []
    for module_name, message_name in REQUIRED_MODULES.items():
        if message_name in OPTIONAL_SCHEMAS:
            continue
        expected_type = _type_from_message(message_name)
        if expected_type not in converters:
            missing.append(expected_type)
    if missing:
        raise SerializationError(
            "Missing protobuf converters for: " + ", ".join(sorted(set(missing))),
            data_type=",".join(sorted(set(missing))),
        )


_CONVERTER_MAP, _SCHEMA_CLASS_MAP = _build_converter_registry()

_DEFAULT_SCHEMA_VERSION = "v0.1.0"
_SCHEMA_VALIDATOR = SchemaValidator(_DEFAULT_SCHEMA_VERSION)


def _resolve_schema_name(schema_message: Message | None, type_name: str) -> str | None:
    """Return protobuf schema identifier for diagnostics."""

    if schema_message is not None and hasattr(schema_message, "DESCRIPTOR"):
        descriptor = schema_message.DESCRIPTOR
        if descriptor is not None:
            return descriptor.full_name

    schema_class = _SCHEMA_CLASS_MAP.get(type_name)
    if schema_class is not None and hasattr(schema_class, "DESCRIPTOR"):
        descriptor = schema_class.DESCRIPTOR
        if descriptor is not None:
            return descriptor.full_name

    return f"{type_name.lower()}_pb2.{type_name}"


def _ensure_message(instance, type_name: str, context: str) -> Message:
    """Validate converter/to_proto output is a protobuf Message instance."""

    if isinstance(instance, Message):
        return instance

    if hasattr(instance, "SerializeToString") and callable(
        getattr(instance, "SerializeToString")
    ):
        return instance

    raise ProtobufEncodeError(
        f"{context} returned non-protobuf instance; expected protobuf Message",
        data_type=type_name,
        schema_name=_resolve_schema_name(None, type_name),
        schema_version=_DEFAULT_SCHEMA_VERSION,
    )


def get_converter(type_name: str):
    """
    Get the protobuf converter function for a data type.

    Args:
        type_name: The data type class name as string (e.g., 'Trade', 'Ticker')

    Returns:
        Converter function or None if not found

    Example:
        >>> converter = get_converter('Trade')
        >>> proto_msg = converter(trade_obj)
    """
    return _CONVERTER_MAP.get(type_name)


def serialize_to_protobuf(obj):
    """
    Serialize any cryptofeed data object to protobuf.

    Args:
        obj: Any cryptofeed data object (Trade, Ticker, etc.)

    Returns:
        Serialized protobuf message (bytes)

    Raises:
        SerializationError: If no converter exists for the object's type
        ProtobufEncodeError: If conversion or serialization fails
    """
    type_name = type(obj).__name__

    # First, check if the object exposes a to_proto() method (test doubles)
    if hasattr(obj, "to_proto") and callable(getattr(obj, "to_proto")):
        try:
            proto_msg = obj.to_proto()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ProtobufEncodeError(
                "to_proto() raised an exception",
                data_type=type_name,
                schema_version=_DEFAULT_SCHEMA_VERSION,
            ) from exc

        proto_msg = _ensure_message(proto_msg, type_name, "to_proto()")
        _SCHEMA_VALIDATOR.validate(proto_msg, schema_version=_DEFAULT_SCHEMA_VERSION)

        try:
            return proto_msg.SerializeToString()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ProtobufEncodeError(
                "SerializeToString() failed",
                data_type=type_name,
                schema_name=_resolve_schema_name(proto_msg, type_name),
                schema_version=_DEFAULT_SCHEMA_VERSION,
            ) from exc

    # Otherwise, use the converter lookup
    converter = get_converter(type_name)

    if not converter:
        raise SerializationError(
            "No protobuf converter registered for data type.",
            data_type=type_name,
        )

    try:
        proto_msg = converter(obj)
    except Exception as exc:
        raise ProtobufEncodeError(
            "Converter raised an exception",
            data_type=type_name,
            schema_name=_resolve_schema_name(None, type_name),
            schema_version=_DEFAULT_SCHEMA_VERSION,
        ) from exc

    proto_msg = _ensure_message(proto_msg, type_name, "converter")
    _SCHEMA_VALIDATOR.validate(proto_msg, schema_version=_DEFAULT_SCHEMA_VERSION)

    try:
        return proto_msg.SerializeToString()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ProtobufEncodeError(
            "SerializeToString() failed",
            data_type=type_name,
            schema_name=_resolve_schema_name(proto_msg, type_name),
            schema_version=_DEFAULT_SCHEMA_VERSION,
        ) from exc


__all__ = [
    "get_converter",
    "serialize_to_protobuf",
]
