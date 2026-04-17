from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal


VehicleCompletionState = Literal[
    "manually_entered",
    "partially_autofilled",
    "mostly_autofilled",
    "verified",
]

VALID_VEHICLE_COMPLETION_STATES: tuple[VehicleCompletionState, ...] = (
    "manually_entered",
    "partially_autofilled",
    "mostly_autofilled",
    "verified",
)

VEHICLE_PRIMARY_FIELDS: tuple[str, ...] = (
    "make_display",
    "model_display",
    "generation_or_platform",
    "production_year",
    "mileage",
    "customer_phone",
    "customer_name",
    "vin",
    "engine_code",
    "engine_model",
    "engine_displacement_l",
    "engine_power_hp",
    "gearbox_type",
    "gearbox_model",
    "drivetrain",
    "fuel_type",
    "oil_engine_capacity_l",
    "oil_gearbox_capacity_l",
    "coolant_capacity_l",
    "steering_system_type",
    "brake_front_type",
    "brake_rear_type",
    "wheel_bolt_pattern",
    "oem_notes",
    "source_summary",
    "source_confidence",
    "source_links_or_refs",
    "data_completion_state",
)

VEHICLE_COMPACT_FIELDS: tuple[str, ...] = (
    "make_display",
    "model_display",
    "production_year",
    "mileage",
    "vin",
    "engine_model",
    "gearbox_model",
    "drivetrain",
    "oem_notes",
)

VEHICLE_META_FIELDS: tuple[str, ...] = (
    "manual_fields",
    "autofilled_fields",
    "tentative_fields",
    "field_sources",
    "raw_input_text",
    "raw_image_text",
    "image_parse_status",
    "warnings",
)

VEHICLE_ALL_FIELDS: tuple[str, ...] = (*VEHICLE_PRIMARY_FIELDS, *VEHICLE_META_FIELDS)

VEHICLE_TEXT_LIMIT = 120
VEHICLE_NOTE_LIMIT = 1200
VEHICLE_RAW_TEXT_LIMIT = 6000
VEHICLE_SOURCE_LINK_LIMIT = 12
VEHICLE_SOURCE_LINK_TEXT_LIMIT = 240
VIN_SOFT_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{11,17})\b", re.IGNORECASE)
_VIN_ALLOWED_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]+$", re.IGNORECASE)


def normalize_vehicle_text(value, *, limit: int = VEHICLE_TEXT_LIMIT) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit]


def normalize_vehicle_notes(value, *, limit: int = VEHICLE_NOTE_LIMIT) -> str:
    text = str(value or "").strip()
    return text[:limit]


def normalize_vehicle_raw_text(value, *, limit: int = VEHICLE_RAW_TEXT_LIMIT) -> str:
    text = str(value or "").strip()
    return text[:limit]


def normalize_vehicle_int(value) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def normalize_vehicle_float(value) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    raw = str(value).strip().replace(",", ".")
    raw = re.sub(r"[^0-9.]+", "", raw)
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return round(parsed, 2)


def normalize_vehicle_links(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = re.split(r"[\n,;]+", value)
    elif isinstance(value, list):
        candidates = [str(item) for item in value]
    else:
        return []
    links: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        link = str(raw or "").strip()
        if not link:
            continue
        link = link[:VEHICLE_SOURCE_LINK_TEXT_LIMIT]
        if link in seen:
            continue
        seen.add(link)
        links.append(link)
        if len(links) >= VEHICLE_SOURCE_LINK_LIMIT:
            break
    return links


def normalize_vehicle_field_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[\s,;]+", value)
    elif isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        name = str(raw or "").strip()
        if not name or name not in VEHICLE_PRIMARY_FIELDS or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def normalize_vehicle_field_sources(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for field_name, raw_source in value.items():
        if field_name not in VEHICLE_PRIMARY_FIELDS:
            continue
        source = normalize_vehicle_text(raw_source, limit=80)
        if not source:
            continue
        result[field_name] = source
    return result


def normalize_source_confidence(value) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return 0.0
    parsed = max(0.0, min(1.0, parsed))
    return round(parsed, 2)


def normalize_completion_state(value) -> VehicleCompletionState:
    candidate = str(value or "").strip().lower()
    if candidate not in VALID_VEHICLE_COMPLETION_STATES:
        return "manually_entered"
    return candidate  # type: ignore[return-value]


def soft_normalize_vin(value) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value or "").strip()).upper()[:17]


def soft_validate_vin(value) -> tuple[str, bool, list[str]]:
    vin = soft_normalize_vin(value)
    if not vin:
        return "", False, []
    warnings: list[str] = []
    suspicious = False
    if len(vin) != 17:
        suspicious = True
        warnings.append("VIN выглядит неполным или имеет нестандартную длину.")
    if not _VIN_ALLOWED_PATTERN.match(vin):
        suspicious = True
        warnings.append("VIN содержит подозрительные символы.")
    if any(char in vin for char in ("I", "O", "Q")):
        suspicious = True
        warnings.append("VIN содержит символы I, O или Q, что обычно недопустимо.")
    return vin, suspicious, warnings


def build_vehicle_display(make_display: str, model_display: str, production_year: int | None = None) -> str:
    parts = [normalize_vehicle_text(make_display), normalize_vehicle_text(model_display)]
    parts = [part for part in parts if part]
    display = " ".join(parts)
    if production_year and display:
        return f"{display} {production_year}"
    return display


@dataclass(slots=True)
class VehicleProfile:
    make_display: str = ""
    model_display: str = ""
    generation_or_platform: str = ""
    production_year: int | None = None
    mileage: int | None = None
    customer_phone: str = ""
    customer_name: str = ""
    vin: str = ""
    engine_code: str = ""
    engine_model: str = ""
    engine_displacement_l: float | None = None
    engine_power_hp: int | None = None
    gearbox_type: str = ""
    gearbox_model: str = ""
    drivetrain: str = ""
    fuel_type: str = ""
    oil_engine_capacity_l: float | None = None
    oil_gearbox_capacity_l: float | None = None
    coolant_capacity_l: float | None = None
    steering_system_type: str = ""
    brake_front_type: str = ""
    brake_rear_type: str = ""
    wheel_bolt_pattern: str = ""
    oem_notes: str = ""
    source_summary: str = ""
    source_confidence: float = 0.0
    source_links_or_refs: list[str] = field(default_factory=list)
    data_completion_state: VehicleCompletionState = "manually_entered"
    manual_fields: list[str] = field(default_factory=list)
    autofilled_fields: list[str] = field(default_factory=list)
    tentative_fields: list[str] = field(default_factory=list)
    field_sources: dict[str, str] = field(default_factory=dict)
    raw_input_text: str = ""
    raw_image_text: str = ""
    image_parse_status: str = "not_attempted"
    warnings: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            [
                self.make_display,
                self.model_display,
                self.generation_or_platform,
                self.production_year,
                self.mileage,
                self.customer_phone,
                self.customer_name,
                self.vin,
                self.engine_code,
                self.engine_model,
                self.engine_displacement_l,
                self.engine_power_hp,
                self.gearbox_type,
                self.gearbox_model,
                self.drivetrain,
                self.fuel_type,
                self.oil_engine_capacity_l,
                self.oil_gearbox_capacity_l,
                self.coolant_capacity_l,
                self.steering_system_type,
                self.brake_front_type,
                self.brake_rear_type,
                self.wheel_bolt_pattern,
                self.oem_notes,
            ]
        )

    def display_name(self) -> str:
        return build_vehicle_display(self.make_display, self.model_display, self.production_year)

    def to_dict(self) -> dict[str, Any]:
        vin, vin_is_suspect, vin_warnings = soft_validate_vin(self.vin)
        warnings = list(self.warnings)
        for warning in vin_warnings:
            if warning not in warnings:
                warnings.append(warning)
        return {
            "make_display": self.make_display,
            "model_display": self.model_display,
            "generation_or_platform": self.generation_or_platform,
            "production_year": self.production_year,
            "mileage": self.mileage,
            "customer_phone": self.customer_phone,
            "customer_name": self.customer_name,
            "vin": vin,
            "vin_is_suspect": vin_is_suspect,
            "engine_code": self.engine_code,
            "engine_model": self.engine_model,
            "engine_displacement_l": self.engine_displacement_l,
            "engine_power_hp": self.engine_power_hp,
            "gearbox_type": self.gearbox_type,
            "gearbox_model": self.gearbox_model,
            "drivetrain": self.drivetrain,
            "fuel_type": self.fuel_type,
            "oil_engine_capacity_l": self.oil_engine_capacity_l,
            "oil_gearbox_capacity_l": self.oil_gearbox_capacity_l,
            "coolant_capacity_l": self.coolant_capacity_l,
            "steering_system_type": self.steering_system_type,
            "brake_front_type": self.brake_front_type,
            "brake_rear_type": self.brake_rear_type,
            "wheel_bolt_pattern": self.wheel_bolt_pattern,
            "oem_notes": self.oem_notes,
            "source_summary": self.source_summary,
            "source_confidence": self.source_confidence,
            "source_links_or_refs": list(self.source_links_or_refs),
            "data_completion_state": self.data_completion_state,
            "manual_fields": list(self.manual_fields),
            "autofilled_fields": list(self.autofilled_fields),
            "tentative_fields": list(self.tentative_fields),
            "field_sources": dict(self.field_sources),
            "raw_input_text": self.raw_input_text,
            "raw_image_text": self.raw_image_text,
            "image_parse_status": self.image_parse_status,
            "warnings": warnings,
            "display_name": self.display_name(),
            "has_any_data": not self.is_empty(),
        }

    def to_compact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        compact = {
            field_name: payload.get(field_name)
            for field_name in VEHICLE_COMPACT_FIELDS
        }
        compact.update(
            {
                "display_name": payload.get("display_name"),
                "has_any_data": payload.get("has_any_data"),
                "source_summary": payload.get("source_summary"),
                "source_confidence": payload.get("source_confidence"),
                "data_completion_state": payload.get("data_completion_state"),
                "manual_fields": list(payload.get("manual_fields") or []),
                "autofilled_fields": list(payload.get("autofilled_fields") or []),
                "tentative_fields": list(payload.get("tentative_fields") or []),
                "warnings": list(payload.get("warnings") or []),
            }
        )
        return compact

    def to_storage_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("vin_is_suspect", None)
        payload.pop("display_name", None)
        payload.pop("has_any_data", None)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "VehicleProfile":
        if not isinstance(payload, dict):
            return cls()
        vin, _, vin_warnings = soft_validate_vin(payload.get("vin"))
        warnings: list[str] = []
        for raw_warning in payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []:
            warning = normalize_vehicle_text(raw_warning, limit=200)
            if warning and warning not in warnings:
                warnings.append(warning)
        for warning in vin_warnings:
            if warning not in warnings:
                warnings.append(warning)
        return cls(
            make_display=normalize_vehicle_text(payload.get("make_display")),
            model_display=normalize_vehicle_text(payload.get("model_display")),
            generation_or_platform=normalize_vehicle_text(payload.get("generation_or_platform")),
            production_year=normalize_vehicle_int(payload.get("production_year")),
            mileage=normalize_vehicle_int(payload.get("mileage")),
            customer_phone=normalize_vehicle_text(payload.get("customer_phone")),
            customer_name=normalize_vehicle_text(payload.get("customer_name")),
            vin=vin,
            engine_code=normalize_vehicle_text(payload.get("engine_code")),
            engine_model=normalize_vehicle_text(payload.get("engine_model")),
            engine_displacement_l=normalize_vehicle_float(payload.get("engine_displacement_l")),
            engine_power_hp=normalize_vehicle_int(payload.get("engine_power_hp")),
            gearbox_type=normalize_vehicle_text(payload.get("gearbox_type")),
            gearbox_model=normalize_vehicle_text(payload.get("gearbox_model")),
            drivetrain=normalize_vehicle_text(payload.get("drivetrain")),
            fuel_type=normalize_vehicle_text(payload.get("fuel_type")),
            oil_engine_capacity_l=normalize_vehicle_float(payload.get("oil_engine_capacity_l")),
            oil_gearbox_capacity_l=normalize_vehicle_float(payload.get("oil_gearbox_capacity_l")),
            coolant_capacity_l=normalize_vehicle_float(payload.get("coolant_capacity_l")),
            steering_system_type=normalize_vehicle_text(payload.get("steering_system_type")),
            brake_front_type=normalize_vehicle_text(payload.get("brake_front_type")),
            brake_rear_type=normalize_vehicle_text(payload.get("brake_rear_type")),
            wheel_bolt_pattern=normalize_vehicle_text(payload.get("wheel_bolt_pattern")),
            oem_notes=normalize_vehicle_notes(payload.get("oem_notes")),
            source_summary=normalize_vehicle_notes(payload.get("source_summary"), limit=400),
            source_confidence=normalize_source_confidence(payload.get("source_confidence")),
            source_links_or_refs=normalize_vehicle_links(payload.get("source_links_or_refs")),
            data_completion_state=normalize_completion_state(payload.get("data_completion_state")),
            manual_fields=normalize_vehicle_field_names(payload.get("manual_fields")),
            autofilled_fields=normalize_vehicle_field_names(payload.get("autofilled_fields")),
            tentative_fields=normalize_vehicle_field_names(payload.get("tentative_fields")),
            field_sources=normalize_vehicle_field_sources(payload.get("field_sources")),
            raw_input_text=normalize_vehicle_raw_text(payload.get("raw_input_text")),
            raw_image_text=normalize_vehicle_raw_text(payload.get("raw_image_text")),
            image_parse_status=normalize_vehicle_text(payload.get("image_parse_status"), limit=40) or "not_attempted",
            warnings=warnings,
        )
