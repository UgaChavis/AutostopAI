from __future__ import annotations

import json
from typing import Any

from ..vehicle_profile import (
    normalize_completion_state,
    normalize_vehicle_float,
    normalize_source_confidence,
    normalize_vehicle_field_names,
    normalize_vehicle_field_sources,
    normalize_vehicle_int,
    normalize_vehicle_links,
    normalize_vehicle_notes,
    normalize_vehicle_text,
    soft_normalize_vin,
)

BRIDGE_PURPOSE = "card_enrichment"
BRIDGE_ALLOWED_TOP_LEVEL_PATCH_KEYS = ("description", "vehicle", "vehicle_profile")
BRIDGE_ALLOWED_VEHICLE_PROFILE_KEYS = (
    "vin",
    "make_display",
    "model_display",
    "production_year",
    "engine_model",
    "gearbox_model",
    "drivetrain",
    "source_summary",
    "source_confidence",
    "source_links_or_refs",
    "autofilled_fields",
    "field_sources",
    "data_completion_state",
    "oem_notes",
    "generation_or_platform",
    "fuel_type",
    "engine_displacement_l",
    "engine_power_hp",
    "gearbox_type",
    "raw_input_text",
    "raw_image_text",
    "image_parse_status",
    "warnings",
)
BRIDGE_ALLOWED_STATUS_VALUES = ("queued", "running", "needs_review", "completed", "failed")


def build_card_enrichment_task(
    task_id: str,
    card_id: str,
    *,
    trigger: str = "button",
    requested_by: str = "crm_ui",
    task_text: str = "",
    card_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": str(task_id or "").strip(),
        "card_id": str(card_id or "").strip(),
        "purpose": BRIDGE_PURPOSE,
        "trigger": str(trigger or "button").strip() or "button",
        "requested_by": str(requested_by or "crm_ui").strip() or "crm_ui",
    }
    text = str(task_text or "").strip()
    if text:
        payload["task_text"] = text
    if isinstance(card_context, dict) and card_context:
        payload["card_context"] = dict(card_context)
    return payload


def build_card_enrichment_response(
    task_id: str,
    card_id: str,
    *,
    status: str,
    summary: str,
    patch: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    sources: list[str] | None = None,
    needs_review: bool = False,
) -> dict[str, Any]:
    normalized_status = str(status or "").strip().lower()
    normalized_patch = normalize_card_enrichment_patch(patch or {})
    if normalized_status not in BRIDGE_ALLOWED_STATUS_VALUES:
        normalized_status = "completed" if normalized_patch else "needs_review"
    elif normalized_status in {"needs_review"}:
        normalized_status = "completed" if normalized_patch and not needs_review else "needs_review"
    return {
        "task_id": str(task_id or "").strip(),
        "card_id": str(card_id or "").strip(),
        "status": normalized_status,
        "summary": str(summary or "").strip(),
        "patch": normalized_patch,
        "warnings": [str(item).strip() for item in (warnings or []) if str(item or "").strip()],
        "sources": [str(item).strip() for item in (sources or []) if str(item or "").strip()],
        "needs_review": bool(needs_review or (normalized_status == "needs_review" and not normalized_patch)),
    }


def normalize_card_enrichment_patch(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    patch: dict[str, Any] = {}
    description = normalize_vehicle_notes(payload.get("description"))
    if description:
        patch["description"] = description
    vehicle = normalize_vehicle_text(payload.get("vehicle"), limit=120)
    if vehicle:
        patch["vehicle"] = vehicle
    vehicle_profile = payload.get("vehicle_profile") if isinstance(payload.get("vehicle_profile"), dict) else {}
    normalized_profile = _normalize_vehicle_profile_patch(vehicle_profile)
    if normalized_profile:
        patch["vehicle_profile"] = normalized_profile
    return patch


def _normalize_vehicle_profile_patch(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    normalized: dict[str, Any] = {}
    vin = soft_normalize_vin(payload.get("vin"))
    if vin:
        normalized["vin"] = vin
    make_display = normalize_vehicle_text(payload.get("make_display"), limit=80)
    if make_display:
        normalized["make_display"] = make_display
    model_display = normalize_vehicle_text(payload.get("model_display"), limit=80)
    if model_display:
        normalized["model_display"] = model_display
    production_year = payload.get("production_year")
    if production_year not in (None, ""):
        year_text = str(production_year).strip()
        if year_text.isdigit():
            normalized["production_year"] = int(year_text)
        else:
            normalized["production_year"] = year_text[:8]
    engine_model = normalize_vehicle_text(payload.get("engine_model"), limit=120)
    if engine_model:
        normalized["engine_model"] = engine_model
    gearbox_model = normalize_vehicle_text(payload.get("gearbox_model"), limit=120)
    if gearbox_model:
        normalized["gearbox_model"] = gearbox_model
    drivetrain = normalize_vehicle_text(payload.get("drivetrain"), limit=80)
    if drivetrain:
        normalized["drivetrain"] = drivetrain
    source_summary = normalize_vehicle_text(payload.get("source_summary"), limit=500)
    if source_summary:
        normalized["source_summary"] = source_summary
    source_confidence = normalize_source_confidence(payload.get("source_confidence"))
    if source_confidence > 0:
        normalized["source_confidence"] = source_confidence
    source_links = normalize_vehicle_links(payload.get("source_links_or_refs"))
    if source_links:
        normalized["source_links_or_refs"] = source_links
    autofilled_fields = normalize_vehicle_field_names(payload.get("autofilled_fields"))
    if autofilled_fields:
        normalized["autofilled_fields"] = autofilled_fields
    field_sources = normalize_vehicle_field_sources(payload.get("field_sources"))
    if field_sources:
        normalized["field_sources"] = field_sources
    if "data_completion_state" in payload:
        data_completion_state = normalize_completion_state(payload.get("data_completion_state"))
        if data_completion_state:
            normalized["data_completion_state"] = data_completion_state
    generation_or_platform = normalize_vehicle_text(payload.get("generation_or_platform"), limit=120)
    if generation_or_platform:
        normalized["generation_or_platform"] = generation_or_platform
    fuel_type = normalize_vehicle_text(payload.get("fuel_type"), limit=80)
    if fuel_type:
        normalized["fuel_type"] = fuel_type
    engine_displacement_l = normalize_vehicle_float(payload.get("engine_displacement_l"))
    if engine_displacement_l is not None:
        normalized["engine_displacement_l"] = engine_displacement_l
    engine_power_hp = normalize_vehicle_int(payload.get("engine_power_hp"))
    if engine_power_hp is not None:
        normalized["engine_power_hp"] = engine_power_hp
    gearbox_type = normalize_vehicle_text(payload.get("gearbox_type"), limit=120)
    if gearbox_type:
        normalized["gearbox_type"] = gearbox_type
    raw_input_text = payload.get("raw_input_text")
    if raw_input_text:
        if isinstance(raw_input_text, str):
            normalized["raw_input_text"] = normalize_vehicle_notes(raw_input_text, limit=6000)
        else:
            normalized["raw_input_text"] = normalize_vehicle_notes(
                json.dumps(raw_input_text, ensure_ascii=False, sort_keys=True, default=str),
                limit=6000,
            )
    raw_image_text = normalize_vehicle_notes(payload.get("raw_image_text"), limit=6000)
    if raw_image_text:
        normalized["raw_image_text"] = raw_image_text
    image_parse_status = normalize_vehicle_text(payload.get("image_parse_status"), limit=40)
    if image_parse_status:
        normalized["image_parse_status"] = image_parse_status
    warnings_payload = payload.get("warnings")
    if isinstance(warnings_payload, list):
        warnings: list[str] = []
        for warning in warnings_payload:
            warning_text = normalize_vehicle_notes(warning, limit=400)
            if warning_text and warning_text not in warnings:
                warnings.append(warning_text)
        if warnings:
            normalized["warnings"] = warnings
    elif isinstance(warnings_payload, str):
        warning_text = normalize_vehicle_notes(warnings_payload, limit=1200)
        if warning_text:
            normalized["warnings"] = [warning_text]
    oem_notes = normalize_vehicle_notes(payload.get("oem_notes"))
    if oem_notes:
        normalized["oem_notes"] = oem_notes
    return normalized
