from __future__ import annotations

import hashlib
import json
import re
from typing import Any


_VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?:\+7|8)\s*(?:\(\s*\d{3}\s*\)|\d{3})\s*[\- ]?\s*\d{3}\s*[\- ]?\s*\d{2}\s*[\- ]?\s*\d{2}")
_MILEAGE_PATTERN = re.compile(r"(?:пробег|mileage|одометр)\s*[:\-]?\s*([\d\s]{2,12})", re.IGNORECASE)
_IMPORTANT_WALL_MARKERS = (
    "vin",
    "клиент",
    "телефон",
    "диагност",
    "ошиб",
    "симптом",
    "работ",
    "детал",
    "материал",
    "договор",
    "срок",
    "оплат",
    "заказ-наряд",
)
_NORMALIZATION_NOISE_MARKERS = ("ии:", "ai:", "///", "***", "  ")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt"}
_MAX_WALL_FACTS = 8
_MAX_WALL_CHANGES = 6
_MAX_WALL_NOTES = 6
_MAX_CARD_FACTS = 10
_MAX_REPAIR_ITEMS = 6
_MAX_ATTACHMENTS = 8
_MAX_LINES_PER_FIELD = 6


def _clean_text(value: Any, *, limit: int = 400) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _compact_lines(value: Any, *, limit: int = _MAX_LINES_PER_FIELD, line_limit: int = 160) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in str(value or "").splitlines():
        line = _clean_text(raw_line, limit=line_limit)
        normalized = line.casefold()
        if not line or normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _text_has_normalization_opportunity(text: str) -> bool:
    normalized = str(text or "")
    if not normalized.strip():
        return False
    if any(marker in normalized.lower() for marker in _NORMALIZATION_NOISE_MARKERS):
        return True
    return "\n\n\n" in normalized or len(normalized) > 800


def _coalesce(*values: Any) -> str:
    for value in values:
        text = _clean_text(value, limit=160)
        if text:
            return text
    return ""


def _extract_vehicle_facts(card: dict[str, Any], repair_order: dict[str, Any], description: str) -> dict[str, Any]:
    vehicle_profile = card.get("vehicle_profile") if isinstance(card.get("vehicle_profile"), dict) else {}
    vin = _coalesce(vehicle_profile.get("vin"), repair_order.get("vin"))
    if not vin:
        match = _VIN_PATTERN.search(description.upper())
        vin = match.group(0).upper() if match else ""
    mileage = _coalesce(vehicle_profile.get("mileage"), repair_order.get("mileage"))
    if not mileage:
        match = _MILEAGE_PATTERN.search(description)
        mileage = _clean_text(match.group(1), limit=32) if match else ""
    vehicle_label = _coalesce(card.get("vehicle"), repair_order.get("vehicle"))
    vehicle_block = {
        "vehicle_label": vehicle_label,
        "vin": vin.upper(),
        "mileage": mileage,
        "make": _coalesce(vehicle_profile.get("make_display")),
        "model": _coalesce(vehicle_profile.get("model_display")),
        "year": _coalesce(vehicle_profile.get("production_year")),
        "engine": _coalesce(vehicle_profile.get("engine_model")),
        "gearbox": _coalesce(vehicle_profile.get("gearbox_model")),
        "drivetrain": _coalesce(vehicle_profile.get("drivetrain")),
    }
    return vehicle_block


def build_ai_wall_digest_packet(context_payload: dict[str, Any]) -> dict[str, Any]:
    events = context_payload.get("events") if isinstance(context_payload.get("events"), list) else []
    card = context_payload.get("card") if isinstance(context_payload.get("card"), dict) else {}
    description = str(card.get("description", "") or "")
    description_lines = _compact_lines(description, limit=_MAX_WALL_NOTES, line_limit=200)
    facts: list[str] = []
    recent_changes: list[str] = []
    important_notes: list[str] = []
    for item in events[:18]:
        if not isinstance(item, dict):
            continue
        action = _clean_text(item.get("action"), limit=48)
        message = _clean_text(item.get("message"), limit=180)
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        combined = " ".join(bit for bit in (action, message) if bit).strip()
        if not combined:
            continue
        if len(recent_changes) < _MAX_WALL_CHANGES:
            recent_changes.append(combined)
        if len(facts) < _MAX_WALL_FACTS and any(marker in combined.casefold() for marker in _IMPORTANT_WALL_MARKERS):
            facts.append(combined)
        if len(important_notes) < _MAX_WALL_NOTES:
            detail_note = _clean_text(details.get("vehicle") or details.get("title") or details.get("task_id"), limit=120)
            if detail_note:
                important_notes.append(detail_note)
    for line in description_lines:
        if len(important_notes) >= _MAX_WALL_NOTES:
            break
        if any(marker in line.casefold() for marker in _IMPORTANT_WALL_MARKERS):
            important_notes.append(line)
    deduped_notes: list[str] = []
    seen_notes: set[str] = set()
    for item in important_notes:
        normalized = item.casefold()
        if normalized in seen_notes:
            continue
        seen_notes.add(normalized)
        deduped_notes.append(item)
    return {
        "kind": "wall_digest",
        "facts": facts[:_MAX_WALL_FACTS],
        "recent_changes": recent_changes[:_MAX_WALL_CHANGES],
        "important_notes": deduped_notes[:_MAX_WALL_NOTES],
        "normalization_candidate": _text_has_normalization_opportunity(description),
    }


def build_ai_card_context_packet(context_payload: dict[str, Any], *, wall_digest: dict[str, Any] | None = None) -> dict[str, Any]:
    card = context_payload.get("card") if isinstance(context_payload.get("card"), dict) else {}
    repair_order = card.get("repair_order") if isinstance(card.get("repair_order"), dict) else {}
    description = str(card.get("description", "") or "")
    vehicle_block = _extract_vehicle_facts(card, repair_order, description)
    phone_match = _PHONE_PATTERN.search(description)
    customer = _coalesce(card.get("title"))
    facts: list[dict[str, str]] = []
    if vehicle_block["vin"]:
        facts.append({"kind": "vin", "value": vehicle_block["vin"]})
    if vehicle_block["vehicle_label"]:
        facts.append({"kind": "vehicle", "value": vehicle_block["vehicle_label"]})
    if phone_match:
        facts.append({"kind": "phone", "value": _clean_text(phone_match.group(0), limit=32)})
    if vehicle_block["mileage"]:
        facts.append({"kind": "mileage", "value": vehicle_block["mileage"]})
    for line in _compact_lines(description, limit=_MAX_CARD_FACTS, line_limit=180):
        lowered = line.casefold()
        if any(marker in lowered for marker in ("симптом", "жалоб", "стук", "шум", "течь", "ошиб", "диагност", "работ", "детал", "замет")):
            facts.append({"kind": "note", "value": line})
        if len(facts) >= _MAX_CARD_FACTS:
            break
    missing_key_fields: list[str] = []
    if not vehicle_block["vehicle_label"]:
        missing_key_fields.append("vehicle")
    if not vehicle_block["vin"]:
        missing_key_fields.append("vin")
    if not customer:
        missing_key_fields.append("title")
    has_candidate_facts = any(item["kind"] in {"vin", "vehicle", "phone", "mileage"} for item in facts)
    incomplete_vehicle = bool(vehicle_block["vin"]) and not all(
        [vehicle_block["make"], vehicle_block["model"], vehicle_block["year"]]
    )
    return {
        "kind": "card_context",
        "card_id": str(card.get("id", "") or "").strip(),
        "short_id": _clean_text(card.get("short_id"), limit=32),
        "title": _clean_text(card.get("title"), limit=120),
        "summary_label": _coalesce(card.get("vehicle"), card.get("title")),
        "status": _clean_text(card.get("status"), limit=24),
        "column": _clean_text(card.get("column"), limit=40),
        "vehicle": vehicle_block,
        "customer_hint": customer,
        "key_fields": {
            "vehicle": vehicle_block["vehicle_label"],
            "vin": vehicle_block["vin"],
            "mileage": vehicle_block["mileage"],
            "indicator": _clean_text(card.get("indicator"), limit=16),
        },
        "ai_relevant_facts": facts[:_MAX_CARD_FACTS],
        "missing_key_fields": missing_key_fields,
        "has_candidate_facts": has_candidate_facts,
        "vehicle_profile_incomplete": incomplete_vehicle,
        "normalization_candidate": bool((wall_digest or {}).get("normalization_candidate")) or _text_has_normalization_opportunity(description),
    }


def build_ai_repair_order_context_packet(context_payload: dict[str, Any]) -> dict[str, Any] | None:
    card = context_payload.get("card") if isinstance(context_payload.get("card"), dict) else {}
    repair_order = card.get("repair_order") if isinstance(card.get("repair_order"), dict) else {}
    repair_order_text = context_payload.get("repair_order_text") if isinstance(context_payload.get("repair_order_text"), dict) else {}
    has_payload = bool(repair_order) or bool(repair_order_text)
    if not has_payload:
        return None
    works = repair_order.get("works") if isinstance(repair_order.get("works"), list) else []
    materials = repair_order.get("materials") if isinstance(repair_order.get("materials"), list) else []
    def _rows_preview(items: list[Any]) -> list[dict[str, str]]:
        preview: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = _clean_text(item.get("name"), limit=120)
            if not name:
                continue
            preview.append(
                {
                    "name": name,
                    "quantity": _clean_text(item.get("quantity"), limit=24),
                    "price": _clean_text(item.get("price"), limit=32),
                }
            )
            if len(preview) >= _MAX_REPAIR_ITEMS:
                break
        return preview
    payment_status = _coalesce(repair_order.get("payment_status"), repair_order.get("payment_method"))
    return {
        "kind": "repair_order_context",
        "repair_order_id": _coalesce(repair_order.get("id"), repair_order.get("number")),
        "number": _clean_text(repair_order.get("number"), limit=40),
        "label": _coalesce(repair_order.get("number"), repair_order.get("status"), card.get("title")),
        "status": _clean_text(repair_order.get("status"), limit=24),
        "client": _coalesce(repair_order.get("client"), card.get("title")),
        "vehicle": _coalesce(repair_order.get("vehicle"), card.get("vehicle")),
        "vin": _coalesce(repair_order.get("vin")),
        "works": _rows_preview(works),
        "materials": _rows_preview(materials),
        "notes": _compact_lines(
            "\n".join(
                str(item or "")
                for item in (
                    repair_order.get("reason"),
                    repair_order.get("comment"),
                    repair_order.get("note"),
                    repair_order_text.get("text"),
                )
            ),
            limit=_MAX_LINES_PER_FIELD,
            line_limit=180,
        ),
        "payment_status": payment_status,
    }


def build_ai_attachment_intake_packet(context_payload: dict[str, Any]) -> dict[str, Any]:
    attachments = context_payload.get("attachments") if isinstance(context_payload.get("attachments"), list) else []
    repair_order_packet = build_ai_repair_order_context_packet(context_payload)
    linked_repair_order_id = str((repair_order_packet or {}).get("repair_order_id", "") or "").strip()
    items: list[dict[str, Any]] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        file_name = _clean_text(attachment.get("file_name"), limit=160)
        mime_type = _clean_text(attachment.get("mime_type"), limit=80).lower()
        extension = ""
        if "." in file_name:
            extension = "." + file_name.rsplit(".", 1)[-1].lower()
        ai_ready = extension in _IMAGE_EXTENSIONS or extension in _DOCUMENT_EXTENSIONS or mime_type.startswith("image/")
        content_kind = "image" if extension in _IMAGE_EXTENSIONS or mime_type.startswith("image/") else "document"
        items.append(
            {
                "attachment_id": str(attachment.get("id", "") or "").strip(),
                "file_name": file_name,
                "file_type": extension.lstrip(".") or mime_type or "file",
                "content_kind": content_kind,
                "ai_ready": ai_ready,
                "scope": {
                    "card_id": str(attachment.get("card_id", "") or "").strip() or str((context_payload.get("card") or {}).get("id", "") or "").strip(),
                    "repair_order_id": linked_repair_order_id,
                },
            }
        )
        if len(items) >= _MAX_ATTACHMENTS:
            break
    return {
        "kind": "attachments_intake",
        "items": items,
        "total": len(items),
    }


def build_ai_compact_context_packet(
    context_payload: dict[str, Any],
    *,
    scenario_id: str = "",
    source: str = "backend",
) -> dict[str, Any]:
    wall_digest = build_ai_wall_digest_packet(context_payload)
    card_context = build_ai_card_context_packet(context_payload, wall_digest=wall_digest)
    repair_order_context = build_ai_repair_order_context_packet(context_payload)
    attachments_intake = build_ai_attachment_intake_packet(context_payload)
    packet = {
        "kind": "compact_context",
        "source": source,
        "scenario_id": str(scenario_id or "").strip(),
        "card_context": card_context,
        "repair_order_context": repair_order_context,
        "wall_digest": wall_digest,
        "attachments_intake": attachments_intake,
    }
    packet["fingerprint"] = compact_context_fingerprint(packet)
    return packet


def compact_context_fingerprint(packet: dict[str, Any]) -> str:
    digest = hashlib.sha1(
        json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:16]
