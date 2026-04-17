from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from pathlib import PurePath
import re
from typing import Any, Collection, Literal
import uuid

from .repair_order import RepairOrder
from .vehicle_profile import VehicleProfile, build_vehicle_display


ColumnId = str
Indicator = Literal["green", "yellow", "red"]
TagColor = Literal["green", "yellow", "red"]
Status = Literal["ok", "warning", "critical", "expired"]
AuditSource = Literal["ui", "api", "mcp", "system"]
CashDirection = Literal["income", "expense"]

DEFAULT_COLUMN_IDS: tuple[str, ...] = ("inbox", "in_progress", "control", "done")
VALID_INDICATORS: tuple[Indicator, ...] = ("green", "yellow", "red")
VALID_TAG_COLORS: tuple[TagColor, ...] = ("green", "yellow", "red")
VALID_STATUSES: tuple[Status, ...] = ("ok", "warning", "critical", "expired")
VALID_AUDIT_SOURCES: tuple[AuditSource, ...] = ("ui", "api", "mcp", "system")
VALID_CASH_DIRECTIONS: tuple[CashDirection, ...] = ("income", "expense")
DEFAULT_INDICATOR: Indicator = "green"
DEFAULT_TAG_COLOR: TagColor = "green"
DEFAULT_DEADLINE_TOTAL_SECONDS = 24 * 3600
WARNING_THRESHOLD_RATIO = 0.6
CRITICAL_THRESHOLD_RATIO = 0.15
BLINK_THRESHOLD_RATIO = 0.05
DEADLINE_HEAT_BUCKET_STEP_PERCENT = 5
DEADLINE_HEAT_BUCKET_COUNT = 100 // DEADLINE_HEAT_BUCKET_STEP_PERCENT
DEADLINE_HEAT_START_RGB = (83, 191, 122)
DEADLINE_HEAT_END_RGB = (212, 98, 98)
COLUMN_LABEL_LIMIT = 40
CARD_TITLE_LIMIT = 120
CARD_VEHICLE_LIMIT = 60
CARD_DESCRIPTION_LIMIT = 20000
TAG_LIMIT = 3
TAG_LABEL_LIMIT = 24
STICKY_TEXT_LIMIT = 1000
STICKY_DEFAULT_TOTAL_SECONDS = 24 * 3600
ACTOR_NAME_LIMIT = 40
ATTACHMENT_FILE_NAME_LIMIT = 240
CASHBOX_NAME_LIMIT = 80
CASHTRANSACTION_NOTE_LIMIT = 240
ARCHIVE_PREVIEW_LIMIT = 30
ARCHIVED_CARD_RETENTION_LIMIT = 300
AUDIT_EVENT_RETENTION_DAYS = 60
AUDIT_EVENT_RETENTION_LIMIT = 5000
REPAIR_ORDER_FILE_RETENTION_LIMIT = 300
MAX_ATTACHMENT_SIZE_BYTES = 15 * 1024 * 1024
CARD_AI_AUTOFILL_LOG_LIMIT = 24
_COLUMN_ID_PATTERN = re.compile(r"[^a-z0-9_]+")
_SPACES_PATTERN = re.compile(r"\s+")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def normalize_bool(value, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def normalize_int(value, *, default: int = 0, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def normalize_indicator(value) -> Indicator:
    indicator = str(value or "").strip().lower()
    if indicator not in VALID_INDICATORS:
        return DEFAULT_INDICATOR
    return indicator


def normalize_source(value, *, default: AuditSource = "api") -> AuditSource:
    source = str(value or "").strip().lower()
    if source not in VALID_AUDIT_SOURCES:
        return default
    return source  # type: ignore[return-value]


def normalize_cash_direction(value, *, default: CashDirection = "income") -> CashDirection:
    direction = str(value or "").strip().lower()
    if direction not in VALID_CASH_DIRECTIONS:
        return default
    return direction  # type: ignore[return-value]


def normalize_column(value, *, valid_columns: Collection[str] | None = None, default: str = "inbox") -> ColumnId:
    column = str(value or "").strip().lower()
    if not column:
        return default
    if valid_columns is not None and column not in set(valid_columns):
        return default
    return column


def normalize_column_id(value) -> str:
    column_id = str(value or "").strip().lower()
    column_id = _COLUMN_ID_PATTERN.sub("_", column_id)
    column_id = re.sub(r"_+", "_", column_id).strip("_")
    return column_id[:64]


def normalize_text(value, *, default: str = "", limit: int | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    if limit is not None:
        text = text[:limit]
    return text


def normalize_ai_autofill_log(value) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        level = normalize_text(entry.get("level"), default="INFO", limit=8).upper()
        if level not in {"INFO", "RUN", "WAIT", "DONE", "WARN"}:
            level = "INFO"
        message = normalize_text(entry.get("message"), default="", limit=240)
        timestamp = normalize_text(entry.get("timestamp"), default="", limit=64)
        task_id = normalize_text(entry.get("task_id"), default="", limit=64)
        if not message:
            continue
        items.append(
            {
                "level": level,
                "message": message,
                "timestamp": timestamp,
                "task_id": task_id,
            }
        )
    if len(items) > CARD_AI_AUTOFILL_LOG_LIMIT:
        items = items[-CARD_AI_AUTOFILL_LOG_LIMIT:]
    return items


def normalize_money_minor(value, *, default: int = 0, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        parsed = default
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(round(value * 100))
    else:
        text = str(value or "").strip().replace(" ", "").replace(",", ".")
        if not text:
            parsed = default
        else:
            try:
                parsed = int(round(float(text) * 100))
            except (TypeError, ValueError):
                parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def format_money_minor(value: int) -> str:
    normalized = int(value)
    sign = "-" if normalized < 0 else ""
    amount = abs(normalized) / 100
    return sign + f"{amount:,.2f}".replace(",", " ").replace(".", ",") + " ₽"


def split_legacy_card_title(value) -> tuple[str, str]:
    title = normalize_text(value, default="Без названия", limit=CARD_TITLE_LIMIT)
    if " / " not in title:
        return "", title
    vehicle_part, task_part = title.split(" / ", 1)
    vehicle = normalize_text(vehicle_part, default="", limit=CARD_VEHICLE_LIMIT)
    task_title = normalize_text(task_part, default="", limit=CARD_TITLE_LIMIT)
    if vehicle and task_title:
        return vehicle, task_title
    return "", title


def normalize_actor_name(value, *, default: str = "СИСТЕМА") -> str:
    actor = normalize_text(value, default=default, limit=ACTOR_NAME_LIMIT)
    return _SPACES_PATTERN.sub(" ", actor)


def normalize_tag_label(value) -> str:
    normalized = normalize_text(value, limit=TAG_LABEL_LIMIT)
    normalized = normalized.replace(",", " ")
    normalized = _SPACES_PATTERN.sub(" ", normalized)
    return normalized.upper()


def normalize_tag_color(value) -> TagColor:
    color = str(value or "").strip().lower()
    if color not in VALID_TAG_COLORS:
        return DEFAULT_TAG_COLOR
    return color  # type: ignore[return-value]


def normalize_seen_by_users(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for raw_actor, raw_seen_at in value.items():
        actor_name = normalize_actor_name(raw_actor, default="")
        seen_at = parse_datetime(raw_seen_at)
        if not actor_name or seen_at is None:
            continue
        normalized[actor_name] = seen_at.isoformat()
    return normalized


@dataclass(slots=True)
class CardTag:
    label: str
    color: TagColor = DEFAULT_TAG_COLOR

    def __post_init__(self) -> None:
        self.label = normalize_tag_label(self.label)
        self.color = normalize_tag_color(self.color)

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "color": self.color}

    @classmethod
    def from_value(cls, value) -> "CardTag | None":
        if isinstance(value, CardTag):
            return cls(label=value.label, color=value.color)
        if isinstance(value, dict):
            label = normalize_tag_label(value.get("label") or value.get("name"))
            if not label:
                return None
            return cls(label=label, color=normalize_tag_color(value.get("color")))
        label = normalize_tag_label(value)
        if not label:
            return None
        return cls(label=label, color=DEFAULT_TAG_COLOR)


def normalize_tags(value) -> list[CardTag]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    tags_by_label: dict[str, CardTag] = {}
    for raw in value:
        tag = CardTag.from_value(raw)
        if tag is None:
            continue
        tags_by_label[tag.label] = tag
        if len(tags_by_label) >= TAG_LIMIT:
            break
    return list(tags_by_label.values())


def normalize_file_name(value) -> str:
    raw_name = normalize_text(value)
    if not raw_name:
        return ""
    safe_name = PurePath(raw_name).name
    safe_name = safe_name.replace("\x00", "")
    safe_name = re.sub(r'[<>:"/\\\\|?*]+', "_", safe_name)
    safe_name = re.sub(r"\s+", " ", safe_name).strip(" .")
    if not safe_name:
        return ""
    if len(safe_name) <= ATTACHMENT_FILE_NAME_LIMIT:
        return safe_name.rstrip(" .")
    suffix = PurePath(safe_name).suffix
    if suffix and len(suffix) < ATTACHMENT_FILE_NAME_LIMIT:
        stem_limit = ATTACHMENT_FILE_NAME_LIMIT - len(suffix)
        stem = safe_name[: -len(suffix)].rstrip(" .")
        stem = stem[:stem_limit].rstrip(" .")
        if stem:
            return f"{stem}{suffix}".rstrip(" .")
    return safe_name[:ATTACHMENT_FILE_NAME_LIMIT].rstrip(" .")


def indicator_from_status(status: Status) -> Indicator:
    if status == "warning":
        return "yellow"
    if status in {"critical", "expired"}:
        return "red"
    return "green"


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _interpolate_channel(start: int, end: int, ratio: float) -> int:
    return max(0, min(255, round(start + ((end - start) * ratio))))


def _interpolate_rgb(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    bounded_ratio = _clamp_ratio(ratio)
    return (
        _interpolate_channel(start[0], end[0], bounded_ratio),
        _interpolate_channel(start[1], end[1], bounded_ratio),
        _interpolate_channel(start[2], end[2], bounded_ratio),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _rgb_to_rgba(rgb: tuple[int, int, int], alpha: float) -> str:
    bounded_alpha = _clamp_ratio(alpha)
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {bounded_alpha:.3f})"


def calculate_deadline_progress_ratio(remaining_seconds: int, total_seconds: int) -> float:
    total = max(1, int(total_seconds))
    remaining = max(0, int(remaining_seconds))
    return _clamp_ratio(1.0 - (remaining / total))


def calculate_deadline_progress_bucket(progress_ratio: float) -> int:
    return max(0, min(DEADLINE_HEAT_BUCKET_COUNT, math.floor(_clamp_ratio(progress_ratio) * DEADLINE_HEAT_BUCKET_COUNT)))


def deadline_heat_rgb_for_bucket(bucket: int) -> tuple[int, int, int]:
    bounded_bucket = max(0, min(DEADLINE_HEAT_BUCKET_COUNT, int(bucket)))
    ratio = bounded_bucket / DEADLINE_HEAT_BUCKET_COUNT
    return _interpolate_rgb(DEADLINE_HEAT_START_RGB, DEADLINE_HEAT_END_RGB, ratio)


def deadline_heat_color_for_bucket(bucket: int) -> str:
    return _rgb_to_hex(deadline_heat_rgb_for_bucket(bucket))


def deadline_heat_border_color_for_bucket(bucket: int) -> str:
    bounded_bucket = max(0, min(DEADLINE_HEAT_BUCKET_COUNT, int(bucket)))
    ratio = bounded_bucket / DEADLINE_HEAT_BUCKET_COUNT
    return _rgb_to_rgba(deadline_heat_rgb_for_bucket(bounded_bucket), 0.34 + (ratio * 0.54))


def deadline_heat_ring_color_for_bucket(bucket: int) -> str:
    bounded_bucket = max(0, min(DEADLINE_HEAT_BUCKET_COUNT, int(bucket)))
    ratio = bounded_bucket / DEADLINE_HEAT_BUCKET_COUNT
    return _rgb_to_rgba(deadline_heat_rgb_for_bucket(bounded_bucket), 0.08 + (ratio * 0.26))


def deadline_heat_glow_color_for_bucket(bucket: int) -> str:
    bounded_bucket = max(0, min(DEADLINE_HEAT_BUCKET_COUNT, int(bucket)))
    ratio = bounded_bucket / DEADLINE_HEAT_BUCKET_COUNT
    return _rgb_to_rgba(deadline_heat_rgb_for_bucket(bounded_bucket), 0.04 + (ratio * 0.24))


def format_remaining_seconds(seconds: int) -> str:
    total_seconds = max(0, int(seconds))
    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{days}д {hours:02d}:{minutes:02d}:{secs:02d}"


def short_entity_id(value, *, prefix: str) -> str:
    token = str(value or "").strip().split("-", 1)[0].upper()
    token = re.sub(r"[^A-Z0-9]+", "", token)[:8]
    if not token:
        token = "UNKNOWN"
    return f"{prefix}-{token}"


def split_seconds_to_days_hours(seconds: int) -> tuple[int, int]:
    total_seconds = max(0, int(seconds))
    if total_seconds == 0:
        return 0, 1
    rounded_hours = max(1, math.ceil(total_seconds / 3600))
    days, hours = divmod(rounded_hours, 24)
    return days, hours


@dataclass(slots=True)
class Column:
    id: str
    label: str
    position: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, payload: dict, *, fallback_position: int = 0) -> "Column":
        if not isinstance(payload, dict):
            raise TypeError("Column payload must be a dictionary.")
        column_id = normalize_column_id(payload.get("id"))
        label = normalize_text(payload.get("label"), limit=COLUMN_LABEL_LIMIT)
        position = normalize_int(payload.get("position"), default=fallback_position, minimum=0)
        if not column_id or not label:
            raise ValueError("Column id and label are required.")
        return cls(id=column_id, label=label, position=position)


@dataclass(slots=True)
class Attachment:
    id: str
    file_name: str
    stored_name: str
    mime_type: str
    size_bytes: int
    created_at: str
    created_by: str
    removed: bool = False
    removed_at: str = ""
    removed_by: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "stored_name": self.stored_name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "removed": self.removed,
            "removed_at": self.removed_at,
            "removed_by": self.removed_by,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Attachment":
        if not isinstance(payload, dict):
            raise TypeError("Attachment payload must be a dictionary.")
        file_name = normalize_file_name(payload.get("file_name"))
        stored_name = normalize_file_name(payload.get("stored_name"))
        created_at = parse_datetime(payload.get("created_at")) or utc_now()
        if not file_name or not stored_name:
            raise ValueError("Attachment file_name and stored_name are required.")
        return cls(
            id=normalize_text(payload.get("id"), default=str(uuid.uuid4()), limit=128),
            file_name=file_name,
            stored_name=stored_name,
            mime_type=normalize_text(payload.get("mime_type"), default="application/octet-stream", limit=100),
            size_bytes=normalize_int(payload.get("size_bytes"), default=0, minimum=0),
            created_at=created_at.isoformat(),
            created_by=normalize_actor_name(payload.get("created_by"), default="СИСТЕМА"),
            removed=normalize_bool(payload.get("removed"), default=False),
            removed_at=(parse_datetime(payload.get("removed_at")) or "").isoformat()
            if parse_datetime(payload.get("removed_at"))
            else "",
            removed_by=normalize_actor_name(payload.get("removed_by"), default=""),
        )


@dataclass(slots=True)
class StickyNote:
    id: str
    text: str
    x: int
    y: int
    created_at: str
    updated_at: str
    deadline_timestamp: str
    deadline_total_seconds: int = STICKY_DEFAULT_TOTAL_SECONDS

    def deadline_datetime(self) -> datetime:
        deadline = parse_datetime(self.deadline_timestamp)
        if deadline is None:
            deadline = utc_now() + timedelta(seconds=max(1, int(self.deadline_total_seconds)))
            self.deadline_timestamp = deadline.isoformat()
        return deadline

    def remaining_seconds(self, reference_time: datetime | None = None) -> int:
        if reference_time is None:
            reference_time = utc_now()
        return max(0, int((self.deadline_datetime() - reference_time).total_seconds()))

    def remaining_ratio(self, reference_time: datetime | None = None) -> float:
        total = max(1, int(self.deadline_total_seconds))
        return self.remaining_seconds(reference_time) / total

    def opacity(self, reference_time: datetime | None = None) -> float:
        ratio = max(0.0, min(1.0, self.remaining_ratio(reference_time)))
        return round(0.5 + (0.4 * ratio), 3)

    def to_dict(self, reference_time: datetime | None = None) -> dict:
        remaining = self.remaining_seconds(reference_time)
        ratio = self.remaining_ratio(reference_time)
        return {
            "id": self.id,
            "short_id": short_entity_id(self.id, prefix="S"),
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline_timestamp": self.deadline_timestamp,
            "deadline_total_seconds": self.deadline_total_seconds,
            "remaining_seconds": remaining,
            "remaining_ratio": ratio,
            "opacity": self.opacity(reference_time),
            "is_expired": remaining <= 0,
        }

    def to_storage_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline_timestamp": self.deadline_timestamp,
            "deadline_total_seconds": self.deadline_total_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "StickyNote":
        if not isinstance(payload, dict):
            raise TypeError("Sticky note payload must be a dictionary.")
        created_at = parse_datetime(payload.get("created_at")) or utc_now()
        updated_at = parse_datetime(payload.get("updated_at")) or created_at
        deadline_total_seconds = normalize_int(
            payload.get("deadline_total_seconds"),
            default=STICKY_DEFAULT_TOTAL_SECONDS,
            minimum=1,
        )
        deadline = parse_datetime(payload.get("deadline_timestamp"))
        if deadline is None:
            deadline = created_at + timedelta(seconds=deadline_total_seconds)
        return cls(
            id=normalize_text(payload.get("id"), default=str(uuid.uuid4()), limit=128),
            text=normalize_text(payload.get("text"), default="ЗАМЕТКА", limit=STICKY_TEXT_LIMIT),
            x=normalize_int(payload.get("x"), default=0, minimum=0),
            y=normalize_int(payload.get("y"), default=0, minimum=0),
            created_at=created_at.isoformat(),
            updated_at=updated_at.isoformat(),
            deadline_timestamp=deadline.isoformat(),
            deadline_total_seconds=deadline_total_seconds,
        )


@dataclass(slots=True)
class AuditEvent:
    id: str
    timestamp: str
    actor_name: str
    source: AuditSource
    action: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    card_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "actor_name": self.actor_name,
            "source": self.source,
            "action": self.action,
            "message": self.message,
            "details": dict(self.details),
            "card_id": self.card_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AuditEvent":
        if not isinstance(payload, dict):
            raise TypeError("Audit event payload must be a dictionary.")
        timestamp = parse_datetime(payload.get("timestamp")) or utc_now()
        details = payload.get("details")
        if not isinstance(details, dict):
            details = {}
        card_id = normalize_text(payload.get("card_id"), default="", limit=128) or None
        return cls(
            id=normalize_text(payload.get("id"), default=str(uuid.uuid4()), limit=128),
            timestamp=timestamp.isoformat(),
            actor_name=normalize_actor_name(payload.get("actor_name")),
            source=normalize_source(payload.get("source"), default="system"),
            action=normalize_text(payload.get("action"), default="unknown", limit=80),
            message=normalize_text(payload.get("message"), default="Событие", limit=300),
            details=details,
            card_id=card_id,
        )


@dataclass(slots=True)
class CashBox:
    id: str
    name: str
    order: int
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "short_id": short_entity_id(self.id, prefix="CB"),
            "name": self.name,
            "order": self.order,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_storage_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "order": self.order,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CashBox":
        if not isinstance(payload, dict):
            raise TypeError("Cash box payload must be a dictionary.")
        created_at = parse_datetime(payload.get("created_at")) or utc_now()
        updated_at = parse_datetime(payload.get("updated_at")) or created_at
        name = normalize_text(payload.get("name"), limit=CASHBOX_NAME_LIMIT)
        if not name:
            raise ValueError("Cash box name is required.")
        order = normalize_int(payload.get("order"), default=0, minimum=0)
        return cls(
            id=normalize_text(payload.get("id"), default=str(uuid.uuid4()), limit=128),
            name=name,
            order=order,
            created_at=created_at.isoformat(),
            updated_at=updated_at.isoformat(),
        )


@dataclass(slots=True)
class CashTransaction:
    id: str
    cashbox_id: str
    direction: CashDirection
    amount_minor: int
    note: str
    created_at: str
    actor_name: str
    source: AuditSource
    employee_id: str = ""
    employee_name: str = ""
    transaction_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "short_id": short_entity_id(self.id, prefix="CT"),
            "cashbox_id": self.cashbox_id,
            "direction": self.direction,
            "amount_minor": self.amount_minor,
            "amount_display": format_money_minor(self.amount_minor),
            "note": self.note,
            "created_at": self.created_at,
            "actor_name": self.actor_name,
            "source": self.source,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "transaction_kind": self.transaction_kind,
        }

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cashbox_id": self.cashbox_id,
            "direction": self.direction,
            "amount_minor": self.amount_minor,
            "note": self.note,
            "created_at": self.created_at,
            "actor_name": self.actor_name,
            "source": self.source,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "transaction_kind": self.transaction_kind,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CashTransaction":
        if not isinstance(payload, dict):
            raise TypeError("Cash transaction payload must be a dictionary.")
        created_at = parse_datetime(payload.get("created_at")) or utc_now()
        cashbox_id = normalize_text(payload.get("cashbox_id"), default="", limit=128)
        if not cashbox_id:
            raise ValueError("Cash transaction cashbox_id is required.")
        return cls(
            id=normalize_text(payload.get("id"), default=str(uuid.uuid4()), limit=128),
            cashbox_id=cashbox_id,
            direction=normalize_cash_direction(payload.get("direction")),
            amount_minor=normalize_money_minor(payload.get("amount_minor"), minimum=1),
            note=normalize_text(payload.get("note"), default="", limit=CASHTRANSACTION_NOTE_LIMIT),
            created_at=created_at.isoformat(),
            actor_name=normalize_actor_name(payload.get("actor_name")),
            source=normalize_source(payload.get("source"), default="api"),
            employee_id=normalize_text(payload.get("employee_id"), default="", limit=64),
            employee_name=normalize_text(payload.get("employee_name"), default="", limit=80),
            transaction_kind=normalize_text(payload.get("transaction_kind"), default="", limit=32),
        )


@dataclass(slots=True)
class Card:
    id: str
    title: str
    description: str
    column: ColumnId
    archived: bool
    created_at: str
    updated_at: str
    deadline_timestamp: str
    deadline_total_seconds: int = DEFAULT_DEADLINE_TOTAL_SECONDS
    position: int = 0
    vehicle: str = ""
    vehicle_profile: VehicleProfile = field(default_factory=VehicleProfile)
    repair_order: RepairOrder = field(default_factory=RepairOrder)
    tags: list[CardTag] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    is_unread: bool = False
    seen_by_users: dict[str, str] = field(default_factory=dict)
    ai_autofill_active: bool = False
    ai_autofill_until: str = ""
    ai_autofill_prompt: str = ""
    ai_next_run_at: str = ""
    last_ai_run_at: str = ""
    ai_run_count: int = 0
    last_card_fingerprint: str = ""
    ai_autofill_log: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tags = normalize_tags(self.tags)
        self.position = normalize_int(self.position, default=0, minimum=0)
        self.seen_by_users = normalize_seen_by_users(self.seen_by_users)
        self.ai_autofill_active = normalize_bool(self.ai_autofill_active, default=False)
        self.ai_autofill_until = normalize_text(self.ai_autofill_until, default="", limit=64)
        self.ai_autofill_prompt = normalize_text(self.ai_autofill_prompt, default="", limit=800)
        self.ai_next_run_at = normalize_text(self.ai_next_run_at, default="", limit=64)
        self.last_ai_run_at = normalize_text(self.last_ai_run_at, default="", limit=64)
        self.ai_run_count = normalize_int(self.ai_run_count, default=0, minimum=0)
        self.last_card_fingerprint = normalize_text(self.last_card_fingerprint, default="", limit=128)
        self.ai_autofill_log = normalize_ai_autofill_log(self.ai_autofill_log)

    def deadline_datetime(self) -> datetime:
        deadline = parse_datetime(self.deadline_timestamp)
        if deadline is None:
            deadline = utc_now() + timedelta(seconds=max(1, int(self.deadline_total_seconds)))
            self.deadline_timestamp = deadline.isoformat()
        return deadline

    def remaining_seconds(self, reference_time: datetime | None = None) -> int:
        if reference_time is None:
            reference_time = utc_now()
        return max(0, int((self.deadline_datetime() - reference_time).total_seconds()))

    def remaining_ratio(self, reference_time: datetime | None = None) -> float:
        total = max(1, int(self.deadline_total_seconds))
        return self.remaining_seconds(reference_time) / total

    def deadline_progress_ratio(self, reference_time: datetime | None = None) -> float:
        return calculate_deadline_progress_ratio(self.remaining_seconds(reference_time), self.deadline_total_seconds)

    def deadline_progress_bucket(self, reference_time: datetime | None = None) -> int:
        return calculate_deadline_progress_bucket(self.deadline_progress_ratio(reference_time))

    def deadline_progress_step_percent(self, reference_time: datetime | None = None) -> int:
        return self.deadline_progress_bucket(reference_time) * DEADLINE_HEAT_BUCKET_STEP_PERCENT

    def deadline_heat_color(self, reference_time: datetime | None = None) -> str:
        return deadline_heat_color_for_bucket(self.deadline_progress_bucket(reference_time))

    def deadline_heat_border_color(self, reference_time: datetime | None = None) -> str:
        return deadline_heat_border_color_for_bucket(self.deadline_progress_bucket(reference_time))

    def deadline_heat_ring_color(self, reference_time: datetime | None = None) -> str:
        return deadline_heat_ring_color_for_bucket(self.deadline_progress_bucket(reference_time))

    def deadline_heat_glow_color(self, reference_time: datetime | None = None) -> str:
        return deadline_heat_glow_color_for_bucket(self.deadline_progress_bucket(reference_time))

    def status(self, reference_time: datetime | None = None) -> Status:
        remaining = self.remaining_seconds(reference_time)
        if remaining <= 0:
            return "expired"
        remaining_ratio = self.remaining_ratio(reference_time)
        if remaining_ratio <= CRITICAL_THRESHOLD_RATIO:
            return "critical"
        if remaining_ratio <= WARNING_THRESHOLD_RATIO:
            return "warning"
        return "ok"

    def indicator(self, reference_time: datetime | None = None) -> Indicator:
        return indicator_from_status(self.status(reference_time))

    def is_blinking(self, reference_time: datetime | None = None) -> bool:
        remaining = self.remaining_seconds(reference_time)
        if remaining <= 0:
            return True
        return self.remaining_ratio(reference_time) <= BLINK_THRESHOLD_RATIO

    def active_attachments(self) -> list[Attachment]:
        return [attachment for attachment in self.attachments if not attachment.removed]

    def vehicle_display(self) -> str:
        vehicle = self.vehicle.strip()
        if vehicle:
            return vehicle
        profile_display = self.vehicle_profile.display_name().strip()
        if profile_display:
            return profile_display[:CARD_VEHICLE_LIMIT]
        return ""

    def heading(self) -> str:
        vehicle = self.vehicle_display()
        title = self.title.strip()
        if vehicle and title:
            return f"{vehicle} / {title}"
        return title or vehicle or "Без названия"

    def tag_labels(self) -> list[str]:
        return [tag.label for tag in self.tags]

    def tag_items(self) -> list[dict[str, str]]:
        return [tag.to_dict() for tag in self.tags]

    def mark_seen(self, actor_name: str | None, *, seen_at: str | None = None) -> bool:
        normalized_actor = normalize_actor_name(actor_name, default="")
        if not normalized_actor:
            return False
        timestamp = parse_datetime(seen_at) or parse_datetime(self.updated_at) or parse_datetime(self.created_at) or utc_now()
        next_seen_at = timestamp.isoformat()
        if self.seen_by_users.get(normalized_actor) == next_seen_at:
            return False
        self.seen_by_users[normalized_actor] = next_seen_at
        return True

    def has_unseen_update_for(self, actor_name: str | None) -> bool:
        normalized_actor = normalize_actor_name(actor_name, default="")
        if not normalized_actor or self.is_unread:
            return False
        seen_at = parse_datetime(self.seen_by_users.get(normalized_actor))
        updated_at = parse_datetime(self.updated_at)
        if seen_at is None or updated_at is None:
            return False
        return seen_at < updated_at

    def to_dict(
        self,
        reference_time: datetime | None = None,
        *,
        events_count: int = 0,
        include_removed_attachments: bool = False,
        viewer_username: str | None = None,
        compact: bool = False,
    ) -> dict:
        remaining = self.remaining_seconds(reference_time)
        status = self.status(reference_time)
        remaining_ratio = self.remaining_ratio(reference_time)
        deadline_progress_ratio = self.deadline_progress_ratio(reference_time)
        deadline_progress_bucket = self.deadline_progress_bucket(reference_time)
        attachments = self.attachments if include_removed_attachments else self.active_attachments()
        vehicle_display = self.vehicle_display()
        normalized_description = _SPACES_PATTERN.sub(" ", str(self.description or "").replace("\r", " ").replace("\n", " ")).strip()
        description_preview = normalized_description[:480].rstrip()
        if len(normalized_description) > len(description_preview):
            description_preview = description_preview.rstrip(" ,.;:-") + "…"
        payload = {
            "id": self.id,
            "short_id": short_entity_id(self.id, prefix="C"),
            "heading": self.heading(),
            "vehicle": vehicle_display,
            "title": self.title,
            "description": self.description,
            "description_preview": description_preview,
            "column": self.column,
            "position": self.position,
            "archived": self.archived,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline_timestamp": self.deadline_timestamp,
            "signal_timestamp": self.deadline_timestamp,
            "remaining_seconds": remaining,
            "remaining_ratio": remaining_ratio,
            "remaining_display": format_remaining_seconds(remaining),
            "deadline_progress_ratio": deadline_progress_ratio,
            "deadline_progress_percent": int(round(deadline_progress_ratio * 100)),
            "deadline_progress_bucket": deadline_progress_bucket,
            "deadline_progress_step_percent": deadline_progress_bucket * DEADLINE_HEAT_BUCKET_STEP_PERCENT,
            "deadline_heat_color": deadline_heat_color_for_bucket(deadline_progress_bucket),
            "deadline_heat_border_color": deadline_heat_border_color_for_bucket(deadline_progress_bucket),
            "deadline_heat_ring_color": deadline_heat_ring_color_for_bucket(deadline_progress_bucket),
            "deadline_heat_glow_color": deadline_heat_glow_color_for_bucket(deadline_progress_bucket),
            "status": status,
            "indicator": indicator_from_status(status),
            "is_blinking": self.is_blinking(reference_time),
            "tags": self.tag_labels(),
            "tag_items": self.tag_items(),
            "attachment_count": len(self.active_attachments()),
            "events_count": max(0, int(events_count)),
            "is_unread": self.is_unread,
            "has_unseen_update": self.has_unseen_update_for(viewer_username),
            "ai_autofill_active": self.ai_autofill_active,
            "ai_autofill_until": self.ai_autofill_until,
            "ai_autofill_prompt": self.ai_autofill_prompt,
            "ai_next_run_at": self.ai_next_run_at,
            "last_ai_run_at": self.last_ai_run_at,
            "ai_run_count": self.ai_run_count,
            "last_card_fingerprint": self.last_card_fingerprint,
        }
        if compact:
            payload["description"] = description_preview
            payload["vehicle_profile_compact"] = self.vehicle_profile.to_compact_dict()
            return payload

        payload.update(
            {
                "vehicle_raw": self.vehicle,
                "vehicle_profile": self.vehicle_profile.to_dict(),
                "vehicle_profile_compact": self.vehicle_profile.to_compact_dict(),
                "repair_order": self.repair_order.to_dict(),
                "attachments": [attachment.to_dict() for attachment in attachments],
                "ai_autofill_log": list(self.ai_autofill_log),
            }
        )
        return payload

    def to_storage_dict(self) -> dict:
        return {
            "id": self.id,
            "vehicle": self.vehicle,
            "vehicle_profile": self.vehicle_profile.to_storage_dict(),
            "repair_order": self.repair_order.to_storage_dict(),
            "title": self.title,
            "description": self.description,
            "column": self.column,
            "position": self.position,
            "archived": self.archived,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline_timestamp": self.deadline_timestamp,
            "deadline_total_seconds": self.deadline_total_seconds,
            "tags": self.tag_items(),
            "attachments": [attachment.to_dict() for attachment in self.attachments],
            "is_unread": self.is_unread,
            "seen_by_users": dict(self.seen_by_users),
            "ai_autofill_active": self.ai_autofill_active,
            "ai_autofill_until": self.ai_autofill_until,
            "ai_autofill_prompt": self.ai_autofill_prompt,
            "ai_next_run_at": self.ai_next_run_at,
            "last_ai_run_at": self.last_ai_run_at,
            "ai_run_count": self.ai_run_count,
            "last_card_fingerprint": self.last_card_fingerprint,
            "ai_autofill_log": list(self.ai_autofill_log),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        *,
        valid_columns: Collection[str] | None = None,
        default_column: str = "inbox",
        fallback_position: int = 0,
    ) -> "Card":
        if not isinstance(payload, dict):
            raise TypeError("Card payload must be a dictionary.")

        created_at = parse_datetime(payload.get("created_at")) or utc_now()
        updated_at = parse_datetime(payload.get("updated_at")) or created_at
        archived = normalize_bool(payload.get("archived"), default=False)
        deadline_total_seconds = normalize_int(
            payload.get("deadline_total_seconds"),
            default=DEFAULT_DEADLINE_TOTAL_SECONDS,
            minimum=1,
        )
        deadline = parse_datetime(payload.get("deadline_timestamp"))
        if deadline is None:
            deadline = created_at + timedelta(seconds=deadline_total_seconds)

        attachments_payload = payload.get("attachments", [])
        attachments: list[Attachment] = []
        if isinstance(attachments_payload, list):
            for item in attachments_payload:
                try:
                    attachments.append(Attachment.from_dict(item))
                except (TypeError, ValueError):
                    continue

        vehicle = normalize_text(payload.get("vehicle"), default="", limit=CARD_VEHICLE_LIMIT)
        vehicle_profile = VehicleProfile.from_dict(payload.get("vehicle_profile"))
        repair_order = RepairOrder.from_dict(payload.get("repair_order"))
        profile_display = build_vehicle_display(
            vehicle_profile.make_display,
            vehicle_profile.model_display,
            vehicle_profile.production_year,
        )
        title = normalize_text(payload.get("title"), default="Без названия", limit=CARD_TITLE_LIMIT)
        if "vehicle" not in payload and not vehicle:
            vehicle, title = split_legacy_card_title(title)
        if not vehicle and profile_display:
            vehicle = normalize_text(profile_display, default="", limit=CARD_VEHICLE_LIMIT)

        card = cls(
            id=normalize_text(payload.get("id"), default=str(uuid.uuid4()), limit=128),
            title=normalize_text(payload.get("title"), default="Без названия", limit=CARD_TITLE_LIMIT),
            description=normalize_text(payload.get("description"), default="", limit=CARD_DESCRIPTION_LIMIT),
            column=normalize_column(payload.get("column"), valid_columns=valid_columns, default=default_column),
            position=normalize_int(payload.get("position"), default=fallback_position, minimum=0),
            archived=archived,
            created_at=created_at.isoformat(),
            updated_at=updated_at.isoformat(),
            deadline_timestamp=deadline.isoformat(),
            deadline_total_seconds=deadline_total_seconds,
            vehicle=vehicle,
            vehicle_profile=vehicle_profile,
            repair_order=repair_order,
            tags=normalize_tags(payload.get("tag_items", payload.get("tags"))),
            attachments=attachments,
            is_unread=normalize_bool(payload.get("is_unread"), default=False),
            seen_by_users=normalize_seen_by_users(payload.get("seen_by_users")),
            ai_autofill_active=normalize_bool(payload.get("ai_autofill_active"), default=False),
            ai_autofill_until=normalize_text(payload.get("ai_autofill_until"), default="", limit=64),
            ai_autofill_prompt=normalize_text(payload.get("ai_autofill_prompt"), default="", limit=800),
            ai_next_run_at=normalize_text(payload.get("ai_next_run_at"), default="", limit=64),
            last_ai_run_at=normalize_text(payload.get("last_ai_run_at"), default="", limit=64),
            ai_run_count=normalize_int(payload.get("ai_run_count"), default=0, minimum=0),
            last_card_fingerprint=normalize_text(payload.get("last_card_fingerprint"), default="", limit=128),
            ai_autofill_log=normalize_ai_autofill_log(payload.get("ai_autofill_log")),
        )
        card.title = title
        return card
