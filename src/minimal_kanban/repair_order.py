from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


REPAIR_ORDER_NUMBER_LIMIT = 40
REPAIR_ORDER_DATE_LIMIT = 32
REPAIR_ORDER_FIELD_LIMIT = 160
REPAIR_ORDER_COMMENT_LIMIT = 4000
REPAIR_ORDER_ROW_NAME_LIMIT = 240
REPAIR_ORDER_ROW_VALUE_LIMIT = 40
REPAIR_ORDER_ROWS_LIMIT = 100
REPAIR_ORDER_TAG_LIMIT = 5
REPAIR_ORDER_TAG_LABEL_LIMIT = 24
REPAIR_ORDER_TAG_COLOR_LIMIT = 16
REPAIR_ORDER_PAYMENT_ID_LIMIT = 80
REPAIR_ORDER_PAYMENT_NOTE_LIMIT = 240
REPAIR_ORDER_PAYMENTS_LIMIT = 200
REPAIR_ORDER_STATUS_OPEN = "open"
REPAIR_ORDER_STATUS_CLOSED = "closed"
REPAIR_ORDER_STATUS_LIMIT = 16
REPAIR_ORDER_DEFAULT_TAG_COLOR = "green"
REPAIR_ORDER_PAYMENT_METHOD_CASH = "cash"
REPAIR_ORDER_PAYMENT_METHOD_CASHLESS = "cashless"
REPAIR_ORDER_PAYMENT_METHOD_LIMIT = 16
REPAIR_ORDER_PAYMENT_TAX_RATE = Decimal("0.15")
REPAIR_ORDER_ALLOWED_STATUSES = {
    REPAIR_ORDER_STATUS_OPEN,
    REPAIR_ORDER_STATUS_CLOSED,
}
REPAIR_ORDER_ALLOWED_PAYMENT_METHODS = {
    REPAIR_ORDER_PAYMENT_METHOD_CASH,
    REPAIR_ORDER_PAYMENT_METHOD_CASHLESS,
}
REPAIR_ORDER_ALLOWED_TAG_COLORS = {
    "green",
    "yellow",
    "red",
}


def _normalize_single_line(value, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit]


def _normalize_multiline(value, *, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _normalize_tag_label(value) -> str:
    normalized = _normalize_single_line(value, limit=REPAIR_ORDER_TAG_LABEL_LIMIT).replace(",", " ")
    return " ".join(normalized.split()).upper()


def _normalize_tag_color(value) -> str:
    normalized = _normalize_single_line(value, limit=REPAIR_ORDER_TAG_COLOR_LIMIT).lower()
    if normalized not in REPAIR_ORDER_ALLOWED_TAG_COLORS:
        return REPAIR_ORDER_DEFAULT_TAG_COLOR
    return normalized


def normalize_repair_order_status(value, *, default: str = REPAIR_ORDER_STATUS_OPEN) -> str:
    raw = _normalize_single_line(value, limit=REPAIR_ORDER_STATUS_LIMIT).lower()
    if raw in {"", "opened", "active", "открыт", "открыта"}:
        return default
    if raw in {"closed", "archived", "archive", "закрыт", "закрыта"}:
        return REPAIR_ORDER_STATUS_CLOSED
    if raw in REPAIR_ORDER_ALLOWED_STATUSES:
        return raw
    return default


def normalize_repair_order_payment_method(value, *, default: str = REPAIR_ORDER_PAYMENT_METHOD_CASH) -> str:
    raw = _normalize_single_line(value, limit=REPAIR_ORDER_PAYMENT_METHOD_LIMIT).lower()
    if raw in {"", "cash", "cash_only", "cash-only", "нал", "наличный", "наличные"}:
        return default
    if raw in {"cashless", "wire", "bank", "безнал", "безналичный", "безналичные"}:
        return REPAIR_ORDER_PAYMENT_METHOD_CASHLESS
    if raw in REPAIR_ORDER_ALLOWED_PAYMENT_METHODS:
        return raw
    return default


def repair_order_payment_method_from_cashbox_name(
    value, *, default: str = REPAIR_ORDER_PAYMENT_METHOD_CASH
) -> str:
    raw = _normalize_single_line(value, limit=REPAIR_ORDER_FIELD_LIMIT).casefold()
    if not raw:
        return normalize_repair_order_payment_method(default)
    if "безнал" in raw or "cashless" in raw or "wire" in raw or "bank" in raw:
        return REPAIR_ORDER_PAYMENT_METHOD_CASHLESS
    return REPAIR_ORDER_PAYMENT_METHOD_CASH


def repair_order_payment_method_from_payments(
    payments: list["RepairOrderPayment"] | Any, *, default: str = REPAIR_ORDER_PAYMENT_METHOD_CASH
) -> str:
    normalized_payments = payments if isinstance(payments, list) else normalize_repair_order_payments(payments)
    if not normalized_payments:
        return normalize_repair_order_payment_method(default)
    for payment in normalized_payments:
        resolved = repair_order_payment_method_from_cashbox_name(
            getattr(payment, "cashbox_name", ""),
            default=getattr(payment, "payment_method", default),
        )
        if resolved == REPAIR_ORDER_PAYMENT_METHOD_CASHLESS:
            return REPAIR_ORDER_PAYMENT_METHOD_CASHLESS
    return REPAIR_ORDER_PAYMENT_METHOD_CASH


def repair_order_payment_method_label(value) -> str:
    normalized = normalize_repair_order_payment_method(value)
    return "Безналичный" if normalized == REPAIR_ORDER_PAYMENT_METHOD_CASHLESS else "Наличный"


def _parse_decimal(value) -> Decimal | None:
    raw = str(value or "").strip().replace(" ", "").replace(",", ".")
    if not raw:
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _format_decimal(value: Decimal) -> str:
    quantized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _round_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def repair_order_payments_value_by_method(
    payments: list["RepairOrderPayment"] | Any,
    payment_method: str,
) -> Decimal:
    normalized_payments = payments if isinstance(payments, list) else normalize_repair_order_payments(payments)
    normalized_method = normalize_repair_order_payment_method(payment_method)
    return sum(
        (
            payment.amount_value()
            for payment in normalized_payments
            if payment.payment_method == normalized_method
        ),
        Decimal("0"),
    )


def repair_order_payment_summary_value(
    base_total: Decimal,
    payments: list["RepairOrderPayment"] | Any,
) -> dict[str, Decimal]:
    normalized_payments = payments if isinstance(payments, list) else normalize_repair_order_payments(payments)
    base_total = _round_money(base_total)
    base_paid_cash = repair_order_payments_value_by_method(normalized_payments, REPAIR_ORDER_PAYMENT_METHOD_CASH)
    base_paid_noncash = repair_order_payments_value_by_method(normalized_payments, REPAIR_ORDER_PAYMENT_METHOD_CASHLESS)
    base_remaining = max(base_total - base_paid_cash - base_paid_noncash, Decimal("0"))
    taxes_and_fees = _round_money(base_paid_noncash * REPAIR_ORDER_PAYMENT_TAX_RATE)
    return {
        "base_total": base_total,
        "base_paid_cash": _round_money(base_paid_cash),
        "base_paid_noncash": _round_money(base_paid_noncash),
        "base_remaining": _round_money(base_remaining),
        "cash_due": _round_money(base_remaining),
        "noncash_due": _round_money(base_remaining * (Decimal("1") + REPAIR_ORDER_PAYMENT_TAX_RATE)),
        "taxes_and_fees": taxes_and_fees,
        "total_paid": _round_money(base_paid_cash + base_paid_noncash),
        "grand_total": _round_money(base_total + taxes_and_fees),
        "due_total": _round_money(base_remaining),
    }


@dataclass(slots=True)
class RepairOrderRow:
    name: str = ""
    catalog_number: str = ""
    quantity: str = ""
    price: str = ""
    total: str = ""
    executor_id: str = ""
    executor_name: str = ""
    salary_mode_snapshot: str = ""
    base_salary_snapshot: str = ""
    work_percent_snapshot: str = ""
    salary_amount: str = ""
    salary_accrued_at: str = ""

    def __post_init__(self) -> None:
        self.name = _normalize_single_line(self.name, limit=REPAIR_ORDER_ROW_NAME_LIMIT)
        self.catalog_number = _normalize_single_line(self.catalog_number, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.quantity = _normalize_single_line(self.quantity, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        self.price = _normalize_single_line(self.price, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        normalized_total = _normalize_single_line(self.total, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        computed_total = self.computed_total()
        self.total = computed_total if computed_total else normalized_total
        self.executor_id = _normalize_single_line(self.executor_id, limit=64)
        self.executor_name = _normalize_single_line(self.executor_name, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.salary_mode_snapshot = _normalize_single_line(self.salary_mode_snapshot, limit=32)
        self.base_salary_snapshot = _normalize_single_line(self.base_salary_snapshot, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        self.work_percent_snapshot = _normalize_single_line(self.work_percent_snapshot, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        self.salary_amount = _normalize_single_line(self.salary_amount, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        self.salary_accrued_at = _normalize_single_line(self.salary_accrued_at, limit=REPAIR_ORDER_DATE_LIMIT)

    def is_empty(self) -> bool:
        return not any([self.name, self.catalog_number, self.quantity, self.price, self.total])

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "catalog_number": self.catalog_number,
            "quantity": self.quantity,
            "price": self.price,
            "total": self.total,
            "executor_id": self.executor_id,
            "executor_name": self.executor_name,
            "salary_mode_snapshot": self.salary_mode_snapshot,
            "base_salary_snapshot": self.base_salary_snapshot,
            "work_percent_snapshot": self.work_percent_snapshot,
            "salary_amount": self.salary_amount,
            "salary_accrued_at": self.salary_accrued_at,
        }

    def total_value(self) -> Decimal:
        computed_total = self.computed_total()
        if computed_total:
            parsed = _parse_decimal(computed_total)
            if parsed is not None:
                return parsed
        parsed_total = _parse_decimal(self.total)
        return parsed_total if parsed_total is not None else Decimal("0")

    def computed_total(self) -> str:
        quantity = _parse_decimal(self.quantity)
        price = _parse_decimal(self.price)
        if quantity is None or price is None:
            return ""
        return _format_decimal(quantity * price)

    @classmethod
    def from_dict(cls, payload: Any) -> "RepairOrderRow":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            name=payload.get("name", ""),
            catalog_number=payload.get(
                "catalog_number",
                payload.get("catalogNumber", payload.get("part_number", payload.get("partNumber", payload.get("catalog", "")))),
            ),
            quantity=payload.get("quantity", payload.get("qty", "")),
            price=payload.get("price", ""),
            total=payload.get("total", payload.get("sum", "")),
            executor_id=payload.get("executor_id", payload.get("employee_id", "")),
            executor_name=payload.get("executor_name", payload.get("employee_name", "")),
            salary_mode_snapshot=payload.get("salary_mode_snapshot", ""),
            base_salary_snapshot=payload.get("base_salary_snapshot", ""),
            work_percent_snapshot=payload.get("work_percent_snapshot", ""),
            salary_amount=payload.get("salary_amount", ""),
            salary_accrued_at=payload.get("salary_accrued_at", ""),
        )


@dataclass(slots=True)
class RepairOrderTag:
    label: str = ""
    color: str = REPAIR_ORDER_DEFAULT_TAG_COLOR

    def __post_init__(self) -> None:
        self.label = _normalize_tag_label(self.label)
        self.color = _normalize_tag_color(self.color)

    def is_empty(self) -> bool:
        return not self.label

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "color": self.color,
        }

    @classmethod
    def from_value(cls, payload: Any) -> "RepairOrderTag | None":
        if isinstance(payload, cls):
            return cls(label=payload.label, color=payload.color)
        if isinstance(payload, dict):
            label = _normalize_tag_label(payload.get("label") or payload.get("name"))
            if not label:
                return None
            return cls(label=label, color=payload.get("color"))
        label = _normalize_tag_label(payload)
        if not label:
            return None
        return cls(label=label)


@dataclass(slots=True)
class RepairOrderPayment:
    id: str = ""
    amount: str = ""
    paid_at: str = ""
    note: str = ""
    payment_method: str = REPAIR_ORDER_PAYMENT_METHOD_CASH
    actor_name: str = ""
    cashbox_id: str = ""
    cashbox_name: str = ""
    cash_transaction_id: str = ""

    def __post_init__(self) -> None:
        self.id = _normalize_single_line(self.id, limit=REPAIR_ORDER_PAYMENT_ID_LIMIT)
        self.amount = _normalize_single_line(self.amount, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        self.paid_at = _normalize_single_line(self.paid_at, limit=REPAIR_ORDER_DATE_LIMIT)
        self.note = _normalize_multiline(self.note, limit=REPAIR_ORDER_PAYMENT_NOTE_LIMIT)
        self.actor_name = _normalize_single_line(self.actor_name, limit=80)
        self.cashbox_id = _normalize_single_line(self.cashbox_id, limit=128)
        self.cashbox_name = _normalize_single_line(self.cashbox_name, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.cash_transaction_id = _normalize_single_line(self.cash_transaction_id, limit=128)
        self.payment_method = repair_order_payment_method_from_cashbox_name(
            self.cashbox_name,
            default=normalize_repair_order_payment_method(self.payment_method),
        )

    def is_empty(self) -> bool:
        return not any([self.amount, self.note, self.paid_at])

    def amount_value(self) -> Decimal:
        return _parse_decimal(self.amount) or Decimal("0")

    def taxes_value(self) -> Decimal:
        if self.payment_method != REPAIR_ORDER_PAYMENT_METHOD_CASHLESS:
            return Decimal("0")
        return _round_money(self.amount_value() * REPAIR_ORDER_PAYMENT_TAX_RATE)

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "amount": self.amount,
            "amount_display": _format_decimal(self.amount_value()),
            "paid_at": self.paid_at,
            "note": self.note,
            "payment_method": self.payment_method,
            "payment_method_label": repair_order_payment_method_label(self.payment_method),
            "actor_name": self.actor_name,
            "cashbox_id": self.cashbox_id,
            "cashbox_name": self.cashbox_name,
            "cash_transaction_id": self.cash_transaction_id,
        }

    def to_storage_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "amount": self.amount,
            "paid_at": self.paid_at,
            "note": self.note,
            "payment_method": self.payment_method,
            "actor_name": self.actor_name,
            "cashbox_id": self.cashbox_id,
            "cashbox_name": self.cashbox_name,
            "cash_transaction_id": self.cash_transaction_id,
        }

    @classmethod
    def from_dict(cls, payload: Any, *, fallback_id: str = "") -> "RepairOrderPayment":
        if not isinstance(payload, dict):
            return cls(id=fallback_id)
        return cls(
            id=payload.get("id", fallback_id),
            amount=payload.get("amount", payload.get("value", "")),
            paid_at=payload.get("paid_at", payload.get("paidAt", payload.get("date", ""))),
            note=payload.get("note", payload.get("comment", payload.get("description", ""))),
            payment_method=payload.get("payment_method", payload.get("paymentMethod", "")),
            actor_name=payload.get("actor_name", payload.get("actorName", "")),
            cashbox_id=payload.get("cashbox_id", payload.get("cashboxId", "")),
            cashbox_name=payload.get("cashbox_name", payload.get("cashboxName", "")),
            cash_transaction_id=payload.get("cash_transaction_id", payload.get("cashTransactionId", "")),
        )


def normalize_repair_order_rows(value: Any) -> list[RepairOrderRow]:
    if not isinstance(value, list):
        return []
    rows: list[RepairOrderRow] = []
    for item in value:
        row = RepairOrderRow.from_dict(item)
        if row.is_empty():
            continue
        rows.append(row)
        if len(rows) >= REPAIR_ORDER_ROWS_LIMIT:
            break
    return rows


def normalize_repair_order_tags(value: Any) -> list[RepairOrderTag]:
    if not isinstance(value, list):
        return []
    tags_by_label: dict[str, RepairOrderTag] = {}
    for item in value:
        tag = RepairOrderTag.from_value(item)
        if tag is None or tag.is_empty():
            continue
        tags_by_label[tag.label] = tag
        if len(tags_by_label) >= REPAIR_ORDER_TAG_LIMIT:
            break
    return list(tags_by_label.values())


def normalize_repair_order_payments(value: Any) -> list[RepairOrderPayment]:
    if not isinstance(value, list):
        return []
    payments: list[RepairOrderPayment] = []
    for index, item in enumerate(value, start=1):
        payment = RepairOrderPayment.from_dict(item, fallback_id=f"payment-{index}")
        if payment.is_empty():
            continue
        if not payment.id:
            payment.id = f"payment-{index}"
        payments.append(payment)
        if len(payments) >= REPAIR_ORDER_PAYMENTS_LIMIT:
            break
    return payments


@dataclass(slots=True)
class RepairOrder:
    number: str = ""
    date: str = ""
    status: str = REPAIR_ORDER_STATUS_OPEN
    opened_at: str = ""
    closed_at: str = ""
    client: str = ""
    phone: str = ""
    vehicle: str = ""
    license_plate: str = ""
    vin: str = ""
    mileage: str = ""
    payment_method: str = REPAIR_ORDER_PAYMENT_METHOD_CASH
    prepayment: str = ""
    payments: list[RepairOrderPayment] = field(default_factory=list)
    reason: str = ""
    comment: str = ""
    note: str = ""
    tags: list[RepairOrderTag] = field(default_factory=list)
    works: list[RepairOrderRow] = field(default_factory=list)
    materials: list[RepairOrderRow] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.number = _normalize_single_line(self.number, limit=REPAIR_ORDER_NUMBER_LIMIT)
        self.date = _normalize_single_line(self.date, limit=REPAIR_ORDER_DATE_LIMIT)
        self.status = normalize_repair_order_status(self.status)
        self.opened_at = _normalize_single_line(self.opened_at, limit=REPAIR_ORDER_DATE_LIMIT)
        self.closed_at = _normalize_single_line(self.closed_at, limit=REPAIR_ORDER_DATE_LIMIT)
        self.client = _normalize_single_line(self.client, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.phone = _normalize_single_line(self.phone, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.vehicle = _normalize_single_line(self.vehicle, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.license_plate = _normalize_single_line(self.license_plate, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.vin = _normalize_single_line(self.vin, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.mileage = _normalize_single_line(self.mileage, limit=REPAIR_ORDER_FIELD_LIMIT)
        self.payment_method = normalize_repair_order_payment_method(self.payment_method)
        self.prepayment = _normalize_single_line(self.prepayment, limit=REPAIR_ORDER_ROW_VALUE_LIMIT)
        self.payments = normalize_repair_order_payments(self.payments)
        if not self.payments:
            legacy_amount = _parse_decimal(self.prepayment)
            if legacy_amount is not None and legacy_amount != Decimal("0"):
                self.payments = [
                    RepairOrderPayment(
                        id="legacy-prepayment",
                        amount=_format_decimal(legacy_amount),
                        paid_at=self.opened_at or self.date,
                        note="Перенесено из предоплаты",
                        payment_method=self.payment_method,
                    )
                ]
        if self.payments:
            self.payment_method = repair_order_payment_method_from_payments(
                self.payments,
                default=self.payment_method,
            )
            self.prepayment = self.prepayment_amount()
        self.reason = _normalize_multiline(self.reason, limit=REPAIR_ORDER_COMMENT_LIMIT)
        self.comment = _normalize_multiline(self.comment, limit=REPAIR_ORDER_COMMENT_LIMIT)
        self.note = _normalize_multiline(self.note, limit=REPAIR_ORDER_COMMENT_LIMIT)
        self.tags = normalize_repair_order_tags(self.tags)
        self.works = normalize_repair_order_rows(self.works)
        self.materials = normalize_repair_order_rows(self.materials)

    def is_empty(self) -> bool:
        return not any(
            [
                self.number,
                self.date,
                self.opened_at,
                self.closed_at,
                self.client,
                self.phone,
                self.vehicle,
                self.license_plate,
                self.vin,
                self.mileage,
                self.prepayment,
                self.payments,
                self.reason,
                self.comment,
                self.note,
                self.tags,
                self.works,
                self.materials,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "date": self.date,
            "status": self.status,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "client": self.client,
            "phone": self.phone,
            "vehicle": self.vehicle,
            "license_plate": self.license_plate,
            "vin": self.vin,
            "mileage": self.mileage,
            "payment_method": self.payment_method,
            "payment_method_label": repair_order_payment_method_label(self.payment_method),
            "prepayment": self.prepayment_amount() if self.payments else self.prepayment,
            "prepayment_display": self.prepayment_amount(),
            "paid_total": self.prepayment_amount(),
            "paid_total_display": self.prepayment_amount(),
            "payments": [payment.to_storage_dict() for payment in self.payments],
            "reason": self.reason,
            "comment": self.comment,
            "client_information": self.comment,
            "note": self.note,
            "tags": [tag.to_dict() for tag in self.tags],
            "works": [row.to_dict() for row in self.works],
            "materials": [row.to_dict() for row in self.materials],
            "payment_summary": self.payment_summary_amounts(),
            "subtotal_total": self.subtotal_amount(),
            "taxes_total": self.taxes_amount(),
            "works_total": self.works_total_amount(),
            "materials_total": self.materials_total_amount(),
            "grand_total": self.grand_total_amount(),
            "due_total": self.due_total_amount(),
            "has_taxes": self.has_taxes(),
            "has_prepayment": self.has_prepayment(),
            "is_paid": self.is_paid(),
            "payment_status": self.payment_status(),
            "payment_status_label": self.payment_status_label(),
            "has_any_data": not self.is_empty(),
        }

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "date": self.date,
            "status": self.status,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "client": self.client,
            "phone": self.phone,
            "vehicle": self.vehicle,
            "license_plate": self.license_plate,
            "vin": self.vin,
            "mileage": self.mileage,
            "payment_method": self.payment_method,
            "prepayment": self.prepayment_amount() if self.payments else self.prepayment,
            "payments": [payment.to_dict() for payment in self.payments],
            "reason": self.reason,
            "comment": self.comment,
            "note": self.note,
            "tags": [tag.to_dict() for tag in self.tags],
            "works": [row.to_dict() for row in self.works],
            "materials": [row.to_dict() for row in self.materials],
        }

    def works_total_amount(self) -> str:
        return _format_decimal(sum((row.total_value() for row in self.works), Decimal("0")))

    def materials_total_amount(self) -> str:
        return _format_decimal(sum((row.total_value() for row in self.materials), Decimal("0")))

    def subtotal_value(self) -> Decimal:
        works_total = _parse_decimal(self.works_total_amount()) or Decimal("0")
        materials_total = _parse_decimal(self.materials_total_amount()) or Decimal("0")
        return works_total + materials_total

    def subtotal_amount(self) -> str:
        return _format_decimal(self.subtotal_value())

    def cash_payments_value(self) -> Decimal:
        if self.payments:
            return repair_order_payments_value_by_method(self.payments, REPAIR_ORDER_PAYMENT_METHOD_CASH)
        if self.payment_method == REPAIR_ORDER_PAYMENT_METHOD_CASH:
            return _parse_decimal(self.prepayment) or Decimal("0")
        return Decimal("0")

    def cashless_payments_value(self) -> Decimal:
        if self.payments:
            return repair_order_payments_value_by_method(self.payments, REPAIR_ORDER_PAYMENT_METHOD_CASHLESS)
        prepayment = _parse_decimal(self.prepayment) or Decimal("0")
        if self.payment_method != REPAIR_ORDER_PAYMENT_METHOD_CASHLESS:
            return Decimal("0")
        return prepayment

    def taxes_value(self) -> Decimal:
        if self.payment_method != REPAIR_ORDER_PAYMENT_METHOD_CASHLESS:
            return Decimal("0")
        return _round_money(self.subtotal_value() * REPAIR_ORDER_PAYMENT_TAX_RATE)

    def taxes_amount(self) -> str:
        return _format_decimal(self.taxes_value())

    def grand_total_amount(self) -> str:
        return _format_decimal(self.subtotal_value() + self.taxes_value())

    def prepayment_value(self) -> Decimal:
        if self.payments:
            return sum((payment.amount_value() for payment in self.payments), Decimal("0"))
        return _parse_decimal(self.prepayment) or Decimal("0")

    def prepayment_amount(self) -> str:
        return _format_decimal(self.prepayment_value())

    def due_total_value(self) -> Decimal:
        grand_total = _parse_decimal(self.grand_total_amount()) or Decimal("0")
        return grand_total - self.prepayment_value()

    def due_total_amount(self) -> str:
        return _format_decimal(self.due_total_value())

    def payment_summary_value(self) -> dict[str, Decimal]:
        return repair_order_payment_summary_value(self.subtotal_value(), self.payments)

    def payment_summary_amounts(self) -> dict[str, str]:
        summary = self.payment_summary_value()
        return {key: _format_decimal(value) for key, value in summary.items()}

    def has_taxes(self) -> bool:
        return self.taxes_value() != Decimal("0")

    def has_prepayment(self) -> bool:
        return self.prepayment_value() != Decimal("0")

    def is_paid(self) -> bool:
        return self.payment_summary_value()["base_remaining"] <= Decimal("0")

    def payment_status(self) -> str:
        return "paid" if self.is_paid() else "unpaid"

    def payment_status_label(self) -> str:
        return "Оплачен" if self.is_paid() else "Не оплачен"

    @classmethod
    def from_dict(cls, payload: Any) -> "RepairOrder":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            number=payload.get("number", ""),
            date=payload.get("date", ""),
            status=payload.get("status", REPAIR_ORDER_STATUS_OPEN),
            opened_at=payload.get("opened_at", payload.get("openedAt", "")),
            closed_at=payload.get("closed_at", payload.get("closedAt", "")),
            client=payload.get("client", ""),
            phone=payload.get("phone", ""),
            vehicle=payload.get("vehicle", ""),
            license_plate=payload.get("license_plate", payload.get("licensePlate", "")),
            vin=payload.get("vin", ""),
            mileage=payload.get("mileage", payload.get("odometer", "")),
            payment_method=payload.get("payment_method", payload.get("paymentMethod", "")),
            prepayment=payload.get("prepayment", payload.get("advance_payment", payload.get("advancePayment", ""))),
            payments=payload.get("payments", payload.get("payment_history", [])),
            reason=payload.get("reason", payload.get("problem", "")),
            comment=payload.get("client_information", payload.get("clientInformation", payload.get("comment", ""))),
            note=payload.get("note", payload.get("master_comment", payload.get("masterComment", payload.get("internal_comment", payload.get("internalComment", ""))))),
            tags=payload.get("tags", []),
            works=payload.get("works", []),
            materials=payload.get("materials", []),
        )
