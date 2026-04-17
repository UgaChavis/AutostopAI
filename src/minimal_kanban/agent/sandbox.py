from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .runner import AgentRunner
from .storage import AgentStorage


def _now_text() -> str:
    from ..models import utc_now_iso

    return utc_now_iso()


@dataclass
class SandboxCard:
    id: str
    title: str
    description: str = ""
    vehicle: str = ""
    column: str = "Inbox"
    tags: list[str] = field(default_factory=list)
    vehicle_profile: dict[str, Any] = field(default_factory=dict)
    repair_order: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "vehicle": self.vehicle,
            "column": self.column,
            "tags": list(self.tags),
            "vehicle_profile": dict(self.vehicle_profile),
            "repair_order": dict(self.repair_order),
        }


class OfflineBoardApiClient:
    def __init__(self) -> None:
        self.cards: dict[str, SandboxCard] = {}
        self.columns: list[dict[str, Any]] = [
            {"id": "inbox", "label": "Inbox", "cards": []},
            {"id": "diagnostics", "label": "Diagnostics", "cards": []},
            {"id": "done", "label": "Done", "cards": []},
        ]
        self.repair_orders: dict[str, dict[str, Any]] = {}
        self.cashboxes: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        return "offline://sandbox"

    def seed_card(self, card: SandboxCard) -> None:
        self.cards[card.id] = deepcopy(card)
        self._rebuild_columns()

    def health(self) -> dict[str, Any]:
        self.calls.append({"method": "health"})
        return {"ok": True, "mode": "offline"}

    def get_card(self, card_id: str) -> dict[str, Any]:
        self.calls.append({"method": "get_card", "card_id": card_id})
        card = self._get_card(card_id)
        return {"data": {"card": card.to_dict()}}

    def get_card_context(self, card_id: str, *, event_limit: int = 20, include_repair_order_text: bool = True) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "get_card_context",
                "card_id": card_id,
                "event_limit": event_limit,
                "include_repair_order_text": include_repair_order_text,
            }
        )
        card = self._get_card(card_id)
        payload: dict[str, Any] = {
            "card": card.to_dict(),
            "events": list(card.events)[-event_limit:],
            "repair_order_text": {"text": self.repair_orders.get(card_id, {}).get("text", "")},
        }
        if card_id in self.repair_orders:
            payload["repair_order"] = deepcopy(self.repair_orders[card_id])
        return {"data": payload}

    def search_cards(
        self,
        *,
        query: str | None = None,
        include_archived: bool = False,
        column: str | None = None,
        tag: str | None = None,
        indicator: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "search_cards",
                "query": query,
                "include_archived": include_archived,
                "column": column,
                "tag": tag,
                "indicator": indicator,
                "status": status,
                "limit": limit,
            }
        )
        items = []
        query_text = str(query or "").strip().casefold()
        for card in self.cards.values():
            haystack = " ".join([card.id, card.title, card.description, card.vehicle, " ".join(card.tags)]).casefold()
            if query_text and query_text not in haystack:
                continue
            if column and str(card.column).strip().casefold() != str(column).strip().casefold():
                continue
            if tag and str(tag).strip().casefold() not in {item.casefold() for item in card.tags}:
                continue
            items.append(
                {
                    "id": card.id,
                    "title": card.title,
                    "vehicle": card.vehicle,
                    "column": card.column,
                    "column_label": card.column,
                    "tags": list(card.tags),
                    "indicator": None,
                }
            )
            if limit is not None and len(items) >= int(limit):
                break
        return {"data": {"cards": items}}

    def update_card(
        self,
        *,
        card_id: str,
        vehicle: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: list[str | dict[str, object]] | None = None,
        deadline: dict | None = None,
        vehicle_profile: dict[str, object] | None = None,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "update_card",
                "card_id": card_id,
                "vehicle": vehicle,
                "title": title,
                "description": description,
                "tags": tags,
                "vehicle_profile": vehicle_profile,
                "actor_name": actor_name,
            }
        )
        card = self._get_card(card_id)
        changed: list[str] = []
        if vehicle is not None and vehicle != card.vehicle:
            card.vehicle = vehicle
            changed.append("vehicle")
        if title is not None and title != card.title:
            card.title = title
            changed.append("title")
        if description is not None and description != card.description:
            card.description = description
            changed.append("description")
        if tags is not None:
            normalized_tags = [str(item.get("label") if isinstance(item, dict) else item).strip() for item in tags]
            if normalized_tags != card.tags:
                card.tags = normalized_tags
                changed.append("tags")
        if vehicle_profile is not None:
            normalized_profile = dict(vehicle_profile)
            if normalized_profile != card.vehicle_profile:
                card.vehicle_profile = normalized_profile
                changed.append("vehicle_profile")
        card.events.append({"type": "update_card", "actor": actor_name or "offline", "at": _now_text(), "changed": list(changed)})
        self._rebuild_columns()
        return {
            "data": {
                "card": card.to_dict(),
                "changed": list(changed),
                "meta": {"changed_fields": list(changed)},
            }
        }

    def get_board_snapshot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "get_board_snapshot", "args": args, "kwargs": kwargs})
        return {"data": {"columns": [deepcopy(column) for column in self.columns]}}

    def list_columns(self) -> dict[str, Any]:
        self.calls.append({"method": "list_columns"})
        return {"data": {"columns": [deepcopy(column) for column in self.columns]}}

    def review_board(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "review_board", "kwargs": kwargs})
        return {"data": {"summary": "offline review", "cards": []}}

    def list_repair_orders(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "list_repair_orders", "kwargs": kwargs})
        return {"data": {"repair_orders": list(self.repair_orders.values())}}

    def get_repair_order(self, card_id: str) -> dict[str, Any]:
        self.calls.append({"method": "get_repair_order", "card_id": card_id})
        return {"data": deepcopy(self.repair_orders.get(card_id, {}))}

    def update_repair_order(self, *, card_id: str, repair_order: dict[str, object], actor_name: str | None = None) -> dict[str, Any]:
        self.calls.append({"method": "update_repair_order", "card_id": card_id, "actor_name": actor_name})
        self.repair_orders[card_id] = deepcopy(repair_order)
        return {"data": {"card_id": card_id, "repair_order": deepcopy(repair_order), "meta": {"changed_fields": list(repair_order)}}}

    def replace_repair_order_works(self, *, card_id: str, rows: list[dict[str, object]], actor_name: str | None = None) -> dict[str, Any]:
        self.calls.append({"method": "replace_repair_order_works", "card_id": card_id, "actor_name": actor_name})
        repair_order = self.repair_orders.setdefault(card_id, {})
        repair_order["works"] = deepcopy(rows)
        return {"data": {"card_id": card_id, "rows": deepcopy(rows)}}

    def replace_repair_order_materials(self, *, card_id: str, rows: list[dict[str, object]], actor_name: str | None = None) -> dict[str, Any]:
        self.calls.append({"method": "replace_repair_order_materials", "card_id": card_id, "actor_name": actor_name})
        repair_order = self.repair_orders.setdefault(card_id, {})
        repair_order["materials"] = deepcopy(rows)
        return {"data": {"card_id": card_id, "rows": deepcopy(rows)}}

    def set_repair_order_status(self, *, card_id: str, status: str, actor_name: str | None = None) -> dict[str, Any]:
        self.calls.append({"method": "set_repair_order_status", "card_id": card_id, "status": status, "actor_name": actor_name})
        repair_order = self.repair_orders.setdefault(card_id, {})
        repair_order["status"] = status
        return {"data": {"card_id": card_id, "status": status}}

    def _get_card(self, card_id: str) -> SandboxCard:
        normalized = str(card_id or "").strip()
        if not normalized:
            raise KeyError("card_id is required")
        card = self.cards.get(normalized)
        if card is None:
            raise KeyError(f"Unknown sandbox card: {normalized}")
        return card

    def _rebuild_columns(self) -> None:
        cards_by_column: dict[str, list[dict[str, Any]]] = {column["id"]: [] for column in self.columns}
        for card in self.cards.values():
            column_id = "inbox" if card.column.lower() == "inbox" else ("done" if card.column.lower() == "done" else "diagnostics")
            cards_by_column.setdefault(column_id, []).append(
                {
                    "id": card.id,
                    "title": card.title,
                    "vehicle": card.vehicle,
                    "column": card.column,
                    "tags": list(card.tags),
                }
            )
        for column in self.columns:
            column["cards"] = cards_by_column.get(column["id"], [])


class NullModelClient:
    model = "offline-null"

    def next_step(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        del system_prompt, messages, reasoning_effort
        return {
            "vin": "",
            "status": "insufficient",
            "source_summary": "offline sandbox",
            "source_confidence": 0.0,
            "source_links_or_refs": [],
            "warnings": ["offline sandbox"],
        }


class OfflineAgentSandbox:
    def __init__(self, *, base_dir: Path | None = None) -> None:
        self._tempdir: TemporaryDirectory[str] | None = None
        if base_dir is None:
            self._tempdir = TemporaryDirectory(prefix="autostopai-sandbox-")
            base_dir = Path(self._tempdir.name)
        self.storage = AgentStorage(base_dir=base_dir)
        self.board_api = OfflineBoardApiClient()
        self.model_client = NullModelClient()
        self.logger = logging.getLogger("autostopai.sandbox")
        self.runner = AgentRunner(
            storage=self.storage,
            board_api=self.board_api,
            model_client=self.model_client,  # type: ignore[arg-type]
            logger=self.logger,
            actor_name="OFFLINE_SANDBOX",
        )

    def close(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def seed_card(self, card: SandboxCard) -> None:
        self.board_api.seed_card(card)

    def enqueue_card_enrichment_task(self, *, card_id: str, task_text: str, requested_by: str = "sandbox") -> dict[str, Any]:
        return self.storage.enqueue_task(
            task_text=task_text,
            source="sandbox",
            mode="card_enrichment",
            metadata={
                "requested_by": requested_by,
                "purpose": "card_enrichment",
                "scenario_id": "card_enrichment",
                "trigger": "manual",
                "context": {"kind": "card", "card_id": card_id},
                "scope": {"type": "current_card", "card_id": card_id},
                "card_enrichment": {"card_id": card_id, "card_heading": self.board_api.cards[card_id].title},
            },
        )

    def enqueue_card_cleanup(self, *, card_id: str, task_text: str, requested_by: str = "sandbox") -> dict[str, Any]:
        return self.enqueue_card_enrichment_task(card_id=card_id, task_text=task_text, requested_by=requested_by)

    def preview_card(self, *, card_id: str, task_text: str) -> dict[str, Any]:
        return self.preview_card_enrichment(card_id=card_id, task_text=task_text)

    def preview_card_enrichment(self, *, card_id: str, task_text: str) -> dict[str, Any]:
        card_payload = self.board_api.get_card_context(card_id)
        task = {
            "id": "preview",
            "task_text": task_text,
            "mode": "card_enrichment",
            "source": "sandbox",
            "metadata": {
                "purpose": "card_enrichment",
                "context": {"kind": "card", "card_id": card_id},
                "card_enrichment": {"card_id": card_id, "card_heading": self.board_api.cards[card_id].title},
            },
        }
        metadata = task["metadata"]
        task_type = self.runner._router.classify_task(task, metadata)
        context_kind = self.runner._router.context_kind(metadata)
        context_payload = self.runner._response_data(card_payload)
        facts = self.runner._analyze_card_autofill_context(context_payload, task_text=task_text, purpose="card_enrichment")
        facts["task_type"] = task_type
        facts["context_kind"] = context_kind
        facts["autofill_plan"] = self.runner._build_card_autofill_plan(facts)
        compact_facts = {
            "task_type": task_type,
            "context_kind": context_kind,
            "repair_order": dict(facts.get("repair_order") or {}),
            "vehicle_context": dict(facts.get("vehicle_context") or {}),
            "vin": str(facts.get("vin", "") or "").strip(),
            "autofill_plan": dict(facts.get("autofill_plan") or {}),
        }
        plan = self.runner._build_orchestration_plan(
            metadata=metadata,
            task_type=task_type,
            context_kind=context_kind,
            evidence=self.runner._build_orchestration_evidence(
                task=task,
                metadata=metadata,
                task_type=task_type,
                context_kind=context_kind,
                context_data=context_payload if isinstance(context_payload, dict) else {},
                raw_context_ref="preview",
            )[0],
            facts=facts,
        )
        return {
            "task_type": task_type,
            "context_kind": context_kind,
            "facts": compact_facts,
            "plan": plan.to_dict(),
            "card_context": context_payload,
        }

    def run_once(self) -> bool:
        return self.runner.run_once()

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.storage.read_status(),
            "tasks": self.storage.list_tasks(limit=20),
            "runs": self.storage.list_runs(limit=20),
            "actions": self.storage.list_actions(limit=20),
            "cards": {card_id: card.to_dict() for card_id, card in self.board_api.cards.items()},
            "calls": list(self.board_api.calls),
        }

    def __enter__(self) -> OfflineAgentSandbox:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def demo_snapshot() -> dict[str, Any]:
    with OfflineAgentSandbox() as sandbox:
        sandbox.seed_card(
            SandboxCard(
                id="card-1",
                title="Ford Focus 2016",
                description="Порядок в карточке нужен. Клиент ждёт.",
                vehicle="",
            )
        )
        sandbox.enqueue_card_enrichment_task(card_id="card-1", task_text="Наведи порядок в карточке и сохрани только полезное.")
        sandbox.run_once()
        return sandbox.snapshot()
