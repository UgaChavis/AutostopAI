from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from .automotive_tools import AutomotiveLookupService
from ..mcp.client import BoardApiClient


@dataclass(frozen=True)
class AgentToolDefinition:
    name: str
    description: str
    args_schema: dict[str, Any]


class ExternalToolBudgetExceeded(ValueError):
    pass


class AgentToolExecutor:
    def __init__(self, board_api: BoardApiClient, *, actor_name: str = "SERVER_AGENT") -> None:
        self._board_api = board_api
        self._actor_name = actor_name
        self._automotive = AutomotiveLookupService()
        self._external_request_budget_default = 5
        self._external_request_budget = self._external_request_budget_default
        self._tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "ping_connector": self._ping_connector,
            "review_board": self._review_board,
            "list_columns": self._list_columns,
            "get_board_snapshot": self._get_board_snapshot,
            "search_cards": self._search_cards,
            "get_card": self._get_card,
            "get_card_context": self._get_card_context,
            "create_card": self._create_card,
            "update_card": self._update_card,
            "move_card": self._move_card,
            "archive_card": self._archive_card,
            "restore_card": self._restore_card,
            "list_repair_orders": self._list_repair_orders,
            "get_repair_order": self._get_repair_order,
            "update_repair_order": self._update_repair_order,
            "replace_repair_order_works": self._replace_repair_order_works,
            "replace_repair_order_materials": self._replace_repair_order_materials,
            "set_repair_order_status": self._set_repair_order_status,
            "list_cashboxes": self._list_cashboxes,
            "get_cashbox": self._get_cashbox,
            "create_cashbox": self._create_cashbox,
            "delete_cashbox": self._delete_cashbox,
            "create_cash_transaction": self._create_cash_transaction,
            "decode_vin": self._decode_vin,
            "find_part_numbers": self._find_part_numbers,
            "search_part_numbers": self._search_part_numbers,
            "estimate_price_ru": self._estimate_price_ru,
            "lookup_part_prices": self._lookup_part_prices,
            "decode_dtc": self._decode_dtc,
            "search_fault_info": self._search_fault_info,
            "estimate_maintenance": self._estimate_maintenance,
            "search_web": self._search_web,
            "fetch_page_excerpt": self._fetch_page_excerpt,
        }

    @property
    def definitions(self) -> list[AgentToolDefinition]:
        return [
            AgentToolDefinition("ping_connector", "Check local CRM/API reachability.", {}),
            AgentToolDefinition(
                "review_board",
                "Get an operational board summary with alerts and priority cards.",
                {
                    "stale_hours": "optional int",
                    "overload_threshold": "optional int",
                    "priority_limit": "optional int",
                    "recent_event_limit": "optional int",
                },
            ),
            AgentToolDefinition("list_columns", "List board columns.", {}),
            AgentToolDefinition("get_board_snapshot", "Get board snapshot.", {"archive_limit": "optional int"}),
            AgentToolDefinition(
                "search_cards",
                "Search cards by text and filters.",
                {
                    "query": "optional string",
                    "include_archived": "optional bool",
                    "column": "optional string",
                    "tag": "optional string",
                    "indicator": "optional string",
                    "status": "optional string",
                    "limit": "optional int",
                },
            ),
            AgentToolDefinition("get_card", "Get one card by id.", {"card_id": "required string"}),
            AgentToolDefinition(
                "get_card_context",
                "Get one card with related context and events.",
                {
                    "card_id": "required string",
                    "event_limit": "optional int",
                    "include_repair_order_text": "optional bool",
                },
            ),
            AgentToolDefinition(
                "create_card",
                "Create a new card.",
                {
                    "vehicle": "optional string",
                    "title": "required string",
                    "description": "optional string",
                    "column": "optional string",
                    "tags": "optional array",
                    "deadline": "optional object",
                    "vehicle_profile": "optional object",
                },
            ),
            AgentToolDefinition(
                "update_card",
                "Update card fields.",
                {
                    "card_id": "required string",
                    "vehicle": "optional string",
                    "title": "optional string",
                    "description": "optional string",
                    "tags": "optional array",
                    "deadline": "optional object",
                    "vehicle_profile": "optional object",
                },
            ),
            AgentToolDefinition(
                "move_card",
                "Move a card to another column or position.",
                {
                    "card_id": "required string",
                    "column": "required string",
                    "before_card_id": "optional string",
                },
            ),
            AgentToolDefinition("archive_card", "Archive a card.", {"card_id": "required string"}),
            AgentToolDefinition("restore_card", "Restore an archived card.", {"card_id": "required string", "column": "optional string"}),
            AgentToolDefinition(
                "list_repair_orders",
                "List repair orders.",
                {
                    "limit": "optional int",
                    "status": "optional string",
                    "query": "optional string",
                    "sort_by": "optional string",
                    "sort_dir": "optional string",
                },
            ),
            AgentToolDefinition("get_repair_order", "Get repair order by card id.", {"card_id": "required string"}),
            AgentToolDefinition(
                "update_repair_order",
                "Update repair order object for a card.",
                {"card_id": "required string", "repair_order": "required object"},
            ),
            AgentToolDefinition(
                "replace_repair_order_works",
                "Replace repair order works rows.",
                {"card_id": "required string", "rows": "required array"},
            ),
            AgentToolDefinition(
                "replace_repair_order_materials",
                "Replace repair order material rows.",
                {"card_id": "required string", "rows": "required array"},
            ),
            AgentToolDefinition(
                "set_repair_order_status",
                "Set repair order status.",
                {"card_id": "required string", "status": "required string"},
            ),
            AgentToolDefinition("list_cashboxes", "List cashboxes.", {"limit": "optional int"}),
            AgentToolDefinition("get_cashbox", "Get one cashbox with transactions.", {"cashbox_id": "required string", "transaction_limit": "optional int"}),
            AgentToolDefinition("create_cashbox", "Create a cashbox.", {"name": "required string"}),
            AgentToolDefinition("delete_cashbox", "Delete a cashbox.", {"cashbox_id": "required string"}),
            AgentToolDefinition(
                "create_cash_transaction",
                "Create cashbox income or expense.",
                {
                    "cashbox_id": "required string",
                    "direction": "required string",
                    "amount": "required number/string",
                    "note": "optional string",
                },
            ),
            AgentToolDefinition(
                "decode_vin",
                "Decode a VIN using external trusted sources.",
                {"vin": "required string"},
            ),
            AgentToolDefinition(
                "find_part_numbers",
                "Find OEM/catalog part numbers with trusted whitelisted sources.",
                {"query": "required string", "vehicle": "optional string/object", "limit": "optional int"},
            ),
            AgentToolDefinition(
                "search_part_numbers",
                "Search OEM/catalog part numbers for a vehicle and requested part.",
                {
                    "part_query": "required string",
                    "vehicle_context": "optional object",
                    "limit": "optional int",
                },
            ),
            AgentToolDefinition(
                "estimate_price_ru",
                "Estimate Russian-market part prices from trusted whitelisted sources.",
                {"part_number": "required string", "vehicle": "optional string/object", "limit": "optional int"},
            ),
            AgentToolDefinition(
                "lookup_part_prices",
                "Search market prices for a part number or part query.",
                {
                    "part_number_or_query": "required string",
                    "vehicle_context": "optional object",
                    "limit": "optional int",
                },
            ),
            AgentToolDefinition(
                "decode_dtc",
                "Decode an OBD/DTC trouble code using trusted whitelisted sources.",
                {"code": "required string", "vehicle": "optional string/object", "vehicle_context": "optional object", "limit": "optional int"},
            ),
            AgentToolDefinition(
                "search_fault_info",
                "Search fault symptoms and repair notes with trusted whitelisted sources.",
                {"query": "required string", "vehicle": "optional string/object", "vehicle_context": "optional object", "limit": "optional int"},
            ),
            AgentToolDefinition(
                "estimate_maintenance",
                "Build a preliminary maintenance plan for a vehicle.",
                {
                    "service_type": "optional string",
                    "vehicle_context": "optional object",
                },
            ),
            AgentToolDefinition(
                "search_web",
                "Search trusted web sources by free-text query.",
                {
                    "query": "required string",
                    "limit": "optional int",
                    "allowed_domains": "optional array",
                },
            ),
            AgentToolDefinition(
                "fetch_page_excerpt",
                "Fetch and clean a web page excerpt.",
                {"url": "required string", "max_chars": "optional int"},
            ),
        ]

    def describe_for_prompt(
        self,
        *,
        task_type: str | None = None,
        context_kind: str | None = None,
    ) -> str:
        lines: list[str] = []
        for item in self.definitions:
            if not self._definition_allowed(item.name, task_type=task_type, context_kind=context_kind):
                continue
            lines.append(f"- {item.name}: {item.description}")
            lines.append(f"  args: {json.dumps(item.args_schema, ensure_ascii=False)}")
        return "\n".join(lines)

    def execute(self, tool_name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        handler = self._tools.get(str(tool_name or "").strip().lower())
        if handler is None:
            raise KeyError(f"Unknown agent tool: {tool_name}")
        return handler(args or {})

    def reset_task_budget(self) -> None:
        self._external_request_budget = self._external_request_budget_default
        reset_cache = getattr(self._automotive, "reset_task_cache", None)
        if callable(reset_cache):
            reset_cache()

    def _ping_connector(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.health()

    def _review_board(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.review_board(
            stale_hours=self._maybe_int(args.get("stale_hours")),
            overload_threshold=self._maybe_int(args.get("overload_threshold")),
            priority_limit=self._maybe_int(args.get("priority_limit")),
            recent_event_limit=self._maybe_int(args.get("recent_event_limit")),
        )

    def _list_columns(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.list_columns()

    def _get_board_snapshot(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.get_board_snapshot(archive_limit=self._maybe_int(args.get("archive_limit")))

    def _search_cards(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.search_cards(
            query=self._maybe_text(args.get("query")),
            include_archived=bool(args.get("include_archived", False)),
            column=self._maybe_text(args.get("column")),
            tag=self._maybe_text(args.get("tag")),
            indicator=self._maybe_text(args.get("indicator")),
            status=self._maybe_text(args.get("status")),
            limit=self._maybe_int(args.get("limit")),
        )

    def _get_card(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.get_card(self._required_text(args, "card_id"))

    def _get_card_context(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.get_card_context(
            self._required_text(args, "card_id"),
            event_limit=self._maybe_int(args.get("event_limit")) or 20,
            include_repair_order_text=bool(args.get("include_repair_order_text", True)),
        )

    def _create_card(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.create_card(
            vehicle=self._maybe_text(args.get("vehicle")) or "",
            title=self._required_text(args, "title"),
            description=self._maybe_text(args.get("description")) or "",
            column=self._maybe_text(args.get("column")),
            tags=self._maybe_list(args.get("tags")),
            deadline=self._maybe_dict(args.get("deadline")),
            vehicle_profile=self._maybe_dict(args.get("vehicle_profile")),
            actor_name=self._actor_name,
        )

    def _update_card(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.update_card(
            card_id=self._required_text(args, "card_id"),
            vehicle=self._maybe_text(args.get("vehicle")),
            title=self._maybe_text(args.get("title")),
            description=self._maybe_text(args.get("description")),
            tags=self._maybe_list(args.get("tags")),
            deadline=self._maybe_dict(args.get("deadline")),
            vehicle_profile=self._maybe_dict(args.get("vehicle_profile")),
            actor_name=self._actor_name,
        )

    def _move_card(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.move_card(
            card_id=self._required_text(args, "card_id"),
            column=self._required_text(args, "column"),
            before_card_id=self._maybe_text(args.get("before_card_id")),
            actor_name=self._actor_name,
        )

    def _archive_card(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.archive_card(card_id=self._required_text(args, "card_id"), actor_name=self._actor_name)

    def _restore_card(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.restore_card(
            card_id=self._required_text(args, "card_id"),
            column=self._maybe_text(args.get("column")),
            actor_name=self._actor_name,
        )

    def _list_repair_orders(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.list_repair_orders(
            limit=self._maybe_int(args.get("limit")),
            status=self._maybe_text(args.get("status")),
            query=self._maybe_text(args.get("query")),
            sort_by=self._maybe_text(args.get("sort_by")),
            sort_dir=self._maybe_text(args.get("sort_dir")),
        )

    def _get_repair_order(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.get_repair_order(self._required_text(args, "card_id"))

    def _update_repair_order(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.update_repair_order(
            card_id=self._required_text(args, "card_id"),
            repair_order=self._required_dict(args, "repair_order"),
            actor_name=self._actor_name,
        )

    def _replace_repair_order_works(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.replace_repair_order_works(
            card_id=self._required_text(args, "card_id"),
            rows=self._required_list(args, "rows"),
            actor_name=self._actor_name,
        )

    def _replace_repair_order_materials(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.replace_repair_order_materials(
            card_id=self._required_text(args, "card_id"),
            rows=self._required_list(args, "rows"),
            actor_name=self._actor_name,
        )

    def _set_repair_order_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.set_repair_order_status(
            card_id=self._required_text(args, "card_id"),
            status=self._required_text(args, "status"),
            actor_name=self._actor_name,
        )

    def _list_cashboxes(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.list_cashboxes(limit=self._maybe_int(args.get("limit")))

    def _get_cashbox(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.get_cashbox(
            self._required_text(args, "cashbox_id"),
            transaction_limit=self._maybe_int(args.get("transaction_limit")),
        )

    def _create_cashbox(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.create_cashbox(self._required_text(args, "name"), actor_name=self._actor_name)

    def _delete_cashbox(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._board_api.delete_cashbox(self._required_text(args, "cashbox_id"), actor_name=self._actor_name)

    def _create_cash_transaction(self, args: dict[str, Any]) -> dict[str, Any]:
        amount = args.get("amount")
        if amount is None:
            raise ValueError("amount is required")
        return self._board_api.create_cash_transaction(
            cashbox_id=self._required_text(args, "cashbox_id"),
            direction=self._required_text(args, "direction"),
            amount=amount,
            note=self._maybe_text(args.get("note")) or "",
            actor_name=self._actor_name,
        )

    def _decode_vin(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.decode_vin(self._required_text(args, "vin"))

    def _find_part_numbers(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.find_part_numbers(
            query=self._required_text(args, "query"),
            vehicle=self._vehicle_payload(args.get("vehicle") or args.get("vehicle_context")),
            limit=self._maybe_int(args.get("limit")) or 5,
        )

    def _search_part_numbers(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.search_part_numbers(
            vehicle_context=self._maybe_dict(args.get("vehicle_context")),
            part_query=self._required_text(args, "part_query"),
            limit=self._maybe_int(args.get("limit")) or 8,
        )

    def _estimate_price_ru(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.estimate_price_ru(
            part_number=self._required_text(args, "part_number"),
            vehicle=self._vehicle_payload(args.get("vehicle") or args.get("vehicle_context")),
            limit=self._maybe_int(args.get("limit")) or 5,
        )

    def _lookup_part_prices(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.lookup_part_prices(
            vehicle_context=self._maybe_dict(args.get("vehicle_context")),
            part_number_or_query=self._required_text(args, "part_number_or_query"),
            limit=self._maybe_int(args.get("limit")) or 8,
        )

    def _decode_dtc(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.decode_dtc(
            code=self._required_text(args, "code"),
            vehicle_context=self._maybe_dict(args.get("vehicle_context")),
            vehicle=self._vehicle_payload(args.get("vehicle")),
            limit=self._maybe_int(args.get("limit")) or 5,
        )

    def _search_fault_info(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.search_fault_info(
            query=self._required_text(args, "query"),
            vehicle_context=self._maybe_dict(args.get("vehicle_context")),
            vehicle=self._vehicle_payload(args.get("vehicle")),
            limit=self._maybe_int(args.get("limit")) or 5,
        )

    def _estimate_maintenance(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._automotive.estimate_maintenance(
            vehicle_context=self._maybe_dict(args.get("vehicle_context")),
            service_type=self._maybe_text(args.get("service_type")) or "ТО",
        )

    def _search_web(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.search_web(
            query=self._required_text(args, "query"),
            limit=self._maybe_int(args.get("limit")) or 5,
            allowed_domains=self._maybe_list(args.get("allowed_domains")),
        )

    def _fetch_page_excerpt(self, args: dict[str, Any]) -> dict[str, Any]:
        self._consume_external_request_budget()
        return self._automotive.fetch_page_excerpt(
            url=self._required_text(args, "url"),
            max_chars=self._maybe_int(args.get("max_chars")) or 2500,
        )

    def _required_text(self, args: dict[str, Any], key: str) -> str:
        value = self._maybe_text(args.get(key))
        if not value:
            raise ValueError(f"{key} is required")
        return value

    def _required_dict(self, args: dict[str, Any], key: str) -> dict[str, Any]:
        value = self._maybe_dict(args.get(key))
        if value is None:
            raise ValueError(f"{key} is required")
        return value

    def _required_list(self, args: dict[str, Any], key: str) -> list[Any]:
        value = self._maybe_list(args.get(key))
        if value is None:
            raise ValueError(f"{key} is required")
        return value

    def _maybe_text(self, value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

    def _maybe_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _maybe_dict(self, value: Any) -> dict[str, Any] | None:
        return value if isinstance(value, dict) else None

    def _maybe_list(self, value: Any) -> list[Any] | None:
        return value if isinstance(value, list) else None

    def _vehicle_payload(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        text = self._maybe_text(value)
        return {"vehicle": text} if text else None

    def _consume_external_request_budget(self) -> None:
        if self._external_request_budget <= 0:
            raise ExternalToolBudgetExceeded("External web tool budget exceeded for this task.")
        self._external_request_budget -= 1

    def _definition_allowed(self, tool_name: str, *, task_type: str | None, context_kind: str | None) -> bool:
        normalized_type = str(task_type or "").strip().lower()
        normalized_context = str(context_kind or "").strip().lower()
        all_tools = {item.name for item in self.definitions}
        core_board = {
            "ping_connector",
            "review_board",
            "list_columns",
            "get_board_snapshot",
            "search_cards",
            "get_card",
            "get_card_context",
            "list_repair_orders",
            "get_repair_order",
            "list_cashboxes",
            "get_cashbox",
        }
        card_update = {
            "get_card",
            "get_card_context",
            "update_card",
            "move_card",
            "archive_card",
            "restore_card",
        }
        repair_order = {
            "get_repair_order",
            "update_repair_order",
            "replace_repair_order_works",
            "replace_repair_order_materials",
            "set_repair_order_status",
        }
        cashboxes = {
            "list_cashboxes",
            "get_cashbox",
            "create_cashbox",
            "delete_cashbox",
            "create_cash_transaction",
        }
        automotive = {
            "decode_vin",
            "find_part_numbers",
            "search_part_numbers",
            "estimate_price_ru",
            "lookup_part_prices",
            "decode_dtc",
            "search_fault_info",
            "estimate_maintenance",
            "search_web",
            "fetch_page_excerpt",
        }

        if normalized_type == "board_review":
            allowed = core_board
        elif normalized_type == "card_cleanup":
            allowed = core_board | card_update | repair_order | automotive
        elif normalized_type == "vin_decode":
            allowed = card_update | {"get_repair_order"} | {"decode_vin", "search_web", "fetch_page_excerpt"}
        elif normalized_type == "parts_lookup":
            allowed = card_update | {"get_repair_order"} | {
                "decode_vin",
                "find_part_numbers",
                "search_part_numbers",
                "estimate_price_ru",
                "lookup_part_prices",
                "decode_dtc",
                "search_fault_info",
                "search_web",
                "fetch_page_excerpt",
            }
        elif normalized_type == "maintenance_estimate":
            allowed = card_update | {"get_repair_order"} | {
                "estimate_maintenance",
                "search_part_numbers",
                "lookup_part_prices",
                "decode_vin",
            }
        elif normalized_type == "dtc_lookup":
            allowed = card_update | {"get_repair_order"} | {
                "decode_dtc",
                "search_fault_info",
                "search_web",
                "fetch_page_excerpt",
            }
        elif normalized_type == "repair_order_assist":
            allowed = card_update | repair_order | automotive
        elif normalized_type == "cash_review":
            allowed = core_board | cashboxes
        else:
            allowed = all_tools

        if normalized_context == "card":
            allowed |= {"get_card_context", "update_card", "get_repair_order"}
        return tool_name in allowed
