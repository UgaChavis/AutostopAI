from __future__ import annotations

import json
import urllib.error
import urllib.request
from logging import Logger

from ..config import get_api_port, get_api_port_fallback_limit


class BoardApiTransportError(RuntimeError):
    pass


def candidate_api_urls() -> list[str]:
    start_port = get_api_port()
    fallback_limit = get_api_port_fallback_limit()
    return [f"http://127.0.0.1:{port}" for port in range(start_port, start_port + fallback_limit)]


def discover_board_api(*, bearer_token: str | None = None, timeout_seconds: float = 1.0) -> str | None:
    for base_url in candidate_api_urls():
        client = BoardApiClient(base_url, bearer_token=bearer_token, timeout_seconds=timeout_seconds)
        try:
            response = client.health()
        except BoardApiTransportError:
            continue
        if response.get("ok"):
            return base_url
    return None


class BoardApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 10.0,
        logger: Logger | None = None,
        default_source: str = "mcp",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._bearer_token = bearer_token
        self._timeout_seconds = timeout_seconds
        self._logger = logger
        self._default_source = default_source

    def health(self) -> dict:
        return self._request("/api/health", method="GET")

    def list_columns(self) -> dict:
        return self._request("/api/list_columns", method="GET")

    def create_column(self, label: str, *, actor_name: str | None = None) -> dict:
        return self._request_with_identity("/api/create_column", {"label": label}, actor_name=actor_name)

    def rename_column(self, column_id: str, label: str, *, actor_name: str | None = None) -> dict:
        return self._request_with_identity(
            "/api/rename_column",
            {"column_id": column_id, "label": label},
            actor_name=actor_name,
        )

    def delete_column(self, column_id: str, *, actor_name: str | None = None) -> dict:
        return self._request_with_identity("/api/delete_column", {"column_id": column_id}, actor_name=actor_name)

    def create_sticky(
        self,
        *,
        text: str,
        x: int = 0,
        y: int = 0,
        deadline: dict,
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"text": text, "x": x, "y": y, "deadline": deadline}
        return self._request_with_identity("/api/create_sticky", payload, actor_name=actor_name)

    def get_cards(self, *, include_archived: bool = False, compact: bool = False) -> dict:
        return self._request("/api/get_cards", {"include_archived": include_archived, "compact": compact})

    def get_card(self, card_id: str) -> dict:
        return self._request("/api/get_card", {"card_id": card_id})

    def get_card_context(
        self,
        card_id: str,
        *,
        event_limit: int = 20,
        include_repair_order_text: bool = True,
    ) -> dict:
        return self._request(
            "/api/get_card_context",
            {
                "card_id": card_id,
                "event_limit": event_limit,
                "include_repair_order_text": include_repair_order_text,
            },
        )

    def get_board_snapshot(
        self,
        *,
        archive_limit: int | None = None,
        compact: bool | None = None,
        include_archive: bool | None = None,
    ) -> dict:
        payload: dict[str, object] = {}
        if archive_limit is not None:
            payload["archive_limit"] = archive_limit
        if compact is not None:
            payload["compact"] = compact
        if include_archive is not None:
            payload["include_archive"] = include_archive
        if not payload:
            return self._request("/api/get_board_snapshot", method="GET")
        return self._request("/api/get_board_snapshot", payload, method="POST")

    def get_board_context(self) -> dict:
        return self._request("/api/get_board_context", method="GET")

    def review_board(
        self,
        *,
        stale_hours: int | None = None,
        overload_threshold: int | None = None,
        priority_limit: int | None = None,
        recent_event_limit: int | None = None,
    ) -> dict:
        payload: dict[str, object] = {}
        if stale_hours is not None:
            payload["stale_hours"] = stale_hours
        if overload_threshold is not None:
            payload["overload_threshold"] = overload_threshold
        if priority_limit is not None:
            payload["priority_limit"] = priority_limit
        if recent_event_limit is not None:
            payload["recent_event_limit"] = recent_event_limit
        if not payload:
            return self._request("/api/review_board", method="GET")
        return self._request("/api/review_board", payload, method="POST")

    def list_cashboxes(self, *, limit: int | None = None) -> dict:
        return self._request_optional_scalar_filter("/api/list_cashboxes", key="limit", value=limit)

    def get_cashbox(self, cashbox_id: str, *, transaction_limit: int | None = None) -> dict:
        payload: dict[str, object] = {"cashbox_id": cashbox_id}
        if transaction_limit is not None:
            payload["transaction_limit"] = transaction_limit
        return self._request("/api/get_cashbox", payload)

    def create_cashbox(self, name: str, *, actor_name: str | None = None) -> dict:
        return self._request_with_identity("/api/create_cashbox", {"name": name}, actor_name=actor_name)

    def delete_cashbox(self, cashbox_id: str, *, actor_name: str | None = None) -> dict:
        return self._request_with_identity("/api/delete_cashbox", {"cashbox_id": cashbox_id}, actor_name=actor_name)

    def create_cash_transaction(
        self,
        *,
        cashbox_id: str,
        direction: str,
        amount_minor: int | None = None,
        amount: str | int | float | None = None,
        note: str = "",
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {
            "cashbox_id": cashbox_id,
            "direction": direction,
            "note": note,
        }
        if amount_minor is not None:
            payload["amount_minor"] = amount_minor
        elif amount is not None:
            payload["amount"] = amount
        return self._request_with_identity("/api/create_cash_transaction", payload, actor_name=actor_name)

    def update_board_settings(self, *, board_scale: float, actor_name: str | None = None) -> dict:
        payload: dict[str, object] = {"board_scale": board_scale}
        return self._request_with_identity("/api/update_board_settings", payload, actor_name=actor_name)

    def get_gpt_wall(self, *, include_archived: bool = True, event_limit: int | None = None) -> dict:
        payload: dict[str, object] = {"include_archived": include_archived}
        if event_limit is not None:
            payload["event_limit"] = event_limit
        return self._request("/api/get_gpt_wall", payload, method="POST")

    def cleanup_card_content(self, *, card_id: str, actor_name: str | None = None) -> dict:
        return self._request_with_identity("/api/cleanup_card_content", {"card_id": card_id}, actor_name=actor_name)

    def autofill_vehicle_data(
        self,
        *,
        raw_text: str = "",
        image_base64: str | None = None,
        image_filename: str | None = None,
        image_mime_type: str | None = None,
        vehicle_profile: dict[str, object] | None = None,
        vehicle: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"raw_text": raw_text}
        if image_base64 is not None:
            payload["image_base64"] = image_base64
        if image_filename is not None:
            payload["image_filename"] = image_filename
        if image_mime_type is not None:
            payload["image_mime_type"] = image_mime_type
        if vehicle_profile is not None:
            payload["vehicle_profile"] = vehicle_profile
        if vehicle is not None:
            payload["vehicle"] = vehicle
        if title is not None:
            payload["title"] = title
        if description is not None:
            payload["description"] = description
        return self._request("/api/autofill_vehicle_data", payload)

    def autofill_repair_order(
        self,
        *,
        card_id: str,
        overwrite: bool = False,
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"card_id": card_id, "overwrite": overwrite}
        return self._request_with_identity("/api/autofill_repair_order", payload, actor_name=actor_name)

    def get_card_log(self, card_id: str, *, limit: int | None = None) -> dict:
        payload: dict[str, object] = {"card_id": card_id}
        if limit is not None:
            payload["limit"] = limit
        return self._request("/api/get_card_log", payload)

    def get_repair_order(self, card_id: str) -> dict:
        return self._request("/api/get_repair_order", {"card_id": card_id})

    def get_repair_order_text(self, card_id: str) -> dict:
        return self._request("/api/get_repair_order_text", {"card_id": card_id})

    def list_archived_cards(self, *, limit: int | None = None, compact: bool | None = None) -> dict:
        payload: dict[str, object] = {}
        if limit is not None:
            payload["limit"] = limit
        if compact is not None:
            payload["compact"] = compact
        if not payload:
            return self._request("/api/list_archived_cards", method="GET")
        return self._request("/api/list_archived_cards", payload, method="POST")

    def list_repair_orders(
        self,
        *,
        limit: int | None = None,
        status: str | None = None,
        query: str | None = None,
        sort_by: str | None = None,
        sort_dir: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {}
        if limit is not None:
            payload["limit"] = limit
        if status:
            payload["status"] = status
        if query is not None:
            payload["query"] = query
        if sort_by:
            payload["sort_by"] = sort_by
        if sort_dir:
            payload["sort_dir"] = sort_dir
        if not payload:
            return self._request("/api/list_repair_orders", method="GET")
        return self._request("/api/list_repair_orders", payload, method="POST")

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
    ) -> dict:
        payload: dict[str, object] = {"include_archived": include_archived}
        if query is not None:
            payload["query"] = query
        if column:
            payload["column"] = column
        if tag:
            payload["tag"] = tag
        if indicator:
            payload["indicator"] = indicator
        if status:
            payload["status"] = status
        if limit is not None:
            payload["limit"] = limit
        return self._request("/api/search_cards", payload)

    def create_card(
        self,
        *,
        vehicle: str = "",
        title: str,
        description: str = "",
        column: str | None = None,
        tags: list[str | dict[str, object]] | None = None,
        deadline: dict | None = None,
        vehicle_profile: dict[str, object] | None = None,
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {
            "vehicle": vehicle,
            "title": title,
            "description": description,
            "deadline": self._normalize_card_deadline(deadline),
        }
        if column:
            payload["column"] = column
        if tags is not None:
            payload["tags"] = tags
        if vehicle_profile is not None:
            payload["vehicle_profile"] = vehicle_profile
        return self._request_with_identity("/api/create_card", payload, actor_name=actor_name)

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
    ) -> dict:
        payload: dict[str, object] = {"card_id": card_id}
        if vehicle is not None:
            payload["vehicle"] = vehicle
        if title is not None:
            payload["title"] = title
        if description is not None:
            payload["description"] = description
        if tags is not None:
            payload["tags"] = tags
        if deadline is not None:
            payload["deadline"] = deadline
        if vehicle_profile is not None:
            payload["vehicle_profile"] = vehicle_profile
        return self._request_with_identity("/api/update_card", payload, actor_name=actor_name)

    def update_repair_order(
        self,
        *,
        card_id: str,
        repair_order: dict[str, object],
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"card_id": card_id, "repair_order": repair_order}
        return self._request_with_identity("/api/update_repair_order", payload, actor_name=actor_name)

    def set_repair_order_status(
        self,
        *,
        card_id: str,
        status: str,
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"card_id": card_id, "status": status}
        return self._request_with_identity("/api/set_repair_order_status", payload, actor_name=actor_name)

    def replace_repair_order_works(
        self,
        *,
        card_id: str,
        rows: list[dict[str, object]],
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"card_id": card_id, "rows": rows}
        return self._request_with_identity("/api/replace_repair_order_works", payload, actor_name=actor_name)

    def replace_repair_order_materials(
        self,
        *,
        card_id: str,
        rows: list[dict[str, object]],
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"card_id": card_id, "rows": rows}
        return self._request_with_identity("/api/replace_repair_order_materials", payload, actor_name=actor_name)

    def update_sticky(
        self,
        *,
        sticky_id: str,
        text: str | None = None,
        deadline: dict | None = None,
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"sticky_id": sticky_id}
        if text is not None:
            payload["text"] = text
        if deadline is not None:
            payload["deadline"] = deadline
        return self._request_with_identity("/api/update_sticky", payload, actor_name=actor_name)

    def move_sticky(self, *, sticky_id: str, x: int, y: int, actor_name: str | None = None) -> dict:
        payload: dict[str, object] = {"sticky_id": sticky_id, "x": x, "y": y}
        return self._request_with_identity("/api/move_sticky", payload, actor_name=actor_name)

    def delete_sticky(self, *, sticky_id: str, actor_name: str | None = None) -> dict:
        return self._request_with_identity("/api/delete_sticky", {"sticky_id": sticky_id}, actor_name=actor_name)

    def set_card_deadline(self, *, card_id: str, deadline: dict, actor_name: str | None = None) -> dict:
        payload = {"card_id": card_id, "deadline": deadline}
        return self._request_with_identity("/api/set_card_deadline", payload, actor_name=actor_name)

    def set_card_indicator(self, *, card_id: str, indicator: str, actor_name: str | None = None) -> dict:
        payload = {"card_id": card_id, "indicator": indicator}
        return self._request_with_identity("/api/set_card_indicator", payload, actor_name=actor_name)

    def move_card(
        self,
        *,
        card_id: str,
        column: str,
        before_card_id: str | None = None,
        actor_name: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"card_id": card_id, "column": column}
        if before_card_id:
            payload["before_card_id"] = before_card_id
        return self._request_with_identity("/api/move_card", payload, actor_name=actor_name)

    def bulk_move_cards(self, *, card_ids: list[str], column: str, actor_name: str | None = None) -> dict:
        payload: dict[str, object] = {"card_ids": card_ids, "column": column}
        return self._request_with_identity("/api/bulk_move_cards", payload, actor_name=actor_name)

    def archive_card(self, *, card_id: str, actor_name: str | None = None) -> dict:
        return self._request_with_identity("/api/archive_card", {"card_id": card_id}, actor_name=actor_name)

    def restore_card(self, *, card_id: str, column: str | None = None, actor_name: str | None = None) -> dict:
        payload: dict[str, object] = {"card_id": card_id}
        if column:
            payload["column"] = column
        return self._request_with_identity("/api/restore_card", payload, actor_name=actor_name)

    def list_overdue_cards(self, *, include_archived: bool = False) -> dict:
        return self._request("/api/list_overdue_cards", {"include_archived": include_archived}, method="POST")

    def _with_identity(self, payload: dict[str, object], *, actor_name: str | None = None) -> dict[str, object]:
        enriched = dict(payload)
        enriched["source"] = self._default_source
        if actor_name:
            enriched["actor_name"] = actor_name
        return enriched

    def _request_optional_scalar_filter(self, path: str, *, key: str, value: object | None) -> dict:
        if value is None:
            return self._request(path, method="GET")
        return self._request(path, {key: value}, method="POST")

    def _request_with_identity(self, path: str, payload: dict[str, object], *, actor_name: str | None = None) -> dict:
        return self._request(path, self._with_identity(payload, actor_name=actor_name))

    def _normalize_card_deadline(self, deadline: dict | None) -> dict[str, int]:
        if not isinstance(deadline, dict):
            return {"days": 1, "hours": 0, "minutes": 0, "seconds": 0}
        normalized = {
            "days": int(deadline.get("days", 0) or 0),
            "hours": int(deadline.get("hours", 0) or 0),
            "minutes": int(deadline.get("minutes", 0) or 0),
            "seconds": int(deadline.get("seconds", 0) or 0),
        }
        if not any(normalized.values()):
            return {"days": 1, "hours": 0, "minutes": 0, "seconds": 0}
        return normalized

    def _request(self, path: str, payload: dict | None = None, *, method: str = "POST", _allow_retry: bool = True) -> dict:
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        request = urllib.request.Request(
            self._compose_url(path),
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                parsed = self._parse_json_payload(response.read(), path=path)
                self._log("board_api_request path=%s status=%s", path, response.status)
                return parsed
        except urllib.error.HTTPError as exc:
            try:
                payload = self._parse_json_payload(exc.read(), path=path)
                self._log("board_api_request path=%s status=%s error=%s", path, exc.code, payload.get("error"))
                return payload
            finally:
                exc.close()
        except (urllib.error.URLError, TimeoutError) as exc:
            if str(method or "POST").strip().upper() == "GET" and _allow_retry:
                self._log("board_api_request path=%s retry_after_transport_error=%s", path, exc)
                return self._request(path, payload, method=method, _allow_retry=False)
            message = f"Не удалось подключиться к локальному API по адресу {self.base_url}."
            raise BoardApiTransportError(message) from exc

    def _parse_json_payload(self, raw: bytes, *, path: str) -> dict:
        try:
            decoded = raw.decode("utf-8")
            parsed = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as parse_error:
            message = f"Локальный API вернул некорректный JSON для {path}."
            raise BoardApiTransportError(message) from parse_error
        return parsed

    def _log(self, message: str, *args) -> None:
        if self._logger is not None:
            self._logger.info(message, *args)

    def _compose_url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        if self.base_url.endswith("/api") and normalized_path.startswith("/api/"):
            normalized_path = normalized_path[4:]
        return f"{self.base_url}{normalized_path}"
