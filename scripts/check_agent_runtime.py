from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _request_json(url: str, *, method: str = "GET", payload: dict | None = None, headers: dict[str, str] | None = None) -> dict:
    data = None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    with urllib.request.urlopen(request, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _login(base_url: str, username: str, password: str) -> str:
    response = _request_json(
        f"{base_url}/api/login_operator",
        method="POST",
        payload={"username": username, "password": password},
    )
    return response["data"]["session"]["token"]


def _heartbeat_age_seconds(value: str) -> float | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())


def _default_api_url() -> str:
    candidate = (
        os.environ.get("MINIMAL_KANBAN_AGENT_BOARD_API_URL")
        or os.environ.get("AUTOSTOP_AGENT_BOARD_API_URL")
        or "http://127.0.0.1:41731"
    )
    return candidate.rstrip("/")


def _http_error_code(exc: Exception) -> int | None:
    return exc.code if isinstance(exc, urllib.error.HTTPError) else None


def _supports_cleanup_only_flow(base_url: str, token: str) -> bool:
    headers = {"X-Operator-Session": token} if token else {}
    try:
        _request_json(
            f"{base_url}/api/cleanup_card_content",
            method="POST",
            payload={},
            headers=headers,
        )
        return True
    except urllib.error.HTTPError as exc:
        # Any non-404 response means the cleanup endpoint exists and the legacy
        # server-agent runtime is simply retired in this deployment.
        return exc.code != 404


def _evaluate_agent_runtime_mode(
    *,
    base_url: str,
    token: str,
    max_heartbeat_age_seconds: float,
) -> tuple[str, dict[str, str]]:
    headers = {"X-Operator-Session": token} if token else {}
    try:
        payload = _request_json(
            f"{base_url}/api/agent_status",
            headers=headers,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as exc:
        if _http_error_code(exc) == 404 and _supports_cleanup_only_flow(base_url, token):
            return "cleanup_only", {"api_url": base_url, "reason": "agent_status_route_retired"}
        raise

    status = payload["data"]["status"]
    agent = payload["data"]["agent"]
    heartbeat_age = _heartbeat_age_seconds(str(status.get("last_heartbeat", "") or ""))
    if agent.get("enabled") and heartbeat_age is not None and heartbeat_age <= max_heartbeat_age_seconds:
        return "ok", {
            "api_url": base_url,
            "heartbeat_age_seconds": f"{heartbeat_age:.2f}",
            "model": str(agent.get("model", "") or ""),
        }
    if _supports_cleanup_only_flow(base_url, token):
        if not agent.get("enabled"):
            reason = "embedded_agent_disabled"
        else:
            reason = "stale_or_missing_heartbeat"
        return "cleanup_only", {"api_url": base_url, "reason": reason}
    if not agent.get("enabled"):
        return "disabled", {"api_url": base_url}
    return "stale_heartbeat", {
        "api_url": base_url,
        "heartbeat_age_seconds": str(heartbeat_age if heartbeat_age is not None else "unknown"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-api-url", default=_default_api_url())
    parser.add_argument("--operator-username", default="admin")
    parser.add_argument("--operator-password", default="admin")
    parser.add_argument("--max-heartbeat-age-seconds", type=float, default=30.0)
    args = parser.parse_args()

    try:
        token = _login(args.local_api_url.rstrip("/"), args.operator_username, args.operator_password)
        status, details = _evaluate_agent_runtime_mode(
            base_url=args.local_api_url.rstrip("/"),
            token=token,
            max_heartbeat_age_seconds=args.max_heartbeat_age_seconds,
        )
        if status == "ok":
            print(
                "status: ok",
                f"heartbeat_age_seconds={details['heartbeat_age_seconds']}",
                f"api_url={details['api_url']}",
                f"model={details['model']}",
            )
            return 0
        if status == "cleanup_only":
            print(
                "status: cleanup_only",
                f"api_url={details['api_url']}",
                f"reason={details['reason']}",
            )
            return 0
        if status == "disabled":
            print("status: disabled", f"api_url={details['api_url']}")
            return 1
        if status == "stale_heartbeat":
            print(
                "status: stale_heartbeat",
                f"heartbeat_age_seconds={details['heartbeat_age_seconds']}",
                f"api_url={details['api_url']}",
            )
            return 1
        print("status: error", f"api_url={args.local_api_url.rstrip('/')}", f"mode={status}")
        return 1
    except (KeyError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        print("status: error", f"type={type(exc).__name__}", f"detail={exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
