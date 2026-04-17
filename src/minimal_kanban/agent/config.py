from __future__ import annotations

import os
from pathlib import Path

from ..config import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    get_app_data_dir,
    get_board_api_url,
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = (os.environ.get(name) or "").strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = (os.environ.get(name) or "").strip()
    if not raw_value:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.1) -> float:
    raw_value = (os.environ.get(name) or "").strip()
    if not raw_value:
        return default
    try:
        return max(minimum, float(raw_value))
    except ValueError:
        return default


def get_agent_enabled() -> bool:
    return _env_flag("MINIMAL_KANBAN_AGENT_ENABLED", default=False)


def get_agent_name() -> str:
    return (os.environ.get("MINIMAL_KANBAN_AGENT_NAME") or "AUTOSTOP SERVER AGENT").strip() or "AUTOSTOP SERVER AGENT"


def get_agent_data_dir() -> Path:
    return get_app_data_dir() / "agent"


def get_agent_log_file() -> Path:
    return get_agent_data_dir() / "agent.log"


def get_agent_prompt_file() -> Path:
    return get_agent_data_dir() / "system_prompt.md"


def get_agent_memory_file() -> Path:
    return get_agent_data_dir() / "memory.md"


def get_agent_tasks_file() -> Path:
    return get_agent_data_dir() / "tasks.json"


def get_agent_schedules_file() -> Path:
    return get_agent_data_dir() / "schedules.json"


def get_agent_status_file() -> Path:
    return get_agent_data_dir() / "status.json"


def get_agent_runs_file() -> Path:
    return get_agent_data_dir() / "runs.jsonl"


def get_agent_actions_file() -> Path:
    return get_agent_data_dir() / "actions.jsonl"


def get_agent_lock_file() -> Path:
    return get_agent_data_dir() / "agent.lock"


def get_agent_openai_api_key() -> str | None:
    value = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return value or None


def get_agent_openai_model() -> str:
    return (os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


def get_agent_openai_base_url() -> str:
    return (os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/") or DEFAULT_OPENAI_BASE_URL


def get_agent_request_timeout_seconds() -> float:
    return _env_float("MINIMAL_KANBAN_AGENT_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS, minimum=1.0)


def get_agent_poll_interval_seconds() -> float:
    return _env_float("MINIMAL_KANBAN_AGENT_POLL_INTERVAL_SECONDS", 2.0, minimum=0.2)


def get_agent_max_steps() -> int:
    return _env_int("MINIMAL_KANBAN_AGENT_MAX_STEPS", 12, minimum=1)


def get_agent_max_tool_result_chars() -> int:
    return _env_int("MINIMAL_KANBAN_AGENT_MAX_TOOL_RESULT_CHARS", 6000, minimum=500)


def get_agent_board_api_url() -> str | None:
    value = (
        os.environ.get("MINIMAL_KANBAN_AGENT_BOARD_API_URL")
        or os.environ.get("AUTOSTOP_AGENT_BOARD_API_URL")
        or get_board_api_url()
        or ""
    ).strip().rstrip("/")
    return value or None
