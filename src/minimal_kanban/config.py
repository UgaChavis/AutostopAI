from __future__ import annotations

import os
from pathlib import Path

from . import __version__
from .texts import APP_DISPLAY_NAME


APP_NAME = "Minimal Kanban"
APP_SLUG = "minimal-kanban"
APP_VERSION = __version__
APP_DISPLAY_TITLE = APP_DISPLAY_NAME
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 41731
API_PORT_FALLBACK_LIMIT = 10
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 41831
MCP_PORT_FALLBACK_LIMIT = 10
DEFAULT_MCP_PATH = "/mcp"
STATE_FILE_NAME = "state.json"
SETTINGS_FILE_NAME = "settings.json"
USERS_FILE_NAME = "users.json"
LOG_FILE_NAME = "minimal-kanban.log"
MCP_STARTUP_LOG_FILE_NAME = "mcp-startup.log"
MCP_OAUTH_STATE_FILE_NAME = "mcp-oauth-state.json"
ATTACHMENTS_DIR_NAME = "attachments"
DEFAULT_OPENAI_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180


def get_app_data_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    return Path.home() / f".{APP_SLUG}"


def get_logs_dir() -> Path:
    return get_app_data_dir() / "logs"


def get_state_file() -> Path:
    return get_app_data_dir() / STATE_FILE_NAME


def get_settings_file() -> Path:
    return get_app_data_dir() / SETTINGS_FILE_NAME


def get_users_file() -> Path:
    return get_app_data_dir() / USERS_FILE_NAME


def get_log_file() -> Path:
    return get_logs_dir() / LOG_FILE_NAME


def get_mcp_startup_log_file() -> Path:
    return get_logs_dir() / MCP_STARTUP_LOG_FILE_NAME


def get_mcp_oauth_state_file() -> Path:
    return get_app_data_dir() / MCP_OAUTH_STATE_FILE_NAME


def get_attachments_dir() -> Path:
    return get_app_data_dir() / ATTACHMENTS_DIR_NAME


def _read_env_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return max(minimum, parsed)


def get_api_host() -> str:
    return (os.environ.get("MINIMAL_KANBAN_API_HOST") or DEFAULT_API_HOST).strip() or DEFAULT_API_HOST


def get_api_port() -> int:
    return _read_env_int("MINIMAL_KANBAN_API_PORT", DEFAULT_API_PORT, minimum=1)


def get_api_port_fallback_limit() -> int:
    return _read_env_int("MINIMAL_KANBAN_API_PORT_FALLBACK_LIMIT", API_PORT_FALLBACK_LIMIT, minimum=1)


def get_api_bearer_token() -> str | None:
    token = (os.environ.get("MINIMAL_KANBAN_API_BEARER_TOKEN") or "").strip()
    return token or None


def get_api_base_url() -> str | None:
    value = (
        os.environ.get("MINIMAL_KANBAN_API_BASE_URL")
        or os.environ.get("MINIMAL_KANBAN_BOARD_API_URL")
        or ""
    ).strip().rstrip("/")
    return value or None


def get_default_admin_username() -> str:
    return (os.environ.get("MINIMAL_KANBAN_DEFAULT_ADMIN_USERNAME") or "admin").strip() or "admin"


def get_default_admin_password() -> str:
    return (os.environ.get("MINIMAL_KANBAN_DEFAULT_ADMIN_PASSWORD") or "admin").strip() or "admin"


def get_mcp_host() -> str:
    return (os.environ.get("MINIMAL_KANBAN_MCP_HOST") or DEFAULT_MCP_HOST).strip() or DEFAULT_MCP_HOST


def get_mcp_port() -> int:
    return _read_env_int("MINIMAL_KANBAN_MCP_PORT", DEFAULT_MCP_PORT, minimum=1)


def get_mcp_port_fallback_limit() -> int:
    return _read_env_int("MINIMAL_KANBAN_MCP_PORT_FALLBACK_LIMIT", MCP_PORT_FALLBACK_LIMIT, minimum=1)


def get_mcp_path() -> str:
    raw_value = (os.environ.get("MINIMAL_KANBAN_MCP_PATH") or DEFAULT_MCP_PATH).strip()
    if not raw_value:
        return DEFAULT_MCP_PATH
    return raw_value if raw_value.startswith("/") else f"/{raw_value}"


def get_mcp_bearer_token() -> str | None:
    token = (os.environ.get("MINIMAL_KANBAN_MCP_BEARER_TOKEN") or "").strip()
    return token or None


def get_mcp_public_base_url() -> str | None:
    value = (os.environ.get("MINIMAL_KANBAN_MCP_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    return value or None


def get_mcp_tunnel_url() -> str | None:
    value = (os.environ.get("MINIMAL_KANBAN_MCP_TUNNEL_URL") or "").strip().rstrip("/")
    return value or None


def get_mcp_public_endpoint_url() -> str | None:
    value = (os.environ.get("MINIMAL_KANBAN_MCP_PUBLIC_ENDPOINT_URL") or "").strip().rstrip("/")
    return value or None


def get_board_api_url() -> str | None:
    return get_api_base_url()
