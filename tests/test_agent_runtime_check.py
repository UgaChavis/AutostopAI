from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_agent_runtime.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_agent_runtime_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CheckAgentRuntimeScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script_module()

    def test_returns_cleanup_only_when_agent_status_route_is_retired(self) -> None:
        http_404 = self.module.urllib.error.HTTPError(
            url="http://127.0.0.1:41731/api/agent_status",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        with patch.object(self.module, "_request_json", side_effect=http_404), patch.object(
            self.module,
            "_supports_cleanup_only_flow",
            return_value=True,
        ):
            status, details = self.module._evaluate_agent_runtime_mode(
                base_url="http://127.0.0.1:41731",
                token="token",
                max_heartbeat_age_seconds=30.0,
            )

        self.assertEqual(status, "cleanup_only")
        self.assertEqual(details["reason"], "agent_status_route_retired")

    def test_returns_cleanup_only_when_embedded_agent_is_disabled(self) -> None:
        payload = {
            "data": {
                "status": {"last_heartbeat": ""},
                "agent": {"enabled": False, "model": ""},
            }
        }
        with patch.object(self.module, "_request_json", return_value=payload), patch.object(
            self.module,
            "_supports_cleanup_only_flow",
            return_value=True,
        ):
            status, details = self.module._evaluate_agent_runtime_mode(
                base_url="http://127.0.0.1:41731",
                token="token",
                max_heartbeat_age_seconds=30.0,
            )

        self.assertEqual(status, "cleanup_only")
        self.assertEqual(details["reason"], "embedded_agent_disabled")

    def test_returns_ok_for_live_embedded_agent(self) -> None:
        payload = {
            "data": {
                "status": {"last_heartbeat": "2026-04-16T08:00:00+00:00"},
                "agent": {"enabled": True, "model": "gpt-test"},
            }
        }
        with patch.object(self.module, "_request_json", return_value=payload), patch.object(
            self.module,
            "_heartbeat_age_seconds",
            return_value=5.0,
        ):
            status, details = self.module._evaluate_agent_runtime_mode(
                base_url="http://127.0.0.1:41731",
                token="token",
                max_heartbeat_age_seconds=30.0,
            )

        self.assertEqual(status, "ok")
        self.assertEqual(details["heartbeat_age_seconds"], "5.00")
        self.assertEqual(details["model"], "gpt-test")


if __name__ == "__main__":
    unittest.main()
