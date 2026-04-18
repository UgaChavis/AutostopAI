from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_script_module(name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_doctor = _load_script_module("agent_doctor", "agent_doctor.py")


class AgentDoctorTests(unittest.TestCase):
    def test_local_agent_shape_is_vin_only(self) -> None:
        report = agent_doctor._check_local_agent_shape()

        self.assertTrue(report["ok"])
        self.assertEqual(report["router"]["scenario_chain"], ["vin_enrichment"])
        self.assertIn("research_vin", report["policy"]["required_tools"])
        self.assertEqual(report["vin_cache_entry"]["vin"], "1HGCM82633A004352")

    def test_openai_check_uses_model_client_contract(self) -> None:
        class DummyClient:
            def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
                self.model = model or "dummy-model"

            def complete_text(
                self,
                *,
                instructions: str,
                messages: list[dict[str, str]],
                reasoning_effort: str | None = None,
                tools: list[dict[str, object]] | None = None,
            ) -> str:
                return (
                    '{"vin":"1HGCM82633A004352","status":"success","make":"HONDA","model":"Accord",'
                    '"model_year":"2003","engine_model":"J30A4","transmission":"Automatic / 5","drive_type":"FWD",'
                    '"source_links_or_refs":["https://example.com/vin"]}'
                )

            def complete_json(
                self,
                *,
                instructions: str,
                messages: list[dict[str, str]],
                temperature: float = 0.0,
                reasoning_effort: str | None = None,
            ) -> dict[str, object]:
                return {"ok": True, "reply": "hi"}

        with patch.object(agent_doctor, "get_agent_openai_api_key", return_value="key"):
            with patch.object(agent_doctor, "OpenAIJsonAgentClient", DummyClient):
                report = agent_doctor._check_openai("gpt-5.4-mini", sample_prompt="hello")

        self.assertTrue(report["ok"])
        self.assertEqual(report["model"], "gpt-5.4-mini")
        self.assertEqual(report["result"]["reply"], "hi")

    def test_vin_check_normalizes_patch(self) -> None:
        class DummyClient:
            def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
                self.model = model or "dummy-model"

            def complete_text(
                self,
                *,
                instructions: str,
                messages: list[dict[str, str]],
                reasoning_effort: str | None = None,
                tools: list[dict[str, object]] | None = None,
            ) -> str:
                return (
                    '{"vin":"1HGCM82633A004352","status":"success","make":"HONDA","model":"Accord",'
                    '"model_year":"2003","engine_model":"J30A4","transmission":"Automatic / 5","drive_type":"FWD",'
                    '"source_links_or_refs":["https://example.com/vin"]}'
                )

            def complete_json(
                self,
                *,
                instructions: str,
                messages: list[dict[str, str]],
                temperature: float = 0.0,
                reasoning_effort: str | None = None,
                tools: list[dict[str, object]] | None = None,
            ) -> dict[str, object]:
                return {
                    "vin": "1HGCM82633A004352",
                    "status": "success",
                    "make": "HONDA",
                    "model": "Accord",
                    "model_year": "2003",
                    "engine_model": "J30A4",
                    "transmission": "Automatic / 5",
                    "drive_type": "FWD",
                    "source_links_or_refs": ["https://example.com/vin"],
                }

        with patch.object(agent_doctor, "get_agent_openai_api_key", return_value="key"):
            with patch.object(agent_doctor, "OpenAIJsonAgentClient", DummyClient):
                with patch.object(agent_doctor.AutomotiveLookupService, "research_vin", return_value={}):
                    report = agent_doctor._check_vin("1HGCM82633A004352")

        self.assertTrue(report["ok"])
        self.assertEqual(report["vehicle_profile_patch"]["make_display"], "HONDA")
        self.assertEqual(report["vehicle_profile_patch"]["data_completion_state"], "mostly_autofilled")


if __name__ == "__main__":
    unittest.main()
