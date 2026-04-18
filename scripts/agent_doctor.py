from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.agent.config import get_agent_openai_api_key, get_agent_openai_model
from minimal_kanban.agent.automotive_tools import AutomotiveLookupService
from minimal_kanban.agent.openai_client import AgentModelError, OpenAIJsonAgentClient
from minimal_kanban.agent.policy import ToolPolicyEngine
from minimal_kanban.agent.router import AgentTaskRouter
from minimal_kanban.agent.storage import AgentStorage
from minimal_kanban.vehicle_profile import build_vehicle_profile_patch_from_vin_research


def _parse_json_text(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return {}
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _check_openai(model: str, *, sample_prompt: str) -> dict[str, Any]:
    api_key = get_agent_openai_api_key()
    if not api_key:
        return {"ok": False, "error": "OPENAI_API_KEY is not configured"}
    try:
        client = OpenAIJsonAgentClient(api_key=api_key, model=model)
        result = client.complete_json(
            instructions='Return exactly one JSON object: {"ok": true, "reply": "short greeting"}',
            messages=[{"role": "user", "content": sample_prompt}],
            temperature=0.0,
            reasoning_effort="high",
        )
    except AgentModelError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "model": client.model, "result": result}


def _check_vin(vin: str) -> dict[str, Any]:
    api_key = get_agent_openai_api_key()
    if not api_key:
        return {"ok": False, "error": "OPENAI_API_KEY is not configured"}
    try:
        client = OpenAIJsonAgentClient(api_key=api_key, model=get_agent_openai_model())
        evidence_brief = client.complete_text(
            instructions=(
                "You are an automotive VIN research specialist. "
                "Use the web_search tool to research the VIN and return exactly one JSON object with keys: "
                "vin, status, make, model, model_year, engine_model, transmission, drive_type, plant_country, "
                "source_summary, source_confidence, source_links_or_refs, oem_notes, description_line, vehicle_label, warnings. "
                "Use only confirmed facts. If evidence is sparse, return status=insufficient and keep fields blank."
            ),
            messages=[{"role": "user", "content": f"Research this VIN: {vin}"}],
            reasoning_effort="low",
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "low",
                    "filters": {
                        "allowed_domains": [
                            "vpic.nhtsa.dot.gov",
                            "get.vin",
                            "www.vindecoderz.com",
                            "vindecoderz.com",
                            "vininfo.eu",
                            "autodetective.com",
                        ]
                    },
                }
            ],
        )
        decoded = _parse_json_text(evidence_brief)
        try:
            local_payload = AutomotiveLookupService().research_vin(vin, limit=4)
        except Exception:
            local_payload = {}
    except AgentModelError as exc:
        return {"ok": False, "error": str(exc)}
    merged = dict(decoded)
    if isinstance(local_payload, dict) and local_payload:
        if not merged.get("wmi_payload") and isinstance(local_payload.get("wmi_payload"), dict):
            merged["wmi_payload"] = dict(local_payload["wmi_payload"])
        for key in ("source_summary", "source_confidence", "source_links_or_refs"):
            if not merged.get(key) and local_payload.get(key):
                merged[key] = local_payload.get(key)
    patch = build_vehicle_profile_patch_from_vin_research(merged, current_vin=vin)
    return {"ok": True, "vin": vin.upper(), "decoded": decoded, "vehicle_profile_patch": patch}


def _check_local_agent_shape() -> dict[str, Any]:
    router = AgentTaskRouter()
    policy = ToolPolicyEngine()
    with tempfile.TemporaryDirectory(prefix="autostopai-doctor-") as temp_dir:
        storage = AgentStorage(base_dir=Path(temp_dir))
        storage.upsert_vin_cache_entry(
            "1HGCM82633A004352",
            {
                "vin": "1HGCM82633A004352",
                "make": "HONDA",
                "model": "Accord",
                "model_year": "2003",
            },
        )
        cache_entry = storage.get_vin_cache_entry("1hgcm82633a004352")
    plan = policy.build_plan(
        scenario_chain=["vin_enrichment"],
        execution_mode="structured_card",
        followup_enabled=True,
        notes=["agent_doctor"],
    )
    return {
        "ok": True,
        "router": {
            "classify_task": router.classify_task({"task_text": "Обогати карточку по VIN."}, {"purpose": "card_enrichment", "context": {"kind": "card", "card_id": "card-1"}}),
            "context_kind": router.context_kind({"context": {"kind": "card"}}),
            "scenario_chain": router.scenario_chain_for_task(metadata={"purpose": "card_enrichment", "context": {"kind": "card"}}, task_type="card_enrichment", context_kind="card", facts={"vin": "1HGCM82633A004352"}),
        },
        "policy": plan.to_dict(),
        "vin_cache_entry": cache_entry,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run a compact readiness check for the AutostopAI agent module.")
    parser.add_argument("--vin", default="1HGCM82633A004352", help="VIN to research during the check.")
    parser.add_argument("--model", default=get_agent_openai_model(), help="OpenAI model to test.")
    parser.add_argument("--sample-prompt", default="Say hello in one short sentence.", help="Prompt used for the OpenAI smoke check.")
    args = parser.parse_args()

    report = {
        "openai": _check_openai(args.model, sample_prompt=args.sample_prompt),
        "vin": _check_vin(args.vin),
        "local_agent": _check_local_agent_shape(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(section.get("ok") for section in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
