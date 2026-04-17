from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...vehicle_profile import build_vehicle_profile_patch_from_vin_research, normalize_vehicle_notes
from .base import ScenarioContext, ScenarioExecutionResult


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


@dataclass(frozen=True)
class VinEnrichmentScenarioExecutor:
    scenario_id: str = "vin_enrichment"

    def execute(self, context: ScenarioContext) -> ScenarioExecutionResult:
        runtime = context.runtime
        facts = context.facts
        if runtime is None:
            raise ValueError("VinEnrichmentScenarioExecutor requires runtime.")
        vin = str(facts.get("vin", "") or "").strip().upper()
        if not vin:
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="skipped",
                warnings=["vin missing"],
                needs_followup=True,
                followup_reason="vin_missing",
            )

        facts["vin_research_attempted"] = True
        facts["vin_decode_attempted"] = True  # compatibility for legacy runtime checks
        runtime._record_log_action(
            task_id=context.task_id,
            run_id=context.run_id,
            step=0,
            level="RUN",
            phase="tool",
            message="research_vin requested.",
        )

        cached_payload = runtime._read_vin_cache_entry(vin)
        if cached_payload:
            research_payload = dict(cached_payload)
            research_tool_payload = None
            runtime._record_log_action(
                task_id=context.task_id,
                run_id=context.run_id,
                step=1,
                level="INFO",
                phase="tool",
                message="research_vin reused from cache.",
            )
        else:
            research_tool_payload = runtime._run_autofill_tool(
                task_id=context.task_id,
                run_id=context.run_id,
                step=1,
                tool_name="research_vin",
                args={"vin": vin},
                reason="Research the VIN on the web and collect source-backed evidence before writing back to CRM",
            )
            if research_tool_payload is None:
                facts["vin_research_status"] = "failed"
                facts["vin_decode_status"] = "failed"
                runtime._record_log_action(
                    task_id=context.task_id,
                    run_id=context.run_id,
                    step=1,
                    level="WARN",
                    phase="tool",
                    message="research_vin failed.",
                )
                return ScenarioExecutionResult(
                    scenario_id=self.scenario_id,
                    status="failed",
                    facts_updates={"vin_research_status": "failed", "vin_decode_status": "failed"},
                    warnings=["vin research request failed"],
                    needs_followup=True,
                    followup_reason="vin_research_failed",
                )
            research_payload = runtime._response_data(research_tool_payload) or research_tool_payload
            runtime._store_vin_cache_entry(vin, research_payload)

        research_result = self._synthesize_vin_research(runtime=runtime, context=context, research_payload=research_payload)
        research_status = self._vin_research_status(research_result)
        facts["vin_research_status"] = research_status
        facts["vin_decode_status"] = research_status  # compatibility with legacy runtime checks
        if isinstance(facts.get("evidence_model"), dict):
            facts["evidence_model"]["external_result_sufficient"] = research_status == "success"
        if research_status == "success":
            facts["vehicle_context"] = runtime._merge_vehicle_context(
                facts["vehicle_context"],
                research_result,
            )

        orchestration_payload = {
            **research_result,
            "research_payload": research_payload,
            "research_status": research_status,
        }
        tool_results = []
        if research_tool_payload is not None:
            tool_results.append(
                runtime._build_tool_result(
                    "research_vin",
                    research_tool_payload,
                    status="success",
                    reason="Research VIN on the web and collect source-backed evidence before writing back to CRM",
                    scenario_id=self.scenario_id,
                    evidence_ref="vin",
                )
            )

        warnings = list(research_result.get("warnings", [])) if isinstance(research_result.get("warnings"), list) else []
        if research_status == "insufficient":
            warnings.append("vin research returned sparse data")
        followup_reason = "vin_research_insufficient" if research_status == "insufficient" else ("vin_research_failed" if research_status == "failed" else "")
        return ScenarioExecutionResult(
            scenario_id=self.scenario_id,
            status="success",
            tool_calls_used=0 if research_tool_payload is None else 1,
            tool_results=tool_results,
            orchestration_updates={"vin_research": orchestration_payload, "decode_vin": orchestration_payload},
            facts_updates={
                "vin_research_status": research_status,
                "vin_decode_status": research_status,
                "vehicle_context": dict(facts.get("vehicle_context") or {}),
            },
            warnings=warnings,
            needs_followup=research_status in {"insufficient", "failed"},
            followup_reason=followup_reason,
        )

    def _synthesize_vin_research(
        self,
        *,
        runtime: Any,
        context: ScenarioContext,
        research_payload: dict[str, Any],
    ) -> dict[str, Any]:
        card = context.facts.get("card") if isinstance(context.facts.get("card"), dict) else {}
        vehicle_profile = context.facts.get("vehicle_profile") if isinstance(context.facts.get("vehicle_profile"), dict) else {}
        prompt_payload = {
            "vin": context.facts.get("vin", ""),
            "card": {
                "title": card.get("title", ""),
                "description": card.get("description", ""),
                "vehicle": card.get("vehicle", ""),
            },
        }
        system_prompt = "\n".join(
            [
                "You are an automotive VIN research specialist.",
                "Use the web_search tool to research this exact VIN on the internet and cross-check multiple sources when possible.",
                "Prefer sources that explicitly mention the VIN, year, make, model, engine, gearbox, and drivetrain.",
                "Do not invent missing facts, but do not be overly conservative: if the VIN clearly maps to a vehicle, return status success with the strongest confirmed facts.",
                "Only return status insufficient when the VIN cannot be tied to a vehicle with reasonable confidence.",
                "Return exactly one JSON object with keys: vin, status, make, model, model_year, engine_model, transmission, drive_type, plant_country, source_summary, source_confidence, source_links_or_refs, oem_notes, description_line, vehicle_label, warnings.",
            ]
        )
        research_text = runtime._model_client.complete_text(
            instructions=system_prompt,
            messages=[{"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)}],
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
        normalized = _parse_json_text(research_text)
        normalized.setdefault("vin", str(context.facts.get("vin", "") or "").strip().upper())
        normalized.setdefault("status", "insufficient")
        normalized.setdefault("source_summary", str(research_payload.get("source_summary", "") or "VIN web research"))
        normalized.setdefault("source_confidence", research_payload.get("source_confidence", 0.0))
        normalized.setdefault("source_links_or_refs", research_payload.get("source_links_or_refs", []))
        normalized.setdefault("warnings", [])
        return normalized

    def _vin_research_status(self, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return "failed"
        status = str(payload.get("status", "") or "").strip().lower()
        if status in {"success", "ok", "confirmed"}:
            return "success"
        if status in {"insufficient", "partial", "partial_success"}:
            return "insufficient"
        if any(str(payload.get(key, "") or "").strip() for key in ("make", "model", "model_year", "engine_model", "transmission", "drive_type", "plant_country")):
            return "success"
        return "failed"
