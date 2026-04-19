from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...vehicle_profile import build_vehicle_profile_patch_from_vin_research, normalize_vehicle_notes
from ..automotive_tools import AutomotiveLookupService, InternetToolError
from .base import ScenarioContext, ScenarioExecutionResult


VIN_RESEARCH_ALLOWED_DOMAINS: tuple[str, ...] = (
    "vpic.nhtsa.dot.gov",
    "get.vin",
    "www.vindecoderz.com",
    "vindecoderz.com",
    "vininfo.eu",
    "autodetective.com",
    "api.vin",
    "www.api.vin",
    "dataonesoftware.com",
    "www.dataonesoftware.com",
    "decodevin.pro",
    "www.decodevin.pro",
    "auto.vin",
    "www.auto.vin",
    "duckdecode.com",
    "www.duckdecode.com",
    "vincario.com",
    "www.vincario.com",
    "typenscheine.ch",
    "www.typenscheine.ch",
    "dauto.ch",
    "www.dauto.ch",
    "autoua.com.ua",
    "www.autoua.com.ua",
    "zapchast.com.ua",
    "www.zapchast.com.ua",
)


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

        local_research_payload = self._prefetch_local_vin_research(vin)
        if local_research_payload:
            research_payload = self._merge_research_payloads(research_payload, local_research_payload)

        runtime._store_vin_cache_entry(vin, research_payload)

        research_result = self._synthesize_vin_research(
            runtime=runtime,
            context=context,
            research_payload=research_payload,
            search_mode="exact",
        )
        research_status = self._vin_research_status(research_result)
        if research_status != "success":
            fallback_result = self._synthesize_vin_research(
                runtime=runtime,
                context=context,
                research_payload=research_payload,
                search_mode="family",
            )
            fallback_status = self._vin_research_status(fallback_result)
            if self._is_richer_vin_result(fallback_result, research_result):
                research_result = fallback_result
                research_status = fallback_status
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
        scenario_patch = self._build_card_patch(
            facts=facts,
            research_result=research_result,
            research_status=research_status,
        )
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
            patch=scenario_patch,
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
        search_mode: str,
    ) -> dict[str, Any]:
        card = context.facts.get("card") if isinstance(context.facts.get("card"), dict) else {}
        vehicle_profile = context.facts.get("vehicle_profile") if isinstance(context.facts.get("vehicle_profile"), dict) else {}
        vin = str(context.facts.get("vin", "") or "").strip().upper()
        prompt_payload = {
            "vin": vin,
            "search_mode": search_mode,
            "search_plan": {
                "step_1": "search exact VIN",
                "step_2": "search VIN prefix family and homologation records",
                "step_3": "cross-check family-level engine/transmission/drivetrain only if the exact VIN remains unavailable",
            },
            "evidence_digest": self._build_local_evidence_digest(research_payload),
            "card": {
                "title": card.get("title", ""),
                "description": card.get("description", ""),
                "vehicle": card.get("vehicle", ""),
            },
        }
        system_prompt = "\n".join(
            [
                "You are an automotive VIN research specialist.",
                "Use the web_search tool to research the VIN on the internet and cross-check multiple sources when possible.",
                "If search_mode is exact, search the exact VIN first.",
                "If search_mode is family, use local_research as the primary evidence pack and search the VIN prefix family, homologation pages, and nearby serial family records for the same platform.",
                "If local_research includes a WMI or manufacturer hint, use it to guide brand-level searching even when the exact VIN is unavailable.",
                "Prefer sources that explicitly mention the VIN, year, make, model, engine, gearbox, drivetrain, plant, displacement, or homologation family.",
                "Aim to return the strongest defensible result rather than an empty partial. If the exact VIN is not confirmed, return the best family-level or WMI-level match you can support from the evidence.",
                "When evidence_digest shows confirmed family data, use it to fill make, model, model_year, drivetrain, engine_model, gearbox_model, and plant_country if the evidence explicitly supports those fields.",
                "When evidence_digest shows a WMI manufacturer hint, you may use that as a brand clue for family-level research and include it in the final answer.",
                "Prefer a useful best-effort answer with clear warnings over a blank result.",
                "Do not invent missing facts; if a field is not supported, leave it blank.",
                "Return exactly one JSON object with keys: vin, status, make, model, model_year, engine_model, transmission, drive_type, plant_country, source_summary, source_confidence, source_links_or_refs, oem_notes, description_line, vehicle_label, warnings.",
            ]
        )
        research_text = runtime._model_client.complete_text(
            instructions=system_prompt,
            messages=[{"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)}],
            reasoning_effort="high",
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "high",
                    "filters": {
                        "allowed_domains": list(VIN_RESEARCH_ALLOWED_DOMAINS),
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
        wmi_payload = research_payload.get("wmi_payload") if isinstance(research_payload.get("wmi_payload"), dict) else {}
        if isinstance(wmi_payload, dict):
            wmi_make = str(wmi_payload.get("make") or wmi_payload.get("manufacturer") or "").strip()
            if wmi_make and not str(normalized.get("make", "") or "").strip():
                normalized["make"] = wmi_make
                warnings = normalized["warnings"] if isinstance(normalized.get("warnings"), list) else []
                if not isinstance(normalized.get("warnings"), list):
                    normalized["warnings"] = warnings
                warnings.append("Manufacturer inferred from NHTSA DecodeWMI; exact VIN decode was not confirmed.")
                if not str(normalized.get("source_summary", "") or "").strip():
                    normalized["source_summary"] = "NHTSA DecodeWMI manufacturer hint"
                source_links = normalized.get("source_links_or_refs")
                if isinstance(source_links, list):
                    source_url = str(wmi_payload.get("source_url", "") or "").strip()
                    if source_url and source_url not in source_links:
                        source_links.insert(0, source_url)
            wmi_country = str(wmi_payload.get("country") or "").strip()
            if wmi_country and not str(normalized.get("plant_country", "") or "").strip():
                normalized["plant_country"] = wmi_country
            if wmi_make and not str(normalized.get("vehicle_label", "") or "").strip():
                normalized["vehicle_label"] = wmi_make
        return normalized

    def _build_local_evidence_digest(self, research_payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(research_payload, dict):
            return {}
        results = research_payload.get("results") if isinstance(research_payload.get("results"), list) else []
        digest_items: list[dict[str, Any]] = []
        for item in results[:4]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "").strip()
            snippet = str(item.get("snippet", "") or "").strip()
            excerpt = str(item.get("excerpt", "") or item.get("page_excerpt", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            if not any((title, snippet, excerpt, url)):
                continue
            digest_items.append(
                {
                    "title": title[:180],
                    "snippet": snippet[:240],
                    "excerpt": excerpt[:360],
                    "url": url,
                    "domain": str(item.get("domain", "") or "").strip(),
                }
            )
        return {
            "source_summary": str(research_payload.get("source_summary", "") or "local VIN web research"),
            "source_confidence": research_payload.get("source_confidence", 0.0),
            "source_links_or_refs": research_payload.get("source_links_or_refs", []),
            "wmi": research_payload.get("wmi_payload") if isinstance(research_payload.get("wmi_payload"), dict) else {},
            "results": digest_items,
        }

    def _prefetch_local_vin_research(self, vin: str) -> dict[str, Any]:
        try:
            service = AutomotiveLookupService()
            payload = service.research_vin(vin, limit=6)
        except InternetToolError:
            return {}
        if not isinstance(payload, dict):
            return {}
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        links = payload.get("source_links_or_refs") if isinstance(payload.get("source_links_or_refs"), list) else []
        wmi_payload = payload.get("wmi_payload") if isinstance(payload.get("wmi_payload"), dict) else {}
        if not results and not links and not wmi_payload:
            return {}
        return payload

    def _merge_research_payloads(self, primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(primary, dict):
            return dict(secondary) if isinstance(secondary, dict) else {}
        if not isinstance(secondary, dict):
            return dict(primary)
        merged = dict(primary)
        for key in ("results", "queries"):
            if not merged.get(key) and secondary.get(key):
                merged[key] = secondary.get(key)
        for key in ("source_summary", "source_confidence", "source_links_or_refs", "source"):
            if not merged.get(key) and secondary.get(key):
                merged[key] = secondary.get(key)
        return merged

    def _is_richer_vin_result(self, candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        candidate_score = self._vin_result_score(candidate)
        current_score = self._vin_result_score(current)
        return candidate_score > current_score

    def _vin_result_score(self, payload: dict[str, Any] | None) -> int:
        if not isinstance(payload, dict):
            return 0
        score = 0
        for key in ("make", "model", "model_year", "engine_model", "transmission", "drive_type", "plant_country", "description_line", "vehicle_label"):
            if str(payload.get(key, "") or "").strip():
                score += 1
        sources = payload.get("source_links_or_refs") if isinstance(payload.get("source_links_or_refs"), list) else []
        score += min(len([item for item in sources if str(item or "").strip()]), 3)
        if str(payload.get("status", "") or "").strip().lower() == "success":
            score += 4
        elif str(payload.get("status", "") or "").strip().lower().startswith("partial"):
            score += 2
        return score

    def _vin_research_status(self, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return "failed"
        status = str(payload.get("status", "") or "").strip().lower()
        if status in {"success", "ok", "confirmed"}:
            return "success"
        if status.startswith("partial") or status in {"insufficient", "partial", "partial_success", "family", "approx", "estimated"}:
            return "insufficient"
        if any(str(payload.get(key, "") or "").strip() for key in ("make", "model", "model_year", "engine_model", "transmission", "drive_type", "plant_country")):
            return "insufficient"
        return "failed"

    def _build_card_patch(self, *, facts: dict[str, Any], research_result: dict[str, Any], research_status: str) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        source_summary = str(research_result.get("source_summary", "") or "").strip()
        if not source_summary:
            source_summary = "VIN research completed in best-effort mode."
        profile_research_result = dict(research_result)
        if not str(profile_research_result.get("source_summary", "") or "").strip():
            profile_research_result["source_summary"] = source_summary
        if profile_research_result.get("source_confidence") in (None, ""):
            profile_research_result["source_confidence"] = 0.45 if research_status != "success" else 0.78
        vehicle_profile_patch = build_vehicle_profile_patch_from_vin_research(
            profile_research_result,
            existing_profile=facts.get("vehicle_profile") if isinstance(facts.get("vehicle_profile"), dict) else {},
            current_vin=str(facts.get("vin", "") or "").strip(),
            include_vin=research_status == "success",
        )
        if vehicle_profile_patch:
            patch["vehicle_profile"] = vehicle_profile_patch

        vehicle_label = str(research_result.get("vehicle_label", "") or "").strip()
        if not vehicle_label:
            parts = [
                str(research_result.get("make", "") or "").strip(),
                str(research_result.get("model", "") or "").strip(),
                str(research_result.get("model_year", "") or "").strip(),
            ]
            vehicle_label = " ".join(part for part in parts if part).strip()
        if vehicle_label:
            patch["vehicle"] = vehicle_label

        description_line = str(research_result.get("description_line", "") or "").strip()
        if not description_line:
            summary_bits = [
                str(research_result.get(key, "") or "").strip()
                for key in ("make", "model", "model_year", "engine_model", "transmission", "drive_type")
                if str(research_result.get(key, "") or "").strip()
            ]
            if summary_bits:
                prefix = "По VIN подтверждено: " if research_status == "success" else "По VIN выполнено best-effort исследование: "
                description_line = prefix + ", ".join(summary_bits)
            else:
                description_line = "По VIN выполнено best-effort исследование"
        if description_line:
            patch["description"] = description_line if description_line.endswith((".", "!", "?")) else f"{description_line}."
        return patch
