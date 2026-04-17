from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import PatchResult, PlanResult, ToolResult


@dataclass(frozen=True)
class ScenarioPolicy:
    required_tools: tuple[str, ...] = ()
    optional_tools: tuple[str, ...] = ()
    allowed_write_targets: tuple[str, ...] = ()
    forbidden_write_targets: tuple[str, ...] = ()
    source_type: str = "crm"


_SCENARIO_POLICIES: dict[str, ScenarioPolicy] = {
    "vin_enrichment": ScenarioPolicy(
        required_tools=("research_vin",),
        allowed_write_targets=("description", "vehicle", "vehicle_profile"),
        source_type="vin_research",
    ),
}


_TOOL_SOURCE_TYPES = {
    "research_vin": "vin_research",
    "decode_vin": "vin_research",
    "get_card": "crm",
    "get_card_context": "crm",
    "update_card": "crm_write",
    "ping_connector": "crm",
}


class ToolPolicyEngine:
    def build_plan(
        self,
        *,
        scenario_chain: list[str],
        execution_mode: str,
        followup_enabled: bool,
        notes: list[str] | None = None,
    ) -> PlanResult:
        normalized_execution_mode = str(execution_mode or "model_loop").strip().lower() or "model_loop"
        normalized_chain = self._normalize_chain(scenario_chain)
        if not normalized_chain:
            normalized_chain = ["vin_enrichment"]
        normalized_chain = [item for item in normalized_chain if item in _SCENARIO_POLICIES] or ["vin_enrichment"]
        primary = normalized_chain[0]
        required_tools: list[str] = []
        allowed_write_targets: list[str] = []
        for scenario_name in normalized_chain:
            policy = self._policy_for(scenario_name)
            required_tools.extend(policy.required_tools)
            allowed_write_targets.extend(policy.allowed_write_targets)
        required_unique = self._unique(required_tools)
        allowed_unique = self._unique(allowed_write_targets)
        stop_conditions = [f"missing_required_tool:{tool_name}" for tool_name in required_unique]
        if normalized_execution_mode == "model_loop" and allowed_unique:
            stop_conditions.append("forbid_unplanned_writes")
        followup_policy = {
            "enabled": bool(followup_enabled),
            "owner": "card_service" if followup_enabled else "",
            "mode": "adaptive_followup" if followup_enabled else "none",
        }
        return PlanResult(
            scenario_id=primary,
            scenario_chain=normalized_chain,
            execution_mode=normalized_execution_mode,
            needs_external_tools=bool(required_unique),
            required_tools=required_unique,
            optional_tools=[],
            tool_order=list(required_unique),
            allowed_write_targets=allowed_unique,
            forbidden_write_targets=[],
            stop_conditions=stop_conditions,
            followup_policy=followup_policy,
            confidence_mode="confirmed_only",
            write_mode="patch_only",
            notes=list(notes or []),
        )

    def missing_required_tools(self, plan: PlanResult, tool_results: list[ToolResult]) -> list[str]:
        executed = {
            str(item.tool_name or "").strip().lower()
            for item in tool_results
            if str(item.status or "").strip().lower() == "success"
        }
        return [tool_name for tool_name in plan.required_tools if tool_name not in executed]

    def filter_patch(self, plan: PlanResult, patch: PatchResult) -> PatchResult:
        allowed = set(self._unique(plan.allowed_write_targets))
        forbidden = set(self._unique(plan.forbidden_write_targets))
        allowed.difference_update(forbidden)
        filtered_card_patch = {
            key: value
            for key, value in dict(patch.card_patch).items()
            if key in allowed and key not in forbidden
        }
        repair_order_patch = dict(patch.repair_order_patch) if "repair_order" in allowed and "repair_order" not in forbidden else {}
        repair_order_works = [dict(item) for item in patch.repair_order_works if isinstance(item, dict)] if "repair_order_works" in allowed and "repair_order_works" not in forbidden else []
        repair_order_materials = [dict(item) for item in patch.repair_order_materials if isinstance(item, dict)] if "repair_order_materials" in allowed and "repair_order_materials" not in forbidden else []
        return PatchResult(
            card_patch=filtered_card_patch,
            repair_order_patch=repair_order_patch,
            repair_order_works=repair_order_works,
            repair_order_materials=repair_order_materials,
            append_only_notes=list(patch.append_only_notes),
            warnings=list(patch.warnings),
            human_review_needed=bool(patch.human_review_needed),
        )

    def tool_source_type(self, tool_name: str, *, scenario_id: str | None = None) -> str:
        normalized_tool = str(tool_name or "").strip().lower()
        if normalized_tool in _TOOL_SOURCE_TYPES:
            return _TOOL_SOURCE_TYPES[normalized_tool]
        if scenario_id:
            return self._policy_for(scenario_id).source_type
        return "crm"

    def policy_for_scenario(self, scenario_name: str) -> dict[str, Any]:
        policy = self._policy_for(scenario_name)
        return {
            "required_tools": list(policy.required_tools),
            "optional_tools": list(policy.optional_tools),
            "allowed_write_targets": list(policy.allowed_write_targets),
            "forbidden_write_targets": list(policy.forbidden_write_targets),
            "source_type": policy.source_type,
        }

    def _policy_for(self, scenario_name: str) -> ScenarioPolicy:
        normalized_name = str(scenario_name or "").strip().lower()
        return _SCENARIO_POLICIES.get(normalized_name, _SCENARIO_POLICIES["vin_enrichment"])

    def _normalize_chain(self, scenario_chain: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in scenario_chain:
            value = str(item or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _unique(self, items: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = str(item or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
