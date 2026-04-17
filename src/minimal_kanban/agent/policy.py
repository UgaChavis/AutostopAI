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
        required_tools=("decode_vin",),
        allowed_write_targets=("description", "vehicle", "vehicle_profile"),
        source_type="external_vin",
    ),
    "parts_lookup": ScenarioPolicy(
        required_tools=("find_part_numbers",),
        optional_tools=("estimate_price_ru", "lookup_part_prices"),
        allowed_write_targets=("description",),
        source_type="external_parts",
    ),
    "maintenance_lookup": ScenarioPolicy(
        required_tools=("estimate_maintenance",),
        optional_tools=("lookup_part_prices",),
        allowed_write_targets=("description",),
        source_type="external_maintenance",
    ),
    "dtc_lookup": ScenarioPolicy(
        required_tools=("decode_dtc",),
        optional_tools=("search_fault_info",),
        allowed_write_targets=("description",),
        source_type="external_diagnostic",
    ),
    "fault_research": ScenarioPolicy(
        required_tools=("search_fault_info",),
        allowed_write_targets=("description",),
        source_type="external_fault",
    ),
    "normalization": ScenarioPolicy(
        allowed_write_targets=("title", "description", "tags", "vehicle"),
        source_type="crm",
    ),
    "repair_order_assistance": ScenarioPolicy(
        allowed_write_targets=("description", "repair_order", "repair_order_works", "repair_order_materials"),
        source_type="crm",
    ),
    "board_review": ScenarioPolicy(source_type="crm"),
    "cash_review": ScenarioPolicy(source_type="crm"),
    "freeform_manual": ScenarioPolicy(source_type="crm"),
}


_TOOL_SOURCE_TYPES = {
    "decode_vin": "external_vin",
    "find_part_numbers": "external_parts",
    "search_part_numbers": "external_parts",
    "estimate_price_ru": "external_price",
    "lookup_part_prices": "external_price",
    "decode_dtc": "external_diagnostic",
    "search_fault_info": "external_fault",
    "estimate_maintenance": "external_maintenance",
    "search_web": "external_search",
    "fetch_page_excerpt": "external_page",
    "update_card": "crm_write",
    "update_repair_order": "crm_write",
    "replace_repair_order_works": "crm_write",
    "replace_repair_order_materials": "crm_write",
    "set_repair_order_status": "crm_write",
    "create_card": "crm_write",
    "move_card": "crm_write",
    "archive_card": "crm_write",
    "restore_card": "crm_write",
    "create_cashbox": "crm_write",
    "delete_cashbox": "crm_write",
    "create_cash_transaction": "crm_write",
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
            normalized_chain = ["freeform_manual"]
        recognized_chain = [item for item in normalized_chain if item in _SCENARIO_POLICIES]
        if recognized_chain:
            normalized_chain = recognized_chain
        else:
            normalized_chain = ["freeform_manual"]
        primary = next(
            (item for item in normalized_chain if item not in {"normalization", "freeform_manual"}),
            normalized_chain[0],
        )
        required_tools: list[str] = []
        optional_tools: list[str] = []
        allowed_write_targets: list[str] = []
        forbidden_write_targets: list[str] = []
        for scenario_name in normalized_chain:
            policy = self._policy_for(scenario_name)
            required_tools.extend(policy.required_tools)
            optional_tools.extend(policy.optional_tools)
            allowed_write_targets.extend(policy.allowed_write_targets)
            forbidden_write_targets.extend(policy.forbidden_write_targets)
        required_unique = self._unique(required_tools)
        optional_unique = [item for item in self._unique(optional_tools) if item not in required_unique]
        forbidden_unique = self._unique(forbidden_write_targets)
        forbidden_set = set(forbidden_unique)
        allowed_unique = [item for item in self._unique(allowed_write_targets) if item not in forbidden_set]
        stop_conditions = [f"missing_required_tool:{tool_name}" for tool_name in required_unique]
        if normalized_execution_mode == "model_loop" and allowed_unique:
            stop_conditions.append("forbid_unplanned_writes")
        followup_policy = {
            "enabled": bool(followup_enabled),
            "owner": "card_service" if followup_enabled else "",
            "mode": "adaptive_followup" if followup_enabled else "none",
        }
        confidence_mode = "standard"
        write_mode = "patch_only"
        if normalized_execution_mode == "structured_card":
            write_mode = "patch_only_additive"
        if any(item in {"vin_enrichment", "dtc_lookup"} for item in normalized_chain):
            confidence_mode = "confirmed_only"
        elif any(item in {"parts_lookup", "fault_research", "maintenance_lookup"} for item in normalized_chain):
            confidence_mode = "evidence_guided"
        return PlanResult(
            scenario_id=primary,
            scenario_chain=normalized_chain,
            execution_mode=normalized_execution_mode,
            needs_external_tools=bool(required_unique or optional_unique),
            required_tools=required_unique,
            optional_tools=optional_unique,
            tool_order=required_unique + [item for item in optional_unique if item not in required_unique],
            allowed_write_targets=allowed_unique,
            forbidden_write_targets=forbidden_unique,
            stop_conditions=stop_conditions,
            followup_policy=followup_policy,
            confidence_mode=confidence_mode,
            write_mode=write_mode,
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
        return _SCENARIO_POLICIES.get(normalized_name, _SCENARIO_POLICIES["freeform_manual"])

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
