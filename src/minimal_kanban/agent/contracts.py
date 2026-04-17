from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FactEvidence:
    name: str
    value: Any = None
    status: str = "absent"
    source: str = "unknown"
    confidence: float = 0.0
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": str(self.status or "absent"),
            "source": str(self.source or "unknown"),
            "confidence": float(self.confidence),
            "conflicts": list(self.conflicts),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EvidenceResult:
    context_kind: str
    card_id: str = ""
    confirmed_facts: dict[str, Any] = field(default_factory=dict)
    fact_evidence: dict[str, FactEvidence] = field(default_factory=dict)
    missing_data: list[str] = field(default_factory=list)
    scenario_signals: dict[str, dict[str, bool]] = field(default_factory=dict)
    sensitive_fields: list[str] = field(default_factory=list)
    allowed_write_targets: list[str] = field(default_factory=list)
    summary: str = ""
    raw_context_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_kind": self.context_kind,
            "card_id": self.card_id,
            "confirmed_facts": dict(self.confirmed_facts),
            "fact_evidence": {
                str(name): value.to_dict()
                for name, value in self.fact_evidence.items()
                if isinstance(value, FactEvidence)
            },
            "missing_data": list(self.missing_data),
            "scenario_signals": {
                str(name): {
                    "trigger_found": bool(signal.get("trigger_found")),
                    "confidence_enough": bool(signal.get("confidence_enough")),
                }
                for name, signal in self.scenario_signals.items()
                if isinstance(signal, dict)
            },
            "sensitive_fields": list(self.sensitive_fields),
            "allowed_write_targets": list(self.allowed_write_targets),
            "summary": self.summary,
            "raw_context_ref": self.raw_context_ref,
        }


@dataclass(frozen=True)
class PlanResult:
    scenario_id: str
    scenario_chain: list[str]
    execution_mode: str
    needs_external_tools: bool
    required_tools: list[str] = field(default_factory=list)
    optional_tools: list[str] = field(default_factory=list)
    tool_order: list[str] = field(default_factory=list)
    allowed_write_targets: list[str] = field(default_factory=list)
    forbidden_write_targets: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    followup_policy: dict[str, Any] = field(default_factory=dict)
    confidence_mode: str = "standard"
    write_mode: str = "patch_only"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_chain": list(self.scenario_chain),
            "execution_mode": self.execution_mode,
            "needs_external_tools": self.needs_external_tools,
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "tool_order": list(self.tool_order),
            "allowed_write_targets": list(self.allowed_write_targets),
            "forbidden_write_targets": list(self.forbidden_write_targets),
            "stop_conditions": list(self.stop_conditions),
            "followup_policy": dict(self.followup_policy),
            "confidence_mode": self.confidence_mode,
            "write_mode": self.write_mode,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: str
    source_type: str
    confidence: float
    data: dict[str, Any] = field(default_factory=dict)
    raw_ref: str = ""
    evidence_ref: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "source_type": self.source_type,
            "confidence": float(self.confidence),
            "data": dict(self.data),
            "raw_ref": self.raw_ref,
            "evidence_ref": self.evidence_ref,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PatchResult:
    card_patch: dict[str, Any] = field(default_factory=dict)
    repair_order_patch: dict[str, Any] = field(default_factory=dict)
    repair_order_works: list[dict[str, Any]] = field(default_factory=list)
    repair_order_materials: list[dict[str, Any]] = field(default_factory=list)
    append_only_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    human_review_needed: bool = False

    def is_empty(self) -> bool:
        return not any(
            (
                self.card_patch,
                self.repair_order_patch,
                self.repair_order_works,
                self.repair_order_materials,
                self.append_only_notes,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_patch": dict(self.card_patch),
            "repair_order_patch": dict(self.repair_order_patch),
            "repair_order_works": [dict(item) for item in self.repair_order_works if isinstance(item, dict)],
            "repair_order_materials": [dict(item) for item in self.repair_order_materials if isinstance(item, dict)],
            "append_only_notes": list(self.append_only_notes),
            "warnings": list(self.warnings),
            "human_review_needed": bool(self.human_review_needed),
        }


@dataclass(frozen=True)
class VerifyResult:
    applied_ok: bool
    fields_changed: list[str] = field(default_factory=list)
    manual_fields_preserved: bool = True
    scenario_completed: bool = False
    needs_followup: bool = False
    outcome_state: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    context_ref: str = ""
    followup_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied_ok": bool(self.applied_ok),
            "fields_changed": list(self.fields_changed),
            "manual_fields_preserved": bool(self.manual_fields_preserved),
            "scenario_completed": bool(self.scenario_completed),
            "needs_followup": bool(self.needs_followup),
            "outcome_state": self.outcome_state,
            "warnings": list(self.warnings),
            "context_ref": self.context_ref,
            "followup_reason": self.followup_reason,
        }


@dataclass(frozen=True)
class OrchestrationTrace:
    version: str
    trigger: dict[str, Any]
    context_snapshot_id: str
    evidence: EvidenceResult
    plan: PlanResult
    scenario_feedback: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    patch: PatchResult = field(default_factory=PatchResult)
    verify: VerifyResult = field(default_factory=lambda: VerifyResult(applied_ok=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "trigger": dict(self.trigger),
            "context_snapshot_id": self.context_snapshot_id,
            "evidence": self.evidence.to_dict(),
            "plan": self.plan.to_dict(),
            "scenario_feedback": [dict(item) for item in self.scenario_feedback if isinstance(item, dict)],
            "tool_results": [item.to_dict() for item in self.tool_results],
            "patch": self.patch.to_dict(),
            "verify": self.verify.to_dict(),
        }
