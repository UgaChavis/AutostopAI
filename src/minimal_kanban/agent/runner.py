from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from ..mcp.client import BoardApiClient, BoardApiTransportError, discover_board_api
from ..models import utc_now_iso
from ..vehicle_profile import normalize_vehicle_notes
from .config import (
    get_agent_board_api_url,
    get_agent_enabled,
    get_agent_max_steps,
    get_agent_max_tool_result_chars,
    get_agent_name,
    get_agent_openai_model,
    get_agent_poll_interval_seconds,
)
from .contracts import EvidenceResult, FactEvidence, OrchestrationTrace, PatchResult, PlanResult, ToolResult, VerifyResult
from .instructions import build_default_system_prompt
from .openai_client import AgentModelError, OpenAIJsonAgentClient
from .policy import ToolPolicyEngine
from .scenarios import ScenarioContext, build_default_scenario_registry
from .storage import AgentStorage
from .tools import AgentToolExecutor, ExternalToolBudgetExceeded


DEFAULT_SYSTEM_PROMPT = build_default_system_prompt()
_AUTOFILL_VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)
_AUTOFILL_DTC_PATTERN = re.compile(r"\b[PBCU][0-9]{4}\b", re.IGNORECASE)
_AUTOFILL_MILEAGE_PATTERN = re.compile(r"(?:пробег|mileage|одометр)\s*[:\-]?\s*([\d\s]{2,12})", re.IGNORECASE)
_AUTOFILL_MAINTENANCE_PATTERN = re.compile(
    r"\b(?:то|техобслуживание|техническое обслуживание|service|oil service|замена масла)\b",
    re.IGNORECASE,
)
_AUTOFILL_WAIT_HINTS = ("ожид", "в пути", "клиент дума", "согласован", "заказали", "ждем", "ждём")
_AUTOFILL_MAINTENANCE_SCOPE_HINTS = (
    "регламент",
    "замена масла",
    "oil service",
    "service",
    "масло",
    "фильтр",
    "свеч",
    "жидкост",
)
_AUTOFILL_PART_LOOKUP_STRONG_HINTS = (
    "артикул",
    "каталож",
    "oem",
    "подобрать",
    "подбор",
    "номер детали",
    "аналог",
    "цена",
    "проценить",
    "стоимость",
    "найти",
)
_AUTOFILL_SYMPTOM_HINTS = (
    "теч",
    "бежит",
    "стук",
    "шум",
    "гул",
    "вибрац",
    "троит",
    "не завод",
    "перегрев",
    "дым",
    "пина",
    "дерга",
    "рывк",
    "скрип",
    "свист",
    "ошибк",
    "антифриз",
    "не едет",
)
_AUTOFILL_PART_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("радиатор", ("радиатор", "radiator")),
    ("рычаг подвески", ("рычаг", "control arm")),
    ("стойка амортизатора", ("стойк", "амортиз", "shock", "strut")),
    ("ступичный подшипник", ("ступиц", "ступич", "bearing", "hub")),
    ("тормозные колодки", ("колодк", "pads")),
    ("тормозной диск", ("тормозн", "brake disc", "rotor")),
    ("термостат", ("термостат", "thermostat")),
    ("помпа", ("помп", "water pump")),
    ("ремень", ("ремень", "belt")),
    ("цепь грм", ("цеп", "timing chain")),
    ("масло", ("масло", "oil")),
    ("фильтр", ("фильтр", "filter")),
    ("свечи зажигания", ("свеч", "spark")),
    ("аккумулятор", ("аккумулятор", "battery")),
)


class AgentRunner:
    def __init__(
        self,
        *,
        storage: AgentStorage,
        board_api: BoardApiClient,
        model_client: OpenAIJsonAgentClient,
        logger: logging.Logger,
        actor_name: str | None = None,
        max_steps: int | None = None,
        max_tool_result_chars: int | None = None,
    ) -> None:
        self._storage = storage
        self._board_api = board_api
        self._model_client = model_client
        self._logger = logger
        self._actor_name = actor_name or get_agent_name()
        self._max_steps = max_steps or get_agent_max_steps()
        self._max_tool_result_chars = max_tool_result_chars or get_agent_max_tool_result_chars()
        self._tools = AgentToolExecutor(board_api, actor_name=self._actor_name)
        self._policy = ToolPolicyEngine()
        self._scenario_registry = build_default_scenario_registry()

    def run_once(self) -> bool:
        task = self._storage.claim_next_task()
        if task is None:
            self._storage.heartbeat(task_id=None, run_id=None)
            return False
        run_id = f"agrun_{uuid.uuid4().hex[:12]}"
        self._storage.update_status(
            running=True,
            current_task_id=task["id"],
            current_run_id=run_id,
            last_heartbeat=utc_now_iso(),
            last_run_started_at=utc_now_iso(),
            last_error="",
        )
        tool_calls = 0
        started_at = utc_now_iso()
        try:
            summary, result, display, tool_calls, orchestration = self._execute_task(task, run_id=run_id)
            completed = self._storage.complete_task(
                task_id=task["id"],
                run_id=run_id,
                summary=summary,
                result=result,
                display=display,
                tool_calls=tool_calls,
            )
            self._storage.append_run(
                {
                    "id": run_id,
                    "task_id": task["id"],
                    "status": "completed",
                    "started_at": started_at,
                    "finished_at": completed["finished_at"],
                    "source": task["source"],
                    "mode": task["mode"],
                    "task_text": task["task_text"],
                    "summary": summary,
                    "result": result,
                    "display": display,
                    "tool_calls": tool_calls,
                    "model": self._model_client.model,
                    "metadata": task.get("metadata", {}),
                    "orchestration": orchestration,
                }
            )
            self._update_board_control_runtime_after_task(
                task=task,
                orchestration=orchestration,
            )
            self._storage.update_status(
                running=False,
                current_task_id=None,
                current_run_id=None,
                last_heartbeat=utc_now_iso(),
                last_run_finished_at=completed["finished_at"],
                last_error="",
            )
            self._logger.info("agent_task_completed task_id=%s run_id=%s tool_calls=%s", task["id"], run_id, tool_calls)
            return True
        except Exception as exc:
            self._record_log_action(
                task_id=task["id"],
                run_id=run_id,
                step=tool_calls + 1,
                level="WARN",
                phase="failed",
                message=self._task_failed_message(task, exc),
            )
            failed = self._storage.fail_task(
                task_id=task["id"],
                run_id=run_id,
                error=str(exc),
                tool_calls=tool_calls,
            )
            self._storage.append_run(
                {
                    "id": run_id,
                    "task_id": task["id"],
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": failed["finished_at"],
                    "source": task["source"],
                    "mode": task["mode"],
                    "task_text": task["task_text"],
                    "summary": "",
                    "result": "",
                    "error": str(exc),
                    "tool_calls": tool_calls,
                    "model": self._model_client.model,
                    "metadata": task.get("metadata", {}),
                }
            )
            self._update_board_control_runtime_after_failure(task=task, error=str(exc))
            self._storage.update_status(
                running=False,
                current_task_id=None,
                current_run_id=None,
                last_heartbeat=utc_now_iso(),
                last_run_finished_at=failed["finished_at"],
                last_error=str(exc),
            )
            self._logger.exception("agent_task_failed task_id=%s run_id=%s error=%s", task["id"], run_id, exc)
            return True

    def _execute_task(self, task: dict[str, Any], *, run_id: str) -> tuple[str, str, dict[str, Any], int, dict[str, Any]]:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        self._tools.reset_task_budget()
        task_type = self._classify_task(task, metadata)
        context_kind = self._context_kind(metadata)
        return self._execute_orchestrated_task(
            task,
            run_id=run_id,
            metadata=metadata,
            task_type=task_type,
            context_kind=context_kind,
        )

    def _execute_orchestrated_task(
        self,
        task: dict[str, Any],
        *,
        run_id: str,
        metadata: dict[str, Any],
        task_type: str,
        context_kind: str,
    ) -> tuple[str, str, dict[str, Any], int, dict[str, Any]]:
        task_id = str(task.get("id", "") or "").strip()
        tool_calls = 0
        context_payload: dict[str, Any] = {}
        context_data: dict[str, Any] = {}
        context_snapshot_id = f"ctx:{task_id}:board"
        self._record_log_action(
            task_id=task_id,
            run_id=run_id,
            step=0,
            level="RUN",
            phase="start",
            message=self._task_started_message(metadata),
        )
        self._record_log_action(
            task_id=task_id,
            run_id=run_id,
            step=0,
            level="INFO",
            phase="analysis",
            message=self._task_analysis_message(metadata),
        )
        if self._should_preload_context(task_type=task_type, metadata=metadata, context_kind=context_kind):
            card_id = self._cleanup_card_id(metadata) or str(metadata.get("card_id", "") or "").strip()
            context_args = {"card_id": card_id, "event_limit": 20, "include_repair_order_text": True}
            context_tool_name, context_payload = self._load_card_autofill_context(card_id=card_id, context_args=context_args)
            context_data = self._response_data(context_payload)
            context_snapshot_id = self._build_context_snapshot_id(task_id=task_id, card_id=card_id, context_tool_name=context_tool_name)
            tool_calls += 1
            self._record_action(
                task_id=task_id,
                run_id=run_id,
                step=tool_calls,
                tool_name=context_tool_name,
                args=context_args if context_tool_name == "get_card_context" else {"card_id": card_id},
                reason="Read focused context before evidence extraction and planning",
                result_payload=context_payload,
            )
        evidence_result, facts = self._build_orchestration_evidence(
            task=task,
            metadata=metadata,
            task_type=task_type,
            context_kind=context_kind,
            context_data=context_data,
            raw_context_ref=context_snapshot_id,
        )
        plan = self._build_orchestration_plan(
            metadata=metadata,
            task_type=task_type,
            context_kind=context_kind,
            evidence=evidence_result,
            facts=facts,
        )
        scenario_feedback: list[dict[str, Any]] = []
        if plan.execution_mode == "structured_card":
            summary, result, display, delta, tool_results, patch_result, verify_result = self._execute_card_autofill_task(
                task,
                run_id=run_id,
                metadata=metadata,
                facts=facts,
                plan=plan,
            )
            scenario_feedback = list(facts.get("_scenario_feedback") or [])
        else:
            summary, result, display, delta, tool_results, patch_result, verify_result = self._execute_decision_loop_task(
                task,
                run_id=run_id,
                metadata=metadata,
                task_type=task_type,
                context_kind=context_kind,
                evidence=evidence_result,
                plan=plan,
                preloaded_context=context_payload,
            )
        tool_calls += delta
        evidence_result = self._enrich_evidence_with_runtime_facts(evidence_result, facts=facts)
        trace = OrchestrationTrace(
            version="agent_orchestrator_v1",
            trigger={
                "task_id": task_id,
                "source": str(task.get("source", "") or "").strip(),
                "mode": str(task.get("mode", "") or "").strip(),
                "purpose": str(metadata.get("purpose", "") or "").strip(),
                "task_type": task_type,
                "requested_by": str(metadata.get("requested_by", "") or "").strip(),
            },
            context_snapshot_id=context_snapshot_id,
            evidence=evidence_result,
            plan=plan,
            scenario_feedback=scenario_feedback,
            tool_results=tool_results,
            patch=patch_result,
            verify=verify_result,
        )
        return summary, result, display, tool_calls, trace.to_dict()

    def _should_preload_context(self, *, task_type: str, metadata: dict[str, Any], context_kind: str) -> bool:
        if str(metadata.get("purpose", "") or "").strip().lower() in {"card_autofill", "board_control"}:
            return True
        if context_kind == "card":
            return True
        return task_type in {"card_cleanup", "vin_decode", "parts_lookup", "maintenance_estimate", "dtc_lookup", "repair_order_assist"}

    def _build_context_snapshot_id(self, *, task_id: str, card_id: str, context_tool_name: str) -> str:
        normalized_card_id = str(card_id or "").strip() or "board"
        return f"ctx:{task_id}:{normalized_card_id}:{context_tool_name}"

    def _build_orchestration_evidence(
        self,
        *,
        task: dict[str, Any],
        metadata: dict[str, Any],
        task_type: str,
        context_kind: str,
        context_data: dict[str, Any],
        raw_context_ref: str,
    ) -> tuple[EvidenceResult, dict[str, Any]]:
        allowed_write_targets = self._suggest_allowed_write_targets(task_type=task_type, context_kind=context_kind, metadata=metadata)
        if context_kind == "card" and context_data:
            facts = self._analyze_card_autofill_context(context_data, task_text=str(task.get("task_text", "") or ""))
            facts["task_type"] = task_type
            facts["context_kind"] = context_kind
            autofill_plan = self._build_card_autofill_plan(facts)
            facts["autofill_plan"] = autofill_plan
            facts["selected_scenarios"] = autofill_plan.get("scenarios", [])
            confirmed_facts = {
                "vin": str(facts.get("vin", "") or "").strip(),
                "mileage": str(facts.get("mileage", "") or "").strip(),
                "dtc_codes": list(facts.get("dtc_codes") or [])[:3],
                "part_queries": list(facts.get("part_queries") or [])[:3],
                "waiting_state": bool(facts.get("waiting_state")),
                "vehicle_context": dict(facts.get("vehicle_context") or {}),
            }
            summary_bits = [
                f"task_type={task_type}",
                f"vin={'yes' if confirmed_facts['vin'] else 'no'}",
                f"dtc={len(confirmed_facts['dtc_codes'])}",
                f"parts={len(confirmed_facts['part_queries'])}",
            ]
            evidence_result = EvidenceResult(
                context_kind=context_kind,
                card_id=self._cleanup_card_id(metadata),
                confirmed_facts=confirmed_facts,
                fact_evidence=self._build_card_fact_evidence(facts, confirmed_facts=confirmed_facts),
                missing_data=list(facts.get("missing_vehicle_fields") or []),
                scenario_signals=dict(facts.get("scenario_evidence") or {}),
                sensitive_fields=["prices", "part_numbers", "customer_notes", "manual_vehicle_fields"],
                allowed_write_targets=allowed_write_targets,
                summary=", ".join(summary_bits),
                raw_context_ref=raw_context_ref,
            )
            return evidence_result, facts
        generic_facts = {"task_type": task_type, "context_kind": context_kind}
        evidence_result = EvidenceResult(
            context_kind=context_kind or "board",
            confirmed_facts={"task_type": task_type, "mode": str(task.get("mode", "") or "").strip()},
            fact_evidence=self._build_generic_fact_evidence(task_type=task_type, context_kind=context_kind, task=task),
            missing_data=[],
            scenario_signals={},
            sensitive_fields=["cash_amounts", "manual_notes"],
            allowed_write_targets=allowed_write_targets,
            summary=f"task_type={task_type}, context={context_kind or 'board'}",
            raw_context_ref=raw_context_ref,
        )
        return evidence_result, generic_facts

    def _build_card_fact_evidence(
        self,
        facts: dict[str, Any],
        *,
        confirmed_facts: dict[str, Any],
    ) -> dict[str, FactEvidence]:
        vehicle_context = confirmed_facts.get("vehicle_context") if isinstance(confirmed_facts.get("vehicle_context"), dict) else {}
        missing_vehicle_fields = list(facts.get("missing_vehicle_fields") or [])
        evidence_model = facts.get("evidence_model") if isinstance(facts.get("evidence_model"), dict) else {}
        part_queries = list(confirmed_facts.get("part_queries") or [])
        return {
            "vin": FactEvidence(
                name="vin",
                value=confirmed_facts.get("vin", ""),
                status="confirmed" if confirmed_facts.get("vin") else "absent",
                source="card_context",
                confidence=1.0 if confirmed_facts.get("vin") else 0.0,
            ),
            "mileage": FactEvidence(
                name="mileage",
                value=confirmed_facts.get("mileage", ""),
                status="confirmed" if confirmed_facts.get("mileage") else "absent",
                source="vehicle_profile_or_repair_order",
                confidence=0.9 if confirmed_facts.get("mileage") else 0.0,
            ),
            "dtc_codes": FactEvidence(
                name="dtc_codes",
                value=list(confirmed_facts.get("dtc_codes") or []),
                status="confirmed" if confirmed_facts.get("dtc_codes") else "absent",
                source="card_context",
                confidence=0.95 if confirmed_facts.get("dtc_codes") else 0.0,
            ),
            "part_queries": FactEvidence(
                name="part_queries",
                value=part_queries,
                status="inferred" if part_queries else "absent",
                source="heuristic_text_extraction",
                confidence=0.7 if part_queries else 0.0,
                notes=["Derived from symptom and card text analysis."] if part_queries else [],
            ),
            "waiting_state": FactEvidence(
                name="waiting_state",
                value=bool(confirmed_facts.get("waiting_state")),
                status="weak_signal" if confirmed_facts.get("waiting_state") else "absent",
                source="heuristic_text_extraction",
                confidence=0.6 if confirmed_facts.get("waiting_state") else 0.0,
            ),
            "vehicle_context": FactEvidence(
                name="vehicle_context",
                value=dict(vehicle_context),
                status="confirmed" if vehicle_context else "absent",
                source="card_context_aggregate",
                confidence=0.85 if vehicle_context else 0.0,
                conflicts=["missing:" + field_name for field_name in missing_vehicle_fields[:4]],
            ),
            "external_result_sufficient": FactEvidence(
                name="external_result_sufficient",
                value=bool(evidence_model.get("external_result_sufficient")),
                status="confirmed" if evidence_model.get("external_result_sufficient") else "absent",
                source="external_tool_results",
                confidence=1.0 if evidence_model.get("external_result_sufficient") else 0.0,
            ),
        }

    def _build_generic_fact_evidence(
        self,
        *,
        task_type: str,
        context_kind: str,
        task: dict[str, Any],
    ) -> dict[str, FactEvidence]:
        return {
            "task_type": FactEvidence(
                name="task_type",
                value=task_type,
                status="confirmed",
                source="task_metadata",
                confidence=1.0,
            ),
            "mode": FactEvidence(
                name="mode",
                value=str(task.get("mode", "") or "").strip(),
                status="confirmed" if str(task.get("mode", "") or "").strip() else "absent",
                source="task_metadata",
                confidence=1.0 if str(task.get("mode", "") or "").strip() else 0.0,
            ),
            "context_kind": FactEvidence(
                name="context_kind",
                value=context_kind or "board",
                status="confirmed",
                source="task_metadata",
                confidence=1.0,
            ),
        }

    def _enrich_evidence_with_runtime_facts(self, evidence: EvidenceResult, *, facts: dict[str, Any]) -> EvidenceResult:
        fact_evidence = dict(evidence.fact_evidence)
        related_cards = facts.get("related_cards") if isinstance(facts.get("related_cards"), list) else []
        if related_cards:
            fact_evidence["related_cards"] = FactEvidence(
                name="related_cards",
                value=[str(item.get("id", "") or "").strip() for item in related_cards[:6] if isinstance(item, dict)],
                status="inferred",
                source="board_search",
                confidence=min(0.85, 0.45 + 0.05 * len(related_cards)),
                notes=[f"Found {len(related_cards)} related cards during runtime context expansion."],
            )
        vin_status = str(facts.get("vin_decode_status", "") or "").strip().lower()
        if vin_status in {"insufficient", "failed"} and isinstance(facts.get("vehicle_context"), dict):
            fact_evidence["vin_fallback_context"] = FactEvidence(
                name="vin_fallback_context",
                value=dict(facts.get("vehicle_context") or {}),
                status="inferred" if self._has_enough_vehicle_context(
                    dict(facts.get("vehicle_context") or {}),
                    missing_vehicle_fields=list(facts.get("missing_vehicle_fields") or []),
                ) else "weak_signal",
                source="card_context_fallback",
                confidence=0.55 if vin_status == "insufficient" else 0.35,
                notes=["Used because VIN decoding did not return enough confirmed vehicle facts."],
            )
        if fact_evidence == evidence.fact_evidence:
            return evidence
        return EvidenceResult(
            context_kind=evidence.context_kind,
            card_id=evidence.card_id,
            confirmed_facts=dict(evidence.confirmed_facts),
            fact_evidence=fact_evidence,
            missing_data=list(evidence.missing_data),
            scenario_signals=dict(evidence.scenario_signals),
            sensitive_fields=list(evidence.sensitive_fields),
            allowed_write_targets=list(evidence.allowed_write_targets),
            summary=evidence.summary,
            raw_context_ref=evidence.raw_context_ref,
        )

    def _build_orchestration_plan(
        self,
        *,
        metadata: dict[str, Any],
        task_type: str,
        context_kind: str,
        evidence: EvidenceResult,
        facts: dict[str, Any],
    ) -> PlanResult:
        scenario_chain = self._scenario_chain_for_task(
            metadata=metadata,
            task_type=task_type,
            context_kind=context_kind,
            facts=facts,
        )
        notes: list[str] = []
        if evidence.missing_data:
            notes.append("missing_data:" + ", ".join(evidence.missing_data[:4]))
        manual_structured_card = (
            context_kind == "card"
            and bool(str(metadata.get("quick_template", "") or "").strip())
            and task_type in {"card_cleanup", "vin_decode", "parts_lookup", "maintenance_estimate", "dtc_lookup"}
        )
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        if purpose == "card_autofill":
            notes.append("followup_owner=card_service")
            execution_mode = "structured_card"
        elif purpose == "board_control":
            notes.append("background_owner=board_control")
            execution_mode = "structured_card"
        elif manual_structured_card:
            notes.append("manual_card_orchestrator=structured")
            execution_mode = "structured_card"
        else:
            execution_mode = "model_loop"
        plan = self._policy.build_plan(
            scenario_chain=scenario_chain,
            execution_mode=execution_mode,
            followup_enabled=bool(purpose == "card_autofill"),
            notes=notes,
        )
        evidence_targets = [str(item or "").strip() for item in evidence.allowed_write_targets if str(item or "").strip()]
        if evidence_targets:
            allowed_targets = [item for item in plan.allowed_write_targets if item in evidence_targets]
            forbidden_targets = [item for item in plan.forbidden_write_targets if item not in allowed_targets]
            return PlanResult(
                scenario_id=plan.scenario_id,
                scenario_chain=list(plan.scenario_chain),
                execution_mode=plan.execution_mode,
                needs_external_tools=plan.needs_external_tools,
                required_tools=list(plan.required_tools),
                optional_tools=list(plan.optional_tools),
                tool_order=list(plan.tool_order),
                allowed_write_targets=allowed_targets,
                forbidden_write_targets=forbidden_targets,
                stop_conditions=list(plan.stop_conditions),
                followup_policy=dict(plan.followup_policy),
                confidence_mode=plan.confidence_mode,
                write_mode=plan.write_mode,
                notes=list(plan.notes),
            )
        return plan

    def _scenario_chain_for_task(
        self,
        *,
        metadata: dict[str, Any],
        task_type: str,
        context_kind: str,
        facts: dict[str, Any],
    ) -> list[str]:
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        autofill_plan = facts.get("autofill_plan") if isinstance(facts.get("autofill_plan"), dict) else {}
        autofill_scenarios = [
            str(item.get("name", "") or "").strip().lower()
            for item in (autofill_plan.get("scenarios") if isinstance(autofill_plan.get("scenarios"), list) else [])
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        ]
        if purpose == "card_autofill":
            return autofill_scenarios or ["normalization"]
        if purpose == "board_control":
            filtered = [item for item in autofill_scenarios if item in {"vin_enrichment", "normalization"}]
            if filtered:
                return filtered
            if str(facts.get("vin", "") or "").strip():
                return ["vin_enrichment", "normalization"]
            return ["normalization"]
        if task_type == "board_review":
            return ["board_review"]
        if task_type == "cash_review":
            return ["cash_review"]
        if task_type == "repair_order_assist":
            return ["repair_order_assistance"]
        if context_kind == "card":
            if task_type == "vin_decode":
                return [item for item in autofill_scenarios if item in {"vin_enrichment", "normalization"}] or ["vin_enrichment", "normalization"]
            if task_type == "parts_lookup":
                return [item for item in autofill_scenarios if item in {"vin_enrichment", "parts_lookup", "normalization"}] or ["parts_lookup", "normalization"]
            if task_type == "maintenance_estimate":
                return [item for item in autofill_scenarios if item in {"vin_enrichment", "maintenance_lookup", "normalization"}] or ["maintenance_lookup", "normalization"]
            if task_type == "dtc_lookup":
                return [item for item in autofill_scenarios if item in {"dtc_lookup", "fault_research", "normalization"}] or ["dtc_lookup", "normalization"]
            if task_type == "card_cleanup":
                return autofill_scenarios or ["normalization"]
        return ["freeform_manual"]

    def _suggest_allowed_write_targets(self, *, task_type: str, context_kind: str, metadata: dict[str, Any] | None = None) -> list[str]:
        purpose = str((metadata or {}).get("purpose", "") or "").strip().lower()
        if context_kind == "card":
            if purpose == "board_control":
                return ["description", "vehicle", "vehicle_profile"]
            if task_type == "repair_order_assist":
                return ["description", "repair_order", "repair_order_works", "repair_order_materials"]
            return ["title", "description", "tags", "vehicle", "vehicle_profile"]
        return []

    def _execute_decision_loop_task(
        self,
        task: dict[str, Any],
        *,
        run_id: str,
        metadata: dict[str, Any],
        task_type: str,
        context_kind: str,
        evidence: EvidenceResult,
        plan: PlanResult,
        preloaded_context: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, Any], int, list[ToolResult], PatchResult, VerifyResult]:
        prompt_override = self._storage.read_prompt_text().strip()
        memory_text = self._storage.read_memory_text().strip()
        system_prompt = DEFAULT_SYSTEM_PROMPT
        if prompt_override and prompt_override != DEFAULT_SYSTEM_PROMPT:
            system_prompt = f"{system_prompt}\n\nLocal instructions:\n{prompt_override}"
        if memory_text:
            system_prompt = f"{system_prompt}\n\nPersistent memory:\n{memory_text}"
        system_prompt = (
            f"{system_prompt}\n\nAvailable tools:\n"
            f"{self._tools.describe_for_prompt(task_type=task_type, context_kind=context_kind)}"
        )
        system_prompt = f"{system_prompt}\n\n{self._contract_prompt_block(plan=plan, evidence=evidence)}"
        cleanup_task = task_type == "card_cleanup"
        cleanup_card_id = self._cleanup_card_id(metadata)
        cleanup_update_applied = False
        cleanup_apply_prompt_sent = False
        applied_updates: list[str] = []
        tool_results: list[ToolResult] = []
        patch_result = PatchResult()
        verify_result = VerifyResult(applied_ok=False)
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": self._build_user_task_message(task, metadata, task_type=task_type),
            }
        ]
        if preloaded_context:
            messages.append(
                {
                    "role": "user",
                    "content": f"READ CONTEXT SNAPSHOT:\n{self._tool_result_for_model('get_card_context', preloaded_context)}",
                }
            )
        tool_calls = 0
        for step in range(1, self._max_steps + 1):
            self._storage.heartbeat(task_id=task["id"], run_id=run_id)
            decision = self._model_client.next_step(system_prompt=system_prompt, messages=messages)
            decision_type = str(decision.get("type", "") or "").strip().lower()
            if decision_type == "final":
                apply_args = self._extract_card_update_apply(decision, cleanup_card_id=cleanup_card_id)
                if apply_args is not None:
                    tool_calls += 1
                    apply_args, apply_result, current_patch, verify_result = self._execute_contract_write_tool(
                        tool_name="update_card",
                        args=apply_args,
                        plan=plan,
                        cleanup_card_id=cleanup_card_id,
                    )
                    patch_result = self._merge_patch_results(patch_result, current_patch)
                    cleanup_update_applied = True
                    applied_updates.extend(self._summarize_applied_update(apply_args, apply_result))
                    tool_results.append(
                        self._build_tool_result(
                            "update_card",
                            apply_result,
                            status="success",
                            reason="Runner applied structured card update from final response",
                            scenario_id=plan.scenario_id,
                            evidence_ref=evidence.raw_context_ref,
                        )
                    )
                    self._record_action(
                        task_id=task["id"],
                        run_id=run_id,
                        step=step,
                        tool_name="update_card",
                        args=apply_args,
                        reason="Runner applied structured card update from final response",
                        result_payload=apply_result,
                    )
                if cleanup_task and cleanup_card_id and not cleanup_update_applied and not cleanup_apply_prompt_sent:
                    messages.append(
                        {
                            "role": "user",
                            "content": self._card_cleanup_apply_instruction(cleanup_card_id),
                        }
                    )
                    cleanup_apply_prompt_sent = True
                    continue
                missing_required = self._policy.missing_required_tools(plan, tool_results)
                if missing_required:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Policy gate: before the final answer you must execute the required tools for the current scenario: "
                                + ", ".join(missing_required)
                                + "."
                            ),
                        }
                    )
                    continue
                summary = str(decision.get("summary", "") or "").strip() or "Task completed."
                result = str(decision.get("result", "") or "").strip() or summary
                display = self._normalize_display_payload(decision, summary=summary, result=result)
                display = self._append_applied_updates(display, applied_updates)
                self._record_log_action(
                    task_id=task["id"],
                    run_id=run_id,
                    step=step,
                    level="DONE",
                    phase="completed",
                    message=self._task_completed_message(metadata, summary=summary, applied_updates=applied_updates),
                )
                verify_result = self._finalize_verify_result(plan=plan, verify=verify_result, tool_results=tool_results)
                return summary, result, display, tool_calls, tool_results, patch_result, verify_result
            if decision_type != "tool":
                raise AgentModelError("Agent model returned neither a tool call nor a final answer.")
            tool_name = str(decision.get("tool", "") or "").strip()
            args = decision.get("args")
            if not isinstance(args, dict):
                args = {}
            reason = str(decision.get("reason", "") or "").strip()
            tool_calls += 1
            if tool_name in {"update_card", "update_repair_order", "replace_repair_order_works", "replace_repair_order_materials"}:
                args, result_payload, current_patch, verify_result = self._execute_contract_write_tool(
                    tool_name=tool_name,
                    args=args,
                    plan=plan,
                    cleanup_card_id=cleanup_card_id,
                )
                patch_result = self._merge_patch_results(patch_result, current_patch)
            else:
                result_payload = self._tools.execute(tool_name, args)
            if cleanup_task and tool_name == "update_card" and str(args.get("card_id", "") or "").strip() == cleanup_card_id:
                cleanup_update_applied = True
                applied_updates.extend(self._summarize_applied_update(args, result_payload))
            tool_results.append(
                self._build_tool_result(
                    tool_name,
                    result_payload,
                    status="success",
                    reason=reason,
                    scenario_id=plan.scenario_id,
                    evidence_ref=evidence.raw_context_ref,
                )
            )
            self._record_action(
                task_id=task["id"],
                run_id=run_id,
                step=step,
                tool_name=tool_name,
                args=args,
                reason=reason,
                result_payload=result_payload,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"type": "tool", "tool": tool_name, "args": args, "reason": reason},
                        ensure_ascii=False,
                    ),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"TOOL RESULT {tool_name}:\n{self._tool_result_for_model(tool_name, result_payload)}",
                }
            )
        raise AgentModelError(f"Agent exceeded max steps ({self._max_steps}) without returning a final answer.")

    def _execute_card_autofill_task(
        self,
        task: dict[str, Any],
        *,
        run_id: str,
        metadata: dict[str, Any],
        facts: dict[str, Any],
        plan: PlanResult,
    ) -> tuple[str, str, dict[str, Any], int, list[ToolResult], PatchResult, VerifyResult]:
        card_id = self._cleanup_card_id(metadata) or str(metadata.get("card_id", "") or "").strip()
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        if not card_id:
            raise AgentModelError("structured card task requires metadata.context.card_id.")
        tool_calls = 0
        applied_updates: list[str] = []
        tool_results: list[ToolResult] = []
        if facts.get("vin"):
            self._record_log_action(
                task_id=task["id"],
                run_id=run_id,
                step=tool_calls,
                level="INFO",
                phase="analysis",
                message="VIN found.",
            )
        if purpose != "board_control" and self._should_load_card_autofill_related_cards(facts):
            related_args = {
                "query": self._related_cards_query(facts),
                "include_archived": True,
                "limit": 6,
            }
            related_payload = self._tools.execute("search_cards", related_args)
            tool_calls += 1
            self._record_action(
                task_id=task["id"],
                run_id=run_id,
                step=tool_calls,
                tool_name="search_cards",
                args=related_args,
                reason="Collect short board context for the same VIN or vehicle before autofill",
                result_payload=related_payload,
            )
            facts["related_cards"] = self._extract_related_cards_from_search(
                card_id=card_id,
                payload=self._response_data(related_payload),
            )
            if facts["related_cards"]:
                self._record_log_action(
                    task_id=task["id"],
                    run_id=run_id,
                    step=tool_calls,
                    level="INFO",
                    phase="analysis",
                    message=f"Контекст доски: найдено связанных карточек — {len(facts['related_cards'])}.",
                )
        plan_payload = facts.get("autofill_plan") if isinstance(facts.get("autofill_plan"), dict) else {}
        scenarios = plan_payload.get("scenarios") if isinstance(plan_payload.get("scenarios"), list) else []
        if not scenarios:
            scenarios = [{"name": name, "label": str(name or "").upper(), "cost": 0} for name in plan.scenario_chain]
            facts["selected_scenarios"] = scenarios
            facts["autofill_plan"] = {"scenarios": scenarios, "skipped": [], "budget_left": 0}
        self._record_log_action(
            task_id=task["id"],
            run_id=run_id,
            step=tool_calls,
            level="INFO",
            phase="analysis",
            message=self._build_card_autofill_plan_message(scenarios, facts=facts),
        )
        self._record_card_autofill_plan_diagnostics(
            task_id=task["id"],
            run_id=run_id,
            step=tool_calls,
            facts=facts,
        )
        orchestration_results: dict[str, Any] = {}
        scenario_warnings: list[str] = []
        scenario_followup_requested = False
        scenario_followup_reason = ""
        scenario_feedback: list[dict[str, Any]] = []
        for scenario in scenarios:
            scenario_name = str(scenario.get("name", "") or "").strip().lower()
            executor = self._scenario_registry.get(scenario_name)
            if executor is None:
                continue
            scenario_result = executor.execute(
                ScenarioContext(
                    scenario_id=scenario_name,
                    task_id=str(task["id"]),
                    run_id=run_id,
                    metadata=metadata,
                    facts=facts,
                    scenario_payload=scenario if isinstance(scenario, dict) else {},
                    runtime=self,
                )
            )
            tool_calls += int(scenario_result.tool_calls_used)
            if scenario_result.orchestration_updates:
                orchestration_results.update(scenario_result.orchestration_updates)
            if scenario_result.facts_updates:
                facts.update(scenario_result.facts_updates)
            if scenario_result.tool_results:
                tool_results.extend(scenario_result.tool_results)
            scenario_feedback.append(
                {
                    "scenario_id": scenario_name,
                    "status": str(scenario_result.status or "").strip(),
                    "tool_calls_used": int(scenario_result.tool_calls_used or 0),
                    "needs_followup": bool(scenario_result.needs_followup),
                    "followup_reason": str(getattr(scenario_result, "followup_reason", "") or "").strip(),
                    "notes": [str(item or "").strip() for item in scenario_result.notes if str(item or "").strip()][:5],
                    "warnings": [str(item or "").strip() for item in scenario_result.warnings if str(item or "").strip()][:5],
                }
            )
            if scenario_result.notes:
                for note in scenario_result.notes[:3]:
                    note_text = str(note or "").strip()
                    if not note_text:
                        continue
                    self._record_log_action(
                        task_id=task["id"],
                        run_id=run_id,
                        step=max(tool_calls, 1),
                        level="INFO",
                        phase="analysis",
                        message=note_text,
                    )
            if scenario_result.warnings:
                normalized_warnings = [
                    str(item or "").strip()
                    for item in scenario_result.warnings
                    if str(item or "").strip()
                ]
                scenario_warnings.extend(normalized_warnings)
                for warning_text in normalized_warnings[:3]:
                    self._record_log_action(
                        task_id=task["id"],
                        run_id=run_id,
                        step=max(tool_calls, 1),
                        level="WARN",
                        phase="analysis",
                        message=warning_text,
                    )
            if scenario_result.needs_followup:
                scenario_followup_requested = True
                if not scenario_followup_reason:
                    scenario_followup_reason = str(getattr(scenario_result, "followup_reason", "") or "").strip()
        facts["_scenario_feedback"] = scenario_feedback
        update_args, display_sections = self._compose_card_autofill_update(
            card_id=card_id,
            facts=facts,
            orchestration_results=orchestration_results,
        )
        patch_result = PatchResult(card_patch={})
        verify_result = VerifyResult(applied_ok=False)
        if update_args is not None:
            update_args, update_result, current_patch, verify_result = self._execute_contract_write_tool(
                tool_name="update_card",
                args=update_args,
                plan=plan,
                cleanup_card_id=card_id,
            )
            patch_result = self._merge_patch_results(patch_result, current_patch)
            tool_calls += 1
            applied_updates.extend(self._summarize_applied_update(update_args, update_result))
            tool_results.append(
                self._build_tool_result(
                    "update_card",
                    update_result,
                    status="success",
                    reason="Apply deterministic autofill enrichment to the current card",
                    scenario_id=plan.scenario_id,
                    evidence_ref="card_patch",
                )
            )
            self._record_action(
                task_id=task["id"],
                run_id=run_id,
                step=tool_calls,
                tool_name="update_card",
                args=update_args,
                reason="Apply deterministic autofill enrichment to the current card",
                result_payload=update_result,
            )
            if update_args.get("vehicle_profile") or update_args.get("vehicle"):
                self._record_log_action(
                    task_id=task["id"],
                    run_id=run_id,
                    step=tool_calls,
                    level="INFO",
                    phase="update",
                    message="fields updated.",
                )
        else:
            verify_result = self._finalize_verify_result(plan=plan, verify=verify_result, tool_results=tool_results)
        verify_result = self._merge_verify_feedback(
            verify_result,
            warnings=scenario_warnings,
            needs_followup=scenario_followup_requested,
            followup_reason=scenario_followup_reason,
        )
        summary = self._autofill_result_summary(applied_updates, orchestration_results, facts=facts)
        display = {
            "emoji": "",
            "title": "Автосопровождение",
            "summary": summary,
            "tone": "success" if applied_updates else "info",
            "sections": display_sections[:5],
            "actions": [],
        }
        verify_result = self._finalize_verify_result(plan=plan, verify=verify_result, tool_results=tool_results)
        verify_result = self._verify_card_autofill_goal(
            plan=plan,
            verify=verify_result,
            facts=facts,
            orchestration_results=orchestration_results,
        )
        self._record_log_action(
            task_id=task["id"],
            run_id=run_id,
            step=max(tool_calls, 1),
            level="DONE",
            phase="completed",
            message=self._task_completed_message(metadata, summary=summary, applied_updates=applied_updates),
        )
        return summary, summary, display, tool_calls, tool_results, patch_result, verify_result

    def _contract_prompt_block(self, *, plan: PlanResult, evidence: EvidenceResult) -> str:
        lines = [
            "Contract orchestration:",
            f"- execution_mode: {plan.execution_mode}",
            f"- scenario_id: {plan.scenario_id}",
            f"- scenario_chain: {', '.join(plan.scenario_chain) if plan.scenario_chain else 'none'}",
            f"- confidence_mode: {plan.confidence_mode}",
            f"- write_mode: {plan.write_mode}",
            f"- required_tools: {', '.join(plan.required_tools) if plan.required_tools else 'none'}",
            f"- optional_tools: {', '.join(plan.optional_tools) if plan.optional_tools else 'none'}",
            f"- allowed_write_targets: {', '.join(plan.allowed_write_targets) if plan.allowed_write_targets else 'none'}",
            f"- evidence_summary: {evidence.summary or 'n/a'}",
        ]
        if evidence.missing_data:
            lines.append("- missing_data: " + ", ".join(evidence.missing_data[:5]))
        lines.extend(
            [
                "- Follow the server contract: read -> evidence -> plan -> tools -> patch -> write -> verify.",
                "- Do not finish a scenario without its required tools.",
                "- Write only to the allowed targets and preserve manual data outside those targets.",
                "- If no safe write is needed, return a final answer without a write tool.",
            ]
        )
        return "\n".join(lines)

    def _execute_contract_write_tool(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        plan: PlanResult,
        cleanup_card_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], PatchResult, VerifyResult]:
        normalized_tool = str(tool_name or "").strip()
        if normalized_tool == "update_card":
            card_id = str(args.get("card_id", "") or cleanup_card_id or "").strip()
            if not card_id:
                raise AgentModelError("update_card requires card_id in contract writer.")
            patch = PatchResult(
                card_patch={
                    key: value
                    for key, value in args.items()
                    if key in {"title", "description", "tags", "vehicle", "vehicle_profile"}
                }
            )
            filtered_patch = self._policy.filter_patch(plan, patch)
            if not filtered_patch.card_patch:
                raise AgentModelError("Contract policy rejected card write outside allowed targets.")
            write_args = {"card_id": card_id, **filtered_patch.card_patch}
            if plan.execution_mode == "structured_card":
                write_args = self._normalize_card_autofill_update(write_args)
            before_state = self._read_verification_state(card_id)
            result_payload = self._tools.execute("update_card", write_args)
            verify = self._verify_contract_write(
                tool_name=normalized_tool,
                card_id=card_id,
                before_state=before_state,
                patch=filtered_patch,
                plan=plan,
            )
            return write_args, result_payload, filtered_patch, verify
        if normalized_tool == "update_repair_order":
            card_id = str(args.get("card_id", "") or cleanup_card_id or "").strip()
            if not card_id:
                raise AgentModelError("update_repair_order requires card_id in contract writer.")
            patch = PatchResult(repair_order_patch=dict(args.get("repair_order") or {}))
            filtered_patch = self._policy.filter_patch(plan, patch)
            if not filtered_patch.repair_order_patch:
                raise AgentModelError("Contract policy rejected repair order write outside allowed targets.")
            before_state = self._read_verification_state(card_id)
            current_repair_order = before_state.get("repair_order") if isinstance(before_state.get("repair_order"), dict) else {}
            merged_repair_order = dict(current_repair_order)
            merged_repair_order.update(filtered_patch.repair_order_patch)
            write_args = {"card_id": card_id, "repair_order": merged_repair_order}
            result_payload = self._tools.execute("update_repair_order", write_args)
            verify = self._verify_contract_write(
                tool_name=normalized_tool,
                card_id=card_id,
                before_state=before_state,
                patch=filtered_patch,
                plan=plan,
            )
            return write_args, result_payload, filtered_patch, verify
        if normalized_tool in {"replace_repair_order_works", "replace_repair_order_materials"}:
            card_id = str(args.get("card_id", "") or cleanup_card_id or "").strip()
            if not card_id:
                raise AgentModelError(f"{normalized_tool} requires card_id in contract writer.")
            rows = [dict(item) for item in (args.get("rows") if isinstance(args.get("rows"), list) else []) if isinstance(item, dict)]
            patch = PatchResult(
                repair_order_works=rows if normalized_tool == "replace_repair_order_works" else [],
                repair_order_materials=rows if normalized_tool == "replace_repair_order_materials" else [],
            )
            filtered_patch = self._policy.filter_patch(plan, patch)
            expected_rows = filtered_patch.repair_order_works if normalized_tool == "replace_repair_order_works" else filtered_patch.repair_order_materials
            if not expected_rows:
                raise AgentModelError("Contract policy rejected repair order rows write outside allowed targets.")
            before_state = self._read_verification_state(card_id)
            write_args = {"card_id": card_id, "rows": expected_rows}
            result_payload = self._tools.execute(normalized_tool, write_args)
            verify = self._verify_contract_write(
                tool_name=normalized_tool,
                card_id=card_id,
                before_state=before_state,
                patch=filtered_patch,
                plan=plan,
            )
            return write_args, result_payload, filtered_patch, verify
        result_payload = self._tools.execute(normalized_tool, args)
        return args, result_payload, PatchResult(), VerifyResult(applied_ok=False)

    def _read_verification_state(self, card_id: str) -> dict[str, Any]:
        state: dict[str, Any] = {}
        try:
            context_payload = self._board_api.get_card_context(card_id, event_limit=5, include_repair_order_text=True)
            state = self._response_data(context_payload)
        except Exception:
            state = {}
        try:
            card_payload = self._board_api.get_card(card_id)
            card_state = self._response_data(card_payload)
        except Exception:
            card_state = {}
        if "card" not in state:
            state = {"card": state} if isinstance(state, dict) else {}
        if isinstance(card_state, dict):
            incoming_card = card_state.get("card") if isinstance(card_state.get("card"), dict) else card_state
            current_card = state.get("card") if isinstance(state.get("card"), dict) else {}
            if isinstance(incoming_card, dict):
                merged_card = dict(current_card)
                merged_card.update(incoming_card)
                state["card"] = merged_card
        if "card" not in state:
            state = {"card": state} if isinstance(state, dict) else {}
        card = state.get("card") if isinstance(state.get("card"), dict) else {}
        if "repair_order" not in state and isinstance(card, dict) and isinstance(card.get("repair_order"), dict):
            state["repair_order"] = dict(card.get("repair_order") or {})
        return state

    def _verify_contract_write(
        self,
        *,
        tool_name: str,
        card_id: str,
        before_state: dict[str, Any],
        patch: PatchResult,
        plan: PlanResult,
    ) -> VerifyResult:
        after_state = self._read_verification_state(card_id)
        warnings: list[str] = []
        fields_changed: list[str] = []
        manual_fields_preserved = True
        scenario_completed = False
        expected_targets = 0
        before_card = before_state.get("card") if isinstance(before_state.get("card"), dict) else {}
        after_card = after_state.get("card") if isinstance(after_state.get("card"), dict) else {}
        before_repair_order = before_state.get("repair_order") if isinstance(before_state.get("repair_order"), dict) else {}
        after_repair_order = after_state.get("repair_order") if isinstance(after_state.get("repair_order"), dict) else {}
        if tool_name == "update_card":
            expected_targets = len(patch.card_patch)
            for field_name, expected_value in patch.card_patch.items():
                if field_name == "vehicle_profile" and isinstance(expected_value, dict):
                    actual_profile = after_card.get("vehicle_profile") if isinstance(after_card.get("vehicle_profile"), dict) else {}
                    if all(self._values_equal(actual_profile.get(key), value) for key, value in expected_value.items()):
                        fields_changed.append("vehicle_profile")
                    else:
                        warnings.append("vehicle_profile verification mismatch")
                    continue
                actual_value = after_card.get(field_name)
                if field_name == "description" and self._description_patch_applied(actual_value, expected_value):
                    fields_changed.append(field_name)
                elif self._values_equal(actual_value, expected_value):
                    fields_changed.append(field_name)
                else:
                    warnings.append(f"{field_name} verification mismatch")
            if "description" not in patch.card_patch:
                previous_description = str(before_card.get("description", "") or "").strip()
                current_description = str(after_card.get("description", "") or "").strip()
                if previous_description != current_description:
                    manual_fields_preserved = False
                    warnings.append("description changed outside planned patch")
            scenario_completed = (len(fields_changed) == expected_targets) or patch.is_empty()
        elif tool_name == "update_repair_order":
            expected_targets = len(patch.repair_order_patch)
            for field_name, expected_value in patch.repair_order_patch.items():
                if self._values_equal(after_repair_order.get(field_name), expected_value):
                    fields_changed.append(field_name)
                else:
                    warnings.append(f"repair_order.{field_name} verification mismatch")
            scenario_completed = len(fields_changed) == expected_targets
        elif tool_name in {"replace_repair_order_works", "replace_repair_order_materials"}:
            expected_rows = patch.repair_order_works if tool_name == "replace_repair_order_works" else patch.repair_order_materials
            expected_targets = 1 if expected_rows else 0
            actual_rows = after_repair_order.get("works" if tool_name == "replace_repair_order_works" else "materials")
            if isinstance(actual_rows, list) and len(actual_rows) == len(expected_rows):
                fields_changed.append("repair_order_works" if tool_name == "replace_repair_order_works" else "repair_order_materials")
            else:
                warnings.append(f"{tool_name} verification mismatch")
            scenario_completed = len(fields_changed) == expected_targets
        else:
            scenario_completed = False
        non_target_card_fields = {"title", "description", "tags", "vehicle"} - set(patch.card_patch)
        for field_name in non_target_card_fields:
            if field_name and not self._values_equal(before_card.get(field_name), after_card.get(field_name)):
                manual_fields_preserved = False
                warnings.append(f"{field_name} changed outside planned patch")
        applied_ok = scenario_completed
        return VerifyResult(
            applied_ok=applied_ok,
            fields_changed=fields_changed,
            manual_fields_preserved=manual_fields_preserved,
            scenario_completed=scenario_completed,
            needs_followup=False,
            outcome_state="write_applied" if applied_ok else "write_unverified",
            warnings=warnings,
            context_ref=f"verify:{card_id}",
        )

    def _description_patch_applied(self, actual_value: Any, expected_value: Any) -> bool:
        actual = " ".join(str(actual_value or "").split()).casefold()
        expected = " ".join(str(expected_value or "").split()).casefold()
        if not expected:
            return not actual
        if actual == expected:
            return True
        return expected in actual

    def _finalize_verify_result(self, *, plan: PlanResult, verify: VerifyResult, tool_results: list[ToolResult]) -> VerifyResult:
        missing_required = self._policy.missing_required_tools(plan, tool_results)
        warnings = list(verify.warnings)
        followup_reason = str(verify.followup_reason or "").strip()
        requested_followup = bool(verify.needs_followup)
        if missing_required:
            warnings.append("missing required tools: " + ", ".join(missing_required))
            if not followup_reason:
                followup_reason = "missing_required_tools"
        scenario_completed = bool(verify.scenario_completed and not missing_required) or (not plan.required_tools and verify.applied_ok)
        if not scenario_completed and not plan.allowed_write_targets and not missing_required:
            scenario_completed = True
        needs_followup = bool(plan.followup_policy.get("enabled")) and (requested_followup or bool(missing_required) or not scenario_completed)
        if missing_required:
            outcome_state = "blocked_missing_required_tools"
        elif not verify.manual_fields_preserved:
            outcome_state = "needs_human_review"
        elif scenario_completed and verify.applied_ok:
            outcome_state = "completed_confirmed"
        elif scenario_completed:
            outcome_state = "completed_no_write"
        elif verify.applied_ok:
            outcome_state = "completed_partial"
        else:
            outcome_state = "blocked_no_progress"
        return VerifyResult(
            applied_ok=bool(verify.applied_ok),
            fields_changed=list(verify.fields_changed),
            manual_fields_preserved=bool(verify.manual_fields_preserved),
            scenario_completed=scenario_completed,
            needs_followup=needs_followup,
            outcome_state=outcome_state,
            warnings=warnings,
            context_ref=verify.context_ref,
            followup_reason=followup_reason,
        )

    def _merge_verify_feedback(
        self,
        verify: VerifyResult,
        *,
        warnings: list[str] | None = None,
        needs_followup: bool = False,
        followup_reason: str = "",
    ) -> VerifyResult:
        merged_warnings = list(verify.warnings)
        for item in warnings or []:
            warning = str(item or "").strip()
            if warning:
                merged_warnings.append(warning)
        merged_reason = str(verify.followup_reason or "").strip() or str(followup_reason or "").strip()
        return VerifyResult(
            applied_ok=bool(verify.applied_ok),
            fields_changed=list(verify.fields_changed),
            manual_fields_preserved=bool(verify.manual_fields_preserved),
            scenario_completed=bool(verify.scenario_completed),
            needs_followup=bool(verify.needs_followup) or bool(needs_followup),
            outcome_state=str(verify.outcome_state or "").strip() or "unknown",
            warnings=merged_warnings,
            context_ref=verify.context_ref,
            followup_reason=merged_reason,
        )

    def _verify_card_autofill_goal(
        self,
        *,
        plan: PlanResult,
        verify: VerifyResult,
        facts: dict[str, Any],
        orchestration_results: dict[str, Any],
    ) -> VerifyResult:
        warnings = list(verify.warnings)
        followup_reason = str(verify.followup_reason or "").strip()
        outcome_state = str(verify.outcome_state or "").strip() or "unknown"
        scenario_completed = bool(verify.scenario_completed)
        needs_followup = bool(verify.needs_followup)
        scenario_chain = [str(item or "").strip().lower() for item in plan.scenario_chain if str(item or "").strip()]
        primary = str(plan.scenario_id or "").strip().lower()
        if "vin_enrichment" in scenario_chain and str(facts.get("vin", "") or "").strip():
            vin_status = str(facts.get("vin_decode_status", "") or "").strip().lower()
            if vin_status == "insufficient":
                warnings.append("vin enrichment blocked by sparse decoder output")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "vin_decode_insufficient"
                outcome_state = "blocked_missing_source_data"
            elif vin_status == "failed":
                warnings.append("vin enrichment failed before confirmed vehicle facts were produced")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "vin_decode_failed"
                outcome_state = "blocked_missing_source_data"
        if "parts_lookup" in scenario_chain:
            part_lookup = orchestration_results.get("find_part_numbers")
            parts_evidence = self._scenario_evidence(facts, "parts_lookup")
            if parts_evidence["trigger_found"] and not isinstance(part_lookup, dict):
                warnings.append("parts lookup did not produce an external result")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "parts_lookup_failed"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
            elif isinstance(part_lookup, dict) and not self._part_lookup_has_useful_result(part_lookup):
                warnings.append("parts lookup completed without reliable candidate parts")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "parts_lookup_insufficient"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
        if primary == "parts_lookup" and orchestration_results.get("find_part_numbers") and outcome_state == "completed_no_write":
            outcome_state = "completed_partial"
        if "dtc_lookup" in scenario_chain:
            dtc_lookup = orchestration_results.get("decode_dtc")
            dtc_evidence = self._scenario_evidence(facts, "dtc_lookup")
            if dtc_evidence["trigger_found"] and not isinstance(dtc_lookup, dict):
                warnings.append("dtc lookup did not produce an external result")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "dtc_lookup_failed"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
            elif isinstance(dtc_lookup, dict) and not self._search_payload_has_useful_result(dtc_lookup):
                warnings.append("dtc lookup completed without a reliable diagnostic excerpt")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "dtc_lookup_insufficient"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
        if "fault_research" in scenario_chain:
            fault_lookup = orchestration_results.get("search_fault_info")
            fault_evidence = self._scenario_evidence(facts, "fault_research")
            if fault_evidence["trigger_found"] and fault_evidence["confidence_enough"] and not isinstance(fault_lookup, dict) and not facts.get("waiting_state"):
                warnings.append("fault research did not produce an external result")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "fault_research_failed"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
            elif isinstance(fault_lookup, dict) and not self._search_payload_has_useful_result(fault_lookup):
                warnings.append("fault research completed without a reliable symptom result")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "fault_research_insufficient"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
        if "maintenance_lookup" in scenario_chain:
            maintenance_lookup = orchestration_results.get("estimate_maintenance")
            maintenance_evidence = self._scenario_evidence(facts, "maintenance_lookup")
            if maintenance_evidence["trigger_found"] and not isinstance(maintenance_lookup, dict):
                warnings.append("maintenance lookup did not produce an external result")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "maintenance_lookup_failed"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
            elif isinstance(maintenance_lookup, dict) and not self._maintenance_lookup_has_useful_result(maintenance_lookup):
                warnings.append("maintenance lookup completed without a usable service plan")
                scenario_completed = False
                needs_followup = bool(plan.followup_policy.get("enabled"))
                followup_reason = followup_reason or "maintenance_lookup_insufficient"
                outcome_state = "completed_partial" if verify.applied_ok else "blocked_missing_source_data"
        if primary == "fault_research" and orchestration_results.get("search_fault_info") and outcome_state == "completed_no_write":
            outcome_state = "completed_partial"
        return VerifyResult(
            applied_ok=bool(verify.applied_ok),
            fields_changed=list(verify.fields_changed),
            manual_fields_preserved=bool(verify.manual_fields_preserved),
            scenario_completed=scenario_completed,
            needs_followup=needs_followup,
            outcome_state=outcome_state,
            warnings=warnings,
            context_ref=verify.context_ref,
            followup_reason=followup_reason,
        )

    def _part_lookup_has_useful_result(self, payload: dict[str, Any]) -> bool:
        part_numbers = payload.get("part_numbers") if isinstance(payload.get("part_numbers"), list) else []
        if any(isinstance(item, dict) and str(item.get("value", "") or "").strip() for item in part_numbers):
            return True
        return self._search_payload_has_useful_result(payload)

    def _search_payload_has_useful_result(self, payload: dict[str, Any]) -> bool:
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        for item in results:
            if not isinstance(item, dict):
                continue
            if str(item.get("snippet", "") or "").strip() or str(item.get("title", "") or "").strip():
                return True
        return False

    def _maintenance_lookup_has_useful_result(self, payload: dict[str, Any]) -> bool:
        works = payload.get("works") if isinstance(payload.get("works"), list) else []
        materials = payload.get("materials") if isinstance(payload.get("materials"), list) else []
        notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
        return bool(works or materials or notes)

    def _build_tool_result(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        status: str,
        reason: str,
        scenario_id: str,
        evidence_ref: str,
    ) -> ToolResult:
        return ToolResult(
            tool_name=str(tool_name or "").strip(),
            status=str(status or "success").strip().lower(),
            source_type=self._policy.tool_source_type(tool_name, scenario_id=scenario_id),
            confidence=self._tool_confidence(tool_name, payload),
            data=self._tool_contract_data(tool_name, payload),
            raw_ref=f"{scenario_id}:{tool_name}",
            evidence_ref=str(evidence_ref or "").strip(),
            reason=str(reason or "").strip(),
        )

    def _tool_confidence(self, tool_name: str, payload: dict[str, Any]) -> float:
        data = self._response_data(payload)
        normalized_tool = str(tool_name or "").strip().lower()
        if normalized_tool == "decode_vin":
            status = self._vin_decode_status(data)
            return 0.92 if status == "success" else (0.45 if status == "insufficient" else 0.05)
        if normalized_tool in {"find_part_numbers", "search_part_numbers"}:
            return 0.82 if list(data.get("part_numbers") or []) else 0.25
        if normalized_tool in {"estimate_price_ru", "lookup_part_prices"}:
            return 0.78 if isinstance(data.get("price_summary"), dict) else 0.22
        if normalized_tool == "decode_dtc":
            return 0.84 if list(data.get("results") or []) else 0.25
        if normalized_tool == "search_fault_info":
            return 0.7 if list(data.get("results") or []) else 0.2
        if normalized_tool == "estimate_maintenance":
            return 0.74 if list(data.get("works") or []) else 0.25
        if normalized_tool.startswith("update_") or normalized_tool.startswith("replace_"):
            return 1.0
        return 0.65

    def _tool_contract_data(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._response_data(payload)
        if tool_name == "update_card":
            return {
                "changed": data.get("changed"),
                "changed_fields": data.get("meta", {}).get("changed_fields") if isinstance(data.get("meta"), dict) else data.get("changed"),
            }
        if tool_name in {"update_repair_order", "replace_repair_order_works", "replace_repair_order_materials"}:
            return {"ok": bool(payload.get("ok", True)), "card_id": data.get("card_id") or payload.get("card_id")}
        if tool_name in {"find_part_numbers", "search_part_numbers"}:
            return {
                "part_numbers": list(data.get("part_numbers") or [])[:5],
                "vehicle_context": data.get("vehicle_context"),
            }
        if tool_name in {"estimate_price_ru", "lookup_part_prices"}:
            return {"price_summary": data.get("price_summary"), "results_total": len(data.get("results") or [])}
        if tool_name in {"decode_dtc", "search_fault_info"}:
            return {"results_total": len(data.get("results") or []), "query": data.get("query") or data.get("code")}
        if tool_name == "decode_vin":
            return {
                "vin": data.get("vin"),
                "make": data.get("make"),
                "model": data.get("model"),
                "model_year": data.get("model_year"),
            }
        if tool_name == "estimate_maintenance":
            return {
                "service_type": data.get("service_type"),
                "works_total": len(data.get("works") or []),
                "materials_total": len(data.get("materials") or []),
            }
        return data if isinstance(data, dict) else {}

    def _values_equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(right, ensure_ascii=False, sort_keys=True)
        if isinstance(left, list) and isinstance(right, list):
            return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(right, ensure_ascii=False, sort_keys=True)
        return left == right

    def _merge_patch_results(self, left: PatchResult, right: PatchResult) -> PatchResult:
        merged_card_patch = dict(left.card_patch)
        merged_card_patch.update(right.card_patch)
        merged_repair_order_patch = dict(left.repair_order_patch)
        merged_repair_order_patch.update(right.repair_order_patch)
        return PatchResult(
            card_patch=merged_card_patch,
            repair_order_patch=merged_repair_order_patch,
            repair_order_works=[*left.repair_order_works, *right.repair_order_works],
            repair_order_materials=[*left.repair_order_materials, *right.repair_order_materials],
            append_only_notes=[*left.append_only_notes, *right.append_only_notes],
            warnings=[*left.warnings, *right.warnings],
            human_review_needed=bool(left.human_review_needed or right.human_review_needed),
        )

    def _load_card_autofill_context(
        self,
        *,
        card_id: str,
        context_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        try:
            return "get_card_context", self._tools.execute("get_card_context", context_args)
        except Exception:
            card_payload = self._board_api.get_card(card_id)
            card_data = self._response_data(card_payload)
            card = card_data.get("card") if isinstance(card_data.get("card"), dict) else card_data
            context: dict[str, Any] = {
                "card": dict(card) if isinstance(card, dict) else {"id": card_id},
                "events": [],
            }
            if hasattr(self._board_api, "get_repair_order"):
                try:
                    repair_order_payload = self._board_api.get_repair_order(card_id)
                    repair_order_data = self._response_data(repair_order_payload)
                    repair_order = (
                        repair_order_data.get("repair_order")
                        if isinstance(repair_order_data.get("repair_order"), dict)
                        else repair_order_data
                    )
                    if isinstance(repair_order, dict):
                        context["card"]["repair_order"] = repair_order
                except Exception:
                    pass
            if hasattr(self._board_api, "get_repair_order_text"):
                try:
                    repair_order_text_payload = self._board_api.get_repair_order_text(card_id)
                    repair_order_text_data = self._response_data(repair_order_text_payload)
                    if isinstance(repair_order_text_data, dict):
                        context["repair_order_text"] = repair_order_text_data
                except Exception:
                    pass
            return "get_card", {"ok": True, "data": context}

    def _should_load_card_autofill_related_cards(self, facts: dict[str, Any]) -> bool:
        vehicle_context = facts.get("vehicle_context") if isinstance(facts.get("vehicle_context"), dict) else {}
        return bool(
            facts.get("vin")
            or str(vehicle_context.get("vehicle", "") or "").strip()
        )

    def _related_cards_query(self, facts: dict[str, Any]) -> str:
        vin = str(facts.get("vin", "") or "").strip().upper()
        if vin:
            return vin
        vehicle_context = facts.get("vehicle_context") if isinstance(facts.get("vehicle_context"), dict) else {}
        return str(vehicle_context.get("vehicle", "") or "").strip()

    def _extract_related_cards_from_search(self, *, card_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
        related: list[dict[str, Any]] = []
        for item in cards:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "") or "").strip()
            if not item_id or item_id == card_id:
                continue
            related.append(
                {
                    "id": item_id,
                    "vehicle": str(item.get("vehicle", "") or "").strip(),
                    "title": str(item.get("title", "") or "").strip(),
                    "column": str(item.get("column_label", "") or item.get("column", "") or "").strip(),
                }
            )
            if len(related) >= 3:
                break
        return related

    def _run_autofill_tool(
        self,
        *,
        task_id: str,
        run_id: str,
        step: int,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
    ) -> dict[str, Any] | None:
        try:
            payload = self._tools.execute(tool_name, args)
        except ExternalToolBudgetExceeded as exc:
            payload = {
                "ok": False,
                "error": str(exc),
                "data": {
                    "partial": True,
                    "error_code": "external_budget_exceeded",
                    "tool_name": tool_name,
                },
                "meta": {
                    "partial": True,
                    "error_code": "external_budget_exceeded",
                    "tool_name": tool_name,
                },
            }
            self._record_action(
                task_id=task_id,
                run_id=run_id,
                step=step,
                tool_name=tool_name,
                args=args,
                reason=reason,
                result_payload=payload,
            )
            self._record_log_action(
                task_id=task_id,
                run_id=run_id,
                step=step,
                level="WARN",
                phase="tool",
                message=f"{tool_name}: external web budget exhausted; scenario left partial.",
            )
            return payload
        except Exception as exc:
            self._record_log_action(
                task_id=task_id,
                run_id=run_id,
                step=step,
                level="WARN",
                phase="tool",
                message=f"{tool_name}: {str(exc or '').strip() or 'ошибка внешнего шага.'}",
            )
            return None
        self._record_action(
            task_id=task_id,
            run_id=run_id,
            step=step,
            tool_name=tool_name,
            args=args,
            reason=reason,
            result_payload=payload,
        )
        completion_message = self._autofill_tool_completion_message(tool_name, self._response_data(payload) or payload)
        if completion_message:
            self._record_log_action(
                task_id=task_id,
                run_id=run_id,
                step=step,
                level="INFO",
                phase="tool",
                message=completion_message,
            )
        return payload

    def _normalize_display_payload(
        self,
        decision: dict[str, Any],
        *,
        summary: str,
        result: str,
    ) -> dict[str, Any]:
        raw_display = decision.get("display")
        payload = raw_display if isinstance(raw_display, dict) else {}

        def _clean_text(value: Any, *, limit: int = 400) -> str:
            text = str(value or "").strip()
            if not text:
                return ""
            return text[:limit].strip()

        def _clean_items(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            items: list[str] = []
            for entry in value:
                text = _clean_text(entry, limit=220)
                if text:
                    items.append(text)
                if len(items) >= 8:
                    break
            return items

        sections: list[dict[str, Any]] = []
        if isinstance(payload.get("sections"), list):
            for entry in payload["sections"]:
                if not isinstance(entry, dict):
                    continue
                section = {
                    "title": _clean_text(entry.get("title"), limit=72),
                    "body": _clean_text(entry.get("body"), limit=500),
                    "items": _clean_items(entry.get("items")),
                }
                if section["title"] or section["body"] or section["items"]:
                    sections.append(section)
                if len(sections) >= 6:
                    break

        emoji = _clean_text(payload.get("emoji"), limit=6)
        title = _clean_text(payload.get("title"), limit=96) or _clean_text(summary, limit=96)
        lead = _clean_text(payload.get("summary"), limit=320)
        tone = _clean_text(payload.get("tone"), limit=16).lower()
        if tone not in {"info", "success", "warning", "error"}:
            tone = "success"
        actions = _clean_items(payload.get("actions"))[:4]
        normalized = {
            "emoji": emoji,
            "title": title,
            "summary": lead,
            "tone": tone,
            "sections": sections,
            "actions": actions,
        }
        if normalized["title"] or normalized["summary"] or normalized["sections"] or normalized["actions"]:
            return normalized
        return {
            "emoji": "",
            "title": _clean_text(summary, limit=96),
            "summary": _clean_text(result, limit=500),
            "tone": "success",
            "sections": [],
            "actions": [],
        }

    def _preview_payload(self, payload: dict[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(text) <= self._max_tool_result_chars:
            return text
        return f"{text[: self._max_tool_result_chars]}... [truncated]"

    def _response_data(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        return payload

    def _response_meta(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        meta = payload.get("meta")
        return meta if isinstance(meta, dict) else {}

    def _tool_payload_error_code(self, payload: Any) -> str:
        data = self._response_data(payload)
        meta = self._response_meta(payload)
        return str(meta.get("error_code") or data.get("error_code") or "").strip().lower()

    def _is_partial_tool_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        data = self._response_data(payload)
        meta = self._response_meta(payload)
        return bool(meta.get("partial") or data.get("partial"))

    def _is_budget_exceeded_payload(self, payload: Any) -> bool:
        return self._tool_payload_error_code(payload) == "external_budget_exceeded"

    def _record_action(
        self,
        *,
        task_id: str,
        run_id: str,
        step: int,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
        result_payload: dict[str, Any],
    ) -> None:
        started_at = utc_now_iso()
        finished_at = utc_now_iso()
        self._storage.append_action(
            {
                "id": f"agact_{uuid.uuid4().hex[:12]}",
                "task_id": task_id,
                "run_id": run_id,
                "step": step,
                "kind": "tool",
                "tool": tool_name,
                "args": args,
                "reason": reason,
                "started_at": started_at,
                "finished_at": finished_at,
                "result_preview": self._preview_payload(result_payload),
            }
        )

    def _record_log_action(
        self,
        *,
        task_id: str,
        run_id: str,
        step: int,
        level: str,
        phase: str,
        message: str,
    ) -> None:
        text = str(message or "").strip()
        if not text:
            return
        timestamp = utc_now_iso()
        self._storage.append_action(
            {
                "id": f"aglog_{uuid.uuid4().hex[:12]}",
                "task_id": task_id,
                "run_id": run_id,
                "step": step,
                "kind": "log",
                "level": str(level or "INFO").strip().upper(),
                "phase": str(phase or "").strip().lower(),
                "message": text,
                "started_at": timestamp,
                "finished_at": timestamp,
                "result_preview": text,
            }
        )

    def _task_started_message(self, metadata: dict[str, Any]) -> str:
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        if purpose == "card_autofill":
            trigger = str(metadata.get("trigger", "") or "").strip().lower()
            if trigger == "adaptive_followup":
                return "Повторный проход автосопровождения запущен."
            return "Первый проход автосопровождения запущен."
        return "Задача агента запущена."

    def _task_analysis_message(self, metadata: dict[str, Any]) -> str:
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        if str(context.get("kind", "") or "").strip().lower() == "card":
            return "Начат анализ карточки."
        return "Начат анализ доски."

    def _task_completed_message(self, metadata: dict[str, Any], *, summary: str, applied_updates: list[str]) -> str:
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        if purpose == "card_autofill":
            return "Карточка обновлена." if applied_updates else "Изменений не обнаружено."
        text = str(summary or "").strip()
        return text or "Задача завершена."

    def _task_failed_message(self, task: dict[str, Any], error: Exception) -> str:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        if purpose == "card_autofill":
            return "Ошибка автосопровождения."
        message = str(error or "").strip()
        return message or "Ошибка выполнения задачи."

    def _tool_result_for_model(self, tool_name: str, payload: dict[str, Any]) -> str:
        compact = payload if isinstance(payload, dict) else {}
        data = self._response_data(compact)
        if tool_name == "review_board":
            summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
            alerts = data.get("alerts") if isinstance(data.get("alerts"), list) else []
            priorities = data.get("priority_cards") if isinstance(data.get("priority_cards"), list) else []
            return self._preview_payload(
                {
                    "summary": summary,
                    "alerts": alerts[:5],
                    "priority_cards": priorities[:5],
                    "text": data.get("text", "") or compact.get("text", ""),
                }
            )
        if tool_name == "get_card_context":
            card = data.get("card") if isinstance(data.get("card"), dict) else data
            vehicle_profile = card.get("vehicle_profile") if isinstance(card.get("vehicle_profile"), dict) else {}
            vehicle_profile_compact = (
                card.get("vehicle_profile_compact")
                if isinstance(card.get("vehicle_profile_compact"), dict)
                else vehicle_profile
            )
            repair_order = card.get("repair_order") if isinstance(card.get("repair_order"), dict) else {}
            return self._preview_payload(
                {
                    "card": {
                        "id": card.get("id"),
                        "vehicle": card.get("vehicle"),
                        "title": card.get("title"),
                        "description": card.get("description"),
                        "column": card.get("column"),
                        "tags": card.get("tags"),
                        "ai_autofill_prompt": card.get("ai_autofill_prompt"),
                        "ai_autofill_log": (card.get("ai_autofill_log") or [])[-8:],
                        "vin": vehicle_profile.get("vin") or repair_order.get("vin"),
                    },
                    "known_vehicle_facts": {
                        "vin": vehicle_profile_compact.get("vin") or vehicle_profile.get("vin"),
                        "make": vehicle_profile_compact.get("make_display") or vehicle_profile.get("make_display"),
                        "model": vehicle_profile_compact.get("model_display") or vehicle_profile.get("model_display"),
                        "year": vehicle_profile_compact.get("production_year") or vehicle_profile.get("production_year"),
                        "engine": vehicle_profile_compact.get("engine_model") or vehicle_profile.get("engine_model"),
                        "gearbox": vehicle_profile_compact.get("gearbox_model") or vehicle_profile.get("gearbox_model"),
                        "drivetrain": vehicle_profile_compact.get("drivetrain") or vehicle_profile.get("drivetrain"),
                    },
                    "vehicle_profile": vehicle_profile_compact,
                    "repair_order": {
                        "number": repair_order.get("number"),
                        "status": repair_order.get("status"),
                        "works_total": len(repair_order.get("works") or []),
                        "materials_total": len(repair_order.get("materials") or []),
                        "reason": repair_order.get("reason"),
                        "comment": repair_order.get("comment"),
                        "note": repair_order.get("note"),
                    },
                    "events_total": len(data.get("events") or []),
                }
            )
        if tool_name == "search_cards":
            cards = data.get("cards") if isinstance(data.get("cards"), list) else []
            return self._preview_payload(
                {
                    "count": len(cards),
                    "cards": [
                        {
                            "id": item.get("id"),
                            "vehicle": item.get("vehicle"),
                            "title": item.get("title"),
                            "column": item.get("column"),
                            "indicator": item.get("indicator"),
                        }
                        for item in cards[:8]
                        if isinstance(item, dict)
                    ],
                }
            )
        if tool_name in {"find_part_numbers", "search_part_numbers", "estimate_price_ru", "lookup_part_prices", "decode_dtc", "search_fault_info"}:
            results = data.get("results") if isinstance(data.get("results"), list) else []
            normalized_results: list[dict[str, Any]] = []
            for item in results[:6]:
                if not isinstance(item, dict):
                    continue
                normalized_results.append(
                    {
                        "title": item.get("title"),
                        "domain": item.get("domain"),
                        "url": item.get("url"),
                        "snippet": item.get("snippet"),
                        "prices": item.get("prices"),
                    }
                )
            return self._preview_payload(
                {
                    "query": data.get("part_query") or data.get("query"),
                    "vehicle_context": data.get("vehicle_context"),
                    "part_numbers": data.get("part_numbers"),
                    "price_summary": data.get("price_summary"),
                    "results": normalized_results,
                }
            )
        if tool_name == "estimate_maintenance":
            return self._preview_payload(
                {
                    "service_type": data.get("service_type"),
                    "vehicle_context": data.get("vehicle_context"),
                    "works": data.get("works"),
                    "materials": data.get("materials"),
                    "notes": data.get("notes"),
                }
            )
        if tool_name == "update_card":
            return self._preview_payload(
                {
                    "card_id": data.get("card_id") or (data.get("card") or {}).get("id"),
                    "changed": data.get("changed"),
                    "changed_fields": data.get("meta", {}).get("changed_fields") if isinstance(data.get("meta"), dict) else data.get("changed"),
                    "card": data.get("card") if isinstance(data.get("card"), dict) else {},
                }
            )
        return self._preview_payload(compact)

    def _autofill_tool_completion_message(self, tool_name: str, payload: dict[str, Any]) -> str:
        if tool_name == "decode_vin":
            status = self._vin_decode_status(payload)
            if status == "success":
                return "decode_vin success."
            if status == "insufficient":
                return "decode_vin insufficient."
            return "decode_vin failed."
        if tool_name == "find_part_numbers":
            part_numbers = payload.get("part_numbers") if isinstance(payload.get("part_numbers"), list) else []
            return "Найдены кандидаты OEM/каталожных номеров." if part_numbers else "Точный OEM не найден, нужен более точный контекст."
        if tool_name == "estimate_price_ru":
            return "Получен ориентир по ценам РФ." if isinstance(payload.get("price_summary"), dict) else "Ценовой ориентир не найден."
        if tool_name == "decode_dtc":
            return "Найдена расшифровка DTC." if isinstance(payload.get("results"), list) and payload.get("results") else "По DTC найден только общий справочный контекст."
        if tool_name == "estimate_maintenance":
            return "Собран предварительный состав ТО."
        if tool_name == "search_fault_info":
            return "Найден внешний контекст по симптомам." if isinstance(payload.get("results"), list) and payload.get("results") else "По симптомам полезного внешнего контекста не найдено."
        return ""

    def _analyze_card_autofill_context(self, context_data: dict[str, Any], *, task_text: str = "") -> dict[str, Any]:
        card = context_data.get("card") if isinstance(context_data.get("card"), dict) else {}
        repair_order = card.get("repair_order") if isinstance(card.get("repair_order"), dict) else {}
        repair_order_text = context_data.get("repair_order_text") if isinstance(context_data.get("repair_order_text"), dict) else {}
        vehicle_profile = card.get("vehicle_profile") if isinstance(card.get("vehicle_profile"), dict) else {}
        recent_events = context_data.get("events") if isinstance(context_data.get("events"), list) else []
        ai_log_tail = card.get("ai_autofill_log") if isinstance(card.get("ai_autofill_log"), list) else []
        ai_prompt = str(card.get("ai_autofill_prompt", "") or "").strip()
        grounded_description = self._strip_existing_ai_notes(str(card.get("description", "") or ""))
        known_vehicle_facts = {
            "make": str(vehicle_profile.get("make_display", "") or "").strip(),
            "model": str(vehicle_profile.get("model_display", "") or "").strip(),
            "year": str(vehicle_profile.get("production_year", "") or "").strip(),
            "engine": str(vehicle_profile.get("engine_model", "") or "").strip(),
            "gearbox": str(vehicle_profile.get("gearbox_model", "") or "").strip(),
            "drivetrain": str(vehicle_profile.get("drivetrain", "") or "").strip(),
            "vin": str(vehicle_profile.get("vin", "") or repair_order.get("vin", "") or "").strip().upper(),
        }
        grounded_text = "\n".join(
            part
            for part in (
                str(card.get("title", "") or "").strip(),
                str(card.get("vehicle", "") or "").strip(),
                grounded_description,
                str(repair_order.get("reason", "") or "").strip(),
                str(repair_order.get("comment", "") or "").strip(),
                str(repair_order.get("note", "") or "").strip(),
                str(repair_order_text.get("text", "") or "").strip(),
            )
            if part
        )
        ai_log_text = "\n".join(
            str(entry.get("message", "") or "").strip()
            for entry in ai_log_tail[-8:]
            if isinstance(entry, dict) and str(entry.get("message", "") or "").strip()
        )
        continuation_text = "\n".join(part for part in (ai_prompt, ai_log_text, str(task_text or "").strip()) if part)
        analysis_text = "\n".join(part for part in (grounded_text, continuation_text) if part)
        grounded_haystack = grounded_text.casefold()
        waiting_state = any(token in grounded_haystack for token in _AUTOFILL_WAIT_HINTS)
        vin_match = _AUTOFILL_VIN_PATTERN.search(grounded_text.upper())
        vin = known_vehicle_facts["vin"] or (vin_match.group(0) if vin_match else "")
        mileage = self._extract_autofill_mileage(card=card, repair_order=repair_order, source_text=grounded_text)
        dtc_codes = list(dict.fromkeys(match.upper() for match in _AUTOFILL_DTC_PATTERN.findall(grounded_text)))[:2]
        part_queries = self._extract_autofill_part_queries(grounded_text)
        maintenance_trigger_found = self._has_explicit_maintenance_trigger(grounded_text)
        maintenance_scope_hint = self._has_maintenance_scope_hint(grounded_haystack)
        maintenance_query = f"ТО на пробеге {mileage}" if mileage else "ТО"
        if "торм" in grounded_haystack:
            maintenance_query = "ТО и тормоза"
        symptom_trigger_found = self._has_explicit_symptom_trigger(grounded_haystack)
        symptom_query = self._extract_autofill_symptom_query(grounded_text) if symptom_trigger_found else ""
        force_vin_decode = bool(vin) and any(token in grounded_haystack for token in ("vin", "расшифр", "комплектац", "подтверд"))
        missing_vehicle_fields = self._profile_missing_fields(vehicle_profile)
        vehicle_context = self._extract_autofill_vehicle_context(
            card=card,
            repair_order=repair_order,
            vehicle_profile=vehicle_profile,
            vin=vin,
        )
        evidence_model = self._build_card_autofill_evidence_model(
            vin=vin,
            part_queries=part_queries,
            maintenance_trigger_found=maintenance_trigger_found,
            maintenance_scope_hint=maintenance_scope_hint,
            mileage=mileage,
            dtc_codes=dtc_codes,
            symptom_trigger_found=symptom_trigger_found,
            symptom_query=symptom_query,
            vehicle_context=vehicle_context,
            missing_vehicle_fields=missing_vehicle_fields,
            grounded_haystack=grounded_haystack,
        )
        scenario_evidence = self._build_card_autofill_scenario_evidence(
            evidence_model=evidence_model,
            waiting_state=waiting_state,
        )
        return {
            "card": card,
            "repair_order": repair_order,
            "vehicle_profile": vehicle_profile,
            "source_text": grounded_text,
            "grounded_text": grounded_text,
            "analysis_text": analysis_text,
            "continuation_text": continuation_text,
            "ai_prompt": ai_prompt,
            "recent_events": recent_events[-10:],
            "ai_log_tail": ai_log_tail[-8:],
            "previous_ai_notes": self._extract_existing_ai_notes(str(card.get("description", "") or "")),
            "vin": vin,
            "mileage": mileage,
            "dtc_codes": dtc_codes,
            "part_queries": part_queries,
            "maintenance_needed": maintenance_trigger_found,
            "maintenance_query": maintenance_query,
            "symptom_query": symptom_query,
            "waiting_state": waiting_state,
            "force_vin_decode": force_vin_decode,
            "missing_vehicle_fields": missing_vehicle_fields,
            "known_vehicle_facts": known_vehicle_facts,
            "vehicle_context": vehicle_context,
            "evidence_model": evidence_model,
            "scenario_evidence": scenario_evidence,
            "related_cards": [],
        }

    def _select_card_autofill_scenarios(self, facts: dict[str, Any]) -> list[dict[str, Any]]:
        plan = self._build_card_autofill_plan(facts)
        return plan["scenarios"]

    def _build_card_autofill_eligibility(self, facts: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for scenario_name in ("vin_enrichment", "parts_lookup", "maintenance_lookup", "dtc_lookup", "fault_research"):
            evidence = self._scenario_evidence(facts, scenario_name)
            result[scenario_name] = {
                "eligible": bool(evidence["trigger_found"] and evidence["confidence_enough"]),
                "trigger_found": bool(evidence["trigger_found"]),
                "confidence_enough": bool(evidence["confidence_enough"]),
                "reason": self._scenario_skip_reason(scenario_name, facts),
            }
        return result

    def _build_card_autofill_strategy(
        self,
        facts: dict[str, Any],
        *,
        eligibility: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        budget = 5
        scenarios: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        if bool(eligibility.get("vin_enrichment", {}).get("eligible")) and budget >= 1:
            scenarios.append({"name": "vin_enrichment", "label": "VIN", "cost": 1})
            budget -= 1
        else:
            skipped.append({"name": "vin_enrichment", "reason": self._scenario_skip_reason("vin_enrichment", facts)})
        if bool(eligibility.get("parts_lookup", {}).get("eligible")) and budget >= 1:
            with_price = budget >= 2
            scenarios.append(
                {
                    "name": "parts_lookup",
                    "label": "ЗАПЧАСТИ",
                    "cost": 2 if with_price else 1,
                    "query": facts["part_queries"][0],
                    "with_price": with_price,
                    "vin_gate_required": bool(facts.get("vin")),
                }
            )
            budget -= 2 if with_price else 1
        else:
            skipped.append({"name": "parts_lookup", "reason": self._scenario_skip_reason("parts_lookup", facts)})
        if bool(eligibility.get("maintenance_lookup", {}).get("eligible")) and budget >= 1:
            scenarios.append({"name": "maintenance_lookup", "label": "ТО", "cost": 1})
            budget -= 1
        else:
            skipped.append({"name": "maintenance_lookup", "reason": self._scenario_skip_reason("maintenance_lookup", facts)})
        if bool(eligibility.get("dtc_lookup", {}).get("eligible")) and budget >= 1:
            scenarios.append({"name": "dtc_lookup", "label": "DTC", "cost": 1, "code": facts["dtc_codes"][0]})
            budget -= 1
        else:
            skipped.append({"name": "dtc_lookup", "reason": self._scenario_skip_reason("dtc_lookup", facts)})
        if bool(eligibility.get("fault_research", {}).get("eligible")) and not facts["waiting_state"] and budget >= 1:
            scenarios.append({"name": "fault_research", "label": "СИМПТОМЫ", "cost": 1})
        scenarios.append({"name": "normalization", "label": "СТРУКТУРА", "cost": 0})
        return {"scenarios": scenarios, "skipped": skipped, "budget_left": budget}

    def _build_card_autofill_plan(self, facts: dict[str, Any]) -> dict[str, Any]:
        eligibility = self._build_card_autofill_eligibility(facts)
        facts["planning_eligibility"] = eligibility
        return self._normalize_card_autofill_plan_labels(
            self._build_card_autofill_strategy(facts, eligibility=eligibility)
        )

    def _normalize_card_autofill_plan_labels(self, plan: dict[str, Any]) -> dict[str, Any]:
        scenarios = plan.get("scenarios") if isinstance(plan.get("scenarios"), list) else []
        normalized: list[dict[str, Any]] = []
        fallback_labels = {
            "vin_enrichment": "VIN",
            "parts_lookup": "ЗАПЧАСТИ",
            "maintenance_lookup": "ТО",
            "dtc_lookup": "DTC",
            "fault_research": "СИМПТОМЫ",
            "normalization": "СТРУКТУРА",
        }
        for item in scenarios:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            name = str(row.get("name", "") or "").strip().lower()
            label = str(row.get("label", "") or "").strip()
            if not label:
                row["label"] = fallback_labels.get(name, label or name.upper())
            normalized.append(row)
        return {
            "scenarios": normalized,
            "skipped": list(plan.get("skipped") or []),
            "budget_left": int(plan.get("budget_left", 0) or 0),
        }

    def _build_card_autofill_plan_message(self, scenarios: list[dict[str, Any]], *, facts: dict[str, Any]) -> str:
        labels = [
            str(item.get("label", "") or "").strip()
            for item in scenarios
            if isinstance(item, dict) and str(item.get("label", "") or "").strip() and str(item.get("name", "") or "").strip().lower() != "normalization"
        ]
        safe_labels = [label for label in labels if label]
        if not safe_labels:
            message = "План: карточка прочитана, подтвержденных внешних сценариев нет, будет только аккуратная нормализация."
        else:
            message = "План: " + " -> ".join(safe_labels + ["СТРУКТУРА"])
        plan = facts.get("autofill_plan") if isinstance(facts.get("autofill_plan"), dict) else {}
        skipped = plan.get("skipped") if isinstance(plan.get("skipped"), list) else []
        gated = [
            str(item.get("name", "") or "").strip()
            for item in skipped
            if isinstance(item, dict) and str(item.get("reason", "") or "").strip()
        ][:3]
        if gated:
            message += " Gated: " + ", ".join(gated) + "."
        related_cards = facts.get("related_cards") if isinstance(facts.get("related_cards"), list) else []
        if related_cards:
            message += f" Связанных карточек на доске: {len(related_cards)}."
        return message

    def _extract_autofill_vehicle_context(
        self,
        *,
        card: dict[str, Any],
        repair_order: dict[str, Any],
        vehicle_profile: dict[str, Any],
        vin: str,
    ) -> dict[str, Any]:
        return {
            "vehicle": str(card.get("vehicle", "") or repair_order.get("vehicle", "") or "").strip(),
            "make": str(vehicle_profile.get("make_display", "") or "").strip(),
            "model": str(vehicle_profile.get("model_display", "") or "").strip(),
            "year": str(vehicle_profile.get("production_year", "") or "").strip(),
            "engine": str(vehicle_profile.get("engine_model", "") or "").strip(),
            "gearbox": str(vehicle_profile.get("gearbox_model", "") or "").strip(),
            "drivetrain": str(vehicle_profile.get("drivetrain", "") or "").strip(),
            "vin": str(vin or "").strip(),
            "mileage": str(vehicle_profile.get("mileage", "") or repair_order.get("mileage", "") or "").strip(),
            "oil_engine_capacity_l": vehicle_profile.get("oil_engine_capacity_l"),
            "oil_gearbox_capacity_l": vehicle_profile.get("oil_gearbox_capacity_l"),
            "coolant_capacity_l": vehicle_profile.get("coolant_capacity_l"),
        }

    def _extract_autofill_mileage(self, *, card: dict[str, Any], repair_order: dict[str, Any], source_text: str) -> str:
        profile = card.get("vehicle_profile") if isinstance(card.get("vehicle_profile"), dict) else {}
        direct = str(profile.get("mileage", "") or repair_order.get("mileage", "") or "").strip()
        if direct:
            return direct
        match = _AUTOFILL_MILEAGE_PATTERN.search(source_text)
        return " ".join(match.group(1).split()) if match else ""

    def _extract_autofill_part_queries(self, source_text: str) -> list[str]:
        haystack = source_text.casefold()
        matches: list[str] = []
        for label, hints in _AUTOFILL_PART_HINTS:
            if any(token in haystack for token in hints):
                matches.append(label)
            if len(matches) >= 2:
                break
        return matches

    def _strip_existing_ai_notes(self, text: str) -> str:
        cleaned: list[str] = []
        inside_ai_block = False
        for raw_line in str(text or "").splitlines():
            line = str(raw_line or "")
            stripped = " ".join(line.strip().split())
            normalized = stripped.casefold()
            if not stripped:
                inside_ai_block = False
                cleaned.append("")
                continue
            if normalized in {"ии:", "ai:"}:
                inside_ai_block = True
                continue
            if normalized.startswith("ии:") or normalized.startswith("ai:"):
                continue
            if inside_ai_block and stripped.startswith("-"):
                continue
            inside_ai_block = False
            cleaned.append(line.rstrip())
        return "\n".join(cleaned).strip()

    def _has_explicit_maintenance_trigger(self, source_text: str) -> bool:
        haystack = source_text.casefold()
        return bool(_AUTOFILL_MAINTENANCE_PATTERN.search(source_text)) or "регламент" in haystack

    def _has_maintenance_scope_hint(self, haystack: str) -> bool:
        return any(token in haystack for token in _AUTOFILL_MAINTENANCE_SCOPE_HINTS)

    def _has_strong_part_lookup_hint(self, haystack: str, part_queries: list[str]) -> bool:
        if any(token in haystack for token in _AUTOFILL_PART_LOOKUP_STRONG_HINTS):
            return True
        return any(query not in {"масло", "фильтр"} for query in part_queries)

    def _has_explicit_symptom_trigger(self, haystack: str) -> bool:
        return any(token in haystack for token in _AUTOFILL_SYMPTOM_HINTS)

    def _has_enough_vehicle_context(self, vehicle_context: dict[str, Any], *, missing_vehicle_fields: list[str]) -> bool:
        score = 0
        if str(vehicle_context.get("vehicle", "") or "").strip():
            score += 1
        if str(vehicle_context.get("vin", "") or "").strip():
            score += 1
        for field_name in ("make", "model", "year", "engine", "gearbox", "drivetrain"):
            if str(vehicle_context.get(field_name, "") or "").strip():
                score += 1
        return score >= 2 or len(missing_vehicle_fields) <= 2

    def _build_card_autofill_evidence_model(
        self,
        *,
        vin: str,
        part_queries: list[str],
        maintenance_trigger_found: bool,
        maintenance_scope_hint: bool,
        mileage: str,
        dtc_codes: list[str],
        symptom_trigger_found: bool,
        symptom_query: str,
        vehicle_context: dict[str, Any],
        missing_vehicle_fields: list[str],
        grounded_haystack: str,
    ) -> dict[str, Any]:
        part_query_found = bool(part_queries)
        explicit_part_found = part_query_found and self._has_strong_part_lookup_hint(grounded_haystack, part_queries)
        return {
            "vin_found": bool(vin),
            "part_query_found": part_query_found,
            "explicit_part_found": explicit_part_found,
            "maintenance_context_found": maintenance_trigger_found,
            "maintenance_scope_found": maintenance_scope_hint,
            "mileage_found": bool(mileage),
            "dtc_found": bool(dtc_codes),
            "fault_symptoms_found": symptom_trigger_found and bool(symptom_query),
            "enough_vehicle_context": self._has_enough_vehicle_context(
                vehicle_context,
                missing_vehicle_fields=missing_vehicle_fields,
            ),
            "external_result_sufficient": False,
        }

    def _build_card_autofill_scenario_evidence(
        self,
        *,
        evidence_model: dict[str, Any],
        waiting_state: bool,
    ) -> dict[str, dict[str, bool]]:
        vin_found = bool(evidence_model.get("vin_found"))
        part_query_found = bool(evidence_model.get("part_query_found"))
        explicit_part_found = bool(evidence_model.get("explicit_part_found"))
        maintenance_context_found = bool(evidence_model.get("maintenance_context_found"))
        maintenance_scope_found = bool(evidence_model.get("maintenance_scope_found"))
        mileage_found = bool(evidence_model.get("mileage_found"))
        dtc_found = bool(evidence_model.get("dtc_found"))
        fault_symptoms_found = bool(evidence_model.get("fault_symptoms_found"))
        return {
            "vin_enrichment": {
                "trigger_found": vin_found,
                "confidence_enough": vin_found,
            },
            "parts_lookup": {
                "trigger_found": part_query_found,
                "confidence_enough": explicit_part_found,
            },
            "maintenance_lookup": {
                "trigger_found": maintenance_context_found,
                "confidence_enough": maintenance_context_found and (mileage_found or maintenance_scope_found),
            },
            "dtc_lookup": {
                "trigger_found": dtc_found,
                "confidence_enough": dtc_found,
            },
            "fault_research": {
                "trigger_found": fault_symptoms_found,
                "confidence_enough": fault_symptoms_found
                and not explicit_part_found
                and not maintenance_context_found
                and not dtc_found
                and not waiting_state,
            },
        }

    def _scenario_skip_reason(self, name: str, facts: dict[str, Any]) -> str:
        evidence = facts.get("evidence_model") if isinstance(facts.get("evidence_model"), dict) else {}
        if name == "vin_enrichment":
            return "" if evidence.get("vin_found") else "no VIN in card"
        if name == "parts_lookup":
            if not evidence.get("part_query_found"):
                return "no explicit part in card"
            if not evidence.get("explicit_part_found"):
                return "part mention is too weak for lookup"
            return ""
        if name == "maintenance_lookup":
            if not evidence.get("maintenance_context_found"):
                return "no maintenance context in card"
            if not evidence.get("mileage_found") and not evidence.get("maintenance_scope_found"):
                return "maintenance trigger is too weak"
            return ""
        if name == "dtc_lookup":
            return "" if evidence.get("dtc_found") else "no DTC in card"
        if name == "fault_research":
            if not evidence.get("fault_symptoms_found"):
                return "no isolated symptom trigger"
            if facts.get("waiting_state"):
                return "card is in waiting state"
            return "covered by stronger scenarios"
        return ""

    def _card_autofill_can_run_parts_lookup(self, facts: dict[str, Any]) -> bool:
        if not facts.get("vin"):
            return True
        vin_status = str(facts.get("vin_decode_status", "") or "").strip().lower()
        if vin_status == "success":
            return True
        evidence = facts.get("evidence_model") if isinstance(facts.get("evidence_model"), dict) else {}
        return bool(evidence.get("enough_vehicle_context"))

    def _record_card_autofill_plan_diagnostics(
        self,
        *,
        task_id: str,
        run_id: str,
        step: int,
        facts: dict[str, Any],
    ) -> None:
        evidence = facts.get("evidence_model") if isinstance(facts.get("evidence_model"), dict) else {}
        plan = facts.get("autofill_plan") if isinstance(facts.get("autofill_plan"), dict) else {}
        evidence_bits = [
            name
            for name, enabled in (
                ("vin", evidence.get("vin_found")),
                ("part", evidence.get("explicit_part_found")),
                ("maintenance", evidence.get("maintenance_context_found")),
                ("mileage", evidence.get("mileage_found")),
                ("dtc", evidence.get("dtc_found")),
                ("symptoms", evidence.get("fault_symptoms_found")),
            )
            if enabled
        ]
        self._record_log_action(
            task_id=task_id,
            run_id=run_id,
            step=step,
            level="INFO",
            phase="analysis",
            message="Evidence: " + (", ".join(evidence_bits) if evidence_bits else "no external trigger"),
        )
        skipped = plan.get("skipped") if isinstance(plan.get("skipped"), list) else []
        for item in skipped[:3]:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason", "") or "").strip()
            name = str(item.get("name", "") or "").strip()
            if not reason or not name:
                continue
            self._record_log_action(
                task_id=task_id,
                run_id=run_id,
                step=step,
                level="INFO",
                phase="analysis",
                message=f"{name} skipped: {reason}",
            )

    def _scenario_evidence(self, facts: dict[str, Any], name: str) -> dict[str, bool]:
        payload = facts.get("scenario_evidence") if isinstance(facts.get("scenario_evidence"), dict) else {}
        evidence = payload.get(name) if isinstance(payload.get(name), dict) else {}
        return {
            "trigger_found": bool(evidence.get("trigger_found")),
            "confidence_enough": bool(evidence.get("confidence_enough")),
        }

    def _vin_decode_status(self, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return "failed"
        if any(str(payload.get(key, "") or "").strip() for key in ("model", "model_year", "engine_model", "transmission", "drive_type")):
            return "success"
        if any(str(payload.get(key, "") or "").strip() for key in ("make", "plant_country", "vin")):
            return "insufficient"
        return "failed"

    def _extract_existing_ai_notes(self, description_text: str) -> list[str]:
        notes: list[str] = []
        inside_ai_block = False
        for raw_line in str(description_text or "").splitlines():
            line = " ".join(str(raw_line or "").strip().split())
            if not line:
                inside_ai_block = False
                continue
            normalized = line.casefold()
            if normalized in {"ии:", "ai:"}:
                inside_ai_block = True
                continue
            if normalized.startswith("ии:") or normalized.startswith("ai:"):
                notes.append(line.split(":", 1)[1].strip())
                inside_ai_block = False
                continue
            if inside_ai_block:
                notes.append(line.lstrip("- ").strip())
        return [item for item in notes if item]

    def _extract_autofill_symptom_query_legacy_unused(self, source_text: str) -> str:
        lines: list[str] = []
        for raw_line in str(source_text or "").splitlines():
            line = " ".join(str(raw_line or "").strip().split())
            if not line:
                continue
            lower = line.casefold()
            if lower.startswith("vin") or lower.startswith("ии:") or lower.startswith("ai:"):
                continue
            if "артикул" in lower:
                continue
            if "цена" in lower and any(char.isdigit() for char in line):
                continue
            lines.append(line)
            if len(lines) >= 3:
                break
        return "; ".join(lines)[:280]

    def _extract_autofill_symptom_query(self, source_text: str) -> str:
        lines: list[str] = []
        symptom_lines: list[str] = []
        blocked_prefixes = (
            "клиент",
            "customer",
            "телефон",
            "phone",
            "марка",
            "make",
            "модель",
            "model",
            "год",
            "vin",
            "гос. номер",
            "госномер",
            "license plate",
            "пробег",
            "mileage",
        )
        symptom_markers = (
            "течь",
            "антифриз",
            "стук",
            "вибрац",
            "ошибк",
            "неисправ",
            "жалоб",
            "симптом",
            "перегрев",
            "дым",
            "шум",
            "троит",
            "coolant",
            "leak",
            "overheat",
            "noise",
            "fault",
        )
        for raw_line in str(source_text or "").splitlines():
            line = " ".join(str(raw_line or "").strip().split())
            if not line:
                continue
            lower = line.casefold()
            if lower.startswith("vin") or lower.startswith("ии:") or lower.startswith("ai:"):
                continue
            if "артикул" in lower:
                continue
            if "цена" in lower and any(char.isdigit() for char in line):
                continue
            if any(lower.startswith(prefix) for prefix in blocked_prefixes):
                continue
            if self._looks_like_customer_line(lower):
                continue
            if any(marker in lower for marker in symptom_markers):
                symptom_lines.append(line)
                continue
            lines.append(line)
            if len(lines) >= 3 and len(symptom_lines) >= 2:
                break
        preferred = symptom_lines[:2] if symptom_lines else []
        fallback = [line for line in lines if line not in preferred][:1]
        return "; ".join(preferred + fallback)[:280]

    def _looks_like_customer_line(self, lower_line: str) -> bool:
        compact = " ".join(str(lower_line or "").split())
        if not compact:
            return False
        if any(token in compact for token in ("+7", "8 (", "телефон", "phone")):
            return True
        words = [item for item in compact.replace(".", " ").split() if item]
        if 2 <= len(words) <= 4 and all(word.isalpha() for word in words):
            return True
        return False

    def _profile_missing_fields(self, vehicle_profile: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for field_name in ("make_display", "model_display", "production_year", "engine_model", "gearbox_model", "drivetrain"):
            if not str(vehicle_profile.get(field_name, "") or "").strip():
                missing.append(field_name)
        return missing

    def _autofill_vin_should_run(self, facts: dict[str, Any]) -> bool:
        evidence = self._scenario_evidence(facts, "vin_enrichment")
        return evidence["trigger_found"] and evidence["confidence_enough"]

    def _merge_vehicle_context(self, current: dict[str, Any], decoded: dict[str, Any]) -> dict[str, Any]:
        merged = dict(current)
        if not merged.get("make") and decoded.get("make"):
            merged["make"] = str(decoded.get("make", "") or "").strip()
        if not merged.get("model") and decoded.get("model"):
            merged["model"] = str(decoded.get("model", "") or "").strip()
        if not merged.get("year") and decoded.get("model_year"):
            merged["year"] = str(decoded.get("model_year", "") or "").strip()
        if not merged.get("engine") and decoded.get("engine_model"):
            merged["engine"] = str(decoded.get("engine_model", "") or "").strip()
        if not merged.get("gearbox") and decoded.get("transmission"):
            merged["gearbox"] = str(decoded.get("transmission", "") or "").strip()
        if not merged.get("drivetrain") and decoded.get("drive_type"):
            merged["drivetrain"] = str(decoded.get("drive_type", "") or "").strip()
        if not merged.get("vin") and decoded.get("vin"):
            merged["vin"] = str(decoded.get("vin", "") or "").strip()
        if not merged.get("vehicle"):
            merged["vehicle"] = " ".join(part for part in (merged.get("make", ""), merged.get("model", ""), merged.get("year", "")) if part).strip()
        return merged

    def _build_user_task_message(self, task: dict[str, Any], metadata: dict[str, Any], *, task_type: str) -> str:
        lines = [
            f"Task id: {task['id']}",
            f"Mode: {task.get('mode', 'manual')}",
            f"Source: {task.get('source', 'manual')}",
            f"Task type: {task_type}",
        ]
        requested_by = str(metadata.get("requested_by", "") or "").strip()
        if requested_by:
            lines.append(f"Requested by: {requested_by}")
        scheduled_name = str(metadata.get("scheduled_task_name", "") or "").strip()
        if scheduled_name:
            lines.append(f"Scheduled task: {scheduled_name}")
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        if context:
            lines.append("Context metadata:")
            lines.append(json.dumps(context, ensure_ascii=False, indent=2))
            if str(context.get("kind", "")).strip().lower() == "card":
                lines.append("This task was opened from a card. Work with this card first and inside this card first.")
        scope_prompt = self._build_scope_prompt_block(metadata)
        if scope_prompt:
            lines.append(scope_prompt)
        lines.append("Task:")
        lines.append(str(task.get("task_text", "") or "").strip())
        return "\n".join(lines)

    def _build_scope_prompt_block(self, metadata: dict[str, Any]) -> str:
        scope = metadata.get("scope") if isinstance(metadata.get("scope"), dict) else {}
        scope_type = str(scope.get("type", "") or "").strip().lower()
        if scope_type not in {"all_cards", "column", "current_card"}:
            return ""
        scope_payload: dict[str, Any] = {
            "type": scope_type,
            "column": str(scope.get("column", "") or "").strip(),
            "column_label": str(scope.get("column_label", "") or "").strip(),
            "card_id": str(scope.get("card_id", "") or "").strip(),
            "card_label": str(scope.get("card_label", "") or "").strip(),
            "cards": [],
        }
        try:
            if scope_type == "current_card" and scope_payload["card_id"]:
                context_result = self._board_api.get_card_context(
                    scope_payload["card_id"],
                    event_limit=20,
                    include_repair_order_text=True,
                )
                context_data = self._response_data(context_result)
                scope_payload["card"] = context_data.get("card") if isinstance(context_data.get("card"), dict) else {}
                scope_payload["events"] = (context_data.get("events") if isinstance(context_data.get("events"), list) else [])[:12]
                return "Execution scope:\n" + json.dumps(scope_payload, ensure_ascii=False, indent=2)
            if scope_type == "column" and scope_payload["column"]:
                result = self._board_api.search_cards(
                    query=None,
                    include_archived=False,
                    column=scope_payload["column"],
                    tag=None,
                    indicator=None,
                    status=None,
                    limit=40,
                )
                search_data = self._response_data(result)
                cards = search_data.get("cards") if isinstance(search_data.get("cards"), list) else []
            else:
                snapshot = self._board_api.get_board_snapshot(archive_limit=0)
                snapshot_data = self._response_data(snapshot)
                columns = snapshot_data.get("columns") if isinstance(snapshot_data.get("columns"), list) else []
                cards = []
                for column in columns if isinstance(columns, list) else []:
                    items = column.get("cards") if isinstance(column, dict) else []
                    if isinstance(items, list):
                        cards.extend(items)
            scope_payload["cards"] = [
                {
                    "id": item.get("id"),
                    "vehicle": item.get("vehicle"),
                    "title": item.get("title"),
                    "column": item.get("column"),
                    "tags": item.get("tags"),
                }
                for item in (cards if isinstance(cards, list) else [])[:20]
                if isinstance(item, dict)
            ]
        except Exception as exc:
            scope_payload["error"] = str(exc)
        return "Execution scope:\n" + json.dumps(scope_payload, ensure_ascii=False, indent=2)

    def _compose_card_autofill_update(
        self,
        *,
        card_id: str,
        facts: dict[str, Any],
        orchestration_results: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        card = facts["card"]
        current_description = str(card.get("description", "") or "").strip()
        decoded_vin = orchestration_results.get("decode_vin")
        vin_decode_status = str(facts.get("vin_decode_status", "") or "").strip().lower()
        vehicle_patch = self._autofill_vehicle_patch(facts=facts, decoded_vin=decoded_vin, vin_decode_status=vin_decode_status)
        vehicle_label_patch = self._autofill_vehicle_label_patch(facts=facts, decoded_vin=decoded_vin, vin_decode_status=vin_decode_status)
        ai_lines: list[str] = []
        if vin_decode_status == "success" and isinstance(decoded_vin, dict):
            vin_bits: list[str] = []
            if decoded_vin.get("make"):
                vin_bits.append(str(decoded_vin.get("make", "") or "").strip())
            if decoded_vin.get("model"):
                vin_bits.append(str(decoded_vin.get("model", "") or "").strip())
            if decoded_vin.get("model_year"):
                vin_bits.append(str(decoded_vin.get("model_year", "") or "").strip())
            if decoded_vin.get("engine_model") and "engine_model" in vehicle_patch:
                vin_bits.append(f"двигатель: {decoded_vin.get('engine_model')}")
            if decoded_vin.get("transmission") and "gearbox_model" in vehicle_patch:
                vin_bits.append(f"КПП: {decoded_vin.get('transmission')}")
            if decoded_vin.get("drive_type") and "drivetrain" in vehicle_patch:
                vin_bits.append(f"привод: {decoded_vin.get('drive_type')}")
            if decoded_vin.get("plant_country"):
                vin_bits.append(f"сборка: {decoded_vin.get('plant_country')}")
            if vin_bits:
                ai_lines.append("По VIN подтверждено: " + ", ".join(vin_bits) + ".")
        elif facts.get("vin") and facts.get("vin_decode_attempted"):
            if vin_decode_status == "insufficient":
                ai_lines.append("Найден VIN, выполнена внешняя расшифровка, но данных недостаточно для уверенного заполнения модели и агрегатов.")
            elif vin_decode_status == "failed":
                ai_lines.append("Найден VIN, выполнена попытка внешней расшифровки, но сервис не вернул пригодный результат.")
        part_lookup = orchestration_results.get("find_part_numbers")
        if isinstance(part_lookup, dict) and facts["part_queries"]:
            primary_part, analog_parts = self._summarize_part_matches(part_lookup)
            if primary_part:
                part_line = f"{facts['part_queries'][0].capitalize()}: OEM {primary_part}"
                if analog_parts:
                    part_line += f"; аналоги: {analog_parts}."
                else:
                    part_line += "."
                price_lookup = orchestration_results.get("estimate_price_ru")
                if isinstance(price_lookup, dict):
                    price_line = self._summarize_price_summary(price_lookup)
                    if price_line:
                        part_line += f" {price_line}"
                ai_lines.append(part_line)
            else:
                missing_bits = self._humanize_missing_vehicle_fields(facts["missing_vehicle_fields"])
                if missing_bits:
                    ai_lines.append(f"Следующему исполнителю: для точного подбора {facts['part_queries'][0]} уточнить {missing_bits}.")
        maintenance = orchestration_results.get("estimate_maintenance")
        if isinstance(maintenance, dict):
            works = maintenance.get("works") if isinstance(maintenance.get("works"), list) else []
            materials = maintenance.get("materials") if isinstance(maintenance.get("materials"), list) else []
            works_preview = ", ".join(
                str(item.get("name", "") or "").strip()
                for item in works[:3]
                if isinstance(item, dict) and str(item.get("name", "") or "").strip()
            )
            materials_preview = ", ".join(
                str(item.get("name", "") or "").strip()
                for item in materials[:4]
                if isinstance(item, dict) and str(item.get("name", "") or "").strip()
            )
            service_type_label = str(maintenance.get("service_type", "ТО") or "ТО").strip()
            if str(facts.get("maintenance_result_origin", "") or "").strip().lower() == "local_heuristic":
                line = f"Сервисная подсказка по {service_type_label}:"
            else:
                line = f"{service_type_label}:"
            if works_preview:
                line += f" работы — {works_preview}."
            if materials_preview:
                line += f" Расходники — {materials_preview}."
            notes_preview = "; ".join(
                str(item or "").strip()
                for item in (maintenance.get("notes") if isinstance(maintenance.get("notes"), list) else [])[:2]
                if str(item or "").strip()
            )
            if notes_preview:
                line += f" {notes_preview}"
            ai_lines.append(line)
        dtc_result = orchestration_results.get("decode_dtc")
        if isinstance(dtc_result, dict) and facts["dtc_codes"]:
            snippet = self._first_search_snippet(dtc_result)
            if snippet:
                ai_lines.append(f"DTC {facts['dtc_codes'][0]}: {snippet}")
        fault_result = orchestration_results.get("search_fault_info")
        if isinstance(fault_result, dict):
            snippet = self._first_search_snippet(fault_result)
            if snippet:
                ai_lines.append(f"По симптомам: {snippet}")
        oem_notes_patch = self._compose_vehicle_profile_oem_notes(
            facts=facts,
            orchestration_results=orchestration_results,
            current_oem_notes=str(facts["vehicle_profile"].get("oem_notes", "") or ""),
        )
        if oem_notes_patch:
            vehicle_patch["oem_notes"] = oem_notes_patch
        ai_lines.extend(self._compose_card_autofill_follow_up_lines(facts=facts, orchestration_results=orchestration_results))
        filtered_ai_lines = [line for line in ai_lines if self._line_has_new_information(current_description, line)]
        if not filtered_ai_lines and not vehicle_patch and not vehicle_label_patch:
            return None, []
        update_args: dict[str, Any] = {"card_id": card_id}
        if filtered_ai_lines:
            update_args["description"] = "ИИ:\n- " + "\n- ".join(filtered_ai_lines)
        if vehicle_label_patch:
            update_args["vehicle"] = vehicle_label_patch
        if vehicle_patch:
            update_args["vehicle_profile"] = vehicle_patch
        display_sections: list[dict[str, Any]] = []
        if vehicle_patch:
            display_sections.append(
                {
                    "title": "Профиль авто",
                    "body": "",
                    "items": [
                        f"{key}: {value}"
                        for key, value in vehicle_patch.items()
                        if key in {"make_display", "model_display", "production_year", "engine_model", "gearbox_model", "drivetrain", "vin"}
                    ],
                }
            )
        if vehicle_label_patch:
            display_sections.append({"title": "Обновлен ярлык авто", "body": "", "items": [vehicle_label_patch]})
        related_cards = facts.get("related_cards") if isinstance(facts.get("related_cards"), list) else []
        if related_cards:
            display_sections.append(
                {
                    "title": "Контекст доски",
                    "body": "",
                    "items": [
                        " / ".join(part for part in (str(item.get("vehicle", "") or "").strip(), str(item.get("title", "") or "").strip(), str(item.get("column", "") or "").strip()) if part)
                        for item in related_cards[:3]
                        if isinstance(item, dict)
                    ],
                }
            )
        if filtered_ai_lines:
            display_sections.append({"title": "Добавлено в карточку", "body": "", "items": filtered_ai_lines[:6]})
        return update_args, display_sections

    def _autofill_vehicle_label_patch(self, *, facts: dict[str, Any], decoded_vin: dict[str, Any] | None, vin_decode_status: str = "") -> str:
        if vin_decode_status != "success":
            return ""
        current_vehicle = str(facts["card"].get("vehicle", "") or "").strip()
        context = facts.get("vehicle_context") if isinstance(facts.get("vehicle_context"), dict) else {}
        candidate = " ".join(
            part
            for part in (
                str(context.get("make", "") or "").strip(),
                str(context.get("model", "") or "").strip(),
                str(context.get("year", "") or "").strip(),
            )
            if part
        ).strip()
        if not candidate and isinstance(decoded_vin, dict):
            candidate = " ".join(
                part
                for part in (
                    str(decoded_vin.get("make", "") or "").strip(),
                    str(decoded_vin.get("model", "") or "").strip(),
                    str(decoded_vin.get("model_year", "") or "").strip(),
                )
                if part
            ).strip()
        if not candidate or candidate == current_vehicle:
            return ""
        if current_vehicle and candidate.casefold() in current_vehicle.casefold():
            return ""
        return candidate

    def _autofill_vehicle_patch(self, *, facts: dict[str, Any], decoded_vin: dict[str, Any] | None, vin_decode_status: str = "") -> dict[str, Any]:
        if not isinstance(decoded_vin, dict) or vin_decode_status != "success":
            return {}
        patch: dict[str, Any] = {}
        existing = facts["vehicle_profile"]
        field_sources: dict[str, str] = {}
        autofilled_fields: list[str] = []

        def _set_if_missing(field_name: str, value: Any) -> None:
            text = str(value or "").strip()
            if not text or str(existing.get(field_name, "") or "").strip():
                return
            patch[field_name] = text
            autofilled_fields.append(field_name)
            field_sources[field_name] = "official_vin_decode_nhtsa"

        _set_if_missing("vin", decoded_vin.get("vin") or facts["vin"])
        _set_if_missing("make_display", decoded_vin.get("make"))
        _set_if_missing("model_display", decoded_vin.get("model"))
        if not str(existing.get("production_year", "") or "").strip():
            try:
                year_value = int(str(decoded_vin.get("model_year", "") or "").strip())
            except (TypeError, ValueError):
                year_value = None
            if year_value:
                patch["production_year"] = year_value
                autofilled_fields.append("production_year")
                field_sources["production_year"] = "official_vin_decode_nhtsa"
        _set_if_missing("engine_model", decoded_vin.get("engine_model"))
        _set_if_missing("gearbox_model", decoded_vin.get("transmission"))
        _set_if_missing("drivetrain", decoded_vin.get("drive_type"))
        if not patch:
            return {}
        patch["source_summary"] = "official VIN decode"
        patch["source_confidence"] = 0.78
        patch["autofilled_fields"] = autofilled_fields
        patch["field_sources"] = field_sources
        source_refs = [str(decoded_vin.get("source_url", "") or "").strip()]
        patch["source_links_or_refs"] = [item for item in source_refs if item]
        patch["data_completion_state"] = "mostly_autofilled" if len(autofilled_fields) >= 3 else "partially_autofilled"
        return patch

    def _compose_vehicle_profile_oem_notes(
        self,
        *,
        facts: dict[str, Any],
        orchestration_results: dict[str, Any],
        current_oem_notes: str,
    ) -> str:
        notes: list[str] = []
        decoded_vin = orchestration_results.get("decode_vin")
        vin_decode_status = str(facts.get("vin_decode_status", "") or "").strip().lower()
        if vin_decode_status == "success" and isinstance(decoded_vin, dict):
            vin_bits: list[str] = []
            for label, key in (
                ("марка", "make"),
                ("модель", "model"),
                ("год", "model_year"),
                ("двигатель", "engine_model"),
                ("КПП", "transmission"),
                ("привод", "drive_type"),
            ):
                value = str(decoded_vin.get(key, "") or "").strip()
                if value:
                    vin_bits.append(f"{label}: {value}")
            if vin_bits:
                notes.append("VIN decode: " + "; ".join(vin_bits[:5]) + ".")
        part_lookup = orchestration_results.get("find_part_numbers")
        if isinstance(part_lookup, dict) and facts.get("part_queries"):
            primary_part, analog_parts = self._summarize_part_matches(part_lookup)
            if primary_part:
                part_line = f"Запчасть {facts['part_queries'][0]}: OEM {primary_part}"
                if analog_parts:
                    part_line += f"; аналоги: {analog_parts}."
                else:
                    part_line += "."
                price_lookup = orchestration_results.get("estimate_price_ru")
                if isinstance(price_lookup, dict):
                    price_line = self._summarize_price_summary(price_lookup)
                    if price_line:
                        part_line += f" {price_line}"
                notes.append(part_line)
        maintenance = orchestration_results.get("estimate_maintenance")
        if isinstance(maintenance, dict):
            works = maintenance.get("works") if isinstance(maintenance.get("works"), list) else []
            materials = maintenance.get("materials") if isinstance(maintenance.get("materials"), list) else []
            works_preview = ", ".join(
                str(item.get("name", "") or "").strip()
                for item in works[:3]
                if isinstance(item, dict) and str(item.get("name", "") or "").strip()
            )
            materials_preview = ", ".join(
                str(item.get("name", "") or "").strip()
                for item in materials[:4]
                if isinstance(item, dict) and str(item.get("name", "") or "").strip()
            )
            service_type_label = str(maintenance.get("service_type", "ТО") or "ТО").strip()
            origin = str(facts.get("maintenance_result_origin", "") or "").strip().lower()
            prefix = f"Сервисная подсказка по {service_type_label}" if origin == "local_heuristic" else service_type_label
            line = f"{prefix}:"
            if works_preview:
                line += f" работы — {works_preview}."
            if materials_preview:
                line += f" расходники — {materials_preview}."
            if line.strip() != f"{prefix}:":
                notes.append(line)
        dtc_result = orchestration_results.get("decode_dtc")
        if isinstance(dtc_result, dict) and facts["dtc_codes"]:
            snippet = self._first_search_snippet(dtc_result)
            if snippet:
                notes.append(f"DTC {facts['dtc_codes'][0]}: {snippet}")
        fault_result = orchestration_results.get("search_fault_info")
        if isinstance(fault_result, dict):
            snippet = self._first_search_snippet(fault_result)
            if snippet:
                notes.append(f"Симптомы: {snippet}")
        unique_notes = [line for line in notes if self._line_has_new_information(current_oem_notes, line)]
        if not unique_notes:
            return ""
        merged_notes = "\n".join(part for part in [current_oem_notes.strip(), *unique_notes] if part)
        return normalize_vehicle_notes(merged_notes, limit=1200)

    def _humanize_missing_vehicle_fields(self, fields: list[str]) -> str:
        mapping = {
            "model_display": "модель",
            "production_year": "год",
            "engine_model": "двигатель",
            "gearbox_model": "КПП",
            "drivetrain": "привод",
            "make_display": "марку",
        }
        values = [mapping[field_name] for field_name in fields[:3] if field_name in mapping]
        return ", ".join(values)

    def _pick_best_part_number(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("part_numbers") if isinstance(payload.get("part_numbers"), list) else []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value", "") or "").strip()
            if value:
                return value
        return ""

    def _summarize_part_matches(self, payload: dict[str, Any]) -> tuple[str, str]:
        candidates = payload.get("part_numbers") if isinstance(payload.get("part_numbers"), list) else []
        values = [
            str(item.get("value", "") or "").strip()
            for item in candidates[:3]
            if isinstance(item, dict) and str(item.get("value", "") or "").strip()
        ]
        if not values:
            return "", ""
        primary = values[0]
        analogs = ", ".join(values[1:3])
        return primary, analogs

    def _summarize_price_summary(self, payload: dict[str, Any]) -> str:
        price_summary = payload.get("price_summary") if isinstance(payload.get("price_summary"), dict) else {}
        if not price_summary:
            return ""
        offers_total = int(price_summary.get("offers_total", 0) or 0)
        min_rub = int(price_summary.get("min_rub", 0) or 0)
        max_rub = int(price_summary.get("max_rub", 0) or 0)
        if min_rub <= 0 and max_rub <= 0:
            return ""
        if min_rub and max_rub and min_rub != max_rub:
            return f"Ориентир по РФ: {min_rub:,}-{max_rub:,} ₽ ({offers_total} предложений).".replace(",", " ")
        value = max_rub or min_rub
        return f"Ориентир по РФ: около {value:,} ₽ ({offers_total} предложений).".replace(",", " ")

    def _first_search_snippet(self, payload: dict[str, Any]) -> str:
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        for item in results:
            if not isinstance(item, dict):
                continue
            text = str(item.get("snippet", "") or item.get("title", "") or "").strip()
            if text:
                return text[:220]
        return ""

    def _line_has_new_information(self, current_description: str, line: str) -> bool:
        normalized_current = " ".join(str(current_description or "").split()).casefold()
        normalized_line = " ".join(str(line or "").replace("ИИ:", "").replace("AI:", "").split()).casefold()
        return bool(normalized_line) and normalized_line not in normalized_current

    def _compose_card_autofill_follow_up_lines(
        self,
        *,
        facts: dict[str, Any],
        orchestration_results: dict[str, Any],
    ) -> list[str]:
        lines: list[str] = []
        parts_evidence = self._scenario_evidence(facts, "parts_lookup")
        maintenance_evidence = self._scenario_evidence(facts, "maintenance_lookup")
        dtc_evidence = self._scenario_evidence(facts, "dtc_lookup")
        fault_evidence = self._scenario_evidence(facts, "fault_research")
        part_payload = orchestration_results.get("find_part_numbers")
        dtc_payload = orchestration_results.get("decode_dtc")
        fault_payload = orchestration_results.get("search_fault_info")
        if parts_evidence["trigger_found"] and (not isinstance(part_payload, dict) or self._is_partial_tool_payload(part_payload)):
            missing_bits = self._humanize_missing_vehicle_fields(facts["missing_vehicle_fields"])
            if missing_bits:
                lines.append(f"Следующему исполнителю: для точного подбора {facts['part_queries'][0]} уточнить {missing_bits}.")
            elif not facts.get("vin"):
                lines.append(f"Следующему исполнителю: для точного подбора {facts['part_queries'][0]} нужен VIN или точный номер снятой детали.")
        if maintenance_evidence["trigger_found"] and not facts["mileage"]:
            lines.append("Следующему исполнителю: уточнить пробег, чтобы подтвердить состав ТО и расходники.")
        if dtc_evidence["trigger_found"] and (not isinstance(dtc_payload, dict) or self._is_partial_tool_payload(dtc_payload)):
            lines.append(f"Следующему исполнителю: повторно проверить код {facts['dtc_codes'][0]} и приложить скрин диагностики.")
        if fault_evidence["trigger_found"] and fault_evidence["confidence_enough"] and (not isinstance(fault_payload, dict) or self._is_partial_tool_payload(fault_payload)) and not facts["waiting_state"]:
            lines.append("Следующему исполнителю: зафиксировать симптомы точнее — когда проявляется, на холодную или на горячую, под нагрузкой или на месте.")
        if any(self._is_budget_exceeded_payload(payload) for payload in orchestration_results.values() if isinstance(payload, dict)):
            lines.append("ИИ: внешний поиск упёрся в лимит запросов этого прохода; повторный проход продолжит поиск автоматически.")
        return lines[:2]

    def _autofill_result_summary(self, applied_updates: list[str], orchestration_results: dict[str, Any], *, facts: dict[str, Any]) -> str:
        if applied_updates:
            parts: list[str] = []
            if "decode_vin" in orchestration_results:
                parts.append("VIN")
            if "find_part_numbers" in orchestration_results:
                parts.append("запчасти")
            if "estimate_maintenance" in orchestration_results:
                parts.append("ТО")
            if "decode_dtc" in orchestration_results:
                parts.append("DTC")
            if "search_fault_info" in orchestration_results:
                parts.append("симптомы")
            if parts:
                return "Карточка дополнена: " + ", ".join(parts) + "."
            return "Карточка дополнена по автосопровождению."
        vin_status = str(facts.get("vin_decode_status", "") or "").strip().lower()
        if facts.get("vin") and facts.get("vin_decode_attempted"):
            if vin_status == "insufficient":
                return "Внешняя VIN-расшифровка выполнена, но данных недостаточно для уверенного обновления."
            if vin_status == "failed":
                return "Внешняя VIN-расшифровка не вернула пригодный результат."
        if any(self._is_budget_exceeded_payload(payload) for payload in orchestration_results.values() if isinstance(payload, dict)):
            return "Внешний поиск частично отложен: исчерпан лимит запросов текущего прохода."
        if "find_part_numbers" in orchestration_results:
            return "Внешний поиск деталей выполнен, но новых надёжных полей для карточки не найдено."
        if "decode_dtc" in orchestration_results or "search_fault_info" in orchestration_results:
            return "Контекст по диагностике собран, но безопасных изменений для карточки не найдено."
        return "Изменений не обнаружено."

    def _cleanup_card_id(self, metadata: dict[str, Any]) -> str:
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        if str(context.get("kind", "")).strip().lower() != "card":
            return ""
        return str(context.get("card_id", "") or "").strip()

    def _context_kind(self, metadata: dict[str, Any]) -> str:
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        return str(context.get("kind", "") or "board").strip().lower() or "board"

    def _normalize_card_autofill_update(self, args: dict[str, Any]) -> dict[str, Any]:
        card_id = str(args.get("card_id", "") or "").strip()
        if not card_id or "description" not in args:
            return args
        try:
            current_payload = self._board_api.get_card(card_id)
        except Exception:
            return args
        current_data = self._response_data(current_payload)
        current_card = current_data.get("card") if isinstance(current_data.get("card"), dict) else current_data
        current_description = str(current_card.get("description", "") if isinstance(current_card, dict) else "").strip()
        proposed_description = str(args.get("description", "") or "").strip()
        merged_description = self._merge_card_autofill_description(current_description, proposed_description)
        if merged_description == proposed_description:
            return args
        normalized_args = dict(args)
        normalized_args["description"] = merged_description
        return normalized_args

    def _merge_card_autofill_description(self, current_text: str, proposed_text: str) -> str:
        current = str(current_text or "").strip()
        proposed = str(proposed_text or "").strip()
        if not proposed:
            return current
        if not current:
            return self._dedupe_card_autofill_paragraphs(proposed)
        current_normalized = " ".join(current.split())
        proposed_normalized = " ".join(proposed.split())
        if proposed_normalized == current_normalized or proposed_normalized in current_normalized:
            return current
        if current_normalized and current_normalized in proposed_normalized:
            return self._dedupe_card_autofill_paragraphs(proposed)
        if "ИИ:" in proposed or "AI:" in proposed:
            return self._dedupe_card_autofill_paragraphs(f"{current}\n\n{proposed}")
        normalized_ai_block = "\n".join(
            line.strip()
            for line in proposed.splitlines()
            if line.strip()
        )
        return self._dedupe_card_autofill_paragraphs(f"{current}\n\nИИ:\n{normalized_ai_block}")

    def _dedupe_card_autofill_paragraphs(self, text: str) -> str:
        paragraphs = [part.strip() for part in str(text or "").split("\n\n") if str(part or "").strip()]
        if not paragraphs:
            return ""
        deduped: list[str] = []
        seen: set[str] = set()
        for paragraph in paragraphs:
            normalized = " ".join(paragraph.split()).casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(paragraph)
        return "\n\n".join(deduped)

    def _classify_task(self, task: dict[str, Any], metadata: dict[str, Any]) -> str:
        if str(metadata.get("purpose", "") or "").strip().lower() in {"card_autofill", "board_control"}:
            return "card_cleanup"
        text = self._normalized_task_text(str(task.get("task_text", "") or ""))
        if self._is_card_cleanup_task(task, metadata):
            return "card_cleanup"
        if "vin" in text or "расшифру" in text or "decode vin" in text:
            return "vin_decode"
        if "dtc" in text or _AUTOFILL_DTC_PATTERN.search(text.upper()):
            return "dtc_lookup"
        if "запчаст" in text or "каталож" in text or "part number" in text or "oem" in text:
            return "parts_lookup"
        if "техобслуж" in text or "maintenance" in text or "service" in text or "процени то" in text or "то на" in text:
            return "maintenance_estimate"
        if "заказ-наряд" in text or "repair order" in text or "work order" in text:
            return "repair_order_assist"
        if "касс" in text or "оплат" in text or "cash" in text or "payment" in text:
            return "cash_review"
        if "обзор" in text or "просроч" in text or "review board" in text or "review the board" in text:
            return "board_review"
        return "general"

    def _is_card_cleanup_task(self, task: dict[str, Any], metadata: dict[str, Any]) -> bool:
        if not self._cleanup_card_id(metadata):
            return False
        text = self._normalized_task_text(str(task.get("task_text", "") or ""))
        cleanup_markers = (
            "наведи порядок",
            "порядок в карточке",
            "структурир",
            "заполни карточ",
            "cleanup",
            "clean up",
            "tidy up",
            "structure the card",
        )
        for marker in cleanup_markers:
            if marker in text:
                return True
        return ("карточ" in text or "card" in text) and ("структур" in text or "заполни" in text or "поряд" in text)

    def _normalized_task_text(self, value: str) -> str:
        text = " ".join(str(value or "").strip().lower().split())
        if not text:
            return ""
        repaired = self._repair_mojibake_text(text)
        return repaired if self._task_text_score(repaired) > self._task_text_score(text) else text

    def _repair_mojibake_text(self, text: str) -> str:
        candidates = [text]
        for encoding in ("latin1", "cp1251", "cp866"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidates.append(" ".join(repaired.lower().split()))
        best = text
        best_score = self._task_text_score(text)
        for candidate in candidates[1:]:
            score = self._task_text_score(candidate)
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _task_text_score(self, text: str) -> int:
        normalized = str(text or "").lower()
        keywords = (
            "наведи",
            "поряд",
            "карточ",
            "структур",
            "заполни",
            "vin",
            "расшифр",
            "запчаст",
            "каталож",
            "касс",
            "оплат",
            "обзор",
            "просроч",
            "техобслуж",
            "maintenance",
            "service",
        )
        score = sum(8 for keyword in keywords if keyword in normalized)
        score += sum(1 for char in normalized if ("а" <= char <= "я") or char == "ё")
        score -= normalized.count("?") * 4
        score -= normalized.count("�") * 6
        return score

    def _extract_card_update_apply(self, decision: dict[str, Any], *, cleanup_card_id: str) -> dict[str, Any] | None:
        payload = decision.get("apply")
        if not isinstance(payload, dict):
            return None
        if str(payload.get("type", "") or "").strip().lower() != "update_card":
            return None
        card_id = str(payload.get("card_id", "") or "").strip() or cleanup_card_id
        if not card_id:
            return None
        update_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        normalized_payload: dict[str, Any] = {"card_id": card_id}
        for field_name in ("vehicle", "title", "description", "deadline", "tags", "vehicle_profile", "repair_order"):
            if field_name in update_payload:
                normalized_payload[field_name] = update_payload[field_name]
        return normalized_payload if len(normalized_payload) > 1 else None

    def _summarize_applied_update(self, args: dict[str, Any], result_payload: dict[str, Any]) -> list[str]:
        response_data = self._response_data(result_payload)
        changed_payload = response_data.get("changed")
        if not isinstance(changed_payload, list):
            meta = response_data.get("meta") if isinstance(response_data.get("meta"), dict) else {}
            changed_payload = meta.get("changed_fields")
        changed_fields = (
            [str(item or "").strip() for item in changed_payload if str(item or "").strip()]
            if isinstance(changed_payload, list)
            else []
        )
        if not changed_fields:
            changed_fields = [
                field_name
                for field_name in ("vehicle", "title", "description", "deadline", "tags", "vehicle_profile", "repair_order")
                if field_name in args
            ]
        labels = {
            "vehicle": "автомобиль",
            "title": "краткая суть",
            "description": "описание",
            "deadline": "сигнал",
            "tags": "метки",
            "vehicle_profile": "паспорт автомобиля",
            "repair_order": "заказ-наряд",
        }
        return [labels.get(item, item) for item in changed_fields]

    def _append_applied_updates(self, display: dict[str, Any], applied_updates: list[str]) -> dict[str, Any]:
        unique_updates: list[str] = []
        seen: set[str] = set()
        for item in applied_updates:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            unique_updates.append(value)
        if not unique_updates:
            return display
        payload = dict(display)
        sections = list(payload.get("sections") or [])
        sections.insert(
            0,
            {
                "title": "Применено",
                "body": "",
                "items": [f"Обновлено поле: {item}" for item in unique_updates],
            },
        )
        payload["sections"] = sections[:6]
        return payload

    def _card_cleanup_apply_instruction(self, card_id: str) -> str:
        return (
            "This is a card cleanup task opened from a card.\n"
            f"Apply confident changes to card {card_id} with update_card before the final answer.\n"
            "Preserve the existing card text and only add or reorganize useful information.\n"
            "External facts found during this task may be added only when they are clearly grounded by the tool results.\n"
            "AI-added notes or follow-up questions inside the description must be labeled with 'ИИ:' or 'AI:'.\n"
            "If nothing can be safely changed, return a final answer that explicitly says no card fields were changed and why."
        )


    def _update_board_control_runtime_after_task(self, *, task: dict[str, Any], orchestration: dict[str, Any] | None) -> None:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if str(metadata.get("purpose", "") or "").strip().lower() != "board_control":
            return
        card_id = self._cleanup_card_id(metadata)
        if not card_id:
            return
        status = self._storage.read_status()
        runtime = status.get("board_control") if isinstance(status.get("board_control"), dict) else {}
        runtime = dict(runtime)
        cache = dict(runtime.get("card_cache") or {})
        cache_entry = dict(cache.get(card_id) or {})
        patch_payload = orchestration.get("patch") if isinstance(orchestration, dict) and isinstance(orchestration.get("patch"), dict) else {}
        verify_payload = orchestration.get("verify") if isinstance(orchestration, dict) and isinstance(orchestration.get("verify"), dict) else {}
        card_patch = patch_payload.get("card_patch") if isinstance(patch_payload.get("card_patch"), dict) else {}
        wrote_anything = bool(card_patch)
        verify_ok = bool(verify_payload.get("applied_ok"))
        cache_entry["last_result"] = "written" if wrote_anything and verify_ok else ("completed_no_write" if not wrote_anything else "verify_failed")
        cache_entry["last_verify_ok"] = verify_ok
        cache_entry["last_processed_at"] = utc_now_iso()
        cache[card_id] = cache_entry
        runtime["card_cache"] = cache
        if wrote_anything and verify_ok:
            runtime["written_count"] = int(runtime.get("written_count", 0) or 0) + 1
        traces = list(runtime.get("recent_traces") or [])
        traces.insert(
            0,
            {
                "card_id": card_id,
                "status": cache_entry["last_result"],
                "verify_ok": verify_ok,
                "written": wrote_anything,
                "at": utc_now_iso(),
            },
        )
        runtime["recent_traces"] = traces[:24]
        self._storage.update_status(board_control=runtime)

    def _update_board_control_runtime_after_failure(self, *, task: dict[str, Any], error: str) -> None:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if str(metadata.get("purpose", "") or "").strip().lower() != "board_control":
            return
        card_id = self._cleanup_card_id(metadata)
        status = self._storage.read_status()
        runtime = status.get("board_control") if isinstance(status.get("board_control"), dict) else {}
        runtime = dict(runtime)
        cache = dict(runtime.get("card_cache") or {})
        if card_id:
            cache_entry = dict(cache.get(card_id) or {})
            cache_entry["last_result"] = "failed"
            cache_entry["last_verify_ok"] = False
            cache_entry["last_processed_at"] = utc_now_iso()
            cache[card_id] = cache_entry
        runtime["card_cache"] = cache
        runtime["error_count"] = int(runtime.get("error_count", 0) or 0) + 1
        traces = list(runtime.get("recent_traces") or [])
        traces.insert(
            0,
            {
                "card_id": card_id,
                "status": "failed",
                "error": str(error or "").strip(),
                "at": utc_now_iso(),
            },
        )
        runtime["recent_traces"] = traces[:24]
        self._storage.update_status(board_control=runtime)


def build_board_api_client(*, logger: logging.Logger) -> BoardApiClient:
    board_api_url = get_agent_board_api_url() or discover_board_api(timeout_seconds=1.0)
    if not board_api_url:
        raise RuntimeError("Unable to discover a reachable local board API for the server agent.")
    try:
        client = BoardApiClient(board_api_url, logger=logger, default_source="agent")
        health = client.health()
    except BoardApiTransportError as exc:
        raise RuntimeError(f"Board API is not reachable for the server agent: {exc}") from exc
    if not health.get("ok"):
        raise RuntimeError("Board API health check failed for the server agent.")
    return client


def run_agent_loop(*, logger: logging.Logger) -> int:
    if not get_agent_enabled():
        logger.info("agent_runtime_disabled")
        return 0
    storage = AgentStorage()
    idle_sleep = get_agent_poll_interval_seconds()
    if not storage.read_prompt_text().strip():
        storage.write_prompt_text(DEFAULT_SYSTEM_PROMPT)
    if not storage.read_memory_text().strip():
        storage.write_memory_text(
            "CRM URL: https://crm.autostopcrm.ru\n"
            "MCP URL: https://crm.autostopcrm.ru/mcp\n"
            "Default admin: admin/admin\n"
            "Use cashbox names exactly as they exist.\n"
            "If payment goes to cashbox 'Безналичный', the repair order adds 15% taxes and fees from that payment amount.\n"
            "Cashboxes 'Наличный' and 'Карта Мария' do not add taxes and fees.\n"
        )
    board_api = None
    while board_api is None:
        try:
            board_api = build_board_api_client(logger=logger)
        except Exception as exc:
            storage.update_status(
                running=False,
                current_task_id=None,
                current_run_id=None,
                last_heartbeat=utc_now_iso(),
                last_error=str(exc),
            )
            logger.warning("agent_waiting_for_board_api error=%s", exc)
            time.sleep(idle_sleep)
    model_client = OpenAIJsonAgentClient()
    runner = AgentRunner(storage=storage, board_api=board_api, model_client=model_client, logger=logger)
    logger.info("agent_runtime_started model=%s board_api_url=%s", get_agent_openai_model(), board_api.base_url)
    while True:
        try:
            processed = runner.run_once()
        except KeyboardInterrupt:
            break
        except Exception as exc:
            storage.update_status(
                running=False,
                current_task_id=None,
                current_run_id=None,
                last_heartbeat=utc_now_iso(),
                last_error=str(exc),
            )
            logger.exception("agent_runtime_loop_failed error=%s", exc)
            time.sleep(idle_sleep)
            continue
        time.sleep(idle_sleep if not processed else 0.2)
    return 0
