from __future__ import annotations

from dataclasses import dataclass

from .base import ScenarioContext, ScenarioExecutionResult


@dataclass(frozen=True)
class PartsLookupScenarioExecutor:
    scenario_id: str = "parts_lookup"

    def execute(self, context: ScenarioContext) -> ScenarioExecutionResult:
        runtime = context.runtime
        facts = context.facts
        scenario = context.scenario_payload
        if runtime is None:
            raise ValueError("PartsLookupScenarioExecutor requires runtime.")
        part_query = str(scenario.get("query", "") or "").strip() or (facts["part_queries"][0] if facts["part_queries"] else "")
        if not part_query:
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="skipped",
                notes=["parts lookup skipped: no part query"],
            )
        if not runtime._card_autofill_can_run_parts_lookup(facts):
            runtime._record_log_action(
                task_id=context.task_id,
                run_id=context.run_id,
                step=0,
                level="INFO",
                phase="tool",
                message="parts lookup skipped: no trusted vehicle context after VIN gate.",
            )
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="skipped",
                notes=["parts lookup skipped after VIN gate"],
                warnings=["parts lookup is waiting for trusted vehicle context"],
                needs_followup=True,
                followup_reason="parts_lookup_waiting_vehicle_context",
            )
        runtime._record_log_action(
            task_id=context.task_id,
            run_id=context.run_id,
            step=0,
            level="RUN",
            phase="tool",
            message="parts lookup started.",
        )
        part_payload = runtime._run_autofill_tool(
            task_id=context.task_id,
            run_id=context.run_id,
            step=1,
            tool_name="find_part_numbers",
            args={
                "query": part_query,
                "vehicle": facts["vehicle_context"],
                "limit": 5,
            },
            reason="Find OEM and analog part numbers for the main detected part request",
        )
        if part_payload is None:
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="failed",
                warnings=["parts lookup request failed"],
                needs_followup=True,
                followup_reason="parts_lookup_failed",
            )
        orchestration_updates = {"find_part_numbers": runtime._response_data(part_payload) or part_payload}
        if runtime._is_partial_tool_payload(part_payload):
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="partial",
                tool_calls_used=1,
                tool_results=[
                    runtime._build_tool_result(
                        "find_part_numbers",
                        part_payload,
                        status="partial",
                        reason="Find OEM and analog part numbers for the main detected part request",
                        scenario_id=self.scenario_id,
                        evidence_ref="part_queries",
                    )
                ],
                orchestration_updates=orchestration_updates,
                warnings=["parts lookup deferred: external budget exceeded"] if runtime._is_budget_exceeded_payload(part_payload) else ["parts lookup returned partial result"],
                needs_followup=True,
                followup_reason="parts_lookup_budget_deferred" if runtime._is_budget_exceeded_payload(part_payload) else "parts_lookup_partial",
            )
        tool_results = [
            runtime._build_tool_result(
                "find_part_numbers",
                part_payload,
                status="success",
                reason="Find OEM and analog part numbers for the main detected part request",
                scenario_id=self.scenario_id,
                evidence_ref="part_queries",
            )
        ]
        tool_calls_used = 1
        has_useful_parts = runtime._part_lookup_has_useful_result(orchestration_updates["find_part_numbers"])
        if isinstance(facts.get("evidence_model"), dict) and has_useful_parts:
            facts["evidence_model"]["external_result_sufficient"] = True
        if bool(scenario.get("with_price")):
            best_part_number = runtime._pick_best_part_number(orchestration_updates["find_part_numbers"])
            if best_part_number:
                price_payload = runtime._run_autofill_tool(
                    task_id=context.task_id,
                    run_id=context.run_id,
                    step=tool_calls_used + 1,
                    tool_name="estimate_price_ru",
                    args={
                        "part_number": best_part_number,
                        "vehicle": facts["vehicle_context"],
                        "limit": 5,
                    },
                    reason="Estimate Russian-market price for the strongest matched part number",
                )
                if price_payload is not None:
                    tool_calls_used += 1
                    orchestration_updates["estimate_price_ru"] = runtime._response_data(price_payload) or price_payload
                    tool_results.append(
                        runtime._build_tool_result(
                            "estimate_price_ru",
                            price_payload,
                            status="success",
                            reason="Estimate Russian-market price for the strongest matched part number",
                            scenario_id=self.scenario_id,
                            evidence_ref="part_queries",
                        )
                    )
        warnings: list[str] = []
        needs_followup = False
        followup_reason = ""
        if not has_useful_parts:
            warnings.append("parts lookup returned no reliable candidate parts")
            needs_followup = True
            followup_reason = "parts_lookup_insufficient"
        price_payload = orchestration_updates.get("estimate_price_ru")
        if isinstance(price_payload, dict) and runtime._is_budget_exceeded_payload(price_payload):
            warnings.append("price lookup deferred: external budget exceeded")
            needs_followup = True
            if not followup_reason:
                followup_reason = "price_lookup_budget_deferred"
        return ScenarioExecutionResult(
            scenario_id=self.scenario_id,
            status="success",
            tool_calls_used=tool_calls_used,
            tool_results=tool_results,
            orchestration_updates=orchestration_updates,
            warnings=warnings,
            needs_followup=needs_followup,
            followup_reason=followup_reason,
        )
