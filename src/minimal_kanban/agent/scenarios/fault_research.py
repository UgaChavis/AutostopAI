from __future__ import annotations

from dataclasses import dataclass

from .base import ScenarioContext, ScenarioExecutionResult


@dataclass(frozen=True)
class FaultResearchScenarioExecutor:
    scenario_id: str = "fault_research"

    def execute(self, context: ScenarioContext) -> ScenarioExecutionResult:
        runtime = context.runtime
        facts = context.facts
        if runtime is None:
            raise ValueError("FaultResearchScenarioExecutor requires runtime.")
        runtime._record_log_action(
            task_id=context.task_id,
            run_id=context.run_id,
            step=0,
            level="RUN",
            phase="tool",
            message="fault research started.",
        )
        payload = runtime._run_autofill_tool(
            task_id=context.task_id,
            run_id=context.run_id,
            step=1,
            tool_name="search_fault_info",
            args={
                "query": facts["symptom_query"],
                "vehicle": facts["vehicle_context"],
                "limit": 5,
            },
            reason="Search short symptom context and typical causes for the current complaint",
        )
        if payload is None:
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="failed",
                warnings=["fault research request failed"],
                needs_followup=True,
                followup_reason="fault_research_failed",
            )
        orchestration_payload = runtime._response_data(payload) or payload
        if runtime._is_partial_tool_payload(payload):
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="partial",
                tool_calls_used=1,
                orchestration_updates={"search_fault_info": orchestration_payload},
                tool_results=[
                    runtime._build_tool_result(
                        "search_fault_info",
                        payload,
                        status="partial",
                        reason="Search short symptom context and typical causes for the current complaint",
                        scenario_id=self.scenario_id,
                        evidence_ref="symptom_query",
                    )
                ],
                warnings=["fault research deferred: external budget exceeded"] if runtime._is_budget_exceeded_payload(payload) else ["fault research returned partial result"],
                needs_followup=True,
                followup_reason="fault_research_budget_deferred" if runtime._is_budget_exceeded_payload(payload) else "fault_research_partial",
            )
        has_useful_result = runtime._search_payload_has_useful_result(orchestration_payload)
        if isinstance(facts.get("evidence_model"), dict) and has_useful_result:
            facts["evidence_model"]["external_result_sufficient"] = True
        return ScenarioExecutionResult(
            scenario_id=self.scenario_id,
            status="success",
            tool_calls_used=1,
            orchestration_updates={"search_fault_info": orchestration_payload},
            tool_results=[
                runtime._build_tool_result(
                    "search_fault_info",
                    payload,
                    status="success",
                    reason="Search short symptom context and typical causes for the current complaint",
                    scenario_id=self.scenario_id,
                    evidence_ref="symptom_query",
                )
            ],
            warnings=["fault research returned no reliable symptom result"] if not has_useful_result else [],
            needs_followup=not has_useful_result,
            followup_reason="fault_research_insufficient" if not has_useful_result else "",
        )
