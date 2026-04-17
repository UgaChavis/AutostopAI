from __future__ import annotations

from dataclasses import dataclass

from .base import ScenarioContext, ScenarioExecutionResult


@dataclass(frozen=True)
class DtcLookupScenarioExecutor:
    scenario_id: str = "dtc_lookup"

    def execute(self, context: ScenarioContext) -> ScenarioExecutionResult:
        runtime = context.runtime
        facts = context.facts
        scenario = context.scenario_payload
        if runtime is None:
            raise ValueError("DtcLookupScenarioExecutor requires runtime.")
        dtc_code = str(scenario.get("code", "") or "").strip() or (facts["dtc_codes"][0] if facts["dtc_codes"] else "")
        if not dtc_code:
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="skipped",
                notes=["dtc lookup skipped: no DTC code"],
            )
        runtime._record_log_action(
            task_id=context.task_id,
            run_id=context.run_id,
            step=0,
            level="RUN",
            phase="tool",
            message="dtc lookup started.",
        )
        payload = runtime._run_autofill_tool(
            task_id=context.task_id,
            run_id=context.run_id,
            step=1,
            tool_name="decode_dtc",
            args={
                "code": dtc_code,
                "vehicle_context": facts["vehicle_context"],
            },
            reason="Decode the highest-priority detected DTC code",
        )
        if payload is None:
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="failed",
                warnings=["dtc lookup request failed"],
                needs_followup=True,
                followup_reason="dtc_lookup_failed",
            )
        orchestration_payload = runtime._response_data(payload) or payload
        if runtime._is_partial_tool_payload(payload):
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="partial",
                tool_calls_used=1,
                orchestration_updates={"decode_dtc": orchestration_payload},
                tool_results=[
                    runtime._build_tool_result(
                        "decode_dtc",
                        payload,
                        status="partial",
                        reason="Decode the highest-priority detected DTC code",
                        scenario_id=self.scenario_id,
                        evidence_ref="dtc_codes",
                    )
                ],
                warnings=["dtc lookup deferred: external budget exceeded"] if runtime._is_budget_exceeded_payload(payload) else ["dtc lookup returned partial result"],
                needs_followup=True,
                followup_reason="dtc_lookup_budget_deferred" if runtime._is_budget_exceeded_payload(payload) else "dtc_lookup_partial",
            )
        has_useful_result = runtime._search_payload_has_useful_result(orchestration_payload)
        if isinstance(facts.get("evidence_model"), dict) and has_useful_result:
            facts["evidence_model"]["external_result_sufficient"] = True
        return ScenarioExecutionResult(
            scenario_id=self.scenario_id,
            status="success",
            tool_calls_used=1,
            orchestration_updates={"decode_dtc": orchestration_payload},
            tool_results=[
                runtime._build_tool_result(
                    "decode_dtc",
                    payload,
                    status="success",
                    reason="Decode the highest-priority detected DTC code",
                    scenario_id=self.scenario_id,
                    evidence_ref="dtc_codes",
                )
            ],
            warnings=["dtc lookup returned no reliable diagnostic excerpt"] if not has_useful_result else [],
            needs_followup=not has_useful_result,
            followup_reason="dtc_lookup_insufficient" if not has_useful_result else "",
        )
