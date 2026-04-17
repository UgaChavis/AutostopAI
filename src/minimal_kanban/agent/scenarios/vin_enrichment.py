from __future__ import annotations

from dataclasses import dataclass

from .base import ScenarioContext, ScenarioExecutionResult


@dataclass(frozen=True)
class VinEnrichmentScenarioExecutor:
    scenario_id: str = "vin_enrichment"

    def execute(self, context: ScenarioContext) -> ScenarioExecutionResult:
        runtime = context.runtime
        facts = context.facts
        if runtime is None:
            raise ValueError("VinEnrichmentScenarioExecutor requires runtime.")
        facts["vin_decode_attempted"] = True
        runtime._record_log_action(
            task_id=context.task_id,
            run_id=context.run_id,
            step=0,
            level="RUN",
            phase="tool",
            message="decode_vin requested.",
        )
        vin_payload = runtime._run_autofill_tool(
            task_id=context.task_id,
            run_id=context.run_id,
            step=1,
            tool_name="decode_vin",
            args={"vin": facts["vin"]},
            reason="Decode VIN first and reuse the confirmed vehicle facts in later lookup steps",
        )
        if vin_payload is None:
            facts["vin_decode_status"] = "failed"
            runtime._record_log_action(
                task_id=context.task_id,
                run_id=context.run_id,
                step=1,
                level="WARN",
                phase="tool",
                message="decode_vin failed.",
            )
            return ScenarioExecutionResult(
                scenario_id=self.scenario_id,
                status="failed",
                facts_updates={"vin_decode_status": "failed"},
                warnings=["vin decode request failed"],
                needs_followup=True,
                followup_reason="vin_decode_failed",
            )
        orchestration_payload = runtime._response_data(vin_payload) or vin_payload
        vin_status = runtime._vin_decode_status(orchestration_payload)
        facts["vin_decode_status"] = vin_status
        if isinstance(facts.get("evidence_model"), dict):
            facts["evidence_model"]["external_result_sufficient"] = vin_status == "success"
        if vin_status == "success":
            facts["vehicle_context"] = runtime._merge_vehicle_context(
                facts["vehicle_context"],
                orchestration_payload,
            )
        return ScenarioExecutionResult(
            scenario_id=self.scenario_id,
            status="success",
            tool_calls_used=1,
            tool_results=[
                runtime._build_tool_result(
                    "decode_vin",
                    vin_payload,
                    status="success",
                    reason="Decode VIN first and reuse the confirmed vehicle facts in later lookup steps",
                    scenario_id=self.scenario_id,
                    evidence_ref="vin",
                )
            ],
            orchestration_updates={"decode_vin": orchestration_payload},
            facts_updates={
                "vin_decode_status": vin_status,
                "vehicle_context": dict(facts.get("vehicle_context") or {}),
            },
            warnings=["vin decode returned sparse data"] if vin_status == "insufficient" else [],
            needs_followup=vin_status in {"insufficient", "failed"},
            followup_reason="vin_decode_insufficient" if vin_status == "insufficient" else ("vin_decode_failed" if vin_status == "failed" else ""),
        )
