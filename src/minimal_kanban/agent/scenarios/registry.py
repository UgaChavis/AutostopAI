from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .base import ScenarioContext, ScenarioExecutionResult, ScenarioExecutor
from .dtc_lookup import DtcLookupScenarioExecutor
from .fault_research import FaultResearchScenarioExecutor
from .maintenance_lookup import MaintenanceLookupScenarioExecutor
from .parts_lookup import PartsLookupScenarioExecutor
from .vin_enrichment import VinEnrichmentScenarioExecutor


@dataclass(frozen=True)
class PassthroughScenarioExecutor:
    scenario_id: str

    def execute(self, context: ScenarioContext) -> ScenarioExecutionResult:
        return ScenarioExecutionResult(
            scenario_id=self.scenario_id,
            status="registered",
            notes=[f"Scenario '{self.scenario_id}' is registered but still executed by legacy runner logic."],
        )


class ScenarioRegistry:
    def __init__(self, executors: Iterable[ScenarioExecutor] | None = None) -> None:
        self._executors: dict[str, ScenarioExecutor] = {}
        for executor in executors or ():
            self.register(executor)

    def register(self, executor: ScenarioExecutor) -> None:
        scenario_id = str(getattr(executor, "scenario_id", "") or "").strip().lower()
        if not scenario_id:
            raise ValueError("Scenario executor must define a non-empty scenario_id.")
        self._executors[scenario_id] = executor

    def get(self, scenario_id: str) -> ScenarioExecutor | None:
        return self._executors.get(str(scenario_id or "").strip().lower())

    def has(self, scenario_id: str) -> bool:
        return self.get(scenario_id) is not None

    def names(self) -> list[str]:
        return sorted(self._executors)


def build_default_scenario_registry() -> ScenarioRegistry:
    return ScenarioRegistry(
        [
            VinEnrichmentScenarioExecutor(),
            PartsLookupScenarioExecutor(),
            MaintenanceLookupScenarioExecutor(),
            DtcLookupScenarioExecutor(),
            FaultResearchScenarioExecutor(),
            PassthroughScenarioExecutor("normalization"),
            PassthroughScenarioExecutor("repair_order_assistance"),
            PassthroughScenarioExecutor("board_review"),
            PassthroughScenarioExecutor("cash_review"),
            PassthroughScenarioExecutor("freeform_manual"),
        ]
    )
