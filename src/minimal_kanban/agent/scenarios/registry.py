from __future__ import annotations

from typing import Iterable

from .base import ScenarioContext, ScenarioExecutionResult, ScenarioExecutor
from .vin_enrichment import VinEnrichmentScenarioExecutor


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
    return ScenarioRegistry([VinEnrichmentScenarioExecutor()])
