from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ScenarioContext:
    scenario_id: str
    task_id: str
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    scenario_payload: dict[str, Any] = field(default_factory=dict)
    runtime: Any | None = None


@dataclass(frozen=True)
class ScenarioExecutionResult:
    scenario_id: str
    status: str
    tool_calls_used: int = 0
    tool_results: list[Any] = field(default_factory=list)
    orchestration_updates: dict[str, Any] = field(default_factory=dict)
    facts_updates: dict[str, Any] = field(default_factory=dict)
    patch: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_followup: bool = False
    followup_reason: str = ""


class ScenarioExecutor(Protocol):
    scenario_id: str

    def execute(self, context: ScenarioContext) -> ScenarioExecutionResult:
        ...
