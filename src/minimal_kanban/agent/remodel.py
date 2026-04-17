from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = (os.environ.get(name) or "").strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "y", "on"}


class AiScenarioId(str, Enum):
    AI_CHAT = "ai_chat"
    FULL_CARD_ENRICHMENT = "full_card_enrichment"
    BOARD_CONTROL = "board_control"


class AiTriggerKind(str, Enum):
    USER_INVOKED = "user_invoked"
    SCHEDULED = "scheduled"
    BACKGROUND = "background"


class AiActorMode(str, Enum):
    INTERACTIVE = "interactive"
    BACKGROUND = "background"


class AiInteractionStyle(str, Enum):
    CONVERSATIONAL = "conversational"
    DETERMINISTIC = "deterministic"


class AiScopeKind(str, Enum):
    WORKSPACE = "workspace"
    CARD = "card"
    BOARD = "board"


class AiWritePolicy(str, Enum):
    READ_HEAVY_RESTRICTED_WRITE = "read_heavy_restricted_write"
    BOUNDED_WRITE = "bounded_write"
    BOUNDED_BACKGROUND_WRITE = "bounded_background_write"


class AiContextSource(str, Enum):
    CARD_CONTEXT = "card_context"
    REPAIR_ORDER_CONTEXT = "repair_order_context"
    WALL_DIGEST = "wall_digest"
    CURATED_INTERNAL_DOCS = "curated_internal_docs"
    INTERNET_LOOKUP = "internet_lookup"
    DELTA_BOARD_CONTEXT = "delta_board_context"
    ATTACHMENTS = "attachments"


class AiRolloutState(str, Enum):
    DISABLED = "disabled"
    HIDDEN = "hidden"
    AVAILABLE = "available"
    PRIMARY = "primary"
    LEGACY_ONLY = "legacy_only"


@dataclass(frozen=True)
class AiScenarioDefinition:
    scenario_id: AiScenarioId
    display_intent: str
    trigger_kind: AiTriggerKind
    actor_mode: AiActorMode
    interaction_style: AiInteractionStyle
    scope_kind: AiScopeKind
    context_sources: tuple[AiContextSource, ...]
    tool_classes: tuple[str, ...]
    write_policy: AiWritePolicy
    allowed_entry_surfaces: tuple[str, ...]
    legacy_replacement_scope: str
    future_module_owner: str
    boundaries: tuple[str, ...]
    non_goals: tuple[str, ...]
    stage: str = "planned"
    default_enabled: bool = False

    def to_dict(self, *, enabled: bool, rollout_state: AiRolloutState) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id.value,
            "display_intent": self.display_intent,
            "stage": self.stage,
            "enabled": bool(enabled),
            "default_enabled": bool(self.default_enabled),
            "rollout_state": rollout_state.value,
            "trigger_kind": self.trigger_kind.value,
            "actor_mode": self.actor_mode.value,
            "interaction_style": self.interaction_style.value,
            "scope_kind": self.scope_kind.value,
            "context_sources": [item.value for item in self.context_sources],
            "tool_classes": list(self.tool_classes),
            "write_policy": self.write_policy.value,
            "allowed_entry_surfaces": list(self.allowed_entry_surfaces),
            "legacy_replacement_scope": self.legacy_replacement_scope,
            "future_module_owner": self.future_module_owner,
            "boundaries": list(self.boundaries),
            "non_goals": list(self.non_goals),
        }


@dataclass(frozen=True)
class AiFeatureFlags:
    legacy_ux_enabled: bool = True
    ai_chat_enabled: bool = False
    full_card_enrichment_enabled: bool = True
    board_control_enabled: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "legacy_ux_enabled": bool(self.legacy_ux_enabled),
            "ai_chat_enabled": bool(self.ai_chat_enabled),
            "full_card_enrichment_enabled": bool(self.full_card_enrichment_enabled),
            "board_control_enabled": bool(self.board_control_enabled),
        }


@dataclass(frozen=True)
class AiModeState:
    scenario_id: AiScenarioId
    rollout_state: AiRolloutState
    primary_interactive: bool
    legacy_compatible_only: bool
    hidden: bool
    background_only: bool
    enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id.value,
            "rollout_state": self.rollout_state.value,
            "primary_interactive": bool(self.primary_interactive),
            "legacy_compatible_only": bool(self.legacy_compatible_only),
            "hidden": bool(self.hidden),
            "background_only": bool(self.background_only),
            "enabled": bool(self.enabled),
        }


@dataclass(frozen=True)
class AiModeConfig:
    legacy_ux_enabled: bool
    scenario_state: dict[AiScenarioId, AiModeState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_ux_enabled": bool(self.legacy_ux_enabled),
            "scenario_state": {
                scenario.value: state.to_dict()
                for scenario, state in self.scenario_state.items()
            },
        }


@dataclass(frozen=True)
class AiScenarioRegistry:
    scenarios: tuple[AiScenarioDefinition, ...]
    by_id: dict[str, AiScenarioDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mapping = {item.scenario_id.value: item for item in self.scenarios}
        object.__setattr__(self, "by_id", mapping)

    def get(self, scenario_id: str) -> AiScenarioDefinition | None:
        return self.by_id.get(str(scenario_id or "").strip().lower())

    def ids(self) -> list[str]:
        return [item.scenario_id.value for item in self.scenarios]

    def rollout_state_for(self, scenario_id: AiScenarioId, flags: AiFeatureFlags) -> AiRolloutState:
        if scenario_id == AiScenarioId.AI_CHAT:
            if not flags.legacy_ux_enabled and flags.ai_chat_enabled:
                return AiRolloutState.PRIMARY
            if flags.ai_chat_enabled:
                return AiRolloutState.AVAILABLE
            return AiRolloutState.HIDDEN
        if scenario_id == AiScenarioId.FULL_CARD_ENRICHMENT:
            if flags.full_card_enrichment_enabled:
                return AiRolloutState.AVAILABLE
            return AiRolloutState.HIDDEN
        if scenario_id == AiScenarioId.BOARD_CONTROL:
            if flags.board_control_enabled:
                return AiRolloutState.AVAILABLE
            return AiRolloutState.HIDDEN
        return AiRolloutState.DISABLED

    def mode_state_for(self, scenario_id: AiScenarioId, flags: AiFeatureFlags) -> AiModeState:
        rollout_state = self.rollout_state_for(scenario_id, flags)
        enabled = rollout_state in {AiRolloutState.AVAILABLE, AiRolloutState.PRIMARY}
        return AiModeState(
            scenario_id=scenario_id,
            rollout_state=rollout_state,
            primary_interactive=rollout_state == AiRolloutState.PRIMARY,
            legacy_compatible_only=rollout_state == AiRolloutState.LEGACY_ONLY,
            hidden=rollout_state == AiRolloutState.HIDDEN,
            background_only=scenario_id == AiScenarioId.BOARD_CONTROL,
            enabled=enabled,
        )

    def to_dict(self, flags: AiFeatureFlags | None = None) -> dict[str, dict[str, Any]]:
        resolved_flags = flags or get_ai_feature_flags()
        return {
            item.scenario_id.value: item.to_dict(
                enabled=self.mode_state_for(item.scenario_id, resolved_flags).enabled,
                rollout_state=self.rollout_state_for(item.scenario_id, resolved_flags),
            )
            for item in self.scenarios
        }


SCENARIO_DEFINITIONS: tuple[AiScenarioDefinition, ...] = (
    AiScenarioDefinition(
        scenario_id=AiScenarioId.AI_CHAT,
        display_intent="Future full-size conversational AI console for CRM operators.",
        trigger_kind=AiTriggerKind.USER_INVOKED,
        actor_mode=AiActorMode.INTERACTIVE,
        interaction_style=AiInteractionStyle.CONVERSATIONAL,
        scope_kind=AiScopeKind.WORKSPACE,
        context_sources=(
            AiContextSource.CARD_CONTEXT,
            AiContextSource.REPAIR_ORDER_CONTEXT,
            AiContextSource.WALL_DIGEST,
            AiContextSource.CURATED_INTERNAL_DOCS,
            AiContextSource.INTERNET_LOOKUP,
        ),
        tool_classes=("read", "research", "summarize"),
        write_policy=AiWritePolicy.READ_HEAVY_RESTRICTED_WRITE,
        allowed_entry_surfaces=("future_ai_chat_window",),
        legacy_replacement_scope="legacy_agent_modal_manual_tasks",
        future_module_owner="Module 1.2 AI Chat Console",
        boundaries=("not_background", "not_board_daemon", "not_default_write_mode"),
        non_goals=("no legacy modal reuse", "no silent scheduling", "no broad board mutations"),
    ),
    AiScenarioDefinition(
        scenario_id=AiScenarioId.FULL_CARD_ENRICHMENT,
        display_intent="Future bounded card enrichment flow launched from the card indicator.",
        trigger_kind=AiTriggerKind.USER_INVOKED,
        actor_mode=AiActorMode.INTERACTIVE,
        interaction_style=AiInteractionStyle.DETERMINISTIC,
        scope_kind=AiScopeKind.CARD,
        context_sources=(
            AiContextSource.CARD_CONTEXT,
            AiContextSource.REPAIR_ORDER_CONTEXT,
            AiContextSource.WALL_DIGEST,
            AiContextSource.ATTACHMENTS,
        ),
        tool_classes=("read", "bounded_research", "normalize", "patch", "verify"),
        write_policy=AiWritePolicy.BOUNDED_WRITE,
        allowed_entry_surfaces=("future_card_enrichment_trigger", "card_indicator"),
        legacy_replacement_scope="legacy_card_agent_button_and_card_autofill_menu",
        future_module_owner="Module 1.3 Card Enrichment Pipeline",
        boundaries=("not_open_ended_chat", "not_board_scope", "not_menu_of_actions"),
        non_goals=("no freeform assistant mode", "no hidden scheduling", "no broad board-level writes"),
        default_enabled=True,
    ),
    AiScenarioDefinition(
        scenario_id=AiScenarioId.BOARD_CONTROL,
        display_intent="Future background board hygiene mode for narrow delta-driven maintenance.",
        trigger_kind=AiTriggerKind.SCHEDULED,
        actor_mode=AiActorMode.BACKGROUND,
        interaction_style=AiInteractionStyle.DETERMINISTIC,
        scope_kind=AiScopeKind.BOARD,
        context_sources=(
            AiContextSource.DELTA_BOARD_CONTEXT,
            AiContextSource.WALL_DIGEST,
            AiContextSource.CARD_CONTEXT,
            AiContextSource.REPAIR_ORDER_CONTEXT,
        ),
        tool_classes=("read", "bounded_update", "verify"),
        write_policy=AiWritePolicy.BOUNDED_BACKGROUND_WRITE,
        allowed_entry_surfaces=("settings_toggle", "background_scheduler"),
        legacy_replacement_scope="legacy_board_scheduler_and_manual_board_agent_review",
        future_module_owner="Module 1.4 Board Control Mode",
        boundaries=("not_user_chat", "not_free_agent", "not_autonomous_dispatcher"),
        non_goals=("no arbitrary moves", "no arbitrary deletes", "no free repair-order mutation"),
    ),
)

LEGACY_SCENARIO_NAMES: tuple[str, ...] = (
    "board_review",
    "cash_review",
    "dtc_lookup",
    "fault_research",
    "freeform_manual",
    "maintenance_lookup",
    "normalization",
    "parts_lookup",
    "repair_order_assistance",
    "vin_enrichment",
)

LEGACY_AI_ENTRY_POINTS: dict[str, dict[str, Any]] = {
    "board_dock_button": {
        "surface": "web_assets.agentDockButton",
        "role": "legacy_ux_entry_point",
        "replacement": "ai_chat",
    },
    "card_agent_button": {
        "surface": "web_assets.cardAgentButton",
        "role": "legacy_ux_entry_point",
        "replacement": "full_card_enrichment",
    },
    "quick_prompts": {
        "surface": "web_assets.quickAgentPrompts",
        "role": "legacy_ux_shortcut",
        "replacement": "ai_chat / full_card_enrichment",
    },
    "agent_tasks_modal": {
        "surface": "web_assets.agentTasksModal",
        "role": "legacy_ux_scheduler_shell",
        "replacement": "board_control",
    },
    "card_autofill_toggle": {
        "surface": "web_assets.agentAutofillButton",
        "role": "legacy_ux_entry_point",
        "replacement": "full_card_enrichment",
    },
}

class AiBackendComponentKind(str, Enum):
    SERVICE_BOUNDARY = "service_boundary"
    API_SURFACE = "api_surface"
    RUNTIME_CORE = "runtime_core"
    POLICY_ENGINE = "policy_engine"
    STORAGE = "storage"
    CONTEXT_PRIMITIVE = "context_primitive"
    TOOLING = "tooling"
    MODEL_ADAPTER = "model_adapter"
    EXECUTOR = "executor"
    SCHEDULER = "scheduler"
    LEGACY_GLUE = "legacy_glue"
    STATUS_SURFACE = "status_surface"


class AiBackendReuseCategory(str, Enum):
    REUSE_AS_IS = "reuse_as_is"
    REUSE_WITH_ADAPTATION = "reuse_with_adaptation"
    LEGACY_ONLY_OR_RETIRE_LATER = "legacy_only_or_retire_later"


@dataclass(frozen=True)
class AiBackendComponentDefinition:
    component_id: str
    component_kind: AiBackendComponentKind
    current_role: str
    reuse_category: AiBackendReuseCategory
    future_targets: tuple[AiScenarioId, ...]
    notes: tuple[str, ...] = ()
    do_not_break: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_kind": self.component_kind.value,
            "current_role": self.current_role,
            "reuse_category": self.reuse_category.value,
            "future_targets": [item.value for item in self.future_targets],
            "notes": list(self.notes),
            "do_not_break": list(self.do_not_break),
        }


@dataclass(frozen=True)
class AiBackendReuseRegistry:
    components: tuple[AiBackendComponentDefinition, ...]
    by_id: dict[str, AiBackendComponentDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_id", {item.component_id: item for item in self.components})

    def get(self, component_id: str) -> AiBackendComponentDefinition | None:
        return self.by_id.get(str(component_id or "").strip().lower())

    def ids(self) -> list[str]:
        return [item.component_id for item in self.components]

    def grouped_by_category(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {item.value: {} for item in AiBackendReuseCategory}
        for item in self.components:
            grouped[item.reuse_category.value][item.component_id] = item.to_dict()
        return grouped

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {item.component_id: item.to_dict() for item in self.components}


BACKEND_COMPONENT_DEFINITIONS: tuple[AiBackendComponentDefinition, ...] = (
    AiBackendComponentDefinition(
        component_id="card_service",
        component_kind=AiBackendComponentKind.SERVICE_BOUNDARY,
        current_role="source_of_truth for cards, repair orders, attachments, and domain writes",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Keep business writes in CardService.",),
        do_not_break=("do_not_move_domain_logic_to_agent_layer", "preserve_existing_validation_rules"),
    ),
    AiBackendComponentDefinition(
        component_id="local_api",
        component_kind=AiBackendComponentKind.API_SURFACE,
        current_role="read/write boundary exposed to UI, MCP, and agent runtime",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("This is the internal boundary for all AI writes.",),
        do_not_break=("keep_card_service_as_source_of_truth", "preserve_auth_and_operator_checks"),
    ),
    AiBackendComponentDefinition(
        component_id="orchestration_contracts",
        component_kind=AiBackendComponentKind.RUNTIME_CORE,
        current_role="read -> evidence -> plan -> tools -> patch -> write -> verify contracts",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Scenario modules should depend on these contracts instead of ad hoc dict shapes.",),
        do_not_break=("preserve_patch_write_verify_discipline", "keep_backward_serialization_helpers_stable"),
    ),
    AiBackendComponentDefinition(
        component_id="policy_engine",
        component_kind=AiBackendComponentKind.POLICY_ENGINE,
        current_role="required tool gates, scenario policy, and patch filtering",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Future modules should extend policy inputs, not bypass the gate.",),
        do_not_break=("preserve_required_tool_checks", "do_not_allow_unbounded_writes"),
    ),
    AiBackendComponentDefinition(
        component_id="snapshot_service",
        component_kind=AiBackendComponentKind.CONTEXT_PRIMITIVE,
        current_role="compact board/card/wall snapshot and digest assembly",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Foundation for wall digest and delta-oriented reads.",),
        do_not_break=("preserve_revision_signatures", "keep_compact_payload_paths_stable"),
    ),
    AiBackendComponentDefinition(
        component_id="agent_storage",
        component_kind=AiBackendComponentKind.STORAGE,
        current_role="tasks, schedules, runs, actions, status, prompt, and memory persistence",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Persistence layer stays in place for new scenario states and traces.",),
        do_not_break=("preserve_file_locking", "preserve_json_payload_shapes_for_current_runtime"),
    ),
    AiBackendComponentDefinition(
        component_id="agent_runtime_api",
        component_kind=AiBackendComponentKind.API_SURFACE,
        current_role="agent_status, runs, actions, tasks, and enqueue route registration",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Runtime exposure stays stable while new scenario entry points are added later.",),
        do_not_break=("keep_agent_status_contract_stable", "preserve_operator_session_checks"),
    ),
    AiBackendComponentDefinition(
        component_id="runner_model_loop",
        component_kind=AiBackendComponentKind.RUNTIME_CORE,
        current_role="claim, read, evidence, plan, tools, patch, write, verify execution loop",
        reuse_category=AiBackendReuseCategory.REUSE_WITH_ADAPTATION,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Will be adapted into scenario dispatch rather than replaced outright.",),
        do_not_break=("keep_tool_call_accounting", "preserve_trace_and_verification_flow"),
    ),
    AiBackendComponentDefinition(
        component_id="runner_autofill_executors",
        component_kind=AiBackendComponentKind.EXECUTOR,
        current_role="legacy autofill scenario executors and follow-up passes",
        reuse_category=AiBackendReuseCategory.REUSE_WITH_ADAPTATION,
        future_targets=(AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Current autofill logic is the closest foundation for full_card_enrichment.",),
        do_not_break=("preserve_current_autofill_verification_semantics", "keep_partial_result_reporting"),
    ),
    AiBackendComponentDefinition(
        component_id="control_scheduler",
        component_kind=AiBackendComponentKind.SCHEDULER,
        current_role="worker, scheduler, heartbeat, and task claim orchestration",
        reuse_category=AiBackendReuseCategory.REUSE_WITH_ADAPTATION,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Will continue to own runtime supervision, but with later board_control specialization.",),
        do_not_break=("preserve_heartbeat_semantics", "keep_throttle_and_claim_logic_stable"),
    ),
    AiBackendComponentDefinition(
        component_id="automotive_tools",
        component_kind=AiBackendComponentKind.TOOLING,
        current_role="VIN, parts, DTC, fault, and maintenance lookup helpers",
        reuse_category=AiBackendReuseCategory.REUSE_WITH_ADAPTATION,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT),
        notes=("Useful foundation for future enrichment and read-heavy chat answers.",),
        do_not_break=("keep_lookup_budgeting", "preserve_source_whitelist_behavior"),
    ),
    AiBackendComponentDefinition(
        component_id="web_tools",
        component_kind=AiBackendComponentKind.TOOLING,
        current_role="internet search and fetch helpers used by bounded research",
        reuse_category=AiBackendReuseCategory.REUSE_WITH_ADAPTATION,
        future_targets=(AiScenarioId.AI_CHAT,),
        notes=("Will be reused by the chat path under stronger policy gates.",),
        do_not_break=("preserve_domain_whitelist_support", "keep_web_budget_limits"),
    ),
    AiBackendComponentDefinition(
        component_id="openai_client",
        component_kind=AiBackendComponentKind.MODEL_ADAPTER,
        current_role="LLM API client used by the worker runtime",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Generic model transport should stay stable across the remodel.",),
        do_not_break=("preserve_model_selection", "preserve_timeout_and_retry_behavior"),
    ),
    AiBackendComponentDefinition(
        component_id="instructions",
        component_kind=AiBackendComponentKind.MODEL_ADAPTER,
        current_role="system prompt and task prompt assembly",
        reuse_category=AiBackendReuseCategory.REUSE_WITH_ADAPTATION,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT, AiScenarioId.BOARD_CONTROL),
        notes=("Prompt assembly will be split by scenario in later modules.",),
        do_not_break=("preserve_current_instruction_fallbacks", "keep_task_prompt_templates_callable"),
    ),
    AiBackendComponentDefinition(
        component_id="source_registry",
        component_kind=AiBackendComponentKind.TOOLING,
        current_role="allowed source registry and source metadata for web/automotive lookups",
        reuse_category=AiBackendReuseCategory.REUSE_AS_IS,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT),
        notes=("This is a stable allow-list foundation for controlled research.",),
        do_not_break=("preserve_source_whitelist_contract", "preserve_source_metadata_shape"),
    ),
    AiBackendComponentDefinition(
        component_id="manual_prompt_bridge",
        component_kind=AiBackendComponentKind.LEGACY_GLUE,
        current_role="freeform textarea -> agent_enqueue_task translation",
        reuse_category=AiBackendReuseCategory.LEGACY_ONLY_OR_RETIRE_LATER,
        future_targets=(AiScenarioId.AI_CHAT,),
        notes=("This bridge is tied to the old manual modal UX and should shrink later.",),
        do_not_break=("keep_legacy_compatibility_until_ai_chat_exists",),
    ),
    AiBackendComponentDefinition(
        component_id="quick_prompt_bridge",
        component_kind=AiBackendComponentKind.LEGACY_GLUE,
        current_role="canned quick prompt preprocessing and prompt injection",
        reuse_category=AiBackendReuseCategory.LEGACY_ONLY_OR_RETIRE_LATER,
        future_targets=(AiScenarioId.AI_CHAT, AiScenarioId.FULL_CARD_ENRICHMENT),
        notes=("This is a convenience shim over the old modal flow.",),
        do_not_break=("keep_quick_prompt_shortcuts_stable_until_replacement_exists",),
    ),
    AiBackendComponentDefinition(
        component_id="autofill_bridge",
        component_kind=AiBackendComponentKind.LEGACY_GLUE,
        current_role="set_card_ai_autofill and on-create trigger plumbing",
        reuse_category=AiBackendReuseCategory.LEGACY_ONLY_OR_RETIRE_LATER,
        future_targets=(AiScenarioId.FULL_CARD_ENRICHMENT,),
        notes=("This is the current card-autofill seam and will be replaced later.",),
        do_not_break=("preserve_existing_autofill_for_backward_compatibility",),
    ),
    AiBackendComponentDefinition(
        component_id="scheduler_task_bridge",
        component_kind=AiBackendComponentKind.LEGACY_GLUE,
        current_role="scheduled task CRUD and run/pause/resume bridge",
        reuse_category=AiBackendReuseCategory.REUSE_WITH_ADAPTATION,
        future_targets=(AiScenarioId.BOARD_CONTROL,),
        notes=("Schedule persistence remains useful, but the UX surface will change.",),
        do_not_break=("keep_existing_schedule_storage_valid", "preserve_run_pause_resume_semantics_until_replacement"),
    ),
)


def build_ai_backend_reuse_registry() -> AiBackendReuseRegistry:
    return AiBackendReuseRegistry(BACKEND_COMPONENT_DEFINITIONS)


class AiEntrySurfaceKind(str, Enum):
    UI = "ui"
    BACKEND = "backend"
    STATUS = "status"
    FUTURE = "future"


class AiEntryExposureState(str, Enum):
    ACTIVE = "active"
    GATED = "gated"
    HIDDEN = "hidden"
    LEGACY_ONLY = "legacy_only"
    REPLACED = "replaced"


class AiEntryDeactivationPolicy(str, Enum):
    KEEP = "keep"
    GATE = "gate"
    LATER_HIDE = "later_hide"
    LATER_REMOVE = "later_remove"


@dataclass(frozen=True)
class AiEntrySurfaceDefinition:
    entry_id: str
    location: str
    current_behavior: str
    scenario_semantics_today: str
    surface_kind: AiEntrySurfaceKind
    legacy_status: str
    deactivation_policy: AiEntryDeactivationPolicy
    replacement_target: str
    replacement_module: str
    rollout_dependency: str
    future_module_owner: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "location": self.location,
            "current_behavior": self.current_behavior,
            "scenario_semantics_today": self.scenario_semantics_today,
            "surface_kind": self.surface_kind.value,
            "legacy_status": self.legacy_status,
            "deactivation_policy": self.deactivation_policy.value,
            "replacement_target": self.replacement_target,
            "replacement_module": self.replacement_module,
            "rollout_dependency": self.rollout_dependency,
            "future_module_owner": self.future_module_owner,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AiEntryExposureRecord:
    entry_id: str
    exposure_state: AiEntryExposureState
    legacy_status: str
    deactivation_policy: AiEntryDeactivationPolicy
    replacement_target: str
    replacement_module: str
    rollout_dependency: str
    surface_kind: AiEntrySurfaceKind
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "exposure_state": self.exposure_state.value,
            "legacy_status": self.legacy_status,
            "deactivation_policy": self.deactivation_policy.value,
            "replacement_target": self.replacement_target,
            "replacement_module": self.replacement_module,
            "rollout_dependency": self.rollout_dependency,
            "surface_kind": self.surface_kind.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AiEntrySurfaceRegistry:
    surfaces: tuple[AiEntrySurfaceDefinition, ...]
    by_id: dict[str, AiEntrySurfaceDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_id", {item.entry_id: item for item in self.surfaces})

    def get(self, entry_id: str) -> AiEntrySurfaceDefinition | None:
        return self.by_id.get(str(entry_id or "").strip().lower())

    def ids(self) -> list[str]:
        return [item.entry_id for item in self.surfaces]


LEGACY_DEACTIVATION_MAP: tuple[AiEntrySurfaceDefinition, ...] = (
    AiEntrySurfaceDefinition(
        entry_id="board_dock_button",
        location="web_assets.agentDockButton -> openAgentModal('board')",
        current_behavior="opens the mixed board agent modal",
        scenario_semantics_today="legacy board-oriented manual AI entry",
        surface_kind=AiEntrySurfaceKind.UI,
        legacy_status="legacy_only",
        deactivation_policy=AiEntryDeactivationPolicy.LATER_HIDE,
        replacement_target=AiScenarioId.AI_CHAT.value,
        replacement_module="Module 1.2 AI Chat Console",
        rollout_dependency="legacy_ux_enabled / ai_chat rollout",
        future_module_owner="Module 1.2 AI Chat Console",
        notes=("Current button remains visible for compatibility.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="card_agent_button",
        location="web_assets.cardAgentButton -> openAgentModal('card')",
        current_behavior="opens the card-scoped legacy agent modal",
        scenario_semantics_today="legacy card-scoped AI entry",
        surface_kind=AiEntrySurfaceKind.UI,
        legacy_status="legacy_only",
        deactivation_policy=AiEntryDeactivationPolicy.LATER_HIDE,
        replacement_target=AiScenarioId.FULL_CARD_ENRICHMENT.value,
        replacement_module="Module 1.3 Card Enrichment Pipeline",
        rollout_dependency="legacy_ux_enabled / full_card_enrichment rollout",
        future_module_owner="Module 1.3 Card Enrichment Pipeline",
        notes=("Replacement will be a bounded enrichment path, not a new modal clone.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="agent_manual_prompt",
        location="web_assets.agentTaskInput + agentRunButton -> /api/agent_enqueue_task",
        current_behavior="submits freeform manual task text to the worker",
        scenario_semantics_today="legacy manual conversational surface",
        surface_kind=AiEntrySurfaceKind.UI,
        legacy_status="legacy_only",
        deactivation_policy=AiEntryDeactivationPolicy.GATE,
        replacement_target=AiScenarioId.AI_CHAT.value,
        replacement_module="Module 1.2 AI Chat Console",
        rollout_dependency="legacy_ux_enabled / ai_chat rollout",
        future_module_owner="Module 1.2 AI Chat Console",
        notes=("This is the main legacy freeform prompt path.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="quick_prompts",
        location="web_assets.quickAgentPrompts + data-agent-prompt handlers",
        current_behavior="prefills canned prompts into the manual agent textarea",
        scenario_semantics_today="legacy shortcut surface for mixed AI tasks",
        surface_kind=AiEntrySurfaceKind.UI,
        legacy_status="legacy_only",
        deactivation_policy=AiEntryDeactivationPolicy.GATE,
        replacement_target=f"{AiScenarioId.AI_CHAT.value} / {AiScenarioId.FULL_CARD_ENRICHMENT.value}",
        replacement_module="Module 1.2 + Module 1.3",
        rollout_dependency="legacy_ux_enabled / ai_chat + enrichment rollout",
        future_module_owner="Module 1.2 AI Chat Console",
        notes=("Quick prompts are currently a shortcut layer, not a new scenario.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="agent_tasks_modal",
        location="web_assets.agentTasksModal -> /api/agent_scheduled_tasks and run/pause/resume endpoints",
        current_behavior="shows manual tasks, schedules, and execution controls",
        scenario_semantics_today="legacy mixed scheduler shell",
        surface_kind=AiEntrySurfaceKind.UI,
        legacy_status="legacy_only",
        deactivation_policy=AiEntryDeactivationPolicy.LATER_HIDE,
        replacement_target=AiScenarioId.BOARD_CONTROL.value,
        replacement_module="Module 1.4 Board Control Mode",
        rollout_dependency="legacy_ux_enabled / board_control rollout",
        future_module_owner="Module 1.4 Board Control Mode",
        notes=("Will later split into background control and separate admin surfaces.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="card_autofill_toggle",
        location="web_assets.agentAutofillButton + mini-prompt panel + /api/set_card_ai_autofill",
        current_behavior="toggles card autofill and opens the mini-prompt panel",
        scenario_semantics_today="legacy card autofill trigger surface",
        surface_kind=AiEntrySurfaceKind.UI,
        legacy_status="legacy_only",
        deactivation_policy=AiEntryDeactivationPolicy.LATER_HIDE,
        replacement_target=AiScenarioId.FULL_CARD_ENRICHMENT.value,
        replacement_module="Module 1.3 Card Enrichment Pipeline",
        rollout_dependency="legacy_ux_enabled / full_card_enrichment rollout",
        future_module_owner="Module 1.3 Card Enrichment Pipeline",
        notes=("This is the old card autofill control path.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="agent_status_surface",
        location="web_assets.agentStatusLabel + agentAutofillStatus",
        current_behavior="shows agent readiness and autofill state in the UI",
        scenario_semantics_today="read-only agent availability surface",
        surface_kind=AiEntrySurfaceKind.STATUS,
        legacy_status="infrastructure",
        deactivation_policy=AiEntryDeactivationPolicy.KEEP,
        replacement_target="none",
        replacement_module="none",
        rollout_dependency="agent runtime availability",
        future_module_owner="Module 1.x shared runtime",
        notes=("This surface is preserved for diagnostics and does not initiate work.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="agent_enqueue_task_api",
        location="/api/agent_enqueue_task",
        current_behavior="accepts manual tasks for the worker queue",
        scenario_semantics_today="backend manual task entry",
        surface_kind=AiEntrySurfaceKind.BACKEND,
        legacy_status="backend_foundation",
        deactivation_policy=AiEntryDeactivationPolicy.KEEP,
        replacement_target=AiScenarioId.AI_CHAT.value,
        replacement_module="Module 1.2 AI Chat Console",
        rollout_dependency="legacy_ux_enabled / ai_chat rollout",
        future_module_owner="Module 1.2 AI Chat Console",
        notes=("Used by current UI and future chat entry seam.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="agent_scheduled_tasks_api",
        location="/api/agent_scheduled_tasks and schedule run/pause/resume endpoints",
        current_behavior="stores and executes scheduled tasks",
        scenario_semantics_today="backend scheduler and task control surface",
        surface_kind=AiEntrySurfaceKind.BACKEND,
        legacy_status="backend_foundation",
        deactivation_policy=AiEntryDeactivationPolicy.GATE,
        replacement_target=AiScenarioId.BOARD_CONTROL.value,
        replacement_module="Module 1.4 Board Control Mode",
        rollout_dependency="board_control rollout",
        future_module_owner="Module 1.4 Board Control Mode",
        notes=("Will later become narrower and board-control-specific.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="set_card_ai_autofill_api",
        location="/api/set_card_ai_autofill",
        current_behavior="enables or disables card autofill and persists mini-prompt text",
        scenario_semantics_today="backend card autofill control surface",
        surface_kind=AiEntrySurfaceKind.BACKEND,
        legacy_status="backend_foundation",
        deactivation_policy=AiEntryDeactivationPolicy.LATER_HIDE,
        replacement_target=AiScenarioId.FULL_CARD_ENRICHMENT.value,
        replacement_module="Module 1.3 Card Enrichment Pipeline",
        rollout_dependency="full_card_enrichment rollout",
        future_module_owner="Module 1.3 Card Enrichment Pipeline",
        notes=("This is the current card-trigger backend seam.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="card_created_auto_trigger",
        location="CardService.handle_card_created -> AgentControlService.enqueue_card_autofill_task",
        current_behavior="launches legacy on-create card autofill schedules",
        scenario_semantics_today="backend card-created auto trigger",
        surface_kind=AiEntrySurfaceKind.BACKEND,
        legacy_status="backend_foundation",
        deactivation_policy=AiEntryDeactivationPolicy.LATER_HIDE,
        replacement_target=AiScenarioId.FULL_CARD_ENRICHMENT.value,
        replacement_module="Module 1.3 Card Enrichment Pipeline",
        rollout_dependency="full_card_enrichment rollout",
        future_module_owner="Module 1.3 Card Enrichment Pipeline",
        notes=("Scheduled follow-up and on-create triggers remain part of current autofill plumbing.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="future_ai_chat_window",
        location="future AI chat console",
        current_behavior="future chat entry surface placeholder",
        scenario_semantics_today="future interactive conversational path",
        surface_kind=AiEntrySurfaceKind.FUTURE,
        legacy_status="future_only",
        deactivation_policy=AiEntryDeactivationPolicy.GATE,
        replacement_target=AiScenarioId.AI_CHAT.value,
        replacement_module="Module 1.2 AI Chat Console",
        rollout_dependency="ai_chat rollout",
        future_module_owner="Module 1.2 AI Chat Console",
        notes=("Not yet wired in UI.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="future_card_enrichment_trigger",
        location="future card indicator / future card enrichment entry",
        current_behavior="future bounded card enrichment trigger placeholder",
        scenario_semantics_today="future deterministic card-scoped path",
        surface_kind=AiEntrySurfaceKind.FUTURE,
        legacy_status="future_only",
        deactivation_policy=AiEntryDeactivationPolicy.GATE,
        replacement_target=AiScenarioId.FULL_CARD_ENRICHMENT.value,
        replacement_module="Module 1.3 Card Enrichment Pipeline",
        rollout_dependency="full_card_enrichment rollout",
        future_module_owner="Module 1.3 Card Enrichment Pipeline",
        notes=("Not yet wired in UI.",),
    ),
    AiEntrySurfaceDefinition(
        entry_id="future_board_control_toggle",
        location="future settings toggle for board control mode",
        current_behavior="future background board-control toggle placeholder",
        scenario_semantics_today="future scheduled board maintenance path",
        surface_kind=AiEntrySurfaceKind.FUTURE,
        legacy_status="future_only",
        deactivation_policy=AiEntryDeactivationPolicy.GATE,
        replacement_target=AiScenarioId.BOARD_CONTROL.value,
        replacement_module="Module 1.4 Board Control Mode",
        rollout_dependency="board_control rollout",
        future_module_owner="Module 1.4 Board Control Mode",
        notes=("Not yet wired in UI.",),
    ),
)


def _entry_replacement_scenarios(entry: AiEntrySurfaceDefinition) -> tuple[AiScenarioId, ...]:
    raw_target = str(entry.replacement_target or "").strip().lower()
    if not raw_target or raw_target == "none":
        return ()
    resolved: list[AiScenarioId] = []
    for scenario in AiScenarioId:
        if scenario.value in raw_target and scenario not in resolved:
            resolved.append(scenario)
    return tuple(resolved)


def _entry_has_enabled_replacement(entry: AiEntrySurfaceDefinition, mode_config: AiModeConfig) -> bool:
    for scenario_id in _entry_replacement_scenarios(entry):
        scenario_state = mode_config.scenario_state.get(scenario_id)
        if scenario_state and scenario_state.enabled:
            return True
    return False


def _entry_has_primary_replacement(entry: AiEntrySurfaceDefinition, mode_config: AiModeConfig) -> bool:
    for scenario_id in _entry_replacement_scenarios(entry):
        scenario_state = mode_config.scenario_state.get(scenario_id)
        if scenario_state and scenario_state.primary_interactive:
            return True
    return False


def build_ai_entry_surface_registry() -> AiEntrySurfaceRegistry:
    return AiEntrySurfaceRegistry(LEGACY_DEACTIVATION_MAP)


def _entry_rollout_state(entry: AiEntrySurfaceDefinition, flags: AiFeatureFlags, mode_config: AiModeConfig) -> AiEntryExposureState:
    if entry.surface_kind == AiEntrySurfaceKind.STATUS:
        return AiEntryExposureState.ACTIVE
    if entry.entry_id == "agent_enqueue_task_api":
        return AiEntryExposureState.ACTIVE
    replacement_enabled = _entry_has_enabled_replacement(entry, mode_config)
    replacement_primary = _entry_has_primary_replacement(entry, mode_config)
    if entry.surface_kind == AiEntrySurfaceKind.BACKEND:
        if entry.entry_id in {"agent_scheduled_tasks_api", "set_card_ai_autofill_api", "card_created_auto_trigger"}:
            if replacement_primary:
                return AiEntryExposureState.REPLACED
            if replacement_enabled:
                return AiEntryExposureState.GATED
            return AiEntryExposureState.LEGACY_ONLY if flags.legacy_ux_enabled else AiEntryExposureState.HIDDEN
        return AiEntryExposureState.ACTIVE
    if entry.surface_kind == AiEntrySurfaceKind.FUTURE:
        if entry.entry_id == "future_card_enrichment_trigger" and replacement_enabled:
            return AiEntryExposureState.ACTIVE
        if replacement_enabled or replacement_primary:
            return AiEntryExposureState.GATED
        return AiEntryExposureState.HIDDEN
    if replacement_primary:
        return AiEntryExposureState.REPLACED
    if replacement_enabled:
        return AiEntryExposureState.LEGACY_ONLY if flags.legacy_ux_enabled else AiEntryExposureState.REPLACED
    if flags.legacy_ux_enabled:
        if entry.deactivation_policy == AiEntryDeactivationPolicy.KEEP:
            return AiEntryExposureState.ACTIVE
        return AiEntryExposureState.LEGACY_ONLY
    if entry.deactivation_policy == AiEntryDeactivationPolicy.KEEP:
        return AiEntryExposureState.ACTIVE
    if entry.deactivation_policy == AiEntryDeactivationPolicy.GATE:
        return AiEntryExposureState.HIDDEN
    if entry.deactivation_policy == AiEntryDeactivationPolicy.LATER_HIDE:
        return AiEntryExposureState.HIDDEN
    return AiEntryExposureState.HIDDEN


def get_ai_entry_surface_map() -> dict[str, dict[str, Any]]:
    return {item.entry_id: item.to_dict() for item in LEGACY_DEACTIVATION_MAP}


def get_ai_entry_exposure_map(flags: AiFeatureFlags | None = None, mode_config: AiModeConfig | None = None) -> dict[str, dict[str, Any]]:
    resolved_flags = flags or get_ai_feature_flags()
    resolved_mode = mode_config or get_ai_mode_config(resolved_flags)
    return {
        item.entry_id: AiEntryExposureRecord(
            entry_id=item.entry_id,
            exposure_state=_entry_rollout_state(item, resolved_flags, resolved_mode),
            legacy_status=item.legacy_status,
            deactivation_policy=item.deactivation_policy,
            replacement_target=item.replacement_target,
            replacement_module=item.replacement_module,
            rollout_dependency=item.rollout_dependency,
            surface_kind=item.surface_kind,
            reason=f"{item.deactivation_policy.value}:{item.replacement_target}",
        ).to_dict()
        for item in LEGACY_DEACTIVATION_MAP
    }


def get_ai_legacy_deactivation_map() -> dict[str, dict[str, Any]]:
    return {
        item.entry_id: {
            "entry_id": item.entry_id,
            "current_behavior": item.current_behavior,
            "legacy_status": item.legacy_status,
            "deactivation_policy": item.deactivation_policy.value,
            "replacement_target": item.replacement_target,
            "replacement_module": item.replacement_module,
            "rollout_dependency": item.rollout_dependency,
            "future_module_owner": item.future_module_owner,
            "surface_kind": item.surface_kind.value,
            "scenario_semantics_today": item.scenario_semantics_today,
        }
        for item in LEGACY_DEACTIVATION_MAP
    }


def build_ai_scenario_registry() -> AiScenarioRegistry:
    return AiScenarioRegistry(SCENARIO_DEFINITIONS)


def get_ai_feature_flags() -> AiFeatureFlags:
    return AiFeatureFlags(
        legacy_ux_enabled=_env_flag("MINIMAL_KANBAN_AI_LEGACY_UX_ENABLED", default=True),
        ai_chat_enabled=_env_flag("MINIMAL_KANBAN_AI_CHAT_ENABLED", default=False),
        full_card_enrichment_enabled=_env_flag("MINIMAL_KANBAN_FULL_CARD_ENRICHMENT_ENABLED", default=True),
        board_control_enabled=_env_flag("MINIMAL_KANBAN_BOARD_CONTROL_ENABLED", default=False),
    )


def get_ai_mode_config(flags: AiFeatureFlags | None = None) -> AiModeConfig:
    resolved_flags = flags or get_ai_feature_flags()
    registry = build_ai_scenario_registry()
    scenario_state = {
        item.scenario_id: registry.mode_state_for(item.scenario_id, resolved_flags)
        for item in registry.scenarios
    }
    return AiModeConfig(legacy_ux_enabled=resolved_flags.legacy_ux_enabled, scenario_state=scenario_state)


def get_ai_effective_mode(flags: AiFeatureFlags | None = None) -> dict[str, Any]:
    resolved_flags = flags or get_ai_feature_flags()
    mode_config = get_ai_mode_config(resolved_flags)
    scenario_registry = build_ai_scenario_registry()
    scenario_map = scenario_registry.to_dict(resolved_flags)
    primary = next(
        (
            state.scenario_id.value
            for state in mode_config.scenario_state.values()
            if state.primary_interactive
        ),
        "legacy_agent_modal_manual_tasks" if resolved_flags.legacy_ux_enabled else "none",
    )
    legacy_compatible_only = [
        scenario_id.value
        for scenario_id, state in mode_config.scenario_state.items()
        if state.legacy_compatible_only
    ]
    hidden = [
        scenario_id.value
        for scenario_id, state in mode_config.scenario_state.items()
        if state.hidden
    ]
    background_only = [
        scenario_id.value
        for scenario_id, state in mode_config.scenario_state.items()
        if state.background_only
    ]
    available = [scenario_id for scenario_id, item in scenario_map.items() if bool(item.get("enabled"))]
    return {
        "legacy_ux_enabled": bool(resolved_flags.legacy_ux_enabled),
        "primary_interactive_path": primary,
        "legacy_compatible_only": legacy_compatible_only,
        "hidden": hidden,
        "background_only": background_only,
        "available_scenarios": available,
        "mode_config": mode_config.to_dict(),
        "entry_exposure": get_ai_entry_exposure_map(resolved_flags, mode_config),
    }


def get_ai_scenario_map(flags: AiFeatureFlags | None = None) -> dict[str, dict[str, Any]]:
    return build_ai_scenario_registry().to_dict(flags)


def get_ai_legacy_entry_point_map() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in LEGACY_AI_ENTRY_POINTS.items()}


def get_ai_backend_component_registry() -> dict[str, dict[str, Any]]:
    return build_ai_backend_reuse_registry().to_dict()


def get_ai_backend_reuse_map() -> dict[str, dict[str, Any]]:
    return build_ai_backend_reuse_registry().grouped_by_category()


def get_ai_remodel_status_payload() -> dict[str, Any]:
    flags = get_ai_feature_flags()
    backend_component_registry = get_ai_backend_component_registry()
    return {
        "phase": "module_1_4_backend_reuse",
        "legacy_ux_enabled": bool(flags.legacy_ux_enabled),
        "feature_flags": flags.to_dict(),
        "mode_config": get_ai_mode_config(flags).to_dict(),
        "effective_mode": get_ai_effective_mode(flags),
        "scenario_registry": get_ai_scenario_map(flags),
        "entry_surface_registry": get_ai_entry_surface_map(),
        "entry_exposure": get_ai_entry_exposure_map(flags, get_ai_mode_config(flags)),
        "legacy_deactivation_map": get_ai_legacy_deactivation_map(),
        "legacy_entry_points": get_ai_legacy_entry_point_map(),
        "backend_component_registry": backend_component_registry,
        "backend_reuse": get_ai_backend_reuse_map(),
        "backend_legacy_only": {
            key: value
            for key, value in backend_component_registry.items()
            if str(value.get("reuse_category")) == AiBackendReuseCategory.LEGACY_ONLY_OR_RETIRE_LATER.value
        },
        "legacy_scenario_names": list(LEGACY_SCENARIO_NAMES),
    }
