# VIN Enrichment Bridge

This document pins the minimal AI-side bridge contract for the green-button CRM flow.

Repository origin:

- `https://github.com/UgaChavis/AutostopAI.git`

Current deployment model:

- local development in this checkout
- GitHub as the source of truth for the code branch
- the server deployment pulls the same commit into its CRM checkout and then restarts the runtime

## Scope

The active flow is only:

`button -> enqueue task -> get_card_context -> VIN research -> compact patch -> update_card`

The AI-side worker reads one card, researches VIN data with web search, and returns a compact CRM-ready patch.

## Task payload

```json
{
  "task_id": "agtask_123",
  "card_id": "card_123",
  "purpose": "card_enrichment",
  "trigger": "button",
  "requested_by": "crm_ui",
  "task_text": "optional",
  "card_context": "optional_hint_only"
}
```

## Response payload

```json
{
  "task_id": "agtask_123",
  "card_id": "card_123",
  "status": "completed",
  "summary": "short result",
  "patch": {
    "description": "...",
    "vehicle": "...",
    "vehicle_profile": {}
  },
  "warnings": [],
  "sources": [],
  "needs_review": false
}
```

## AI-side rules

- Always call `get_card_context(card_id)` first.
- Treat `card_context` as a hint only.
- Use web research for VIN enrichment.
- Return only confirmed facts.
- Do not touch `repair_order`, parts, DTC, or maintenance.
- Keep the card description short and readable.

## Allowed write fields

- `description`
- `vehicle`
- `vehicle_profile`

## Vehicle profile guidance

Write only confirmed values and keep the patch compact.

Common safe fields:

- `vin`
- `make_display`
- `model_display`
- `production_year`
- `engine_model`
- `gearbox_model`
- `drivetrain`
- `source_summary`
- `source_confidence`
- `source_links_or_refs`
- `autofilled_fields`
- `field_sources`
- `data_completion_state`
- `oem_notes`

## Current behavior notes

- The worker reads the card and returns the result in the same response when the task completes.
- Deferred "I'll send it later" style promises are sanitized out of final text.
- If the task cannot be completed, the response should explicitly say why and stop there.
- The current success case is a bounded patch plus verification, not a freeform rewrite.

## Status model

- `queued`
- `running`
- `needs_review`
- `completed`
- `failed`

## Implementation notes

The AI-side bridge helpers live in:

- `src/minimal_kanban/agent/bridge.py`
- `src/minimal_kanban/vehicle_profile.py`

The main operational docs are:

- [`README.md`](../README.md)
- [`docs/AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md)
- [`docs/PROJECT_MEMORY.md`](PROJECT_MEMORY.md)
- [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md)
