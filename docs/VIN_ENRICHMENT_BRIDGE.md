# VIN Enrichment Bridge

This document pins the minimal AI-side bridge contract for the green-button CRM flow.

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

