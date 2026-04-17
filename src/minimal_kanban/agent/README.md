# Agent Module Map

This module is the server-side AI layer for AutoStop CRM.

For the working logic, read the main runbook:

- [`docs/AGENT_RUNBOOK.md`](../../docs/AGENT_RUNBOOK.md)

## Active path

- `control.py` enqueues `card_enrichment` tasks from the green button.
- `runner.py` executes the card-only orchestration loop.
- `router.py` routes card enrichment to `vin_enrichment`.
- `scenarios/vin_enrichment.py` performs VIN web research and synthesis.
- `vehicle_profile.py` normalizes VIN research output into CRM-ready fields.
- `storage.py` keeps runtime state and the local VIN cache.
- `tools.py` exposes only the tools used by the active flow.

## Editing order

1. `instructions.py`
2. `router.py`
3. `policy.py`
4. `scenarios/vin_enrichment.py`
5. `vehicle_profile.py`
6. `runner.py`
7. `sandbox.py`

## Legacy compatibility

The module still contains some legacy names such as `autofill` and `card_autofill` in compatibility helpers. They should not be expanded.

## Rule of thumb

- Prefer one card-only scenario.
- Do not add new scenarios unless they are part of the CRM button flow.
- Keep external lookups deterministic, cached, and normalized before writing back to CRM.
- Keep card descriptions short and readable.
