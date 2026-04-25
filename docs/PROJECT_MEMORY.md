# Project Memory

Date: 2026-04-25

## Repository

- GitHub origin: `https://github.com/UgaChavis/AutostopAI.git`
- Current branch: `main`
- Tracking: `origin/main`

## What This Repo Is

This repository contains the extracted server-side AI runtime used by AutoStop CRM.

The active shape is a bounded card enrichment agent, not a freeform autonomous assistant.

## Current Working State

- the green-button card flow is card-only
- VIN enrichment is the supported active scenario
- the runtime reads card context first, then researches VIN, then writes a compact patch
- the worker should return a complete result in the same response
- deferred "I'll send it later" style promises are removed from final user-facing text
- repair-order, parts, DTC, and maintenance lookups stay out of the green-button path

## Important Paths

- `README.md`
- `docs/AGENT_RUNBOOK.md`
- `docs/VIN_ENRICHMENT_BRIDGE.md`
- `src/minimal_kanban/agent/README.md`
- `src/minimal_kanban/agent/instructions.py`
- `src/minimal_kanban/agent/runner.py`
- `src/minimal_kanban/agent/scenarios/vin_enrichment.py`
- `src/minimal_kanban/vehicle_profile.py`
- `tests/test_offline_agent_sandbox.py`

## Operational Rules

- keep the prompt short and practical
- keep the patch bounded to `description`, `vehicle`, and `vehicle_profile`
- normalize VIN research before writing to CRM
- verify the write-back after every successful update
- keep card descriptions short and readable

## Validation

Use these checks before shipping changes:

```powershell
python -m unittest tests.test_vin_enrichment_bridge tests.test_vin_cache_storage tests.test_vehicle_profile_vin_patch tests.test_smoke tests.test_offline_agent_sandbox tests.test_agent_runtime_check tests.test_agent_openai_config tests.test_agent_doctor
python -m ruff check src/minimal_kanban/agent tests/test_offline_agent_sandbox.py
```

Useful smoke commands:

```powershell
python scripts\agent_doctor.py
python scripts\offline_agent_preview.py --preview
python scripts\check_openai_connection.py --model gpt-5.4-mini
```

## Deployment Model

Local source of truth:

- edit here first
- run the focused tests
- commit and push to GitHub

Server sync target:

- standard CRM checkout path: `/opt/autostopcrm`
- sync by pulling the same commit into that checkout
- restart or redeploy the runtime that executes the agent

If the server checkout lives elsewhere, use the same `git pull` plus service restart flow there.

## Recent Fixes

- conversation memory was added for chat continuation context
- final responses now sanitize out future-follow-up promises
- card enrichment now reports concrete updated fields in completion messages
- VIN enrichment remains the only active scenario in the card flow

