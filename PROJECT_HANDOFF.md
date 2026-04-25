# Project Handoff

## Current Status

The agent runtime is in a partial-success state that is worth preserving:

- green-button card flow works as a bounded card enrichment path
- VIN enrichment is the main supported scenario
- the worker returns a complete answer in the same turn
- follow-up promises are removed from final user-facing text
- the result is verified after write-back

## What To Remember

- GitHub origin: `https://github.com/UgaChavis/AutostopAI.git`
- Local branch: `main`
- Server checkout target used by the CRM deployment: `/opt/autostopcrm`

## Files To Read First

1. `README.md`
2. `docs/AGENT_RUNBOOK.md`
3. `docs/VIN_ENRICHMENT_BRIDGE.md`
4. `docs/PROJECT_MEMORY.md`
5. `src/minimal_kanban/agent/README.md`

## Sync Order

1. edit locally
2. run the focused tests
3. commit and push to GitHub
4. pull the same commit into the server checkout
5. restart or redeploy the runtime

## Good Checkpoints

- `python scripts\agent_doctor.py`
- `python scripts\check_agent_runtime.py`
- `python -m unittest tests.test_offline_agent_sandbox`
- `python -m unittest tests.test_vin_enrichment_bridge tests.test_vin_cache_storage tests.test_vehicle_profile_vin_patch tests.test_smoke tests.test_offline_agent_sandbox tests.test_agent_runtime_check tests.test_agent_openai_config tests.test_agent_doctor`

## Operational Rule

Do not treat a task as finished if the bot only says it will continue later.
The completion message must contain the actual result, or the blocker, in the same response.

