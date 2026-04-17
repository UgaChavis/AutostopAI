# AutostopAI Agent Runbook

This document is the working map for the server-side AI agent module in AutoStop CRM.

## Scope

The agent module does one job today:

- read a single CRM card
- extract VIN from card text or current card fields
- research that VIN on the web with the model
- normalize the research into CRM-ready fields
- write back a short card description, vehicle label, and vehicle profile patch

It does not currently own:

- repair-order orchestration
- parts lookup
- DTC lookup
- maintenance planning
- board-wide autonomous cleanup

Those paths exist only as compatibility or historical code and should not be expanded unless the CRM product explicitly reintroduces them.

## Mental Model

Think of the agent as a small pipeline:

1. `control.py` receives a green-button request and creates a queued task.
2. `storage.py` persists that task and tracks runtime state.
3. `runner.py` claims the task and executes the orchestration loop.
4. `router.py` classifies the work as `card_enrichment`.
5. `vin_enrichment.py` runs the only active scenario.
6. `openai_client.py` asks the model to do web research with `web_search`.
7. `vehicle_profile.py` converts model output into CRM-safe field patches.
8. `runner.py` writes the patch back to the board API and records the trace.

The card can be closed in the UI immediately after the button is pressed. The work continues in the background because the task is queued and owned by the agent runtime, not by the open card tab.

## Active Flow

### 1. Button press

The CRM green button calls `enqueue_card_enrichment_task(...)` in `control.py`.

The payload should include:

- `card_id`
- optional `task_text`
- optional `card_heading`
- optional `vehicle`
- optional `requested_by`

The control layer adds:

- `purpose = card_enrichment`
- `scenario_id = card_enrichment`
- `context.kind = card`
- `scope.type = current_card`

### 2. Queue write

`storage.py` writes a task object to the agent queue.

Important state fields:

- `status`
- `run_id`
- `summary`
- `result`
- `display`
- `tool_calls`
- `metadata`

The queue is persistent, so the task survives UI navigation and short-lived process interruptions.

### 3. Claim and run

`runner.py` calls `claim_next_task()` and runs exactly one queued task per `run_once()`.

The runner:

- sets `running=true`
- records heartbeat and start time
- loads card context
- extracts VIN
- researches VIN through the model
- normalizes the result
- builds a bounded patch
- writes back to CRM
- verifies what was written
- stores execution traces and runtime logs

### 4. VIN research

`vin_enrichment.py` is the active scenario.

Current behavior:

- it first checks the local VIN cache
- if there is no cache hit, it asks the model to use `web_search`
- it asks for confirmed facts only
- it allows partial or insufficient results when the sources are weak
- it normalizes the payload into `vehicle_profile`

The output is intentionally conservative. If the web sources do not confirm a field, the field stays blank.

### 5. Write-back

Allowed CRM write targets are currently:

- `description`
- `vehicle`
- `vehicle_profile`

The update should stay short and readable:

- preserve useful original card text
- add a compact `ИИ:` block when appropriate
- do not duplicate the whole source text if the card is long
- do not write repair-order content in this flow

## Data Flow

### Inputs

The agent reads:

- card title
- card description
- vehicle label
- current vehicle profile
- card events/context
- VIN if present in the card text or existing profile

### Outputs

The agent writes:

- a shorter, cleaner card description
- a normalized vehicle label
- a vehicle profile patch with confirmed fields
- optional oem notes
- runtime trace and action log entries

### VIN normalization

`vehicle_profile.py` turns VIN research output into CRM-safe fields.

Principles:

- copy only confirmed values
- strip markdown links and URL noise
- normalize years to integers
- normalize confidence to a bounded numeric score
- keep source references compact

## Prompt Rules

The system prompt in `instructions.py` should keep the agent constrained:

- use VIN web research only
- do not invent engine or gearbox facts
- preserve the current card context
- keep the final description short
- mark AI-added card content with `ИИ:` or `AI:`
- do not expand into repair-order or board-wide behavior

If the card is long, the resulting description should be compact, not a verbatim copy of the original card.

## Tool Contract

The tool surface is intentionally small for the active path.

Primary tools:

- `get_card_context`
- `get_card`
- `update_card`
- `research_vin`

Compatibility tools still exist in the codebase, but the active VIN-only path should not rely on them.

## Queue and Runtime

The task lifecycle is:

1. enqueue
2. pending
3. claimed
4. running
5. completed or failed

Runtime state lives in the agent storage files, not in the UI tab.

Important operational files:

- `tasks.json`
- `status.json`
- `runs.jsonl`
- `actions.jsonl`
- `vin_cache.json`

## Offline Test Surface

Use the offline sandbox for deterministic checks without CRM:

- `scripts/offline_agent_preview.py`
- `scripts/agent_doctor.py`
- `tests/test_offline_agent_sandbox.py`

Use the live smoke checks only when the OpenAI key and board API are available.

## Live Checks

The two most useful live checks are:

- `python scripts\agent_doctor.py`
- `python -m unittest discover -s tests -p "test_*.py"`

When the board API is available, the agent should also be exercised with a real card containing a VIN and a noisy description.

## Common Failure Modes

1. OpenAI key missing.
2. Board API not reachable.
3. VIN is present but the model returns sparse web evidence.
4. The card description is too long and needs compaction.
5. Legacy compatibility names appear in logs even though the active path is still VIN-only.

## Editing Order

When changing the agent, edit in this order:

1. `instructions.py`
2. `router.py`
3. `policy.py`
4. `vin_enrichment.py`
5. `vehicle_profile.py`
6. `runner.py`
7. `sandbox.py`

That order keeps the high-level behavior stable while the lower layers are adjusted.

## What To Avoid

- Do not add new autonomous scenarios unless the CRM button flow requires them.
- Do not reintroduce repair-order writes into the green-button path.
- Do not rely on web search results without normalizing them into CRM-safe fields.
- Do not keep raw citation refs or markdown links in final card text if a clean value is available.

## Current Status

The active path is ready for CRM-side integration work, but the CRM backend/button bridge still needs to be wired into the main application.

The current agent-side runtime is healthy when:

- OpenAI responds through `gpt-5.4-mini`
- VIN research returns a usable payload
- the test suite passes
- the offline sandbox still produces a card enrichment preview

