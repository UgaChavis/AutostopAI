# AutostopAI

This repository contains the extracted server-agent runtime from AutoStop CRM.

What lives here:

- the agent control loop and task runner
- OpenAI JSON client wrapper
- the VIN enrichment scenario
- compact context, policy, storage, and tool executors
- minimal shared board-model helpers required by the agent

What does not live here:

- desktop UI
- board web app
- API server
- MCP server runtime

## Local usage

Create a virtual environment, install dependencies, and run the agent loop:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The agent uses the local board API from AutoStop CRM when `MINIMAL_KANBAN_AGENT_BOARD_API_URL`
or `AUTOSTOP_AGENT_BOARD_API_URL` is set. If neither is set, it will try to discover a local API.
For OpenAI access, the agent reads `OPENAI_API_KEY` first and then falls back to
`MINIMAL_KANBAN_OPENAI_API_KEY_FILE`, `OPENAI_API_KEY_FILE`, or
`MINIMAL_KANBAN_AGENT_OPENAI_API_KEY_FILE` if one of those points to a text file.
The green-button CRM flow is modeled as `card_enrichment`: it reads only the card, extracts VIN,
enriches vehicle data, and writes back a concise description plus vehicle profile fields.

## Offline sandbox

You can inspect the agent's routing and planning logic without connecting to CRM:

```powershell
python scripts\offline_agent_preview.py
python scripts\offline_agent_preview.py --preview
```

This uses an in-memory board fixture and prints the computed task preview, plan, and runtime snapshot.
The active card flow is VIN-only: other legacy scenarios remain only as compatibility code and are not part of the main path.
The agent also keeps a local VIN cache in its app-data directory so repeated VIN research results can be reused across tasks.

## OpenAI connection check

Run a minimal API test without CRM:

```powershell
python scripts\check_openai_connection.py --model gpt-5.4-mini
```

## Agent doctor

Run a compact readiness check for the agent module:

```powershell
python scripts\agent_doctor.py
```

This checks the OpenAI call, a live VIN web research pass, the VIN cache, and the active VIN-only routing shape.

## Development notes

- The code was extracted from the AutoStop CRM main repository.
- Keep the agent repo focused on orchestration, tools, and enrichment logic.
- Keep the board API contract stable when adding new agent actions.
- For the detailed agent logic, editing order, runtime checks, and failure modes, read [`docs/AGENT_RUNBOOK.md`](docs/AGENT_RUNBOOK.md).

