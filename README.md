# AutostopAI

This repository contains the extracted server-agent runtime from AutoStop CRM.

What lives here:

- the agent control loop and task runner
- OpenAI JSON client wrapper
- VIN / parts / diagnostics scenarios
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

## Development notes

- The code was extracted from the AutoStop CRM main repository.
- Keep the agent repo focused on orchestration, tools, and enrichment logic.
- Keep the board API contract stable when adding new agent actions.

