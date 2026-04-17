from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.agent.config import get_agent_openai_api_key, get_agent_openai_model
from minimal_kanban.agent.openai_client import AgentModelError, OpenAIJsonAgentClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the OpenAI connection used by the AutostopAI agent.")
    parser.add_argument("--model", default=get_agent_openai_model(), help="Model name to test.")
    parser.add_argument("--key-file", default="", help="Optional text file containing the API key.")
    args = parser.parse_args()

    api_key = get_agent_openai_api_key()
    if not api_key and args.key_file:
        path = Path(args.key_file).expanduser()
        if path.exists():
            api_key = path.read_text(encoding="utf-8").strip() or None
    if not api_key:
        print(json.dumps({"ok": False, "error": "OPENAI_API_KEY is not configured"}, ensure_ascii=False, indent=2))
        return 1

    try:
        client = OpenAIJsonAgentClient(api_key=api_key, model=args.model)
        result = client.complete_json(
            instructions='Return exactly one JSON object: {"ok": true, "reply": "short greeting"}',
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            temperature=0.0,
        )
    except AgentModelError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"ok": True, "model": client.model, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
