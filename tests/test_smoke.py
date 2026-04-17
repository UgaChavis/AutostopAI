from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AgentSmokeTests(unittest.TestCase):
    def test_agent_modules_import(self) -> None:
        from minimal_kanban.agent.control import AgentControlService
        from minimal_kanban.agent.openai_client import OpenAIJsonAgentClient
        from minimal_kanban.agent.runner import AgentRunner

        self.assertIsNotNone(AgentControlService)
        self.assertIsNotNone(OpenAIJsonAgentClient)
        self.assertIsNotNone(AgentRunner)


if __name__ == "__main__":
    unittest.main()
