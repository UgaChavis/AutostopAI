from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.agent.config import get_agent_openai_api_key


class AgentOpenAIConfigTests(unittest.TestCase):
    def test_reads_key_from_env(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            self.assertEqual(get_agent_openai_api_key(), "env-key")

    def test_reads_key_from_file_when_env_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autostopai-key-") as temp_dir:
            key_file = Path(temp_dir) / "key.txt"
            key_file.write_text("Project: AutoStop AI\nsk-file-key-1234567890abcdef\n", encoding="utf-8")
            with patch.dict(os.environ, {"MINIMAL_KANBAN_OPENAI_API_KEY_FILE": str(key_file)}, clear=True):
                self.assertEqual(get_agent_openai_api_key(), "sk-file-key-1234567890abcdef")

    def test_prefers_env_over_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autostopai-key-") as temp_dir:
            key_file = Path(temp_dir) / "key.txt"
            key_file.write_text("file-key\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "env-key",
                    "MINIMAL_KANBAN_OPENAI_API_KEY_FILE": str(key_file),
                },
                clear=True,
            ):
                self.assertEqual(get_agent_openai_api_key(), "env-key")


if __name__ == "__main__":
    unittest.main()
