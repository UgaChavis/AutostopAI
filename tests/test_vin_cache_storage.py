from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.agent.storage import AgentStorage


class VinCacheStorageTests(unittest.TestCase):
    def test_vin_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autostopai-vin-cache-") as temp_dir:
            storage = AgentStorage(base_dir=Path(temp_dir))
            storage.upsert_vin_cache_entry(
                "JTDKB20U093123456",
                {
                    "make": "Toyota",
                    "model": "Prius",
                    "model_year": "2009",
                    "source": "NHTSA vPIC",
                },
            )
            entry = storage.get_vin_cache_entry("jtdkb20u093123456")
            cache = storage.read_vin_cache()

        self.assertIsNotNone(entry)
        self.assertEqual(entry["vin"], "JTDKB20U093123456")
        self.assertEqual(entry["make"], "Toyota")
        self.assertIn("JTDKB20U093123456", cache)


if __name__ == "__main__":
    unittest.main()
