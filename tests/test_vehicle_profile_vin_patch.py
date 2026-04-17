from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.vehicle_profile import build_vehicle_profile_patch_from_vin_research


class VehicleProfileVinPatchTests(unittest.TestCase):
    def test_builds_only_missing_fields(self) -> None:
        patch = build_vehicle_profile_patch_from_vin_research(
            {
                "vin": "1HGCM82633A004352",
                "make": "HONDA",
                "model": "Accord",
                "model_year": "2003",
                "engine_model": "J30A4",
                "transmission": "Automatic / 5",
                "drive_type": "FWD",
                "source_url": "https://example.com/vin",
            },
            existing_profile={"model_display": "Accord"},
            current_vin="",
        )

        self.assertEqual(patch["vin"], "1HGCM82633A004352")
        self.assertEqual(patch["make_display"], "HONDA")
        self.assertNotIn("model_display", patch)
        self.assertEqual(patch["production_year"], 2003)
        self.assertEqual(patch["source_summary"], "VIN web research")
        self.assertEqual(patch["source_confidence"], 0.78)
        self.assertEqual(patch["source_links_or_refs"], ["https://example.com/vin"])

    def test_uses_source_confidence_hint_when_available(self) -> None:
        patch = build_vehicle_profile_patch_from_vin_research(
            {
                "vin": "1HGCM82633A004352",
                "make": "HONDA",
                "model": "Accord",
                "model_year": "2003",
                "source_confidence": "high",
            },
            current_vin="",
        )

        self.assertEqual(patch["source_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
