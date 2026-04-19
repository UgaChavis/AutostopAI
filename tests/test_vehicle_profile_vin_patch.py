from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.vehicle_profile import build_vehicle_profile_patch_from_vin_research
from minimal_kanban.agent.scenarios.vin_enrichment import VinEnrichmentScenarioExecutor
from minimal_kanban.agent.automotive_tools import AutomotiveLookupService


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
            include_vin=True,
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

    def test_partial_research_does_not_inject_vin_when_not_confirmed(self) -> None:
        patch = build_vehicle_profile_patch_from_vin_research(
            {
                "vin": "1HGCM82633A004352",
                "make": "HONDA",
                "model": "Accord",
                "model_year": "2003",
                "status": "partial_match",
            },
            current_vin="1HGCM82633A004352",
        )

        self.assertNotIn("vin", patch)

    def test_richer_vin_result_wins(self) -> None:
        executor = VinEnrichmentScenarioExecutor()
        current = {"status": "insufficient", "make": "Toyota"}
        candidate = {
            "status": "partial",
            "make": "Toyota",
            "model": "Land Cruiser 4.0",
            "model_year": "2013",
            "drive_type": "AWD",
            "source_links_or_refs": ["https://example.com/a", "https://example.com/b"],
        }

        self.assertTrue(executor._is_richer_vin_result(candidate, current))

    def test_wmi_payload_can_fill_make_and_country(self) -> None:
        patch = build_vehicle_profile_patch_from_vin_research(
            {
                "status": "partial",
                "wmi_payload": {
                    "make": "Toyota",
                    "manufacturer": "Toyota",
                    "country": "Japan",
                    "source_url": "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeWMI/JTE?format=json",
                },
            },
            current_vin="JTEBU3FJX05027767",
        )

        self.assertEqual(patch["make_display"], "Toyota")
        self.assertEqual(patch["plant_country"], "Japan")

    def test_autofilled_fields_do_not_duplicate_when_wmi_refills_make(self) -> None:
        patch = build_vehicle_profile_patch_from_vin_research(
            {
                "status": "partial",
                "make": "Toyota",
                "wmi_payload": {
                    "make": "Toyota",
                    "manufacturer": "Toyota",
                    "country": "Japan",
                },
            },
            current_vin="JTEBU3FJX05027767",
        )

        self.assertEqual(patch["make_display"], "Toyota")
        self.assertEqual(patch["plant_country"], "Japan")
        self.assertEqual(patch["autofilled_fields"].count("make_display"), 1)
        self.assertEqual(len(patch["autofilled_fields"]), len(set(patch["autofilled_fields"])))

    def test_sparse_research_still_emits_best_effort_patch(self) -> None:
        executor = VinEnrichmentScenarioExecutor()
        patch = executor._build_card_patch(
            facts={"vin": "JTEBU3FJX05027767", "vehicle_profile": {}},
            research_result={"warnings": ["Sparse VIN search"]},
            research_status="insufficient",
        )

        self.assertIn("description", patch)
        self.assertTrue(patch["description"].startswith("По VIN выполнено best-effort исследование"))
        self.assertIn("vehicle_profile", patch)
        self.assertEqual(patch["vehicle_profile"]["source_summary"], "VIN web research")
        self.assertIn("raw_input_text", patch["vehicle_profile"])
        self.assertEqual(patch["vehicle_profile"]["warnings"], ["Sparse VIN search"])

    def test_vin_research_queries_include_family_sources(self) -> None:
        service = AutomotiveLookupService()
        queries = service._vin_research_queries("JTEBU3FJX05027767")

        self.assertTrue(any('"JTEBU3FJX05" Toyota VIN family' in query for query in queries))
        self.assertTrue(any('"JTEBU3FJX05" Toyota homologation' in query for query in queries))
        self.assertTrue(any('site:typenscheine.ch "JTEBU3FJX05"' in query for query in queries))
        self.assertTrue(any('site:dauto.ch "JTEBU3FJX05"' in query for query in queries))
        self.assertTrue(any('site:7zap.com "JTEBU3FJX05"' in query for query in queries))


if __name__ == "__main__":
    unittest.main()
