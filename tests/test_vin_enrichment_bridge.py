from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.agent.bridge import (
    build_card_enrichment_response,
    build_card_enrichment_task,
    normalize_card_enrichment_patch,
)


class VinEnrichmentBridgeTests(unittest.TestCase):
    def test_build_card_enrichment_task(self) -> None:
        task = build_card_enrichment_task(
            "agtask_123",
            "card_123",
            trigger="button",
            requested_by="crm_ui",
            task_text="check vin",
            card_context={"card": {"id": "card_123"}},
        )

        self.assertEqual(task["task_id"], "agtask_123")
        self.assertEqual(task["card_id"], "card_123")
        self.assertEqual(task["purpose"], "card_enrichment")
        self.assertEqual(task["trigger"], "button")
        self.assertEqual(task["requested_by"], "crm_ui")
        self.assertEqual(task["task_text"], "check vin")
        self.assertIn("card_context", task)

    def test_normalize_card_enrichment_patch_drops_unknown_keys(self) -> None:
        patch = normalize_card_enrichment_patch(
            {
                "description": "  VIN found.  ",
                "vehicle": " Honda Accord 2003 ",
                "title": "should be dropped",
                "vehicle_profile": {
                    "vin": "1hgcm82633a004352",
                    "make_display": "HONDA",
                    "model_display": "Accord",
                    "production_year": "2003",
                    "engine_model": "V6 SOHC 24V",
                    "unknown_field": "drop me",
                    "source_links_or_refs": ["https://example.com/a", ""],
                    "autofilled_fields": ["vin", "make_display", "unknown_field"],
                    "field_sources": {"vin": "vin_web_research", "unknown_field": "drop"},
                    "data_completion_state": "mostly_autofilled",
                    "oem_notes": "short note",
                },
            }
        )

        self.assertEqual(patch["description"], "VIN found.")
        self.assertEqual(patch["vehicle"], "Honda Accord 2003")
        self.assertNotIn("title", patch)
        self.assertEqual(patch["vehicle_profile"]["vin"], "1HGCM82633A004352")
        self.assertNotIn("unknown_field", patch["vehicle_profile"])
        self.assertEqual(patch["vehicle_profile"]["data_completion_state"], "mostly_autofilled")

    def test_build_card_enrichment_response_is_minimal(self) -> None:
        response = build_card_enrichment_response(
            "agtask_123",
            "card_123",
            status="completed",
            summary="done",
            patch={"vehicle_profile": {"vin": "1hgcm82633a004352", "unknown": "drop"}},
            warnings=["ok"],
            sources=["source-a"],
            needs_review=False,
        )

        self.assertEqual(response["task_id"], "agtask_123")
        self.assertEqual(response["card_id"], "card_123")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["summary"], "done")
        self.assertEqual(response["warnings"], ["ok"])
        self.assertEqual(response["sources"], ["source-a"])
        self.assertFalse(response["needs_review"])
        self.assertEqual(response["patch"]["vehicle_profile"]["vin"], "1HGCM82633A004352")
        self.assertNotIn("unknown", response["patch"]["vehicle_profile"])

    def test_build_card_enrichment_response_promotes_supported_patch(self) -> None:
        response = build_card_enrichment_response(
            "agtask_123",
            "card_123",
            status="unsupported",
            summary="check manually",
            patch={"description": "ok", "extra": "drop"},
            warnings=[],
            sources=[],
            needs_review=False,
        )

        self.assertEqual(response["status"], "completed")
        self.assertFalse(response["needs_review"])
        self.assertEqual(response["patch"], {"description": "ok"})


if __name__ == "__main__":
    unittest.main()
