from __future__ import annotations

from pathlib import Path
import logging
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.agent.router import AgentTaskRouter
from minimal_kanban.agent.sandbox import OfflineAgentSandbox, SandboxCard
from minimal_kanban.agent.runner import AgentRunner
from minimal_kanban.agent.sandbox import NullModelClient, OfflineBoardApiClient
from minimal_kanban.agent.storage import AgentStorage


class OfflineAgentRouterTests(unittest.TestCase):
    def test_routes_card_enrichment_tasks_without_crm_dependencies(self) -> None:
        router = AgentTaskRouter()
        metadata = {"purpose": "card_enrichment", "context": {"kind": "card", "card_id": "card-1"}}
        task = {"task_text": "Наведи порядок в карточке", "metadata": metadata}

        self.assertEqual(router.classify_task(task, metadata), "card_enrichment")
        self.assertEqual(router.context_kind(metadata), "card")
        self.assertTrue(router.should_preload_context(task_type="card_enrichment", metadata=metadata, context_kind="card"))
        self.assertEqual(router.suggest_allowed_write_targets(task_type="card_enrichment", context_kind="card", metadata=metadata), ["description", "vehicle", "vehicle_profile"])
        self.assertEqual(router.extract_vin("VIN: JTDKB20U093123456"), "JTDKB20U093123456")


class OfflineAgentSandboxTests(unittest.TestCase):
    def test_preview_uses_offline_board_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autostopai-test-") as temp_dir:
            with OfflineAgentSandbox(base_dir=Path(temp_dir)) as sandbox:
                sandbox.seed_card(
                    SandboxCard(
                        id="card-1",
                        title="Toyota Camry 2015",
                        description="Наведи порядок в карточке.",
                    )
                )
                snapshot = sandbox.preview_card(
                    card_id="card-1",
                    task_text="Наведи порядок в карточке и оставь только полезное.",
                )

        self.assertEqual(snapshot["context_kind"], "card")
        self.assertEqual(snapshot["task_type"], "card_enrichment")
        self.assertIn("plan", snapshot)
        self.assertIn("card_context", snapshot)
        self.assertEqual(snapshot["card_context"]["card"]["id"], "card-1")

    def test_run_once_completes_without_crm_connection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autostopai-test-") as temp_dir:
            with OfflineAgentSandbox(base_dir=Path(temp_dir)) as sandbox:
                sandbox.seed_card(
                    SandboxCard(
                        id="card-1",
                        title="Toyota Camry 2015",
                        description="Клиент ожидает. Наведи порядок в карточке.",
                    )
                )
                sandbox.enqueue_card_enrichment_task(
                    card_id="card-1",
                    task_text="Наведи порядок в карточке и сохрани только полезное.",
                )
                processed = sandbox.run_once()
                snapshot = sandbox.snapshot()

        self.assertTrue(processed)
        self.assertEqual(snapshot["status"]["running"], False)
        self.assertEqual(snapshot["tasks"][0]["status"], "completed")
        self.assertGreaterEqual(len(snapshot["runs"]), 1)
        self.assertTrue(any(call["method"] == "get_card_context" for call in snapshot["calls"]))
        self.assertFalse(any(call["method"] == "update_card" for call in snapshot["calls"]))

    def test_card_enrichment_ignores_repair_order_text(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autostopai-test-") as temp_dir:
            with OfflineAgentSandbox(base_dir=Path(temp_dir)) as sandbox:
                sandbox.seed_card(
                    SandboxCard(
                        id="card-1",
                        title="Toyota Camry 2015",
                        description="VIN: JTDKB20U093123456",
                        repair_order={"text": "repair order should not affect card_enrichment"},
                    )
                )
                preview = sandbox.preview_card_enrichment(
                    card_id="card-1",
                    task_text="Обогати карточку по VIN.",
                )

        self.assertEqual(preview["task_type"], "card_enrichment")
        self.assertEqual(preview["facts"]["repair_order"], {})
        self.assertEqual(preview["facts"]["vehicle_context"]["vin"], "JTDKB20U093123456")
        self.assertEqual(preview["plan"]["scenario_chain"], ["vin_enrichment"])

    def test_long_description_is_compacted_when_merged_with_ai_text(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autostopai-test-") as temp_dir:
            storage = AgentStorage(base_dir=Path(temp_dir))
            runner = AgentRunner(
                storage=storage,
                board_api=OfflineBoardApiClient(),
                model_client=NullModelClient(),
                logger=logging.getLogger("autostopai.test"),
            )
            long_text = (
                "Клиент приехал на диагностику, жалуется на посторонний шум в передней части, "
                "говорит, что уже менял масло и фильтры, но проблема осталась. Просит проверить, "
                "не нужна ли замена по подвеске. VIN в тексте: 1HGCM82633A004352. Дополнительно просит пройти ТО."
            )
            merged = runner._merge_card_autofill_description(
                long_text,
                "ИИ:\n- По VIN подтверждено: Honda, Accord, 2003.",
            )

        self.assertLess(len(merged), len(long_text))
        self.assertIn("По VIN подтверждено", merged)
        self.assertIn("ИИ:", merged)


if __name__ == "__main__":
    unittest.main()
