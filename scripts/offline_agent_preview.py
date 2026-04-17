from __future__ import annotations

import argparse
import json

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_kanban.agent.sandbox import OfflineAgentSandbox, SandboxCard, demo_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AutostopAI agent offline without CRM.")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demonstration scenario.")
    parser.add_argument("--preview", action="store_true", help="Print the offline planning snapshot only.")
    args = parser.parse_args()

    if args.demo or args.preview:
        if args.preview:
            with OfflineAgentSandbox() as sandbox:
                sandbox.seed_card(
                    SandboxCard(
                        id="card-1",
                        title="Nissan X-Trail 2018",
                        description="Заполни карточку и сохрани только полезное.",
                        vehicle="",
                    )
                )
                snapshot = sandbox.preview_card(
                    card_id="card-1",
                    task_text="Наведи порядок в карточке автосервиса без CRM.",
                )
        else:
            snapshot = demo_snapshot()
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return 0

    with OfflineAgentSandbox() as sandbox:
        sandbox.seed_card(
            SandboxCard(
                id="card-1",
                title="Toyota Camry 2015",
                description="Клиент ожидает. Наведи порядок в карточке.",
                vehicle="",
            )
        )
        preview = sandbox.preview_card(card_id="card-1", task_text="Наведи порядок в карточке и уточни полезные данные.")
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(json.dumps({"calls": sandbox.board_api.calls}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
