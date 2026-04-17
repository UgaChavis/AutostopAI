from __future__ import annotations

from typing import Any

import re


_AUTOFILL_VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)


class AgentTaskRouter:
    def classify_task(self, task: dict[str, Any], metadata: dict[str, Any]) -> str:
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        if purpose == "card_enrichment":
            return "card_enrichment"
        if self._is_card_cleanup_task(task, metadata):
            return "card_enrichment"
        return "general"

    def context_kind(self, metadata: dict[str, Any]) -> str:
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        return str(context.get("kind", "") or "board").strip().lower() or "board"

    def should_preload_context(self, *, task_type: str, metadata: dict[str, Any], context_kind: str) -> bool:
        if self.context_kind(metadata) == "card":
            return True
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        return purpose == "card_enrichment" or task_type in {"card_enrichment", "card_cleanup"}

    def scenario_chain_for_task(
        self,
        *,
        metadata: dict[str, Any],
        task_type: str,
        context_kind: str,
        facts: dict[str, Any],
    ) -> list[str]:
        purpose = str(metadata.get("purpose", "") or "").strip().lower()
        if purpose == "card_enrichment" and context_kind == "card":
            return ["vin_enrichment"]
        if context_kind == "card" and task_type in {"card_cleanup", "card_enrichment"}:
            return ["vin_enrichment"]
        return []

    def suggest_allowed_write_targets(self, *, task_type: str, context_kind: str, metadata: dict[str, Any] | None = None) -> list[str]:
        if context_kind != "card":
            return []
        return ["description", "vehicle", "vehicle_profile"]

    def extract_vin(self, source_text: str) -> str:
        match = _AUTOFILL_VIN_PATTERN.search(str(source_text or "").upper())
        return match.group(0) if match else ""

    def _is_card_cleanup_task(self, task: dict[str, Any], metadata: dict[str, Any]) -> bool:
        if not self._cleanup_card_id(metadata):
            return False
        text = self._normalized_task_text(str(task.get("task_text", "") or ""))
        cleanup_markers = (
            "наведи порядок",
            "порядок в карточке",
            "структурир",
            "заполни карточ",
            "cleanup",
            "clean up",
            "tidy up",
            "structure the card",
        )
        for marker in cleanup_markers:
            if marker in text:
                return True
        return ("карточ" in text or "card" in text) and ("структур" in text or "заполни" in text or "поряд" in text)

    def _cleanup_card_id(self, metadata: dict[str, Any]) -> str:
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        if str(context.get("kind", "")).strip().lower() != "card":
            return ""
        return str(context.get("card_id", "") or "").strip()

    def _normalized_task_text(self, value: str) -> str:
        text = " ".join(str(value or "").strip().lower().split())
        if not text:
            return ""
        repaired = self._repair_mojibake_text(text)
        return repaired if self._task_text_score(repaired) > self._task_text_score(text) else text

    def _repair_mojibake_text(self, text: str) -> str:
        candidates = [text]
        for encoding in ("latin1", "cp1251", "cp866"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidates.append(" ".join(repaired.lower().split()))
        best = text
        best_score = self._task_text_score(text)
        for candidate in candidates[1:]:
            score = self._task_text_score(candidate)
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _task_text_score(self, text: str) -> int:
        normalized = str(text or "").lower()
        keywords = (
            "наведи",
            "поряд",
            "карточ",
            "структур",
            "заполни",
            "vin",
            "расшифр",
        )
        score = sum(8 for keyword in keywords if keyword in normalized)
        score += sum(1 for char in normalized if ("а" <= char <= "я") or char == "ё")
        score -= normalized.count("?") * 4
        score -= normalized.count("�") * 6
        return score
