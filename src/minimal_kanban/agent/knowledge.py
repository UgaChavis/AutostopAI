from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .automotive_tools import AutomotiveLookupService, InternetToolError
from .source_registry import trusted_domains


@dataclass(frozen=True)
class CuratedDocumentDefinition:
    document_id: str
    title: str
    relative_path: str
    description: str
    use_case: str
    keywords: tuple[str, ...]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{3,}", re.UNICODE)
_DOC_CONTEXT_KEYWORDS = (
    "документ",
    "документы",
    "guide",
    "guidance",
    "instruction",
    "instructions",
    "инструкция",
    "инструкции",
    "регламент",
    "план",
    "plan",
    "scenario",
    "module",
    "settings",
    "api",
    "handoff",
    "knowledge",
    "remodel",
    "reference",
)
_INTERNET_CONTEXT_KEYWORDS = (
    "интернет",
    "web",
    "browse",
    "lookup",
    "search",
    "найди",
    "проверь",
    "актуаль",
    "сейчас",
    "цена",
    "каталог",
    "вин",
    "vin",
    "ошибка",
    "dtc",
    "part",
    "parts",
    "oem",
)

CURATED_DOCUMENTS: tuple[CuratedDocumentDefinition, ...] = (
    CuratedDocumentDefinition(
        document_id="master_plan",
        title="MASTER-PLAN",
        relative_path="MASTER-PLAN.md",
        description="Главная рамка новой AI-модели и перехода по модулям.",
        use_case="Module 1 overview and product direction.",
        keywords=("ai", "plan", "module", "scenario", "remodel", "model"),
    ),
    CuratedDocumentDefinition(
        document_id="project_handoff",
        title="PROJECT_HANDOFF",
        relative_path="PROJECT_HANDOFF.md",
        description="Текущее состояние проекта, ограничения и handoff notes.",
        use_case="Operational handoff and current working rules.",
        keywords=("handoff", "state", "runtime", "status", "workflow", "project"),
    ),
    CuratedDocumentDefinition(
        document_id="readme_settings",
        title="README_SETTINGS",
        relative_path="README_SETTINGS.md",
        description="Настройки и runtime switches для локального контура.",
        use_case="Runtime and settings guidance.",
        keywords=("settings", "config", "runtime", "toggle", "mode", "switch"),
    ),
    CuratedDocumentDefinition(
        document_id="api_guide",
        title="API_GUIDE",
        relative_path="API_GUIDE.md",
        description="Локальный API и основные точки интеграции.",
        use_case="API surface and request/response usage.",
        keywords=("api", "route", "server", "request", "response", "tool"),
    ),
)


def get_curated_documents() -> list[dict[str, Any]]:
    return [definition_to_payload(item) for item in CURATED_DOCUMENTS]


def definition_to_payload(item: CuratedDocumentDefinition, *, excerpt: str = "") -> dict[str, Any]:
    payload = {
        "document_id": item.document_id,
        "title": item.title,
        "source_path": item.relative_path,
        "description": item.description,
        "use_case": item.use_case,
        "relative_path": item.relative_path,
        "excerpt": excerpt,
        "keywords": list(item.keywords),
    }
    return payload


def build_ai_chat_knowledge_packet(
    *,
    prompt: str,
    context: dict[str, Any] | None = None,
    prompt_profile: dict[str, Any] | None = None,
    lookup_service: AutomotiveLookupService | None = None,
    document_limit: int = 3,
    internet_limit: int = 3,
) -> dict[str, Any]:
    normalized_prompt = _normalize_text(prompt)
    compact_context = context if isinstance(context, dict) else {}
    documents_requested = _should_use_documents(normalized_prompt, compact_context)
    internet_requested = _should_use_internet(normalized_prompt, compact_context)
    selected_documents = (
        _select_curated_documents(normalized_prompt, compact_context, limit=document_limit)
        if documents_requested
        else []
    )
    internet_packet = (
        _lookup_controlled_internet(
            prompt=normalized_prompt,
            context=compact_context,
            lookup_service=lookup_service,
            limit=internet_limit,
        )
        if internet_requested
        else _empty_internet_packet(normalized_prompt, compact_context)
    )
    source_labels = ["CRM"]
    if selected_documents:
        source_labels.append("documents")
    if internet_packet["used"]:
        source_labels.append("internet")
    crm_summary = _build_crm_summary(compact_context)
    return {
        "kind": "ai_chat_knowledge",
        "prompt": normalized_prompt,
        "prompt_profile_kind": str(
            (prompt_profile or {}).get("kind")
            or (prompt_profile or {}).get("prompt_profile_kind")
            or "ai_chat"
        ).strip()
        or "ai_chat",
        "source_labels": source_labels,
        "crm": crm_summary,
        "documents": {
            "available": bool(CURATED_DOCUMENTS),
            "requested": documents_requested,
            "used": bool(selected_documents),
            "count": len(selected_documents),
            "items": selected_documents,
        },
        "internet": internet_packet,
        "policy": {
            "documents_requested": documents_requested,
            "internet_requested": internet_requested,
            "external_knowledge_used": bool(selected_documents) or bool(internet_packet["used"]),
            "external_knowledge_allowed": True,
        },
    }


def _build_crm_summary(context: dict[str, Any]) -> dict[str, Any]:
    card_context = (
        context.get("card_context") if isinstance(context.get("card_context"), dict) else {}
    )
    repair_order_context = (
        context.get("repair_order_context")
        if isinstance(context.get("repair_order_context"), dict)
        else {}
    )
    wall_digest = context.get("wall_digest") if isinstance(context.get("wall_digest"), dict) else {}
    attachments = (
        context.get("attachments_intake")
        if isinstance(context.get("attachments_intake"), dict)
        else {}
    )
    return {
        "kind": str(context.get("kind") or "compact_context").strip() or "compact_context",
        "surface": str(context.get("surface") or "ai_chat").strip() or "ai_chat",
        "source_kind": str(context.get("source_kind") or "workspace").strip() or "workspace",
        "card_id": str(context.get("card_id") or card_context.get("card_id") or "").strip(),
        "repair_order_id": str(
            context.get("repair_order_id") or repair_order_context.get("repair_order_id") or ""
        ).strip(),
        "card_label": str(
            context.get("card_label") or card_context.get("summary_label") or ""
        ).strip(),
        "repair_order_label": str(
            context.get("repair_order_label") or repair_order_context.get("summary_label") or ""
        ).strip(),
        "context_label": str(context.get("context_label") or "").strip(),
        "wall_label": str(
            wall_digest.get("label") or wall_digest.get("summary_label") or ""
        ).strip(),
        "attachments_label": str(
            attachments.get("label") or attachments.get("summary_label") or ""
        ).strip(),
    }


def _normalize_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\s+", " ", text)


def _context_text(context: dict[str, Any]) -> str:
    parts = [
        str(context.get("prompt") or ""),
        str(context.get("card_label") or ""),
        str(context.get("context_label") or ""),
        str(context.get("repair_order_label") or ""),
    ]
    card_context = (
        context.get("card_context") if isinstance(context.get("card_context"), dict) else {}
    )
    repair_order_context = (
        context.get("repair_order_context")
        if isinstance(context.get("repair_order_context"), dict)
        else {}
    )
    wall_digest = context.get("wall_digest") if isinstance(context.get("wall_digest"), dict) else {}
    parts.extend(
        [
            str(card_context.get("summary_label") or ""),
            str(card_context.get("title") or ""),
            str(
                card_context.get("ai_relevant_facts", {}).get("client")
                if isinstance(card_context.get("ai_relevant_facts"), dict)
                else ""
            ),
            str(repair_order_context.get("summary_label") or ""),
            str(
                repair_order_context.get("ai_relevant_facts", {}).get("machine")
                if isinstance(repair_order_context.get("ai_relevant_facts"), dict)
                else ""
            ),
            str(wall_digest.get("label") or wall_digest.get("summary_label") or ""),
            str(wall_digest.get("summary_text") or ""),
        ]
    )
    return _normalize_text(" ".join(part for part in parts if part))


def _context_terms(context: dict[str, Any]) -> list[str]:
    text = _context_text(context)
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text) if len(token) >= 4]


def _prompt_terms(prompt: str, context: dict[str, Any]) -> list[str]:
    tokens = [token.casefold() for token in _TOKEN_PATTERN.findall(prompt)]
    tokens.extend(_context_terms(context))
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _should_use_documents(prompt: str, context: dict[str, Any]) -> bool:
    haystack = " ".join([prompt, _context_text(context)]).casefold()
    return any(keyword in haystack for keyword in _DOC_CONTEXT_KEYWORDS)


def _should_use_internet(prompt: str, context: dict[str, Any]) -> bool:
    haystack = " ".join([prompt, _context_text(context)]).casefold()
    if any(keyword in haystack for keyword in _INTERNET_CONTEXT_KEYWORDS):
        return True
    return bool(re.search(r"\bvin\b", haystack, re.IGNORECASE)) or bool(
        re.search(r"\bdtc\b", haystack, re.IGNORECASE)
    )


def _select_curated_documents(
    prompt: str, context: dict[str, Any], *, limit: int
) -> list[dict[str, Any]]:
    terms = _prompt_terms(prompt, context)
    scored: list[tuple[int, CuratedDocumentDefinition, str]] = []
    for definition in CURATED_DOCUMENTS:
        text = _load_document_text(definition.relative_path)
        score = _score_document(definition, text, terms)
        if score <= 0:
            continue
        excerpt = _build_document_excerpt(text, terms)
        scored.append((score, definition, excerpt))
    if not scored:
        for definition in CURATED_DOCUMENTS[:limit]:
            text = _load_document_text(definition.relative_path)
            scored.append((1, definition, _build_document_excerpt(text, terms)))
    scored.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    selected: list[dict[str, Any]] = []
    for score, definition, excerpt in scored[: max(1, limit)]:
        selected.append(
            {
                **definition_to_payload(definition, excerpt=excerpt),
                "relevance_score": score,
            }
        )
    return selected


def _score_document(definition: CuratedDocumentDefinition, text: str, terms: list[str]) -> int:
    haystack = " ".join(
        [
            definition.title.casefold(),
            definition.description.casefold(),
            definition.use_case.casefold(),
            " ".join(definition.keywords).casefold(),
            text.casefold(),
        ]
    )
    score = 0
    for term in terms:
        if len(term) < 3:
            continue
        if term in haystack:
            score += 1
    return score


@lru_cache(maxsize=32)
def _load_document_text(relative_path: str) -> str:
    path = _REPO_ROOT / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _build_document_excerpt(text: str, terms: list[str], *, max_chars: int = 1200) -> str:
    source = _normalize_text(text)
    if not source:
        return ""
    if len(source) <= max_chars:
        return source
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return source[:max_chars].strip()
    matched_indexes: list[int] = []
    for index, line in enumerate(lines):
        lower = line.casefold()
        if any(term in lower for term in terms):
            matched_indexes.append(index)
    if not matched_indexes:
        return _normalize_text("\n".join(lines[:12]))[:max_chars].strip()
    collected: list[str] = []
    seen: set[str] = set()
    for index in matched_indexes[:3]:
        for candidate in lines[max(0, index - 1) : min(len(lines), index + 2)]:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            collected.append(normalized)
            if len("\n".join(collected)) >= max_chars:
                return "\n".join(collected)[:max_chars].strip()
    return "\n".join(collected)[:max_chars].strip()


def _internet_query(prompt: str, context: dict[str, Any]) -> str:
    parts = [prompt]
    card_context = (
        context.get("card_context") if isinstance(context.get("card_context"), dict) else {}
    )
    repair_order_context = (
        context.get("repair_order_context")
        if isinstance(context.get("repair_order_context"), dict)
        else {}
    )
    for value in (
        card_context.get("summary_label"),
        card_context.get("ai_relevant_facts", {}).get("machine")
        if isinstance(card_context.get("ai_relevant_facts"), dict)
        else "",
        repair_order_context.get("summary_label"),
        repair_order_context.get("ai_relevant_facts", {}).get("vehicle")
        if isinstance(repair_order_context.get("ai_relevant_facts"), dict)
        else "",
        repair_order_context.get("ai_relevant_facts", {}).get("vin")
        if isinstance(repair_order_context.get("ai_relevant_facts"), dict)
        else "",
    ):
        text = str(value or "").strip()
        if text:
            parts.append(text)
    query = _normalize_text(" ".join(parts))
    return query[:240]


def _select_allowed_domains(prompt: str, context: dict[str, Any]) -> list[str] | None:
    haystack = " ".join([prompt, _context_text(context)]).casefold()
    if "vin" in haystack:
        domains = trusted_domains(kind="vin")
        return domains or None
    if any(
        keyword in haystack
        for keyword in (
            "catalog",
            "part",
            "parts",
            "oem",
            "price",
            "цена",
            "каталог",
            "деталь",
            "запчаст",
        )
    ):
        domains = trusted_domains(kind="catalog") + trusted_domains(kind="price")
        return domains or None
    if any(
        keyword in haystack
        for keyword in ("dtc", "ошибк", "fault", "symptom", "симптом", "диагност")
    ):
        domains = trusted_domains(kind="dtc") + trusted_domains(kind="fault")
        return domains or None
    return None


def _empty_internet_packet(prompt: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        "requested": False,
        "used": False,
        "count": 0,
        "query": _internet_query(prompt, context),
        "allowed_domains": [],
        "items": [],
    }


def _lookup_controlled_internet(
    *,
    prompt: str,
    context: dict[str, Any],
    lookup_service: AutomotiveLookupService | None,
    limit: int,
) -> dict[str, Any]:
    service = lookup_service or AutomotiveLookupService()
    query = _internet_query(prompt, context)
    allowed_domains = _select_allowed_domains(prompt, context)
    try:
        search_payload = service.search_web(
            query=query, limit=max(1, limit), allowed_domains=allowed_domains
        )
    except InternetToolError as exc:
        return {
            "available": True,
            "requested": True,
            "used": False,
            "count": 0,
            "query": query,
            "allowed_domains": allowed_domains or [],
            "items": [],
            "error": str(exc),
        }
    results = search_payload.get("results") if isinstance(search_payload, dict) else []
    normalized_results: list[dict[str, Any]] = []
    for index, item in enumerate(results[: max(1, limit)] if isinstance(results, list) else []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = _normalize_text(item.get("snippet") or item.get("body") or "")
        domain = str(item.get("domain") or urlsplit(url).netloc).strip()
        excerpt = ""
        if index == 0 and url:
            try:
                fetched = service.fetch_page_excerpt(url=url, max_chars=1200)
                excerpt = _normalize_text(
                    fetched.get("excerpt") if isinstance(fetched, dict) else ""
                )
            except InternetToolError:
                excerpt = ""
        normalized_results.append(
            {
                "title": title,
                "url": url,
                "domain": domain,
                "snippet": snippet,
                "excerpt": excerpt,
            }
        )
    return {
        "available": True,
        "requested": True,
        "used": bool(normalized_results),
        "count": len(normalized_results),
        "query": query,
        "allowed_domains": allowed_domains or [],
        "items": normalized_results,
    }
