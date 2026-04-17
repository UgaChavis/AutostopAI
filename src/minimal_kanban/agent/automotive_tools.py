from __future__ import annotations

from copy import deepcopy
import re
import json
from typing import Any
from urllib.parse import quote

import httpx

from .source_registry import PARTS_CATALOG_SOURCES, PARTS_PRICE_SOURCES, trusted_domains
from .web_tools import DuckDuckGoSearchClient, InternetToolError


_PART_NUMBER_PATTERN = re.compile(r"\b[A-Z0-9-]{5,18}\b")


_PRICE_PATTERN = re.compile(r"(\d[\d\s]{2,}(?:[.,]\d{1,2})?)\s*(₽|руб(?:\.|лей|ля)?|KZT|₸|\$|€)", re.I)
_DEFAULT_SERVICE_TYPE = "ТО"
_MAINTENANCE_HINTS = ("то", "техобслуж", "техничес", "service", "oil")
_BRAKE_HINTS = ("торм", "brake")
_SUSPENSION_HINTS = ("подвес", "ходов", "suspension")
_SPARK_HINTS = ("свеч", "spark")
_PART_NUMBER_STOPWORDS = {
    "RADIATOR",
    "EXTERIOR",
    "INTERIOR",
    "ENGINE",
    "FILTER",
    "COOLANT",
    "FRONT",
    "REAR",
    "RIGHT",
    "LEFT",
    "PRICE",
    "OEM",
    "PART",
    "NUMBER",
    "CATALOG",
}
_PART_QUERY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("радиатор", ("radiator", "coolant radiator")),
    ("рычаг", ("control arm", "suspension arm")),
    ("стойка", ("shock absorber", "strut")),
    ("ступиц", ("wheel bearing", "hub bearing")),
    ("колодк", ("brake pads",)),
    ("диск", ("brake disc", "brake rotor")),
    ("термостат", ("thermostat",)),
    ("помп", ("water pump",)),
    ("ремень", ("belt", "serpentine belt")),
    ("цеп", ("timing chain",)),
    ("масло", ("engine oil",)),
    ("фильтр", ("filter", "oil filter")),
    ("свеч", ("spark plug",)),
    ("аккумулятор", ("battery",)),
)


class AutomotiveLookupService:
    def __init__(self, *, timeout_seconds: float = 12.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._search = DuckDuckGoSearchClient(timeout_seconds=timeout_seconds)
        self._task_cache: dict[str, dict[str, Any]] = {}

    def reset_task_cache(self) -> None:
        self._task_cache.clear()

    def _cache_key(self, method_name: str, payload: dict[str, Any]) -> str:
        return f"{method_name}:{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)}"

    def _cached_result(self, method_name: str, payload: dict[str, Any], factory) -> dict[str, Any]:
        cache_key = self._cache_key(method_name, payload)
        cached = self._task_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached)
        result = factory()
        self._task_cache[cache_key] = deepcopy(result)
        return deepcopy(result)

    def decode_vin(self, vin: str) -> dict[str, Any]:
        return self.research_vin(vin)

    def research_vin(self, vin: str, limit: int = 5) -> dict[str, Any]:
        normalized_vin = str(vin or "").strip().upper()
        if len(normalized_vin) < 11:
            raise InternetToolError("VIN is required and must be at least 11 characters.")
        return self._cached_result(
            "research_vin",
            {"vin": normalized_vin, "limit": int(limit or 5)},
            lambda: self._research_vin_uncached(normalized_vin, limit=max(1, limit or 5)),
        )

    def search_part_numbers(self, *, vehicle_context: dict[str, Any] | None, part_query: str, limit: int = 8) -> dict[str, Any]:
        normalized_query = self._required_query(part_query)
        context = self._normalize_vehicle_context(vehicle_context)
        return self._cached_result(
            "search_part_numbers",
            {"vehicle_context": context, "part_query": normalized_query, "limit": int(limit or 8)},
            lambda: self._search_part_numbers_uncached(context=context, normalized_query=normalized_query, limit=limit),
        )

    def find_part_numbers(self, *, query: str, vehicle: dict[str, Any] | str | None = None, limit: int = 5) -> dict[str, Any]:
        vehicle_context = vehicle if isinstance(vehicle, dict) else {"vehicle": str(vehicle or "").strip()}
        return self.search_part_numbers(vehicle_context=vehicle_context, part_query=query, limit=limit)

    def lookup_part_prices(
        self,
        *,
        vehicle_context: dict[str, Any] | None,
        part_number_or_query: str,
        limit: int = 8,
    ) -> dict[str, Any]:
        normalized_query = self._required_query(part_number_or_query)
        context = self._normalize_vehicle_context(vehicle_context)
        return self._cached_result(
            "lookup_part_prices",
            {"vehicle_context": context, "part_number_or_query": normalized_query, "limit": int(limit or 8)},
            lambda: self._lookup_part_prices_uncached(context=context, normalized_query=normalized_query, limit=limit),
        )

    def estimate_price_ru(self, *, part_number: str, vehicle: dict[str, Any] | str | None = None, limit: int = 5) -> dict[str, Any]:
        vehicle_context = vehicle if isinstance(vehicle, dict) else {"vehicle": str(vehicle or "").strip()}
        payload = self.lookup_part_prices(
            vehicle_context=vehicle_context,
            part_number_or_query=part_number,
            limit=limit,
        )
        payload["part_number"] = str(part_number or "").strip()
        payload["market"] = "ru"
        return payload

    def estimate_maintenance(
        self,
        *,
        vehicle_context: dict[str, Any] | None,
        service_type: str = _DEFAULT_SERVICE_TYPE,
    ) -> dict[str, Any]:
        return self._cached_result(
            "estimate_maintenance",
            {"vehicle_context": self._normalize_vehicle_context(vehicle_context), "service_type": self._normalize_service_type(service_type)},
            lambda: self._estimate_maintenance_uncached(vehicle_context=vehicle_context, service_type=service_type),
        )

    def _decode_vin_uncached(self, normalized_vin: str) -> dict[str, Any]:
        url = (
            "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/"
            + quote(normalized_vin)
            + "?format=json"
        )
        try:
            with httpx.Client(timeout=self._timeout_seconds, headers={"User-Agent": "Mozilla/5.0 AutoStopCRM/1.0"}) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InternetToolError(f"VIN decode failed: {exc}") from exc
        payload = response.json()
        results = payload.get("Results") or []
        row = results[0] if isinstance(results, list) and results else {}
        if not isinstance(row, dict):
            row = {}
        return {
            "vin": normalized_vin,
            "make": self._text(row.get("Make")),
            "model": self._text(row.get("Model")),
            "model_year": self._text(row.get("ModelYear")),
            "vehicle_type": self._text(row.get("VehicleType")),
            "body_class": self._text(row.get("BodyClass")),
            "engine_model": self._join_values(row.get("EngineModel"), row.get("DisplacementL"), row.get("FuelTypePrimary")),
            "transmission": self._join_values(row.get("TransmissionStyle"), row.get("TransmissionSpeeds")),
            "drive_type": self._text(row.get("DriveType")),
            "plant_country": self._text(row.get("PlantCountry")),
            "plant_company": self._text(row.get("Manufacturer")),
            "error_code": self._text(row.get("ErrorCode")),
            "source": "NHTSA vPIC",
            "source_url": url,
        }

    def _research_vin_uncached(self, normalized_vin: str, *, limit: int) -> dict[str, Any]:
        queries = [
            f'"{normalized_vin}" VIN specifications',
            f'"{normalized_vin}" vehicle specifications engine transmission',
            f'"{normalized_vin}" make model year',
        ]
        allowed_domains = [
            "vpic.nhtsa.dot.gov",
            "get.vin",
            "www.vindecoderz.com",
            "vindecoderz.com",
            "vincheck.info",
            "www.vincheck.info",
            "vehiclehistory.com",
            "www.vehiclehistory.com",
        ]
        raw_results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for query in queries:
            payload = self._search_web_uncached(query=query, limit=limit, allowed_domains=allowed_domains)
            results = payload.get("results") if isinstance(payload, dict) else []
            if not isinstance(results, list):
                continue
            for item in results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                excerpt = ""
                try:
                    fetched = self._fetch_page_excerpt_uncached(url=url, max_chars=1600)
                    excerpt = self._text(fetched.get("excerpt"))
                except InternetToolError:
                    excerpt = ""
                raw_results.append(
                    {
                        "title": self._text(item.get("title")),
                        "url": url,
                        "domain": self._text(item.get("domain")),
                        "snippet": self._text(item.get("snippet")),
                        "excerpt": excerpt,
                    }
                )
                if len(raw_results) >= limit:
                    break
            if len(raw_results) >= limit:
                break
        if len(raw_results) < max(2, limit // 2):
            for query in queries:
                payload = self._search_web_uncached(query=query, limit=limit, allowed_domains=[])
                results = payload.get("results") if isinstance(payload, dict) else []
                if not isinstance(results, list):
                    continue
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url", "") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    excerpt = ""
                    try:
                        fetched = self._fetch_page_excerpt_uncached(url=url, max_chars=1600)
                        excerpt = self._text(fetched.get("excerpt"))
                    except InternetToolError:
                        excerpt = ""
                    raw_results.append(
                        {
                            "title": self._text(item.get("title")),
                            "url": url,
                            "domain": self._text(item.get("domain")),
                            "snippet": self._text(item.get("snippet")),
                            "excerpt": excerpt,
                        }
                    )
                    if len(raw_results) >= limit:
                        break
                if len(raw_results) >= limit:
                    break
        return {
            "vin": normalized_vin,
            "queries": queries,
            "results": raw_results,
            "source_summary": "Public VIN web research",
            "source_confidence": 0.0,
            "source_links_or_refs": [item["url"] for item in raw_results if item.get("url")],
            "source": "web_research",
        }

    def _search_part_numbers_uncached(self, *, context: dict[str, Any], normalized_query: str, limit: int) -> dict[str, Any]:
        query_variants = self._expand_part_query_variants(normalized_query)
        queries: list[str] = []
        for variant in query_variants[:3]:
            if context.get("vin"):
                queries.append(f'{context["vin"]} {variant} OEM part number')
            queries.append(self._build_vehicle_query(context, variant, suffix="OEM part number"))
            queries.append(self._build_vehicle_query(context, variant, suffix="catalog"))
        results = self._search_domains(
            queries,
            allowed_domains=trusted_domains(kind="catalog"),
            per_query_limit=max(2, limit),
            total_limit=limit,
        )
        enriched_results = self._enrich_part_catalog_results(results)
        return {
            "vehicle_context": context,
            "part_query": normalized_query,
            "query_variants": query_variants,
            "results": enriched_results,
            "part_numbers": self._extract_part_numbers_from_results(enriched_results),
            "source_group": [item.label for item in PARTS_CATALOG_SOURCES],
        }

    def _lookup_part_prices_uncached(self, *, context: dict[str, Any], normalized_query: str, limit: int) -> dict[str, Any]:
        quoted = f'"{normalized_query}"'
        queries = [
            f"{quoted} price",
            self._build_vehicle_query(context, normalized_query, suffix="price"),
        ]
        search_results = self._search_domains(
            queries,
            allowed_domains=trusted_domains(kind="price") + trusted_domains(kind="catalog"),
            per_query_limit=max(3, limit),
            total_limit=limit,
        )
        enriched: list[dict[str, Any]] = []
        for item in search_results:
            price_matches = self._extract_prices(" ".join([item.get("title", ""), item.get("snippet", "")]))
            page_excerpt = ""
            if not price_matches:
                try:
                    fetched = self._search.fetch_page_excerpt(item.get("url", ""), max_chars=1800)
                    page_excerpt = str(fetched.get("excerpt", "") or "")
                    price_matches = self._extract_prices(page_excerpt)
                except InternetToolError:
                    page_excerpt = ""
            enriched.append(
                {
                    **item,
                    "prices": price_matches,
                    "page_excerpt": page_excerpt[:500] if page_excerpt else "",
                }
            )
        return {
            "vehicle_context": context,
            "query": normalized_query,
            "results": enriched,
            "price_summary": self._summarize_price_results(enriched),
            "source_group": [item.label for item in PARTS_PRICE_SOURCES],
        }

    def _estimate_maintenance_uncached(self, *, vehicle_context: dict[str, Any] | None, service_type: str) -> dict[str, Any]:
        context = self._normalize_vehicle_context(vehicle_context)
        normalized_service = self._normalize_service_type(service_type)
        lower = normalized_service.casefold()
        works = [{"name": "Диагностика и осмотр автомобиля", "quantity": "1"}]
        materials = [
            {"name": "Моторное масло", "quantity": "1"},
            {"name": "Масляный фильтр", "quantity": "1"},
        ]
        notes = [
            "Список работ и материалов предварительный.",
            "Для точной цены по материалам используйте lookup_part_prices после уточнения каталожных номеров.",
        ]
        mileage_text = self._text(context.get("mileage"))
        try:
            mileage_value = int(str(mileage_text).replace(" ", "")) if mileage_text else 0
        except ValueError:
            mileage_value = 0
        if lower == "то" or self._contains_any(lower, _MAINTENANCE_HINTS):
            self._append_unique_rows(
                works,
                [
                    {"name": "Замена моторного масла", "quantity": "1"},
                    {"name": "Замена масляного фильтра", "quantity": "1"},
                    {"name": "Контроль технических жидкостей", "quantity": "1"},
                ],
            )
            self._append_unique_rows(
                materials,
                [
                    {"name": "Прокладка сливной пробки", "quantity": "1"},
                    {"name": "Фильтр салона", "quantity": "1"},
                    {"name": "Воздушный фильтр двигателя", "quantity": "1"},
                ],
            )
        if self._contains_any(lower, _BRAKE_HINTS):
            self._append_unique_rows(works, [{"name": "Осмотр тормозной системы", "quantity": "1"}])
            self._append_unique_rows(materials, [{"name": "Тормозная жидкость", "quantity": "1"}])
        if self._contains_any(lower, _SUSPENSION_HINTS):
            self._append_unique_rows(works, [{"name": "Диагностика подвески", "quantity": "1"}])
        if self._contains_any(lower, _SPARK_HINTS):
            self._append_unique_rows(materials, [{"name": "Комплект свечей зажигания", "quantity": "1"}])
            self._append_unique_rows(works, [{"name": "Замена свечей зажигания", "quantity": "1"}])
        if mileage_value >= 60_000:
            self._append_unique_rows(materials, [{"name": "Тормозная жидкость", "quantity": "1"}, {"name": "Свечи зажигания", "quantity": "1 комплект"}])
            self._append_unique_rows(works, [{"name": "Проверка свечей зажигания и катушек", "quantity": "1"}])
        if mileage_value >= 90_000:
            self._append_unique_rows(materials, [{"name": "Антифриз", "quantity": "по объему"}, {"name": "Ремень навесного оборудования", "quantity": "1"}])
            self._append_unique_rows(works, [{"name": "Проверка приводного ремня и роликов", "quantity": "1"}])
        if mileage_value >= 120_000:
            self._append_unique_rows(works, [{"name": "Проверка масла АКПП/МКПП", "quantity": "1"}])
        if context.get("vin"):
            notes.append("VIN доступен: можно выполнить точный подбор расходников и каталожных номеров.")
        if mileage_text:
            notes.append(f"Пробег в карточке: {mileage_text} км.")
        engine_oil_capacity = self._text(context.get("oil_engine_capacity_l"))
        if engine_oil_capacity:
            notes.append(f"Объем моторного масла: {engine_oil_capacity} л.")
        gearbox_oil_capacity = self._text(context.get("oil_gearbox_capacity_l"))
        if gearbox_oil_capacity:
            notes.append(f"Объем масла КПП: {gearbox_oil_capacity} л.")
        coolant_capacity = self._text(context.get("coolant_capacity_l"))
        if coolant_capacity:
            notes.append(f"Объем охлаждающей жидкости: {coolant_capacity} л.")
        return {
            "vehicle_context": context,
            "service_type": normalized_service,
            "works": works,
            "materials": materials,
            "notes": notes,
        }

    def search_web(self, *, query: str, limit: int = 5, allowed_domains: list[str] | None = None) -> dict[str, Any]:
        normalized_query = self._required_query(query)
        normalized_domains = sorted({str(item or "").strip() for item in (allowed_domains or []) if str(item or "").strip()})
        return self._cached_result(
            "search_web",
            {"query": normalized_query, "limit": int(limit or 5), "allowed_domains": normalized_domains},
            lambda: self._search_web_uncached(query=normalized_query, limit=limit, allowed_domains=normalized_domains),
        )

    def decode_dtc(
        self,
        *,
        code: str,
        vehicle_context: dict[str, Any] | None = None,
        vehicle: dict[str, Any] | str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        normalized_code = self._required_query(code).upper()
        context = self._normalize_vehicle_context(
            vehicle_context if isinstance(vehicle_context, dict) else (vehicle if isinstance(vehicle, dict) else {"vehicle": str(vehicle or "").strip()})
        )
        return self._cached_result(
            "decode_dtc",
            {"code": normalized_code, "vehicle_context": context, "limit": int(limit or 5)},
            lambda: self._decode_dtc_uncached(code=normalized_code, context=context, limit=limit),
        )

    def search_fault_info(
        self,
        *,
        query: str,
        vehicle_context: dict[str, Any] | None = None,
        vehicle: dict[str, Any] | str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        normalized_query = self._required_query(query)
        context = self._normalize_vehicle_context(
            vehicle_context if isinstance(vehicle_context, dict) else (vehicle if isinstance(vehicle, dict) else {"vehicle": str(vehicle or "").strip()})
        )
        return self._cached_result(
            "search_fault_info",
            {"query": normalized_query, "vehicle_context": context, "limit": int(limit or 5)},
            lambda: self._search_fault_info_uncached(query=normalized_query, context=context, limit=limit),
        )

    def fetch_page_excerpt(self, *, url: str, max_chars: int = 2500) -> dict[str, Any]:
        normalized_url = str(url or "").strip()
        return self._cached_result(
            "fetch_page_excerpt",
            {"url": normalized_url, "max_chars": int(max_chars or 2500)},
            lambda: self._fetch_page_excerpt_uncached(url=normalized_url, max_chars=max_chars),
        )

    def _search_web_uncached(self, *, query: str, limit: int, allowed_domains: list[str]) -> dict[str, Any]:
        results = [item.to_dict() for item in self._search.search(query, limit=limit, allowed_domains=allowed_domains)]
        return {
            "query": str(query or "").strip(),
            "results": results,
        }

    def _decode_dtc_uncached(self, *, code: str, context: dict[str, Any], limit: int) -> dict[str, Any]:
        query = " ".join(part for part in (context.get("vehicle", ""), code, "DTC code meaning") if part).strip()
        return {
            "code": code,
            "vehicle_context": context,
            "query": query,
            "results": [item.to_dict() for item in self._search.search(query, limit=limit, allowed_domains=trusted_domains(kind="dtc"))],
            "source_group": ["DTC lookup"],
        }

    def _search_fault_info_uncached(self, *, query: str, context: dict[str, Any], limit: int) -> dict[str, Any]:
        search_query = self._build_vehicle_query(context, query)
        return {
            "query": query,
            "vehicle_context": context,
            "search_query": search_query,
            "results": [item.to_dict() for item in self._search.search(search_query, limit=limit, allowed_domains=trusted_domains(kind="fault"))],
            "source_group": ["Fault search"],
        }

    def _fetch_page_excerpt_uncached(self, *, url: str, max_chars: int) -> dict[str, Any]:
        return self._search.fetch_page_excerpt(url, max_chars=max_chars)

    def _required_query(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise InternetToolError("part query is required")
        return text

    def _normalize_vehicle_context(self, vehicle_context: dict[str, Any] | None) -> dict[str, Any]:
        payload = vehicle_context if isinstance(vehicle_context, dict) else {}
        make = self._text(payload.get("make") or payload.get("make_display"))
        model = self._text(payload.get("model") or payload.get("model_display"))
        year = self._text(payload.get("year") or payload.get("production_year"))
        vehicle = self._text(payload.get("vehicle"))
        if not vehicle:
            vehicle = " ".join(part for part in (make, model, year) if part).strip()
        return {
            "make": make,
            "model": model,
            "year": year,
            "engine": self._text(payload.get("engine") or payload.get("engine_model")),
            "vin": self._text(payload.get("vin")),
            "vehicle": vehicle,
            "mileage": self._text(payload.get("mileage")),
            "oil_engine_capacity_l": self._text(payload.get("oil_engine_capacity_l")),
            "oil_gearbox_capacity_l": self._text(payload.get("oil_gearbox_capacity_l")),
            "coolant_capacity_l": self._text(payload.get("coolant_capacity_l")),
        }

    def _build_vehicle_query(self, context: dict[str, Any], item_query: str, *, suffix: str = "") -> str:
        parts = [
            context.get("make", ""),
            context.get("model", ""),
            context.get("year", ""),
            context.get("engine", ""),
            item_query,
            suffix,
        ]
        return " ".join(part for part in parts if part).strip()

    def _expand_part_query_variants(self, part_query: str) -> list[str]:
        normalized = str(part_query or "").strip()
        lower = normalized.casefold()
        variants = [normalized]
        for token, aliases in _PART_QUERY_ALIASES:
            if token not in lower:
                continue
            for alias in aliases:
                if alias not in variants:
                    variants.append(alias)
        return variants[:4]

    def _search_domains(
        self,
        queries: list[str],
        *,
        allowed_domains: list[str],
        per_query_limit: int,
        total_limit: int,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for query in queries:
            if not query:
                continue
            query_variants = [query]
            for domain in allowed_domains[:2]:
                query_variants.append(f"site:{domain} {query}")
            for query_variant in query_variants:
                try:
                    batch = self._search.search(query_variant, limit=per_query_limit, allowed_domains=allowed_domains)
                except InternetToolError:
                    continue
                for result in batch:
                    if result.url in seen_urls:
                        continue
                    seen_urls.add(result.url)
                    items.append(result.to_dict())
                    if len(items) >= total_limit:
                        return items
                if batch:
                    break
        return items

    def _extract_prices(self, text: str) -> list[dict[str, str]]:
        prices: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for amount, currency in _PRICE_PATTERN.findall(str(text or "")):
            key = (amount.strip(), currency.strip())
            if key in seen:
                continue
            seen.add(key)
            prices.append({"amount": amount.strip(), "currency": currency.strip()})
        return prices[:5]

    def _extract_part_numbers_from_results(self, results: list[dict[str, Any]]) -> list[dict[str, str]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            text = " ".join(
                str(part or "")
                for part in (
                    item.get("title"),
                    item.get("snippet"),
                    item.get("page_excerpt"),
                )
            ).upper()
            for value in self._extract_part_numbers(text):
                key = value.casefold()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "value": value,
                        "domain": str(item.get("domain", "") or "").strip(),
                        "url": str(item.get("url", "") or "").strip(),
                        "_score": self._part_number_score(value),
                    }
                )
        candidates.sort(
            key=lambda item: (
                int(item.get("_score", 0) or 0),
                len(str(item.get("value", "") or "")),
            ),
            reverse=True,
        )
        return [
            {
                "value": str(item.get("value", "") or ""),
                "domain": str(item.get("domain", "") or ""),
                "url": str(item.get("url", "") or ""),
            }
            for item in candidates[:6]
        ]

    def _enrich_part_catalog_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            updated = dict(item)
            existing_text = " ".join(
                str(part or "")
                for part in (
                    updated.get("title"),
                    updated.get("snippet"),
                    updated.get("page_excerpt"),
                )
            )
            if self._extract_part_numbers(existing_text):
                enriched.append(updated)
                continue
            try:
                excerpt_payload = self._search.fetch_page_excerpt(str(updated.get("url", "") or ""), max_chars=1800)
            except InternetToolError:
                enriched.append(updated)
                continue
            updated["page_excerpt"] = str(excerpt_payload.get("excerpt", "") or "")[:600]
            enriched.append(updated)
        return enriched

    def _extract_part_numbers(self, text: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for raw in _PART_NUMBER_PATTERN.findall(str(text or "").upper()):
            candidate = str(raw or "").strip("- ")
            if not self._is_plausible_part_number(candidate):
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(candidate)
        return values

    def _is_plausible_part_number(self, candidate: str) -> bool:
        value = str(candidate or "").strip().upper()
        if len(value) < 5 or len(value) > 18:
            return False
        if len(value) == 17:
            return False
        if value in _PART_NUMBER_STOPWORDS:
            return False
        if re.fullmatch(r"(?:19|20)\d{2}", value):
            return False
        if re.fullmatch(r"\d{1,2}-\d{1,2}", value):
            return False
        letters = sum(1 for char in value if "A" <= char <= "Z")
        digits = sum(1 for char in value if char.isdigit())
        if digits == 0:
            return False
        if letters == 0:
            return 8 <= len(value) <= 15
        if digits < 3:
            return False
        if len(value) <= 6 and digits <= 2:
            return False
        return True

    def _part_number_score(self, value: str) -> int:
        candidate = str(value or "").strip().upper()
        letters = sum(1 for char in candidate if "A" <= char <= "Z")
        digits = sum(1 for char in candidate if char.isdigit())
        hyphens = candidate.count("-")
        if letters == 0 and 8 <= len(candidate) <= 15:
            return 100
        if digits >= 6 and letters >= 1:
            return 90 - hyphens
        if digits >= 4 and letters >= 1:
            return 75 - hyphens
        if digits >= 3:
            return 60 - hyphens
        return 10

    def _summarize_price_results(self, results: list[dict[str, Any]]) -> dict[str, int] | None:
        amounts: list[int] = []
        offers_total = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            for price in item.get("prices") if isinstance(item.get("prices"), list) else []:
                if not isinstance(price, dict):
                    continue
                rub_amount = self._rub_amount(price.get("amount"), price.get("currency"))
                if rub_amount is None:
                    continue
                offers_total += 1
                amounts.append(rub_amount)
        if not amounts:
            return None
        amounts.sort()
        return {
            "offers_total": offers_total,
            "min_rub": amounts[0],
            "max_rub": amounts[-1],
            "avg_rub": int(round(sum(amounts) / len(amounts))),
        }

    def _rub_amount(self, amount: Any, currency: Any) -> int | None:
        currency_text = str(currency or "").strip().casefold()
        if "руб" not in currency_text and "₽" not in currency_text and "rub" not in currency_text:
            return None
        normalized = str(amount or "").strip().replace(" ", "").replace(",", ".")
        if not normalized:
            return None
        try:
            return int(round(float(normalized)))
        except (TypeError, ValueError):
            return None

    def _normalize_service_type(self, value: str) -> str:
        text = str(value or "").strip()
        if text in {"ТО", "РўРћ", "Ð¢Ðž"}:
            return "ТО"
        return text or _DEFAULT_SERVICE_TYPE

    def _text(self, value: Any) -> str:
        return str(value or "").strip()

    def _join_values(self, *values: Any) -> str:
        parts: list[str] = []
        for value in values:
            text = self._text(value)
            if text:
                parts.append(text)
        return " / ".join(parts)

    def _contains_any(self, haystack: str, needles: tuple[str, ...]) -> bool:
        return any(needle in haystack for needle in needles)

    def _append_unique_rows(self, rows: list[dict[str, str]], additions: list[dict[str, str]]) -> None:
        existing = {str(item.get("name") or "").strip().casefold() for item in rows}
        for item in additions:
            name = str(item.get("name") or "").strip()
            if not name or name.casefold() in existing:
                continue
            rows.append(item)
            existing.add(name.casefold())
