from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx


_RESULT_BLOCK_PATTERN = re.compile(r'<div class="result(?:__body)?[^"]*".*?</div>\s*</div>', re.S)
_RESULT_LINK_PATTERN = re.compile(
    r'<a rel="nofollow" class="result__a" href="(?P<href>[^"]+)">(?P<title>.*?)</a>',
    re.S,
)
_RESULT_SNIPPET_PATTERN = re.compile(
    r'<a class="result__snippet" href="[^"]+">(?P<snippet>.*?)</a>|<div class="result__snippet">(?P<snippet_alt>.*?)</div>',
    re.S,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_PATTERN = re.compile(r"<(?:script|style)\b.*?>.*?</(?:script|style)>", re.S | re.I)
_MULTISPACE_PATTERN = re.compile(r"\s+")


class InternetToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
        }


class DuckDuckGoSearchClient:
    def __init__(self, *, timeout_seconds: float = 12.0) -> None:
        self._timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        allowed_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        query_text = str(query or "").strip()
        if not query_text:
            raise InternetToolError("query is required")
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query_text)}"
        try:
            with httpx.Client(timeout=self._timeout_seconds, headers={"User-Agent": "Mozilla/5.0 AutoStopCRM/1.0"}) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InternetToolError(f"Web search failed: {exc}") from exc
        return self._parse_results(response.text, limit=limit, allowed_domains=allowed_domains)

    def fetch_page_excerpt(self, url: str, *, max_chars: int = 2500) -> dict[str, Any]:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            raise InternetToolError("url is required")
        try:
            with httpx.Client(timeout=self._timeout_seconds, headers={"User-Agent": "Mozilla/5.0 AutoStopCRM/1.0"}) as client:
                response = client.get(normalized_url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InternetToolError(f"Page fetch failed: {exc}") from exc
        text = self._clean_html_text(response.text)
        return {
            "url": str(response.url),
            "domain": urlparse(str(response.url)).netloc.lower(),
            "excerpt": text[:max_chars],
        }

    def _parse_results(
        self,
        html_text: str,
        *,
        limit: int,
        allowed_domains: list[str] | None,
    ) -> list[SearchResult]:
        allowed = [item.casefold() for item in (allowed_domains or []) if item]
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for block in _RESULT_BLOCK_PATTERN.findall(html_text):
            link_match = _RESULT_LINK_PATTERN.search(block)
            if not link_match:
                continue
            snippet_match = _RESULT_SNIPPET_PATTERN.search(block)
            resolved_url = self._resolve_duckduckgo_url(link_match.group("href"))
            if not resolved_url or resolved_url in seen_urls:
                continue
            domain = urlparse(resolved_url).netloc.lower()
            if allowed and not any(domain == item or domain.endswith(f".{item}") for item in allowed):
                continue
            seen_urls.add(resolved_url)
            results.append(
                SearchResult(
                    title=self._clean_html_text(link_match.group("title")),
                    url=resolved_url,
                    snippet=self._clean_html_text(
                        (snippet_match.group("snippet") if snippet_match else "")
                        or (snippet_match.group("snippet_alt") if snippet_match else "")
                    ),
                    domain=domain,
                )
            )
            if len(results) >= limit:
                break
        return results

    def _resolve_duckduckgo_url(self, href: str) -> str:
        raw_href = html.unescape(str(href or ""))
        if raw_href.startswith("//duckduckgo.com/l/?"):
            parsed = urlparse("https:" + raw_href)
            encoded = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(encoded) if encoded else ""
        if raw_href.startswith("/l/?"):
            parsed = urlparse("https://duckduckgo.com" + raw_href)
            encoded = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(encoded) if encoded else ""
        return raw_href

    def _clean_html_text(self, value: str) -> str:
        text = _SCRIPT_STYLE_PATTERN.sub(" ", str(value or ""))
        text = _TAG_PATTERN.sub(" ", text)
        text = html.unescape(text)
        text = _MULTISPACE_PATTERN.sub(" ", text)
        return text.strip()
