"""Journal search utilities using OpenAlex public API."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


OPENALEX_BASE_URL = "https://api.openalex.org/works"


@dataclass(frozen=True)
class SearchCandidate:
    title: str
    abstract: str
    venue: str
    year: int | None
    cited_by_count: int
    doi: str
    url: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "venue": self.venue,
            "year": self.year,
            "cited_by_count": self.cited_by_count,
            "doi": self.doi,
            "url": self.url,
        }


def search_journals(topic: str, *, page: int, per_page: int = 10) -> list[dict[str, Any]]:
    """Return normalized candidate papers for a topic."""
    query = quote(topic.strip())
    url = (
        f"{OPENALEX_BASE_URL}?search={query}"
        f"&filter=type:article,has_abstract:true,is_retracted:false"
        f"&sort=relevance_score:desc&per-page={per_page}&page={page}"
    )
    req = Request(url, headers={"User-Agent": "research-assistant/1.0"})
    with urlopen(req, timeout=12) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    results = payload.get("results") or []
    out: list[dict[str, Any]] = []
    for item in results:
        title = (item.get("display_name") or "").strip()
        if not title:
            continue
        abstract = _reconstruct_abstract(item.get("abstract_inverted_index") or {})
        if not abstract:
            continue
        host = item.get("primary_location", {}).get("source", {}) or {}
        venue = (host.get("display_name") or "").strip()
        out.append(
            SearchCandidate(
                title=title,
                abstract=abstract,
                venue=venue,
                year=item.get("publication_year"),
                cited_by_count=int(item.get("cited_by_count") or 0),
                doi=(item.get("doi") or "").strip(),
                url=(item.get("id") or "").strip(),
            ).as_dict()
        )
    return out


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    if not inverted_index:
        return ""
    max_pos = -1
    for positions in inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""

    words = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = token
    return " ".join(w for w in words if w).strip()
