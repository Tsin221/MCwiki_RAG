from __future__ import annotations

import re


_STRATEGIES = ("literal_phrase", "entity_phrase", "token_or", "title_boost")
_STRUCTURED_PATTERNS = (
    re.compile(
        r"(?:Java|Bedrock|基岩|教育)\s*版?\s*\d+(?:\.\d+)+(?:-(?:pre|rc)\d+)?",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{2}w\d{2}[a-z]\b", re.IGNORECASE),
    re.compile(r"(?<!\w)/[a-z][a-z0-9_.-]*", re.IGNORECASE),
    re.compile(r"\b[a-z0-9_.-]+:[a-z0-9_./-]+\b", re.IGNORECASE),
    re.compile(r"\b(?:NBT|SNBT|[A-Z][A-Za-z0-9_]{2,})\b"),
)
_DECORATIONS = re.compile(
    r"请问|麻烦|能不能|可以|是否|顺便|告诉我|我想知道|请介绍|介绍一下|"
    r"具体|一下|是什么|有哪些|怎么用|如何使用|怎么样|分别|主要"
)
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9_.:/-]{3,}|[\u3400-\u9fff]{3,}")


def supported_strategies() -> tuple[str, ...]:
    return _STRATEGIES


def escape_fts_phrase(value: str) -> str:
    """Quote arbitrary text as one literal FTS5 phrase."""
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _structured_terms(query: str) -> list[str]:
    return _ordered_unique(
        [match.group(0).replace(" ", "") for pattern in _STRUCTURED_PATTERNS for match in pattern.finditer(query)]
    )


def _entity_terms(query: str) -> list[str]:
    terms = _structured_terms(query)
    cleaned = _DECORATIONS.sub("", query)
    cleaned = re.sub(r"[?？!！,，。:：;；()（）\[\]{}<>《》\"']+", " ", cleaned)
    fragments = [fragment.strip(" 的呢吗呀啊了") for fragment in cleaned.split()]
    terms.extend(fragment for fragment in fragments if len(fragment) >= 3)
    return _ordered_unique(terms)[:12]


def _token_terms(query: str) -> list[str]:
    terms = _structured_terms(query)
    cleaned = _DECORATIONS.sub("", query)
    for word in _WORD_PATTERN.findall(cleaned):
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            terms.extend(word[index : index + 3] for index in range(len(word) - 2))
        else:
            terms.append(word)
    return _ordered_unique(terms)[:32]


def build_match_query(query: str, strategy: str) -> str:
    """Build a fully quoted FTS5 MATCH expression for a named strategy."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown BM25 query strategy: {strategy}")
    normalized = query.strip()
    if not normalized:
        return ""
    if strategy == "literal_phrase":
        return escape_fts_phrase(normalized)
    terms = _entity_terms(normalized) if strategy == "entity_phrase" else _token_terms(normalized)
    if not terms:
        terms = [normalized]
    return " OR ".join(escape_fts_phrase(term) for term in terms)
