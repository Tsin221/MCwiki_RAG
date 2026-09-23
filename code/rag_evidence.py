from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


EvidenceStrategy = Literal[
    "ranked_first",
    "source_cap",
    "adjacent_merge",
    "redundancy_suppression",
]
SUPPORTED_STRATEGIES: tuple[EvidenceStrategy, ...] = (
    "ranked_first",
    "source_cap",
    "adjacent_merge",
    "redundancy_suppression",
)
_NON_WORD = re.compile(r"[^0-9a-z\u3400-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    chunk_id: str
    title: str
    text: str
    source: str
    document_id: str | None = None
    chunk_index: int | None = None


@dataclass(frozen=True, slots=True)
class AnswerEvidence:
    id: int
    chunk_id: str
    title: str
    url: str
    text: str
    excerpt: str
    component_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    strategy: EvidenceStrategy
    max_context_chars: int = 6_000
    max_chunks_per_source: int = 2
    redundancy_threshold: float = 0.8
    min_merge_overlap_chars: int = 20

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"unsupported strategy: {self.strategy}")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero")
        if self.max_chunks_per_source <= 0:
            raise ValueError("max_chunks_per_source must be greater than zero")
        if not 0 <= self.redundancy_threshold <= 1:
            raise ValueError("redundancy_threshold must be between zero and one")
        if self.min_merge_overlap_chars <= 0:
            raise ValueError("min_merge_overlap_chars must be greater than zero")


@dataclass(frozen=True, slots=True)
class _AssembledCandidate:
    chunk_ids: tuple[str, ...]
    title: str
    text: str
    source: str


def _excerpt(text: str, *, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 1].rstrip()}…"


def _normalized_text(text: str) -> str:
    return _NON_WORD.sub("", text).casefold()


def _character_shingles(text: str, size: int = 3) -> set[str]:
    normalized = _normalized_text(text)
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def normalized_text_overlap(left: str, right: str) -> float:
    """Return a dependency-free character-shingle overlap score for Chinese text."""
    left_normalized = _normalized_text(left)
    right_normalized = _normalized_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return min(len(left_normalized), len(right_normalized)) / max(
            len(left_normalized), len(right_normalized)
        )
    left_shingles = _character_shingles(left)
    right_shingles = _character_shingles(right)
    union = left_shingles | right_shingles
    return len(left_shingles & right_shingles) / len(union) if union else 0.0


def _as_assembled(candidate: EvidenceCandidate) -> _AssembledCandidate:
    return _AssembledCandidate(
        chunk_ids=(candidate.chunk_id,),
        title=candidate.title,
        text=candidate.text.strip(),
        source=candidate.source,
    )


def _ranked_unique(candidates: Sequence[EvidenceCandidate]) -> list[EvidenceCandidate]:
    selected: list[EvidenceCandidate] = []
    seen_text: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.text.split())
        if not normalized or normalized in seen_text:
            continue
        seen_text.add(normalized)
        selected.append(candidate)
    return selected


def _source_cap(
    candidates: Sequence[EvidenceCandidate], limit: int
) -> list[EvidenceCandidate]:
    counts: Counter[str] = Counter()
    selected: list[EvidenceCandidate] = []
    for candidate in candidates:
        if counts[candidate.source] >= limit:
            continue
        counts[candidate.source] += 1
        selected.append(candidate)
    return selected


def _merge_text(left: str, right: str, min_overlap: int) -> str:
    max_overlap = min(len(left), len(right))
    for overlap in range(max_overlap, min_overlap - 1, -1):
        if left[-overlap:] == right[:overlap]:
            return left + right[overlap:]
    return f"{left}\n\n{right}"


def _merge_adjacent(
    candidates: Sequence[EvidenceCandidate], min_overlap: int
) -> list[_AssembledCandidate]:
    ranked = list(enumerate(candidates))
    by_location = {
        (item.document_id, item.chunk_index): (rank, item)
        for rank, item in ranked
        if item.document_id is not None and item.chunk_index is not None
    }
    consumed: set[int] = set()
    assembled: list[tuple[int, _AssembledCandidate]] = []

    for rank, item in ranked:
        if rank in consumed:
            continue
        group = [(rank, item)]
        if item.document_id is not None and item.chunk_index is not None:
            index = item.chunk_index - 1
            while (item.document_id, index) in by_location:
                group.append(by_location[(item.document_id, index)])
                index -= 1
            index = item.chunk_index + 1
            while (item.document_id, index) in by_location:
                group.append(by_location[(item.document_id, index)])
                index += 1
        group.sort(key=lambda entry: entry[1].chunk_index or 0)
        consumed.update(group_rank for group_rank, _ in group)

        text = group[0][1].text.strip()
        for _, neighbor in group[1:]:
            text = _merge_text(text, neighbor.text.strip(), min_overlap)
        assembled.append(
            (
                min(group_rank for group_rank, _ in group),
                _AssembledCandidate(
                    chunk_ids=tuple(neighbor.chunk_id for _, neighbor in group),
                    title=item.title,
                    text=text,
                    source=item.source,
                ),
            )
        )
    assembled.sort(key=lambda entry: entry[0])
    return [item for _, item in assembled]


def _suppress_redundancy(
    candidates: Sequence[EvidenceCandidate], threshold: float
) -> list[EvidenceCandidate]:
    selected: list[EvidenceCandidate] = []
    for candidate in candidates:
        if any(
            normalized_text_overlap(candidate.text, previous.text) >= threshold
            for previous in selected
        ):
            continue
        selected.append(candidate)
    return selected


def select_evidence(
    candidates: Sequence[EvidenceCandidate], config: SelectionConfig
) -> list[AnswerEvidence]:
    """Select and assemble evidence deterministically under a strict text budget."""
    ranked = _ranked_unique(candidates)
    if config.strategy == "source_cap":
        ranked = _source_cap(ranked, config.max_chunks_per_source)
    elif config.strategy == "redundancy_suppression":
        ranked = _suppress_redundancy(ranked, config.redundancy_threshold)

    assembled = (
        _merge_adjacent(ranked, config.min_merge_overlap_chars)
        if config.strategy == "adjacent_merge"
        else [_as_assembled(item) for item in ranked]
    )

    evidence: list[AnswerEvidence] = []
    used_chars = 0
    for item in assembled:
        remaining = config.max_context_chars - used_chars
        if remaining <= 0:
            break
        if evidence and len(item.text) > remaining:
            break
        text = item.text[:remaining]
        evidence.append(
            AnswerEvidence(
                id=len(evidence) + 1,
                chunk_id=item.chunk_ids[0],
                component_chunk_ids=item.chunk_ids,
                title=item.title,
                url=item.source,
                text=text,
                excerpt=_excerpt(text),
            )
        )
        used_chars += len(text)
    return evidence
