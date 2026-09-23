# Evidence assembly offline comparison (2026-09-23)

## Decision

Switch the production default from `ranked_first` to `adjacent_merge`.
The selected strategy keeps retrieval candidates unchanged, preserves every original chunk ID,
and improves annotated context precision without reducing relevant-evidence recall.

Source caps are not enabled. Every tested cap from 1 through 4 reduced v2 relevant-evidence recall,
so this experiment does not justify a default per-source quota.

## Reproduction inputs

- Dataset: `data/evaluation/v2/questions.json` (48 cases).
- Candidate snapshot: `evidence-candidates-2026-09-23.json`.
- Candidate generation: unchanged BM25 20 + semantic 20, RRF, final candidate limit 8.
- Context budget: 12,000 characters.
- Full per-case selected evidence: `evidence-comparison-2026-09-23.json`.
- No answer-model calls were used for the offline comparison.

## Results

| Strategy | Relevant evidence recall | Context precision | Expected source recall | Evidence ID validity | Evidence items |
|---|---:|---:|---:|---:|---:|
| ranked-first | 0.4915 | 0.0863 | 0.9524 | 1.0000 | 384 |
| adjacent merge | 0.4915 | 0.1089 | 0.9524 | 1.0000 | 290 |
| source cap 1 | 0.1525 | 0.0523 | 0.9524 | 1.0000 | 208 |
| source cap 2 | 0.2712 | 0.0667 | 0.9524 | 1.0000 | 283 |
| source cap 3 | 0.3390 | 0.0719 | 0.9524 | 1.0000 | 324 |
| source cap 4 | 0.4237 | 0.0820 | 0.9524 | 1.0000 | 353 |
| redundancy suppression 0.6 | 0.4915 | 0.0866 | 0.9524 | 1.0000 | 382 |
| redundancy suppression 0.7 | 0.4915 | 0.0863 | 0.9524 | 1.0000 | 383 |
| redundancy suppression 0.8 | 0.4915 | 0.0863 | 0.9524 | 1.0000 | 383 |
| redundancy suppression 0.9 | 0.4915 | 0.0863 | 0.9524 | 1.0000 | 384 |

Context precision is the proportion of displayed evidence items containing at least one annotated
relevant chunk. Adjacent merge improves it by 26.2% relative while reducing displayed evidence
items by 24.5%. Recall is computed from the original component chunk IDs, not synthetic merged IDs.

The absolute v2 recall reflects the unchanged top-8 candidate generator and strict chunk-level
annotations; evidence assembly cannot recover relevant chunks absent from those candidates.

Reproduce the comparison from the saved snapshot:

```powershell
cd code
uv run python rag_evidence_evaluation.py --snapshot ..\data\evaluation\v2\evidence-candidates-2026-09-23.json --output ..\data\evaluation\v2\evidence-comparison-2026-09-23.json
```

## Strategy details

- `ranked_first`: preserves the prior normalized-exact-text deduplication and budget behavior.
- `source_cap`: tested caps 1-4; all lost complementary chunks, including multi-evidence cases.
- `adjacent_merge`: merges only consecutive chunks sharing the same ingestion `document_id`, removes
  the exact splitter overlap, and records all component chunk IDs.
- `redundancy_suppression`: uses normalized Unicode alphanumeric/CJK character trigrams and Jaccard
  overlap. This is dependency-free and works directly on Chinese characters, but removed too little
  duplication to justify becoming the default.

## Fixed 18-case regression

The saved September 22 candidates were replayed first: expected source recall stayed 15/15 for all
tested strategies. After the offline gate passed, a real 18-case run was saved as
`data/evaluation/answer_quality/raw-2026-09-23-evidence-assembly.json`.

- Expected source in evidence: 15/15.
- Strict expected-source citation: 14/15 (same known Java 1.21 source-equivalence case).
- Citation ID validity: 136/136 (100%).
- Reliable abstention: 3/3.
- Manual correctness/completeness/faithfulness: 2.00/2 each.

## Risks and rollback

Adjacent merging requires retrieval metadata (`document_id`, `chunk_index`). Candidates lacking that
metadata remain separate, so old callers retain ranked-first behavior. Roll back by setting
`MCWIKI_EVIDENCE_STRATEGY=ranked_first`; no retrieval index or data migration is required.
