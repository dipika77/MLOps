# Agentic evaluation results (mock LLM)

- Task completion rate: **100%** (7/7 queries)
- Tool-call correctness: **100%** (7/7 trajectory steps)
- Average trajectory length: **1.9 iterations** (cap: 4)
- Multi-agent (Researcher+Verifier) total tokens across 7 queries: **3060**
- Single-agent (W15 baseline) total tokens across 7 queries: **1260**
- Coordination overhead of the second (Verifier) agent: **+143%** tokens vs. the single-agent baseline

## Per-query results

| id | kind | verification | iterations | tool-calls correct | tokens | latency (ms) | completed |
|----|------|--------------|-----------|---------------------|--------|---------------|-----------|
| q1 | single_source | sufficient | 2 | 1/1 | 540 | 1 | yes |
| q2 | calculation | not_verified | 2 | 1/1 | 360 | 0 | yes |
| q3 | cross_source | conflicting | 2 | 1/1 | 540 | 0 | yes |
| q4 | cross_source | sufficient | 2 | 1/1 | 540 | 0 | yes |
| q5 | ambiguous | clarification_needed | 1 | 1/1 | 180 | 0 | yes |
| q6 | single_source | sufficient | 2 | 1/1 | 540 | 0 | yes |
| q7 | failure_injection | insufficient | 2 | 1/1 | 360 | 0 | yes |

## Failure log

No failures recorded.
