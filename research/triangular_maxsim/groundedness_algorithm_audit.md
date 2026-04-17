# Groundedness Algorithm Audit

## Plain-English Summary

`voyager-index` groundedness is a post-generation support check.

For each response token, the scorer asks: "What is the best-matching support
token anywhere in the supplied context?" The headline score,
`reverse_context`, is the weighted average of those per-token best matches.

For `raw_context`, the service first packs adjacent sentences into
`256`-token-target support units. Those units can then be scored in grouped
batches and merged exactly by taking the per-response-token maximum across the
batches. That exact merge property is what makes chunked raw-context scoring
safe for the headline metric.

The new `consensus_hardened` score is intentionally secondary. It does not
replace `reverse_context`. Instead, it looks at how many distinct support units
offer strong evidence for each response token and then applies a small
conservative discount when support is narrow. It is a robustness hint, not a
formal statistical confidence estimate.

## Exact Formulas

Let response token embeddings be `R = {r_t}` and support units be
`U = {C_u}` where each support unit `C_u = {c_{u,j}}`.

Use cosine or dot-product similarity `s(., .)` over normalized token vectors.

Per-support-unit token support:

```text
m_{t,u} = max_j s(r_t, c_{u,j})
```

Headline per-token groundedness:

```text
g_t = max_u m_{t,u}
```

Headline scalar score with token weights `w_t`:

```text
reverse_context(R | U) = (sum_t w_t g_t) / (sum_t w_t)
```

Chunk merge over grouped support batches `B_k`:

```text
g_t^(k) = max_{u in B_k} m_{t,u}
g_t = max_k g_t^(k)
```

That equality is exact because `max` over the union of support tokens is the
same as `max` over batch-local maxima.

Secondary breadth diagnostics use the per-support-unit maxima `m_{t,u}`:

```text
count_above_tau(t) = sum_u 1[m_{t,u} >= tau]
soft_breadth(t) = sum_u sigmoid(alpha * (m_{t,u} - tau))
z_{t,u} = max(m_{t,u} - tau, 0)
effective_support_units(t) = (sum_u z_{t,u})^2 / sum_u z_{t,u}^2
```

Current calibration constants:

```text
tau = 0.85
alpha = 20.0
beta = 4.0
lambda = 0.03
```

Breadth normalization:

```text
b_t = 1 - exp(-max(effective_support_units(t) - 1, 0) / beta)
```

Secondary conservative score:

```text
consensus_hardened_t = g_t * (1 - lambda * (1 - b_t))
consensus_hardened(R | U) = (sum_t w_t consensus_hardened_t) / (sum_t w_t)
```

This construction keeps the secondary score close to `reverse_context` while
slightly discounting narrow single-unit support.

## What Is Algorithmically Exact

- Sentence-packed `raw_context` chunking preserves sentence boundaries and
  stable offsets; overflow sentences move to the next packed unit intact.
- `reverse_context` merge across grouped support batches is exact.
- Evidence attribution remains exact for the winning support token after merge.
- `consensus_hardened` merge is exact because it is computed from the full
  concatenated matrix of per-support-unit maxima `m_{t,u}`, not from lossy
  post-aggregated scalars.
- The hardest-case verification run showed zero diff for:
  - per-token `reverse_context`
  - scalar `reverse_context`
  - per-token `consensus_hardened`
  - scalar `consensus_hardened`
  - per-token `effective_support_units`
  - top-evidence and token-evidence mappings

## What Is Calibrated Or Heuristic

- Token weights are rule-based, not learned.
- `consensus_hardened` is not an IID significance test. Repeated paraphrases or
  duplicated evidence across support units can still inflate breadth.
- Dense similarity remains weak on negation, role swaps, exact dates, numbers,
  and entity substitutions.
- Query-conditioned channels remain diagnostic-only. On the current hard suite,
  triangular scoring did not beat the naive reverse-context baseline.

## Verification Results

### Automated checks

- `pytest tests/test_groundedness_service.py`
- Result: `13 passed`

Covered areas:

- new `256` packed-window default
- sentence carry behavior at boundaries
- mocked `vllm-factory` ModernColBERT `/pooling` request/response contract
- grouped-batch exact merge parity
- consensus exact merge parity
- tokenizer helper regression coverage
- encoder-limit warning behavior

### Hardest-case exactness proof

From `groundedness_service_validation__long_ambiguous_packed_256_consensus.md`:

- verification batch size: `16` support units
- selected hardest previous case: `LG3`
- reverse-context per-token max abs diff: `0.00000000`
- reverse-context scalar abs diff: `0.00000000`
- consensus per-token max abs diff: `0.00000000`
- consensus scalar abs diff: `0.00000000`
- effective-support-units max abs diff: `0.00000000`
- evidence mapping exact match: `True`
- top-evidence exact match: `True`

### Long-context quality and latency

Audit configuration:

- model: `lightonai/GTE-ModernColBERT-v1`
- packed raw-context chunk budget: `256`
- production score batch size: `64`
- latency repeats per case: `3`
- mean context length: about `7.8k` tokens

Measured results:

- anchor AUROC (`reverse_context`): `1.0000`
- anchor AUROC (`consensus_hardened`): `1.0000`
- anchor AUROC (`reverse_query_context`): `1.0000`
- anchor AUROC (`triangular`): `0.0000`
- score-only latency p50 / p95: `90.94 ms` / `97.18 ms`
- user-facing go/no-go at the current gate (`p95 <= 25 ms`): `False`

Interpretation:

- The exact merge math is sound.
- The naive `reverse_context` headline remains the right primary score.
- `consensus_hardened` is useful as secondary context, but it does not create a
  large extra separation on the current ambiguous/entity-swap long-context
  cases. Treat it as an explanatory robustness cue, not a stronger detector.
- Triangular scoring remains underpowered on this suite.
- Long-context scoring quality is usable for evidence views, but latency still
  misses the current production gate on this model.

## Transport Audit Note

The `vllm-factory` ModernColBERT provider is now implemented in the service and
covered by mocked unit tests for:

- `GET /health`
- `POST /pooling`
- `task="plugin"`
- `data.text`
- `data.is_query`
- multi-vector payload decoding

No live `vllm-factory` endpoint was configured in this audit environment, so the
numeric quality/latency results above were produced with the local
`lightonai/GTE-ModernColBERT-v1` provider rather than a live remote pooling
deployment.
