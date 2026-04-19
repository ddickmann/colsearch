<!--
Continuous progress log for the "Riemannian Low-Bit ROQ" plan.

Read top-to-bottom: pinned headers first, then newest entries. Total catch-up
time is minutes regardless of how long the project has been running.

Update rules (mirrored from the plan, do not relax):

- One entry per experiment, never per seed or per dataset.
- Tables max 5 rows. Full metric set lives in the cited JSON.
- Why is 1-3 sentences. Longer analysis goes in a separate note linked here.
- Negative results follow the same template; failures must be visible.
- Auto-stub from harness emits config + table + artifact links;
  engineer fills Verdict + Why + Gate-impact within 1 working day.
- Pinned headers (Current State, Promoted, Killed) are updated by hand at
  every gate. One commit per update.
-->

# PROGRESS — Riemannian Low-Bit ROQ

## Current State

- **Phase:** end-to-end production lane shipped behind opt-in `Compression.RROQ158`. A5 + A6 both integrated; B/X cross-cuts deferred until the dataset gap is closed.
- **Most recent gate:** `beir-prod-rroq158-2026-04-19` — full BEIR end-to-end through real LEMUR routing on nfcorpus + scidocs (3 variants × 5 rroq158 seeds). Verdict: **KEEP-EXPERIMENTAL** (within gates on small/easy nfcorpus, fails both quality and latency gates on large/hard scidocs; the kernel itself is sound at 1.26 ms/query on 2k×512×32 token-pairs, the regression is in the codec on harder datasets).
- **A-best candidates:** ternary (A2.5)
- **B-best candidate:** rroq158-K1024 (B3)
- **Production candidate:** none — rroq158 ships as `Compression.RROQ158` opt-in for users who can tolerate the quality gap on hard datasets in exchange for ~5.5× disk savings. Default codec stays ROQ4.
- **Hardware budget (operator constraint):** 24 GB CPU RAM, 24 GB GPU VRAM each — A5000 (24 GB) is the fixed GPU box; sweeps must respect this for A1 sample size, BEIR shard build params, k_candidates, and the C1.5 bake-off matrix.
- **Open questions:**
  - Does K=4096 / K=8192 close the scidocs quality gap without blowing the disk budget? (current K=1024 has 256 KB centroid table; K=8192 = 2 MB, still trivial.)
  - Why does the rroq158 wrapper's measured p95 latency in the BEIR loop (78 ms) diverge so much from the steady-state probe (2.81 ms / query) — Triton autotune cache miss across the LEMUR/kernel boundary, or PyTorch allocator stalls under the 2k-candidate gather pattern? Fixing this is the prerequisite for re-evaluating production fitness.
- **Restored after CPU OOM:** `runners.py` rebuilt; the 488 MB `tests/fixtures/token_sample_1m.npy` (real ColBERT tokens, 200k each from nfcorpus / scifact / arguana / scidocs / fiqa) survived; no cell reports were lost (none had been generated).

## Promoted

- `phase-0-harness` — multi-seed harness, paired-bootstrap, candidate-recall logging, cold/warm p95, distortion bench, PROGRESS.md auto-stubs all green on 34 unit tests; runners (`run_a1.py`, `run_a6.py`, `run_c1_5.py`) smoke-tested.
- `a1-lite-pilot` — 20-cell A1 distortion pilot on 8 192 real ColBERT tokens / 128 queries: angular error monotone in bit-width (1-bit p50=29.3°, 1.58-bit=23.7°, 2-bit=20.4°); FWHT helps 2-bit (–4° p50) and is recall-neutral for 1-bit. Full 5-seed × 5-dataset BEIR follow-up still pending the operator's next memory window.
- `a2-roq2-asym-kernel` — 2-bit asymmetric Triton kernel on A5000: 0.29 ms p50, 55k QPS, parity 0.0 vs NumPy reference. Replaces the unusable symmetric 2-bit kernel in `voyager_index/_internal/kernels/triton_roq.py` for production.
- `a2.5-ternary-kernel` — 1.58-bit ternary asymmetric Triton kernel on A5000: 0.26 ms p50, 61k QPS, parity 2.3e-5 vs dequant baseline. Faster *and* smaller than 2-bit; primary candidate for shipping.
- `a3-ternary-anisotropic-flag` — kept the `TernaryConfig.fit_method='anisotropic'` switch wired but as opt-in: marginal +1–2% IP-RMS over `tau_frac=0.5` doesn't justify making it the default at index time.
- `b3-rroq158-K1024` — Riemannian ternary with K=1024 spherical centroids + tangent-residual ternary codes. Recovers 31% of the ternary→roq4 `rank_corr@100` gap (0.253 → 0.365) and 31% of the `NN50*` gap (0.219 → 0.348) at +1.25 B/tok overhead. Composes cleanly with the ternary kernel — same residual encoding, just with a per-token centroid_id added.
- `x1-distill-multi-view` — 3-layer MLP (~1.2 K params) trained with pairwise hinge on 6 features (rroq158 score, qc, qr, ‖r̂‖, |qd-qc|, raw-ternary score). On top of rroq158-K1024 it lifts `NN50*` from 0.348 → 0.427 (recovers 50% of the gap) and `NN5*` from 0.381 → 0.391. The *raw-ternary* score is the critical extra feature — it provides decorrelated noise from the rroq158 view. Inference cost is dominated by the extra ternary score itself, not the MLP.
- `b3-kernel-rroq158-fused-triton` — production-grade fused two-stage Triton kernel for the rroq158 score formula (host-side `qc_table = q_amb @ centroids.T` + `q_rot = FWHT(q_amb)`; device-side per-(q,d) CTA with BLOCK_D autotune over doc tokens, ternary popcount residual, and `cos_norm·qc + sin_norm·resi` combine). Microbench on A5000 at 32 q-tok × 32 d-tok × 512 docs: **p50 = 0.15 ms / 3.4 M docs/s**, parity ≤ 1e-4 vs `reference_score_rroq158`. Plumbed end-to-end behind `Compression.RROQ158` (build/store/scorer/bench driver). At BEIR-scale shapes (32 q-tok × 512 d-tok × 2 k docs) the warm kernel-only call is **1.26 ms** (a clean 8× sub-linear from the microbench, as expected from BLOCK_D parallelism). Total wrapper hot path: **2.81 ms / query**.
- `a5-rroq158-prod-plumbing` — `Compression.RROQ158` shipped as an opt-in production codec. Encoder lifted to `voyager_index/_internal/inference/quantization/rroq158.py` with chunked spherical k-means (`fit_sample_cap=100k`, `encode_chunk=32k`); fitted centroids + FWHT seed persisted to `<index>/rroq158_meta.npz`. Per-token storage on disk: 46 B (vs 256 B fp16, 64 B ROQ4) → ~5.5× / ~1.4× compression respectively. Build branch in `_manager/lifecycle.py`; `pack_shard_rroq158` + `load_shard_rroq158` in `_store/`; `score_rroq158_topk` dispatch in `scorer.py`. Reuses fp16 LEMUR artifacts in the bench driver (matches plan: routing artifacts are codec-agnostic). Defensive fallback to FP16 when the encoder fails or the payload is missing.

## Killed

- `nn1_preservation` as a primary A1 metric — self-matches contaminate it (queries sampled from the corpus all rank themselves first), giving 1-bit NN1≈1.0 at 30° angular error. Use angular_p50 / IP-RMS / NN5 / NN100 instead; if NN1 is needed in a follow-up, drop the diagonal (`true_ips[i, q_idx[i]] = -inf`) before argpartition.
- `a3-roq2-anisotropic-as-default` — Newton fitter in `fit_anisotropic_min_max` produces *worse* IP-RMS than uniform at every η > 1 tested (44% worse at η=2, 200% worse at η=8). Off the C1.5 matrix until the gradient bug (`eta * parallel` vs `(eta - 1) * parallel`, `codes_centered` vs `codes`) is fixed and re-validated.
- `a4-norm-salience-as-production-default` — token_recall@K tracks the random-prune baseline (1 - prune_rate) within 2 points across all sweep rates, on the post-encoder fixture. Norm carries no per-token signal once the embeddings have been L2-normed. Re-test only after A5 makes raw pre-projection embeddings available.
- `b0-tangent-pair-score` — `s = <q,d> - λ·θ²(q,d)` is a *monotonic* transform of `<q,d>` (since θ = arccos), so it cannot change ranks. Sweeping λ ∈ {0.05, 0.1, 0.25} returned identical `rank_corr@100`, `NN5*`, `NN50*` to the unmodified ternary baseline. KEEP only at the *router* layer (compares scores across different centroids); KILL as a per-pair rerank score. This was a planning bug from B0 — it conflated routing-score adjustment with pair-score adjustment.
- `x1-distill-rroq158-features-only` — same MLP architecture, trained on the 5 rroq158-derived features without the raw-ternary score. Regressed `rank_corr@100` from 0.365 → 0.264 and gave only marginal `NN50*` gains (0.348 → 0.395). The five features are nearly-linearly-related — the MLP can't extract a second view from a single quantization. The fix was to add an *independently-rotated* ternary score (multi-view).
- `x1-per-centroid-bias` — single scalar per centroid, score' = score + bias[c_d]. Pairwise hinge stayed at 0.041 across 8 epochs (no convergence) and eval `rank_corr@100` was -0.125. The bias parameter overpowers the score range; needs a clamp / much smaller learning rate before this is even worth re-running.

## Open `[VERDICT-PENDING]` entries

_(empty — auto-populated when the harness emits stub entries that have not
yet been completed by an engineer)_

---

## [2026-04-19] beir-prod-rroq158 — End-to-end through real LEMUR routing on nfcorpus + scidocs

**Config:** `benchmarks/run_rroq158_prod_sweep.py` driving `benchmarks/beir_benchmark.py` with `--compression {fp16,roq4,rroq158}` and `--rroq158-seed 42..46`. GPU-corpus mode on A5000, n_shards=32, k_candidates=2000, max_docs_exact=2000, top_k=100, `OPTIMAL_GPU` defaults. rroq158: K=1024 spherical centroids, group_size=32, FWHT rotator. LEMUR routing artifacts built once on fp16 corpus (seed=42, 10 epochs) and reused across all three codecs (per plan §4 LEMUR-reuse strategy).
**Datasets / seeds:** nfcorpus (323 queries, 3 633 docs, 0.86 M tokens) and scidocs (1 000 queries, 25 657 docs, 4.84 M tokens). fp16 / roq4 are deterministic at seed=42 (LEMUR train + GPU-corpus search have no further randomness, so a single run captures their full distribution); rroq158 was run for 5 seeds because the FWHT rotator and spherical k-means initialisation are seed-dependent.
**Baselines:** fp16 (whole-corpus MaxSim) and roq4 (current production codec).

| dataset  | variant   | R@10              | NDCG@10           | p95 ms        | QPS  | bytes/tok | Δ R@10 vs fp16 | Δ p95 vs fp16 |
|----------|-----------|-------------------|-------------------|---------------|-----:|----------:|---------------:|--------------:|
| nfcorpus | fp16      | 0.3404            | 0.3833            | 3.88          |  188 |       256 |          +0.00 |          +0.0% |
| nfcorpus | roq4      | 0.3379            | 0.3800            | 3.84          |  286 |        64 |          −0.26 |          −1.1% |
| nfcorpus | rroq158   | 0.3359 ± 0.0019   | 0.3729 ± 0.0029   | 4.20 ± 0.04   |  267 |        46 |          −0.45 |          +8.2% |
| scidocs  | fp16      | 0.2070            | 0.1977            | 4.37          |  176 |       256 |          +0.00 |          +0.0% |
| scidocs  | roq4      | 0.2076            | 0.1973            | 4.38          |  243 |        64 |          +0.05 |          +0.1% |
| scidocs  | rroq158   | 0.1925 ± 0.0021   | 0.1850 ± 0.0015   | 77.94 ± 0.75  |   75 |        46 |          **−1.45** |     **+1682%** |

**Verdict:** **KEEP-EXPERIMENTAL.** Ship the production-tree plumbing (`Compression.RROQ158`, encoder, kernel, store/scorer integration) as an **opt-in** codec for users who can tolerate the quality gap on hard datasets in exchange for ~5.5× disk savings. Do **not** promote to default. Default remains ROQ4. The `SearchConfig.distill_rerank` MV-distill toggle is wired but stays default-off — it still regresses Recall@10 on real BEIR (carries the recovery-bench finding through to production).

**Why:**
1. **The codec quality is dataset-dependent in a way the offline distortion bench did not capture.** On nfcorpus, rroq158 R@10 is within the 0.5-pt gate (−0.45 pt). On scidocs — larger (7× more docs), harder (lowest fp16 R@10 in BEIR) — the gap blows out to −1.45 pt. The recovery-2026-04-19 bench was 8 K tokens drawn uniformly across BEIR datasets; it predicted "K=1024 should hold" but did not stratify by per-dataset hardness. Two plausible mechanisms: (a) K=1024 is too few centroids for scidocs's tighter intra-cluster spread (scidocs is a citation-recommendation task, dense in topic-space), and/or (b) the LEMUR shortlist already filters out the easy positives at the routing stage, so the remaining 2 k candidates need *more* discriminative scoring than rroq158 provides.
2. **The kernel is fast where it should be.** Microbench on A5000 (32×32×512): 0.15 ms p50, 3.4 M docs/sec. At BEIR shapes (32 × 512 × 2 k = 32 M token-pairs) the warm kernel-only call is 1.26 ms — a clean 8× sub-linear speedup vs the microbench despite 61× more work, because BLOCK_D parallelism amortises the per-CTA overhead. Total wrapper hot path is 2.81 ms / query, dominated by the CPU-side `encode_query_for_rroq158` (0.89 ms — fresh torch FWHT rotator instantiated per query, then a small CPU matmul for `qc_table`).
3. **The 78 ms p95 in the BEIR loop is NOT the kernel.** The probe (`/tmp/probe_rroq158_latency.py`) measured 2.81 ms steady-state per query end-to-end including the wrapper, but the production loop measured 14 ms average and 78 ms p95. The discrepancy is consistent with Triton autotune cache not being shared across the LEMUR routing call boundary (autotune key includes `n_d_tokens` which can vary slightly across queries when the LEMUR shortlist returns < `max_docs_exact`), or with PyTorch allocator stalls under the 7-tensor `index_select` gather pattern at 2 k candidates. This is a **fixable engineering issue, not a kernel design issue** — but it does not affect the verdict because the **quality regression is the binding constraint**.
4. **Disk savings are real and meaningful.** rroq158 ships at 46 B / token (sign + nonzero + group scales + centroid_id + cos_norm + sin_norm) vs 256 B for fp16 and 64 B for ROQ4. On scidocs (4.84 M tokens) that is 220 MB vs 1.20 GB fp16 vs 310 MB ROQ4 — a **5.5× / 1.4× compression** that lets large indexes stay GPU-resident on the same hardware. For users where disk is the binding constraint and they can absorb a ~1.5 pt R@10 hit on hard datasets, that's a sensible trade.
5. **MV-distill is plumbed but does not help on real BEIR.** Verified Phase 2: even after fixing the train/eval distribution bug (training pairs now drawn from the rroq158 top-K shortlist rather than random negatives), MV-distill regresses R@10 on nfcorpus brute-force (0.346 → 0.116). It stays in the codebase as `SearchConfig.distill_rerank` (default `False`) so future iterations can re-enable it without re-plumbing, but it is not a recommended default. The offline `NN50*` recovery (50% of the gap) does not survive contact with BEIR's actual relevance distribution.

**Action:**
- Land the rroq158 production lane as the next commit (already at `39fabf7`). Ship `Compression.RROQ158` as opt-in. Document the quality caveat for hard datasets prominently.
- Open follow-up tickets for the two open questions:
  - K-sweep on scidocs: try K ∈ {2048, 4096, 8192} and re-measure R@10. Centroid table cost scales linearly (K=8192 = 2 MB GPU-resident, still trivial).
  - Wrapper latency: cache the CPU centroid copy, pre-instantiate the FWHT rotator per index, and benchmark whether the autotune cache miss across the LEMUR boundary is the real culprit. Goal: bring p95 from 78 ms back to the steady-state 2.81 ms.
- Move the B3 / X1 / a5 / a6 plan todos to **completed** with the verdict above. Re-prioritise B1 / B2 (spherical / tangent routing) and B5 (per-cluster PCA) to the next round — they may close the K=1024 gap on scidocs without forcing K to grow.
- Honest caveats:
  - Quality variance across rroq158 seeds is small (R@10 std ≈ 0.002 on both datasets), so the −1.45 pt scidocs gap is real, not noise.
  - The LEMUR routing artefacts were trained once on fp16 and reused across codecs. This is the right apples-to-apples comparison for "what does the codec cost you?", but a router co-trained with rroq158 candidates *might* close some of the scidocs gap (the router currently favours candidates that are easy to score in fp16, which need not be easy to score in rroq158).
  - We measured only two BEIR datasets. The plan called for "wins on at least 3 of 5"; we have one within-gate (nfcorpus) and one decisively out-of-gate (scidocs). The remaining three (arguana, fiqa, scifact) are not measured here, but the scidocs result is decisive enough to keep us at EXPERIMENTAL.

**Artifacts:** `reports/beir_rroq158_nfcorpus.json`, `reports/beir_rroq158_scidocs.json`, `reports/kernel_rroq158.json`, `tests/test_rroq158_kernel.py`, commit `39fabf7`.
**Gate impact:**
- B-track: rroq158-K1024 stops at "opt-in production"; B5 (per-cluster PCA) and B4 (mixed-precision) re-prioritised as gap-closers for hard datasets.
- X-track: MV-distill stays opt-in; X2 (cross-encoder rerank with a small Transformer head) added to the next-round backlog as a higher-capacity recovery alternative.
- A-track: A5/A6 closed as completed (production plumbing + BEIR end-to-end). Future rroq158 iterations re-enter through B-track tickets.
- Combined-track: C2 (k_candidates stress test) and C3 (CPU/streamed serving replay) deferred until the dataset gap is closed — there is no point in proving rroq158 holds up under a smaller shortlist if it cannot match fp16 on the larger one.

---

## [2026-04-19] recovery — Riemannian + distillation gap-recovery on top of ternary

**Config:** `research/low_bit_roq/bench_recovery.py` — 8 192 real ColBERT tokens × 192 train queries × 64 eval queries (held out) from the same fixture, dim=128, group_size=32, K ∈ {256, 1 024}, λ ∈ {0.05, 0.1, 0.25}, seed=0. Memory cap: 16 GB CPU. Wall: 90 s.
**Datasets / seeds:** offline distortion only, single seed. Eval queries are disjoint from training queries; self-pairs are masked.
**Baselines:** ternary (rank_corr@100=0.253, NN50*=0.219), roq4 (0.620 / 0.634).

| variant                                | bits |    K | rank@100 | NN5* | NN50* | extra B/tok | NN50* gap recovery |
| -------------------------------------- | ---: | ---: | -------: | ---: | ----: | ----------: | -----------------: |
| ternary (base)                         | 1.58 |    0 |    0.253 | 0.241 | 0.219 |        0.00 |                  0% |
| ternary + B0-tangent (any λ)           | 1.58 |    0 |    0.253 | 0.241 | 0.219 |        0.00 |                  0% |
| rroq158-K1024                          | 1.58 | 1024 |    0.365 | 0.381 | 0.348 |        1.25 |                 31% |
| rroq158-K1024 + X1-MLP (5 features)    | 1.58 | 1024 |    0.264 | 0.359 | 0.395 |        1.25 |                 42% |
| **rroq158-K1024 + ternary + X1-MV**    | 1.58 | 1024 |    0.297 | 0.391 | **0.427** |    1.25 |             **50%** |

**Verdict:** PROMOTE `rroq158-K1024 + multi-view X1 distill` as the recovery candidate that composes with the ternary kernel (A2.5). KILL `B0-pair-tangent` as a per-pair score and re-scope B0 strictly as a routing-layer experiment.

**Why:**
1. **B0 at the pair level is mathematically a no-op.** `s = cos - λ·arccos(cos)²` is monotonic in `cos`, so for any pair (q, d) it preserves rank. The L2-normed cosine, the arccos, and the squared-arccos are all monotonic; subtracting a monotonic function of cos from cos is still monotonic. Useful only when comparing *across* centroids in the routing stage (different `c` → different θ shift), not as a per-pair rerank correction.
2. **rroq158-K1024 is the cheapest meaningful recovery.** It buys 31% of the NN50* gap for +1.25 B/tok (a single int10 centroid_id per token) without any training-time component. The reference scorer matches the analytic `<q, exp_c(r̂)>` formula exactly; the existing 2-bit-asym kernel can score the residual codes with one extra term per token (centroid lookup).
3. **Multi-view distill recovers HALF the NN50* gap.** The single-view MLP variants (rroq158 features only, residual MLP, per-centroid embedding) all *regressed* `rank_corr@100` because their input features are nearly-linearly-related — `qc`, `qr`, and the rroq158 score are derived from the same K-centroid encoding. Adding the *raw ternary score* (computed in a different rotation, with different codes) gives the MLP a decorrelated second view; that's where the gain comes from. The trade-off is that the MV head needs *two* per-pair scores at rerank time (one rroq158 + one ternary), which doubles the kernel call. With the rerank shortlist size at ~1 k–4 k, that's still <0.1 ms on the A5000.
4. **`rank_corr@100` and `NN50*` measure different things in this regime.** All MLP variants improved NN50* (more true neighbours found) but worsened rank_corr (rougher within-shortlist order). For the production flow `ternary top-K → fp16 rerank top-K → return top-10`, NN50* is the constraint that matters: it bounds the recall ceiling of the rerank stage. Rank_corr@100 only matters if you skip the fp16 rerank, which we don't.
5. **2-bit at K=1024 (rroq2) is a shade better than ternary K=1024 on raw rank_corr** (0.373 vs 0.365) but at 2× the residual storage (16 → 32 bytes/token). Not worth the bit budget given the kernel-latency parity from A2.5.

**Action:**
- Add `rroq158-K1024` to the C1.5 bake-off matrix as a primary candidate. Drop the rroq2-K1024 arm — same disk as rroq4 with worse rank_corr.
- Build a Triton kernel that scores rroq158 residuals using the existing `triton_roq_ternary` kernel (residual codes are ternary-shaped; only the centroid lookup is new). Persist `centroid_id` as an int10 next to the ternary blob; +1.25 B/tok.
- The X1 MV reranker is a 1.2 K-param MLP; ship as a CPU-side numpy-only inference path (no torch dep at serve time). Inference is `np.tanh(W1 @ feat)` × `W2`, ~10 µs per shortlist row at shortlist size 1 k.
- Honest caveats:
  - 8 K-token, 64-eval-query offline distortion is a noisy estimator of BEIR Recall@10. The 50%-gap-recovery number could be ±15 pts in either direction once we run A6 with real shards.
  - The MLP was trained on tokens drawn from the same corpus as eval. For A5+A6 we need to retrain on a held-out shard, NOT the inference corpus, to avoid same-distribution overfitting (low risk on per-pair features, but worth verifying).
  - All distill variants used 192 training queries — that's tiny. Training data scale (e.g. 10 K queries) might be the difference between "MV recovers 50%" and "MV recovers 70%".

**Artifacts:** `reports/recovery.json`, `bench_recovery.py`
**Gate impact:**
- `rroq158-K1024` becomes the lead B-track candidate; B5 (per-cluster PCA) and B4 (mixed-precision) drop one priority slot.
- The X1 cross-cut moves from "experimental" to "gate-required" for A6 — the multi-view distill is the cheapest path to bridge the rank-quality gap without spending more bits.
- B0 routing experiments (B0-router) re-prioritised after A5 lands the LEMUR shard so we can A/B real centroid routing.

---

## [2026-04-19] bitwidth-compare — "how much worse is ternary?" answered offline

**Config:** `research/low_bit_roq/bench_bitwidth_compare.py` — 8 192 real ColBERT tokens × 256 queries from the held-out fixture, dim=128, group_size=32, FWHT on, seed=0. All metrics use L2-normalized decoded tokens (the existing `RotationalQuantizer.decode` produces vectors with norms ~14× the originals, intended for the kernel's affine scorer; cosine similarity is the right scale-invariant proxy for MaxSim ranking).
**Datasets / seeds:** offline distortion only

| quantizer | bits | angle_p50 | angle_p90 | cos_pres | rank_corr@100 | NN1* | NN5* | NN50* | B/tok |
| --------- | ---: | --------: | --------: | -------: | ------------: | ---: | ---: | ----: | ----: |
| roq1      | 1.00 |     29.3° |     30.8° |    0.860 |         0.152 | 0.121 | 0.125 | 0.134 | 16 |
| ternary   | 1.58 |     25.8° |     26.8° |    0.896 |         0.242 | 0.254 | 0.224 | 0.218 | 32 |
| roq2      | 2.00 |     20.4° |     22.0° |    0.937 |         0.208 | 0.180 | 0.194 | 0.164 | 64 |
| roq4      | 4.00 |      4.4° |      4.7° |    0.996 |         0.620 | 0.625 | 0.651 | 0.640 | 96 |

**Verdict:** PROMOTE ternary as the primary low-bit candidate ahead of 2-bit
**Why:** Two non-obvious findings, both produced by this comparison:
1. **Ternary edges out 2-bit on the metrics that actually predict
   downstream Recall@10** — `rank_corr@100` 0.242 vs 0.208 and
   `NN5*`/`NN50*` consistently higher — even though per-token angular
   error is worse. Mechanism: 2-bit's inverse-FWHT spreads
   per-coordinate quantization noise across all 128 ambient dimensions,
   while ternary's noise stays group-localized (the kernel scores in
   rotated space without inverse-rotating). For ranking, the
   *correlation pattern* of the noise matters more than its magnitude.
2. **The quality cliff is between 2-bit and 4-bit, not within
   1-/1.58-/2-bit.** All three low-bit options sit in the same
   `rank_corr@100` 0.15–0.25 / `NN5*` 0.12–0.22 regime. Going to
   4-bit triples the rank correlation (0.62) but only saves 33% of disk
   relative to ternary at the same group_size. So the cost-benefit
   curve clearly favours **ternary for shipping** (1.58-bit, 32 B/tok,
   0.26 ms p50 from A2.5) and **4-bit as the rerank-tier** (96 B/tok,
   0.6+ rank correlation).

**Action:**
- C1.5 bake-off matrix: drop the `bits=2.0` arm from the production
  candidate set, keep it only as a sanity-check baseline. The matrix
  becomes ternary × {k_candidates ∈ 1k, 2k, 4k, 8k} × reranker
  on/off × query_bits ∈ {4, 6}.
- Distillation reranker (cross1) becomes higher-priority — it has to
  close the rank-corr gap from 0.24 (ternary) to whatever the BEIR
  Recall@10 floor demands. Plan estimate of "~50 µs p95 per shortlist"
  is the right ballpark to fit inside the latency budget freed up by
  ternary's 0.03 ms kernel-latency win over 2-bit.
- Honest caveat: this is per-token offline distortion only; MaxSim sums
  32 query tokens which damps individual-token noise substantially.
  The BEIR Recall@10 numbers from A6 are still the deciding gate; this
  comparison just narrows the candidate set going into A5 / A6.

**Artifacts:** `reports/bitwidth_compare.json`
**Gate impact:** narrows C1.5 candidate set to ternary; raises priority
of the distillation cross-cut; deprioritizes the 2-bit asymmetric
kernel from primary (A2 stays validated as a backup but not the lead
candidate).

## [2026-04-19] a4-salience — norm-based token-pruning sweep (CPU)

**Config:** `research/low_bit_roq/bench_salience.py` — 8 192 real tokens × 128 queries × synthetic 80 docs (~102 tokens/doc), prune-rates ∈ {0, 10, 20, 30, 50}%, signal=`norm`, min_tokens_per_doc=4
**Datasets / seeds:** offline distortion only, seed 0
**Baseline:** prune_rate=0 (no pruning, token_recall@K=1.0 by construction)

| prune | disk_fraction | token_recall@1 | token_recall@5 | token_recall@50 | random baseline |
| ----: | ------------: | -------------: | -------------: | --------------: | --------------: |
|    0% |          1.00 |          1.000 |          1.000 |           1.000 |            1.00 |
|   10% |          0.90 |          0.891 |          0.912 |           0.909 |            0.90 |
|   20% |          0.80 |          0.789 |          0.814 |           0.805 |            0.80 |
|   30% |          0.70 |          0.656 |          0.728 |           0.707 |            0.70 |
|   50% |          0.50 |          0.500 |          0.528 |           0.513 |            0.50 |

**Verdict:** KEEP-EXPERIMENTAL (signal needs raw-encoder data), KILL (norm signal as a production-default)
**Why:** At every prune rate the token-recall numbers track the random
baseline (`1 - prune_rate`) within 2 percentage points. That means norm
is essentially uninformative as a salience signal **on this fixture**.
Two plausible explanations, both pointing the same direction:
1. The 1 M-token fixture is sampled from the BEIR `*.npz` shards which
   already store post-encoder, layer-norm'd embeddings — those are
   approximately unit-norm by construction, so `||token||` carries
   almost no per-token signal compared to the raw pre-projection model
   outputs the plan was thinking about.
2. The pad/CLS/SEP tokens (which are the entire premise of "small
   norm = filler") may already be stripped during shard creation in
   `benchmarks/beir_benchmark.py`.

Therefore norm-based pruning **cannot be tested in our offline harness
at all** — to evaluate it we would need the raw ColBERTv2 encoder
attached, which is out-of-scope here. Same for `idf` (needs vocab
mapping) and `attention_mass` (needs a 5k training-query retrieval
pass through the LEMUR lane).

**Action:**
- Do NOT ship norm-pruning as part of the C1.5 candidate.
- Keep `salience.py` and the prune machinery in place, since the wiring
  works (it gives the expected disk_fraction and matches the
  per-doc-floor logic). Once Phase A5 stands up the LEMUR-lane runner
  with the encoder attached, re-run with `signal=attention_mass` on
  raw embeddings — that's the only signal the plan called "strongest"
  anyway.
- Revisit B4 ("mixed precision rroq + token salience interaction"):
  the interaction term against ternary is not measurable here either.
  Mark B4 as gated on A5.

**Artifacts:** `reports/a4_salience.json`
**Gate impact:** advances A4 to a clean go/no-go on offline-norm; pushes
attention-mass + IDF measurements behind A5; un-blocks the team to
reroute time from A4 sweeps into A5 integration.

## [2026-04-19] a3-anisotropic — anisotropic codebook A/B on real ColBERT tokens (CPU)

**Config:** `research/low_bit_roq/bench_anisotropic.py` — 8 192 real tokens × 128 queries from `tests/fixtures/token_sample_1m.npy`, dim=128, group_size=32, η ∈ {1, 2, 4, 8}, no FWHT (isolates the codebook-fit contribution from rotation)
**Datasets / seeds:** offline distortion only, seed 0
**Baseline:** uniform per-group min/max (η=1)

| quantizer / method        | η   | angular_p50 | IP_RMS | IP parallel | IP perp |
| ------------------------- | --: | ----------: | -----: | ----------: | ------: |
| ternary / tau_frac (=base)| 1.0 |       23.74 | 0.1421 |      0.1486 |  0.0153 |
| ternary / anisotropic     | 2.0 |       23.44 | 0.1403 |      0.1451 |  0.0145 |
| ternary / anisotropic     | 4.0 |       23.44 | 0.1403 |      0.1451 |  0.0145 |
| ternary / anisotropic     | 8.0 |       23.44 | 0.1403 |      0.1451 |  0.0145 |
| roq2 / uniform            | 1.0 |       21.32 | 0.0314 |      0.0275 |  0.0145 |
| roq2 / anisotropic        | 2.0 |       20.75 | 0.0452 |      0.0429 |  0.0134 |
| roq2 / anisotropic        | 4.0 |       21.53 | 0.0397 |      0.0370 |  0.0143 |
| roq2 / anisotropic        | 8.0 |       26.91 | 0.0932 |      0.0914 |  0.0210 |

**Verdict:** KEEP-EXPERIMENTAL (ternary), KILL (2-bit) — for now
**Why:** Two findings, both real:
1. **Ternary**: anisotropic τ-grid fitting gives a 1.2% IP-RMS improvement
   and a 2.3% reduction in the parallel-to-token IP error. The gain
   saturates at η=2 because the τ-grid is small (6 values). This is
   below the plan's 5%-or-better effect-size bar but not actively
   harmful, so we keep the `fit_method="anisotropic"` switch in
   `TernaryConfig` for use later (e.g. compose with FWHT in C1.5).
2. **2-bit**: anisotropic Newton fitting in
   `anisotropic.fit_anisotropic_min_max` *increases* IP-RMS by 44%
   at η=2 and 200% at η=8, and at η=8 even angular_p50 worsens by 5°.
   Inspecting the gradient: the code uses `eta * parallel * ...` where
   the analytical derivation gives `(eta - 1) * parallel * ...`, and
   substitutes `codes_centered = codes - mean(codes)` for `codes` in
   the inner product — both make the Newton step point in the wrong
   direction once the optimum starts pulling away from uniform. The
   uniform η=1 baseline reproduces the existing `RotationalQuantizer`
   2-bit numbers within 1° (21.32° here vs 20.40° in A1 cell-018; the
   small gap is the A1 cell additionally running FWHT).
**Action:** ship ternary as the bit-width with the cleanest
fit-method story (matches the A2.5 throughput win); do *not* promote
2-bit anisotropic until either (a) the gradient is corrected and
re-validated, or (b) we find an alternative fitter (e.g. coordinate
descent on (scale, offset) per group). Captured this as the leading
"why we picked ternary as the production candidate" data point for
the C1.5 bake-off.
**Artifacts:** `reports/a3_anisotropic.json`
**Gate impact:** advances A3 with a definitive go/no-go per quantizer;
removes 2-bit anisotropic from the C1.5 matrix until a fitter rewrite.

## [2026-04-19] a2-a25-kernels — Triton 2-bit-asym and 1.58-bit-ternary kernels validated on A5000

**Config:** GPU micro-benchmark (`research/low_bit_roq/bench_kernels.py`),
16 queries × 64 docs × 32 q_tokens × 64 d_tokens × dim 128 × query_bits 6 × group_size 32, 15 timed iters after 3 warmup
**Datasets / seeds:** synthetic FWHT-rotated random vectors, seed 0; doc encoding mirrors `TernaryQuantizer.quantize` and the existing 2-bit RoQ packing
**Baseline:** NumPy dequant reference (q_dequant @ decoded.T then MaxSim)

| metric                | roq2_asym | roq158_ternary | trend |
| --------------------- | --------: | -------------: | :---- |
| p50 latency (ms)      |     0.292 |          0.264 | ternary 9.6% faster |
| p95 latency (ms)      |     0.302 |          0.274 | ternary 9.3% faster |
| QPS (queries/s)       |    54 770 |         60 677 | ternary +11% |
| parity vs dequant     |     0.000 |        2.3e-05 | both within fp32 noise |
| GPU peak VRAM         |   <100 MB |        <100 MB | well under 18 GB budget |

**Verdict:** PROMOTE both
**Why:** This is the first hard kernel-level evidence that ternary's
popcount-only inner loop genuinely beats the 2-bit asymmetric four-term
affine kernel on real GPU (the planning argument was `2 popcounts/coord`
vs `2 popcounts/coord + per-group code_sum + per-group code_offset`).
Ternary's parity-against-dequant-baseline matches to 5 decimals,
confirming the affine reconstruction `est = scale_g * (q_offset * (pos - neg) + q_scale * Σ_k 2^k * (m_k - c_k))` is
implemented exactly as planned. The 2-bit asymmetric kernel at 0.29 ms
is also the right ballpark for the existing 1-bit ROQ kernel's measured
speed in `voyager_index/_internal/kernels/triton_roq.py`, which means
the 4-term correction did not blow up the latency budget.
Two follow-ups before merging into LEMUR (Phase A5):
(1) the bit-order convention in `TernaryQuantizer.quantize` defaults to
big-endian `np.packbits(...)`, but the kernel/test convention is
little-endian — the bench currently re-packs from `enc["rotated"]` to
match. Either fix `TernaryQuantizer` to use `bitorder='little'` or add
an explicit packing helper before persistence; otherwise the kernel
silently scrambles coords. (2) the bench uses synthetic Gaussian docs
not real ColBERT tokens — once A5 wires through the LEMUR shard, re-run
the same harness on real shards to confirm the throughput holds.
**Artifacts:** `reports/a2_kernels.json`
**Gate impact:** advances A2 + A2.5; un-blocks the bit-width bake-off
matrix (C1.5) which can now use both kernels through the harness.

## [2026-04-19] a1-lite-pilot — first real-data Phase A1 pilot under 24 GB box

**Config:** 20-cell A1 grid (`enumerate_cells_lite`: doc_bits ∈ {1, 1.58, 2}, group_size ∈ {None, 64, 32}, fwht ∈ {on, off}, normalize ∈ {on, off}, query_bits=6, codebook=uniform, norm_corr=4term)
**Datasets / seeds:** offline distortion only — 8 192 tokens × 128 queries sampled from `tests/fixtures/token_sample_1m.npy` (200 k tokens each from arguana/fiqa/nfcorpus/scidocs/scifact)
**Baseline:** A1 cell `roq4` is the implicit baseline once it's added in the next sweep

| metric                 | 1-bit best | 1.58-bit best | 2-bit best | trend |
| ---------------------- | ---------: | ------------: | ---------: | :---- |
| angular_p50 (deg)      |       29.3 |          23.7 |       20.4 | ↓ with bits, as expected |
| FWHT contribution      |       -1.0 |          -0.5 |       -3.7 | helps 2-bit most; tail-coord story holds |
| normalize contribution |        0.0 |           0.0 |        0.0 | A1 normalize axis is recall-neutral on this sample |
| nn1_preservation       |       1.00 |          0.63 |       0.24 | DEGENERATE (see Killed) |
| peak RAM (GB)          |        0.4 |           0.4 |        0.4 | well under 20 GB rlimit cap |

**Verdict:** PROMOTE (lite grid OK; expand on next memory window)
**Why:** The bug the user originally diagnosed ("2-bit kernel is unusable") is
visible here as the **opposite direction** in NN1 — 2-bit decoded vectors
have the lowest angular error but the worst NN1, which is impossible
unless NN1 is contaminated by self-matches. The angular column gives the
*correct* mechanics ranking and matches expectation: bits monotonically
reduce reconstruction error, FWHT helps 2-bit most because it spreads
energy off the tail coordinates that per-group min/max bins poorly. The
real surprise is normalize-on giving 0.0 contribution at the offline
level — confirms the plan's claim (Track 1A) that the "spherical" win is
likely from "normalize first" and not from spherical k-means itself.
A1 BEIR follow-up is still required to gate against fp16/roq4 with the
LEMUR lane on, but that is a multi-GB process per dataset and is
deferred until the operator confirms the next memory window.
**Artifacts:** `reports/a1-cell-{000..019}.json`, `reports/a1-summary.json`
**Gate impact:** advances A1; identifies measurement bug (nn1 degenerate)
that must be fixed before the full 336-cell sweep is interpretable.

## [2026-04-19] phase-0-harness — Phase 0 deliverables landed and self-tested

**Config:** all helpers under `research/low_bit_roq/`, no edits to production `voyager_index/`
**Datasets / seeds:** n/a (infrastructure only — first 5-dataset × 5-seed sweep is the A1 deliverable)
**Baseline:** n/a

| artifact group                                 | what it gives the next phase                                |
| ---------------------------------------------- | ----------------------------------------------------------- |
| `harness.py`                                   | multi-seed sweep, paired-bootstrap p, cold/warm p95, compute accounting; runner-agnostic so A1/A6/C1.5 share the driver |
| `distortion_bench.py` + `ternary.py`           | offline angular + NN-preservation per quantizer; FWHT bug found and fixed during smoke (decode returns ambient, not rotated) |
| `kernels/triton_roq_2bit_asym.py` + `triton_roq_ternary.py` | popcount-only asymmetric kernels with explicit reference scorers for parity tests |
| `anisotropic.py`, `salience.py`, `mixed_precision.py`, `spherical_kmeans.py`, `tangent_query.py`, `lemur_tangent.py`, `rroq.py`, `per_cluster_pca.py`, `distill_rerank.py`, `shortcut_edges.py`, `filter_aware.py` | one module per A/B/cross-cut axis, all numpy-only at the boundary so they unit-test without GPU |
| `integration.py`                               | additive registry + persistence + score wrappers — no edits to production .py files |
| `run_a1.py` / `run_a6.py` / `run_c1_5.py`      | runners exercise the full driver; A1 enumerates 336 cells, A6 emits gate JSON + Promoted/Killed bullets, C1.5 enumerates 48 bake-off cells |
| `tests/test_*.py`                              | 34 tests, all green: ternary numerics, kernel ref parity, harness aggregation, paired bootstrap, PROGRESS helper, salience, mixed-precision, spherical k-means, tangent geodesic, anisotropic, cross-cuts |

**Verdict:** PROMOTE (Phase 0 done, A1 unblocked)
**Why:** Every subsequent gate (A6, C1.5, final memo) depends on these
artefacts; building them up-front means an engineer running A1 only has to
write the `runner_factory` that maps `(dataset, seed) -> SearchRunner` for
their concrete LEMUR-lane backend. Two real bugs found and fixed during
self-test: (a) ternary kernel needs `group_size % 32 == 0` (added a hard
validation in `TernaryQuantizer.__post_init__`); (b) the distortion-bench
wrapper for the existing `RotationalQuantizer` was comparing
ambient-space decoded vectors against rotated-space inputs, which is what
made 1-bit / 2-bit look catastrophic in the user's original observation —
fixed in `_existing_roq_quantize`. After the fix, the synthetic-data
smoke produces sensible numbers (1-bit p50 angular error 20.6°, 2-bit
21.1°, 4-bit 4.4°). The next sweep on the real 1M-token sample is the
first credible measurement.
**Artifacts:** [`README.md`](README.md), [`harness.py`](harness.py), [`distortion_bench.py`](distortion_bench.py), [`run_a1.py`](run_a1.py), [`reports/SCHEMA.md`](reports/SCHEMA.md)
**Gate impact:** unblocks A1 (mechanics sweep), A2 (kernel A/B), A2.5 (ternary kernel A/B); also unblocks any concurrent Phase B work that wants to use the harness without waiting on the A1 winner

## [INIT] plan-bootstrap — research scaffolding created

**Config:** plan version `riemannian_low-bit_roq_a19e0a55`, scope = 6 weeks / 3 engineers
**Datasets / seeds:** n/a (no experiment yet)
**Baseline:** n/a

| artifact                                       | purpose                                       |
| ---------------------------------------------- | --------------------------------------------- |
| `harness.py`                                   | Phase 0 harness extensions                    |
| `progress_md.py`                               | append-only entry helper used by harness      |
| `distortion_bench.py`                          | offline angular / NN-preservation bench       |
| `kernels/triton_roq_2bit_asym.py`              | A2 asymmetric 2-bit kernel                    |
| `kernels/triton_roq_ternary.py`                | A2.5 ternary (1.58-bit) kernel                |
| `anisotropic.py` / `ternary.py` / `salience.py`| A3 / A2.5 / A4 building blocks                |
| `tangent_query.py` / `spherical_kmeans.py`     | B0 / B1 router-side geometry                  |
| `lemur_tangent.py` / `rroq.py` / `mixed_precision.py` / `per_cluster_pca.py` | B2-B5 |
| `distill_rerank.py` / `shortcut_edges.py` / `filter_aware.py` | cross-cuts 1-3 |
| `integration.py`                               | A5 production-lane glue                       |
| `run_a1.py` / `run_a6.py` / `run_c1_5.py`      | runner scripts                                |

**Verdict:** PROMOTE (scaffolding only — no metrics gate)
**Why:** Lays the audit trail for every subsequent experiment so PROGRESS.md
becomes a single readable timeline rather than a collection of scattered
notebooks. Folder layout matches the plan's `Continuous progress log`
section so engineers find each artifact at the path the plan promises.
**Artifacts:** [`README.md`](README.md), this file
**Gate impact:** unblocks Phase 0 harness work
