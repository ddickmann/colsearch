# Groundedness Service Validation

- profile: `long_ambiguous`
- model: `lightonai/GTE-ModernColBERT-v1`
- prompts: `False`
- packed raw_context chunk tokens: `256`
- latency repeats per case: `3`
- encoder token limit: `300`
- anchor count: `4`
- anchor AUROC (reverse_context): `1.0000`
- anchor AUROC (consensus_hardened): `1.0000`
- anchor AUROC (reverse_query_context): `1.0000`
- anchor AUROC (triangular): `0.0000`
- latency p50/p95 ms: `90.94` / `97.18`
- mean/max context tokens: `7847` / `7888`
- mean packed support units: `33.5`
- user-facing go/no-go: `False`

## Difficulty Summary

| bucket | count | mean reverse_context | mean consensus_hardened | mean triangular | mean context tokens |
|---|---:|---:|---:|---:|---:|
| entity_swap | 2 | 0.9850 | 0.9847 | 0.9626 | 7858 |
| grounded | 2 | 0.9887 | 0.9885 | 0.9569 | 7816 |
| partial | 2 | 0.9793 | 0.9792 | 0.9510 | 7867 |

## Hardest Previous Case Rerun

- selected case: `LG3` (ungrounded)
- rationale: highest previous non-grounded reverse_context score (`0.9908`) in the earlier long-context report
- before: reverse_context `0.9908`, consensus_hardened `0.0000`, support_units `66`
- after packed-256: reverse_context `0.9865`, consensus_hardened `0.9859`, support_units `34`
- verification per-token max abs diff: `0.00000000`
- verification scalar abs diff: `0.00000000`
- verification consensus per-token max abs diff: `0.00000000`
- verification consensus scalar abs diff: `0.00000000`
- verification effective-support-units max abs diff: `0.00000000`
- grouped score batch units: `16`
- evidence mapping exact match: `True`
- top-evidence exact match: `True`

## Example Evidence

### entity_swap
- `LG3`: Construction on the Royal Arena in Copenhagen, where the 2017 European Short Course Swimming Championships will be held, broke ground on 26 July 2013.
  - notes: Date-swap near miss inside an ultra-long context; hard for embeddings and easy for a human to miss.
  - token `Ġ26` -> `Ġa` score `0.9918`
  - token `Ġ2017` -> `Ġ2017` score `0.9907`
  - token `Ġ2013` -> `ĠThe` score `0.9897`
- `LG4`: Treg-cell-specific deletion of integrin αvβ6 did not result in a spontaneous inflammatory phenotype in the studied mice.
  - notes: Single-character entity swap inside a long scientific context block.
  - token `6` -> `ĠÎ±` score `0.9978`
  - token `Î²` -> `ste` score `0.9506`
  - token `Ġresult` -> `8` score `0.9967`

### grounded
- `LG1`: Teardrops, the second single from George Harrison's album Somewhere in England, was released in the United States on 20 July 1981.
  - notes: Long distractor-heavy context with the George Harrison evidence paragraph buried near the middle.
  - token `Ġ20` -> `Ġon` score `0.9950`
  - token `Ġ1981` -> `Ġis` score `0.9885`
  - token `ĠStates` -> `,` score `0.9967`
- `LG2`: Aptamer-functionalized lipid nanoparticles can target specific cell types such as osteoblasts, as demonstrated with the CH6 aptamer-functionalized LNP system.
  - notes: Long biomedical context where the relevant aptamer paragraph must survive thousands of unrelated tokens.
  - token `6` -> `Ġthen` score `0.9917`
  - token `Ġdemonstrated` -> `Ġof` score `0.9951`
  - token `Ġsuch` -> `Ġimplicated` score `0.9945`

### partial
- `LG5`: SHP2 signal-deficient knockin mice display lymphadenopathy and splenomegaly, and were the first transgenic model used in human clinical trials for lupus nephritis.
  - notes: First clause is supported, second clause is a plausible biomedical extrapolation hidden inside a long context.
  - token `2` -> `Ġwere` score `0.9848`
  - token `Ġused` -> `Ġthe` score `0.9942`
  - token `Ġtrials` -> `IT` score `0.9939`
- `LG6`: Be Quick '28 is a football club from Zwolle in the province of Overijssel, and it currently plays in the Dutch Eredivisie top flight.
  - notes: Supported province fact mixed with an unsupported league claim under heavy distractor load.
  - token `28` -> `Ġprovince` score `0.9898`
  - token `Ġfootball` -> `ij` score `0.9966`
  - token `ĠZ` -> `,` score `0.9959`

## Per-Case Scores

| id | label | subcategory | context_tokens | support_units | reverse_context | consensus_hardened | reverse_query_context | triangular | latency_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LG1 | grounded | - | 7808 | 33 | 0.9895 | 0.9894 | 0.9895 | 0.9588 | 75.38 |
| LG2 | grounded | - | 7824 | 33 | 0.9879 | 0.9875 | 0.9879 | 0.9550 | 96.57 |
| LG3 | ungrounded | entity_swap | 7828 | 34 | 0.9865 | 0.9859 | 0.9875 | 0.9611 | 90.77 |
| LG4 | ungrounded | entity_swap | 7888 | 34 | 0.9836 | 0.9836 | 0.9836 | 0.9642 | 77.57 |
| LG5 | ambiguous | partial | 7884 | 34 | 0.9785 | 0.9785 | 0.9785 | 0.9550 | 91.12 |
| LG6 | ambiguous | partial | 7850 | 33 | 0.9800 | 0.9799 | 0.9800 | 0.9470 | 96.90 |
