# PyHealth 2.0 Publication Preflight

Date: 2026-07-29. Published Space:
`kondratevakate/repro-pyhealth-2-0-clinical-deep-learning-toolkit`.

Publication status: `PUBLISHED / PENDING JUDGE`.

Space URL:
`https://huggingface.co/spaces/kondratevakate/repro-pyhealth-2-0-clinical-deep-learning-toolkit`.

Published commit:
`101ec071fcd5ca2b407dd8573e6e1577f590aa09`.

Canonical static host:
`https://kondratevakate-repro-pyhealth-2-0-clinical-deep-6cc09de.static.hf.space/`.

## Score forecast

| Claim | Prepared verdict | Expected points | State |
| --- | --- | ---: | --- |
| C1 toolkit coverage | `VERIFIED` | 2 | Full paper/release inventory complete |
| C2 mortality throughput | `NOT ATTEMPTED` | 0 | Full MIMIC-IV v2.2 and baseline sweep required |
| C3 drug/LOS throughput | `NOT ATTEMPTED` | 0 | Full MIMIC-IV v2.2 and baseline sweep required |
| C4 seven-line/Table 2 attribution | `FALSIFIED` | 2 | Machine-parsed source mismatch plus mutation control |
| C5 community/tutorial/RHealth | `INCONCLUSIVE` | 0 | Historical membership artifact unavailable |

Defensible forecast: `4/10`; plausible judge range `3-5/10`. The upside is a
partial point for the independently supported tutorial/RHealth components of
C5 and is not included in the forecast.

## Passed

- Live challenge API scan found no existing logbook tagged
  `paper-gMLVFN9hl8` among 5,308 Spaces.
- arXiv `2601.16414v2` source archive is pinned at SHA-256
  `8f592e99599e609ced3216700c1d4e878baffcb792c35a63a9c767c8224d078c`.
- Official PyHealth `v2.0.1` checkout is clean at commit
  `ed562121b5bae185b36322c64ce6215c2095dd50`.
- C1 parses paper totals `22/43/28/7` and release exports `24/51/40/8`;
  all ten threshold and witness checks pass.
- C4 parses Table 2 mortality values `34/27/51`, contradicting the anchored
  `7/24/51` claim for both PyHealth rows.
- Four tests pass, including genomics-removal and corrected-Table-2 mutations.
- `prepare_sources.py`, the audit, tests, and validator require only Python's
  standard library.
- Source bundle and publish-bundle file hashes match.
- Local ICML logbook validator passes.
- Browser render shows all pages and the pinned poster without raw markup.
- Browser console has no warnings or errors.
- No API keys, access tokens, passwords, subject IDs, or admission IDs are
  present in the publishable bundle.
- Fetched paper source and official upstream clone are excluded from Git.

## Post-publication checks

- The Space is public, uses the static SDK, and reports runtime `RUNNING`.
- Tags include `icml2026-repro`, `paper-gMLVFN9hl8`, and `arxiv:2601.16414`.
- The canonical static host returns HTTP 200.
- The remote repository contains all 27 expected files.
- All seven files under `repro_pyhealth/` were downloaded from the published
  commit and match their local SHA-256 hashes.
- The live pre-publication scan covered 5,315 challenge Spaces and found zero
  prior attempts for `paper-gMLVFN9hl8`.
- C4 remains narrowly scoped to the anchored composite claim.
- C2/C3 remain `NOT ATTEMPTED`; C5 remains `INCONCLUSIVE`.

Any future edit requires a fresh audit, local validator run, republish, and
remote hash comparison. The current frozen forecast remains `4/10`.

## Frozen integrity hashes

```text
69d7b5c52aeb147208d86cd50f9ca13b97e8db72825091700f8533caa2d49800  audit_claims.py
8a721f57a08ef17da4644a71d1e001d9d87fd21a2a62e937c64aaa0f68c12841  prepare_sources.py
41193c100bde4bfc6b32880dbf1eef929a17f961f4e40379fa66e2bd0fc75a6c  validate_evidence.py
5eab28d38d7a7821a27e95d1ffbc2e835253892e099442ae9590f91c6c5f87b3  tests/test_audit_claims.py
448469a3a584cf3e4395b83df77e3dfdf3c8318a18ea6cf8ea3f6a0803f27b4a  evidence/claims_audit.json
```

Logbook:

```text
a8aaa7658baf56663ec798457ee83e80069c2a6c4825dc07d07dad3058e7a598  .trackio/metadata.json
65c326a7b7ef74976a205843169aee582322a3b55345c877880c2548ab7a5b77  .trackio/logbook/logbook.json
8713f188d2f7b473812b2a0c9201f1fecd7221b749aad92fa2f60083fc38b3e1  executive-summary/page.md
5e994cab27a702ddc80ca2921961fe426d11835e946bfa01cc391019a18246e8  claim-1-toolkit-coverage/page.md
f48769d1ffa340704d5008fc579c017316df047d9db19d5c68e011ae33d93d7d  claim-4-seven-line-table-2-attribution/page.md
```
