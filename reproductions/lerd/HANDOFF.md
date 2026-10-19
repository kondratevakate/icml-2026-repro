# LERD: handoff (draft for picking up in another chat)

Translated from Russian on 2026-10-19; the original is in the git history of this file. Paths follow the
2026-10-19 layout (`reproductions/`, `notes/`).

## Target
- **Paper:** LERD (neurodegeneration / Alzheimer EEG)
- **OpenReview id:** `B5DAV1EA8Z` · **arXiv:** 2602.18195
- **Points:** 12 · **Fit tier:** A (top fit for Kate's profile)
- **Data:** OpenNeuro **ds004504**, open without registration or credentialing.
- Where the lead came from: `notes/PAPERS.md` (lines 41 and 114) and `notes/HANDOFF.md`
  ("Nearest datasets for the targets already planned").

## Status at the time of that chat
- The folder `lerd/` had just been created: **reproduction from scratch**, no code or paper.txt yet.
- Dataset ds004504 **not downloaded yet** (not in `data/`, which held only cifar100n, mfiddr_sample,
  mimic3demo, sleepedf).
- From `HANDOFF.md`: "the synthetic part of claim 1 is reproducible **without downloads**". The real data are
  ds004504.

## Open questions to the user (awaiting an answer), about an MCP server for OpenNeuro
1. **File transport:** S3 (`s3fs`/`boto3`, fast) versus plain HTTPS (no dependencies, slower).
2. **Confirmation threshold for downloads:** download silently versus ask above about 2 GB.
- Further context: the user asked to look at how the MCP for the NIH Imaging Data Commons is built
  (mcpmarket.com/tools/skills/imaging-data-commons-access; it returned HTTP 429, retry needed) and at other
  projects with similar data, to "pick up experience". The reference IDC package: `idc-index`.

## Plan (for the next chat)
1. **Get the paper:** arXiv 2602.18195 → `lerd/paper.txt` (PyMuPDF, NOT from an LLM summary; see the memory
   `feedback-never-reproduce-from-llm-summary`).
2. **Extract the anchored claims** for LERD (section-anchored, 5-6 precise ones); do not use the vague
   claims.json.
3. **The synthetic part of claim 1** is reproducible without downloads; start with it.
4. **Download ds004504** from OpenNeuro (S3: `s3://openneuro.org/ds004504`, or HTTPS). It is EEG (Alzheimer,
   FTD, healthy controls), BIDS format with .set/.fdt files.
5. **MCP for OpenNeuro**: set up a server skeleton in parallel; settle questions 1-2 above.

## Working rules (from memory)
- At the time: logbooks in Russian; no dashes or plus signs in the text; nothing published to a logbook
  without Kate's word; verify the .py before targeting.
- Reproduce from the paper's real specification; do not invent a proxy.
- Write verdicts cleanly and in binary form (refute = confirm = 2 points); scope caveats go separately, not in
  the verdict.

## Repository
At the time `C:\Projects\02_academia\icml-repro` (branch master); datasets in `data/`. Now
`reproductions/lerd/` in this repository.
