# HANDOFF: ICML 2026 Reproduction Challenge (Kate)

Updated 2026-07-21; translated from Russian on 2026-10-19 (the Russian original is in the git history
of this file). Rewritten in full at Kate's request: every problem we ran into, and above all everything
that actually cost points (judging by the real scores, not by guesses).

Paths below follow the 2026-10-19 layout: paper folders are under `reproductions/`, notes under `notes/`.

## Goal and selection filter

The ICML 2026 paper reproduction challenge (HF Space `ICML-2026-agent-repro/challenge`).
Kate's priority: **application to medical data** (directly or indirectly), reproducibility, and that
**the claim can actually be put to use** in Kate's own work. Points and the certificate are a by-product.
The prize is GPU credits ($4000 on HF), not money.

**The one criterion that cannot be skipped is whether the data needed for the reproduction are available.**
Code is NOT a criterion (we reimplement from the text). The text is NOT a criterion (OpenReview can be read
in a browser). If the data are closed (in-house, hospital data, access by committee application), the paper
is out, however interesting it is.

## Kate's profile

Reliability of structural brain MRI, evaluation of segmentation (Dice vs ICC), scanner effects and
harmonisation, neurodegeneration, metrics. MIDL 2026. New interest: **conformal prediction** (wants to
work on it in the lab).

---

# What actually cost points

Two rounds of real judge scores, both analysed line by line. Total over all 11 logbooks:
**34 of 98 possible points (35%)**. Below is what explains each loss, from the most important cause to
the less important ones.

## New requirements for every reproduction (from 2026-07-21, after reading the logbooks of the #1 team on the leaderboard)

Mandatory checklist **before declaring a claim verified or falsified**:

1. **A mutation (destructive) test for every claim.** Do not only compute a number: break the mechanism
   on purpose (replace a component with its naive alternative, invert a condition) and show that the result
   changes in the predicted way. A number without such a test shows correlation, not cause.
2. **Exact enumeration instead of seeds wherever the claim's space is finite.** If all cases can be listed
   (all partitions, all configurations), do it literally rather than sampling 10-12 seeds and claiming they
   "agree". Seeds are what one falls back on when exact enumeration is impossible, not the default.
3. **Full open datasets on CPU, not a toy subsample by default.** If the data are open and fit in the CPU
   budget (example: full MNIST, 60,000 examples), use all of it. Cut down only if size or training really
   needs a GPU.
4. **Limits of applicability go in a separate, clearly titled section** (for example "Evidence boundary"
   at the end of the logbook), not inside the sentence that carries the verdict. The verdict is stated as
   confidently as the actual evidence allows; honest caveats go separately, neither dissolved into the
   verdict nor hidden altogether.
5. **Cite the source exactly: the line or section number in the paper's source text**, not a paraphrase of
   "it says somewhere". Where possible, give a hash of the frozen protocol (seeds, script version) so the
   run can be reproduced bit for bit.
6. **Present a falsification with the same surgical precision as a verification**: give the exact source of
   the discrepancy (the specific symbol, line or number), not "possibly a typo, cannot confirm". A
   falsification earns the same points as a verification, but only if it is decisive.

**Open questions, not investigated, not to be dismissed:** switching models in the middle of a session may
have broken the chain of reasoning; Kate's own instructions to subagents may not have been strict enough;
conditions may have been lost over a long multi-agent session. Analysing a competitor's logbooks gives a
better method, but does not prove that the method is the only reason for the gap in points.

## Cause 1 (CORRECTED 2026-07-21, the first version was wrong): not "hide caveats" but "make the evidence stronger"

The first version of this section claimed "a hedge in the text = 0 points", based on the fact that the
totals for five papers added up to that arithmetic. **Kate rightly pointed out that this was fitting after
the fact**: we have no per-claim breakdown, only the total per paper, and I made up a story that fitted it.
Worse, even if it had been true, the advice "hide honest caveats from the judge" contradicts what is
actually needed, which is results that can be trusted rather than results optimised for the metric. I
should not have presented it as a recommendation.

**Instead of theorising, we looked at the actual logbooks of the #1 team on the leaderboard**
(`neonforestmist`, 656 points over 102 logbooks; we had 34 over 11, so twice as many points per logbook).
Nothing resembling "hiding caveats". On the contrary:

- **A claim openly declared false received full marks.** "FALSIFIED AS WRITTEN: the challenge text has
  2·c̄, the source has √(2·c̄)": a falsification that names the exact line of the source, not "possibly a
  typo, not sure".
- **Mutation tests instead of bare numbers.** Replacing the self-including centroid with leave-one-out flipped
  the result from 456/456 fixed partitions to 1/456. This shows that the mechanism works **causally**, not
  that the number happened to come out that way.
- **Exact enumeration instead of seeds where possible.** Not "10 seeds agree" but a literal enumeration of
  all 456 admissible partitions: 14,592 of 14,592 checks.
- **Full datasets, not toy subsamples, where CPU allows.** All 60,000 MNIST examples in one of the logbooks,
  not a cut-down version.
- **Limits of applicability are present but in a separate section** ("Evidence boundary") at the end of the
  logbook, after all claim verdicts have been given cleanly: not built into the verdict sentence, and not
  hidden either.
- Cryptographic freezing of the protocol (a SHA hash of the frozen seeds) and exact line numbers in the
  source `.tex` for cited theorems: traceability an order of magnitude above ours.

**Revised practical conclusion.** The lever is not "write more confidently on the same evidence" but
**make the evidence categorically stronger**: design a mutation (destructive) test for each claim that breaks
the result if the mechanism is not real; use exact enumeration instead of seeds where the claim's space is
finite; take full open datasets on CPU instead of toy subsamples by default; pin exact source lines and
hashes. With evidence like that, a confident verdict follows by itself, and honest limits of applicability
live in a separate section without blurring the verdict sentence.

Kate also suggested, neither confirmed nor ruled out: (a) switching models mid-session may have broken the
chain of reasoning or lost context; (b) Kate's own verification instructions to subagents may not have been
strict enough; (c) conditions may have been forgotten over a long multi-agent session. None of this was
investigated: the competitor's logbooks give additional evidence about **method**, not proof that method was
the only reason for the gap in points.

## Cause 2: one seed on a stochastic pipeline is not evidence

The original CalPro claim 1 ("no more than 5 pp coverage degradation") showed 0.984 coverage at tau = 0.9 in
its first publication, from **one** seed. Re-checking on 5-8 seeds: mean degradation 17.5 pp, worst 38.8 pp,
the threshold is not met at all; coverage under shift never reached nominal. The headline number was **a lucky
initialisation**, not a reproducible result. The initial low score (2/6 for the old version) was most likely
fair.

**Rule: at least 3-5 seeds before a "reproduced" verdict, especially if the pipeline includes training a
neural network or other randomness.** All three papers with full marks in the first wave (Solvable AE,
Conditional Coverage Diagnostics, Survival Model Evaluation) rested on an exact analytical ground truth where
the seed hardly matters. That is not a coincidence; it is why they were reliable.

## Cause 3: testing an invented proxy instead of the real claim

The project has two claim files: `claims.json` (2-3 vague marketing sentences per paper) and
`claims_anchored.json` (5-6 claims per paper, each precise, with a number and an anchor to a section, table
or figure; **this is what is judged**). The first 8 logbooks were done from `claims.json`; since it does not
allow checking a specific value, we had to invent an operationalisation ourselves.

The worst case is **CalPro**. The real claim A4: "under the X-ray to cryo-EM shift, CalPro coverage degrades
by 3.8 pp against 14-18 pp for the pLDDT baseline", that is **protein structures, per-residue RMSD, the
baseline is AlphaFold pLDDT, and CalPro itself is a graph evidential head with a differentiable conformal
layer and domain priors**. What was reproduced was a one-dimensional synthetic heteroscedastic toy with a
home-made support shift: a different experiment altogether. Both the old positive verdict and the later
multi-seed negative one **measure the wrong thing**, and neither can be submitted. The honest status of A4 is
"not tested".

**Rule: before starting any reproduction, read the paper's entry in `claims_anchored.json` and target it
word for word, together with its section or table anchor.** Read both files only with `encoding='utf-8'`;
the default cp1252 on Windows fails on them.

## Cause 4: a hallucinated specification from WebFetch (two independent cases in one session)

WebFetch returns **an LLM summary of the page, not the page itself**. For reproducing a paper this is
dangerous: twice in one wave of work it **confidently invented a methodology that does not exist**.

- For KM-ARL it invented "a CUSUM detector, threshold 370, change point at position 2500"; none of this is in
  the paper (the real method: a GSR detector with true statistics). What saved us was that the agent
  downloaded the PDF and extracted the real text with PyMuPDF.
- For SurvFD it distorted test scenarios 8-10 (giving `β₁₃x₁x₃²` where the paper has `β₁₃x₁x₂³`) and
  attached the numerical threshold 0.015 to the wrong quantity.

In both cases the results built on the summary had to be withdrawn and recomputed properly. Tellingly, in
the second case WebFetch passed on the theorem statement **correctly**, but by luck, not by reliability; a
partly correct summary is more dangerous than a plainly wrong one, because spot-checking one item gives
false confidence.

**Rule: never reproduce from a WebFetch summary of a paper.** Download the PDF and extract the text with
PyMuPDF (`import fitz`; poppler is not installed), or for OpenReview use the built-in browser
(`mcp__Claude_Browser__*`), which passes the Cloudflare check (curl and the REST API get 403). WebSearch is
fine for **finding** a paper's arXiv id, not as a source of its content.

---

# Technical pitfalls in publishing (not about points, but they cost time)

- **Windows cp1252 mangles emoji.** Edit `metadata.json` and `README.md` only with a UTF-8 editor; a raw
  Python `open("w")` on Windows silently corrupts 🎯, and publishing fails with "Invalid metadata... fails
  Extended_Pictographic".
- **Calling `trackio.init()` by hand inserts a phantom dashboard page** that breaks the page order and the
  validator's check. For the artifact bundle use `finalize_bundle.py` (it logs the artifact and removes the
  phantom from `logbook.json`, `pages/`, `index.md` and metadata).
- **Long claim labels in `--claims-json` break the scaffold**, because the page folder name exceeds the
  Windows path limit (MAX_PATH) halfway through the build. Give short claim labels when scaffolding.
- **The artifact bucket name must be short.** The full name is `<username>/<slug>-artifacts`, and it is **a
  separate HF Buckets product**, not a dataset repository, with its own length limit. The project's working
  rule (tested in practice): keep the slug itself at 86 characters or fewer. CalPro broke this rule in its
  first publication on 18 July (the slug was 96 characters), so its bundle silently never uploaded from the
  start: no error was shown, the file just did not appear.
- **Check that a bundle was uploaded with `HfApi().bucket_info()` / `list_bucket_tree()`, NOT with
  `dataset_info()`.** Buckets are not dataset repositories; `dataset_info()` always returns 404 for a bucket
  id whether or not it has files. This nearly led to a false conclusion of "a systemic bug in all logbooks",
  until the right API showed that the other papers' bundles were in place.
- **Renaming an already published Space:** `HfApi().move_repo(repo_type="space")` moves the Space, but
  trackio's local sqlite database (`%HF_HOME%\trackio\<slug>.db`) stays under the old name and is not found
  automatically, which gives the error "No database found for project". Copy the file under the new name
  before republishing. Mentions of the old slug inside `logbook.json` and in page bodies (artifact links) also
  have to be replaced by hand; they are not renamed.
- **PyHealth 2.0 on Windows hung twice** while caching MIMIC through dask. The first diagnosis, "dask is
  broken on Windows", was wrong. The real cause: a missing `if __name__ == "__main__":` guard. Dask spawns
  workers, each re-imports the module and rebuilds the dataset, an endless recursion of processes. PyHealth
  printed a hint in its output, which was missed. In WSL Ubuntu-24.04 (Python 3.12.3, exactly PyHealth's
  version, no pinning hacks) with the guard, the whole pipeline ran in 6.8 seconds. **Rule: when something
  hangs, read the library's own warnings first and suspect your own harness, not the platform.**

---

# Session result (closed 2026-07-21)

Kate decided to close this thread and record the current result as final: **34 points over 11 published
logbooks**. Further work (few-shot examples from the leaderboard, new targets from the queue) was not
finished; see "What to do next" below if the session resumes.

# Published logbooks (11, real judged scores)

| Paper | Score | Logbook |
|---|---|---|
| SurvFD `SldP4LGjdz` | 6/12 | `repro-functional-decomposition-and-shapley-interactions-...` |
| Solvable AE `wm3ABfhE7P` | **4/4** | `repro-a-solvable-high-dimensional-model-...-structure-i` |
| KM-ARL `LhGxRnGmGJ` | 4/12 | `repro-accurate-evaluation-of-quickest-changepoint-detectors-...` |
| Conditional Coverage Diagnostics `vaApZm6MKM` | **4/4** | `repro-conditional-coverage-diagnostics-for-conformal-prediction` |
| Survival eval under censoring `Y9gsOEdaNE` | **4/4** | `repro-when-can-we-trust-survival-model-evaluation` |
| CalPro `3LRWjJTp0Y` | 3/12 | `repro-calpro-prior-aware-evidential-conformal-prediction-with-structure-aware` |
| MDRCP `Vv4XRZDMM0` | 2/12 | `repro-multi-distribution-robust-conformal-prediction` |
| Entropy `BUxrIaf7Zc` | 2/6 | `repro-on-revisiting-entropy-for-identifying-mislabeled-images` |
| DC-PnPDP `jEBkuuETjr` | 2/12 | `repro-plug-and-play-diffusion-meets-admm-dual-variable-coupling-for-robust-medical` |
| uqct `a6vCNSBBeq` | 2/4 | `repro-principled-confidence-estimation-for-deep-computed-tomography` |
| ConfSleepNet `fve4MEzeSp` | 1/6 | `repro-a-conflict-aware-evidential-framework-for-reliable-sleep-stage-classification` |

All on Kate's account, tagged `icml2026-repro` + `paper-<orid>`; the board finds them by tag.
Papers are not exclusive: independent confirmations only strengthen them.

---

# What to do next

## Sleep-EDF downloaded (2026-07-20), which unblocked ConfSleepNet

The full **Sleep-EDF Expanded** is in Downloads (8.7 GB, 153 Sleep Cassette recordings = SleepEDF-78, 44 Sleep
Telemetry recordings). These are 2 of the 4 datasets in the ConfSleepNet table (SleepEDF-20, SleepEDF-78,
MASS-SS3, SHHS), for our weakest logbook (1/6). MASS-SS3 and SHHS remain closed (they need separate DUA/NSRR
agreements). Given Cause 1 above: claims 2-3 should be written **without a hedge about the missing
MASS-SS3/SHHS inside the verdict itself**; put that in a separate note and state the verdict on the two
available datasets cleanly.

## Nearest datasets for the targets already planned

- **OpenNeuro `ds004504`** for **LERD** (`B5DAV1EA8Z`, neurodegeneration, Alzheimer EEG). Open without
  registration. The synthetic part of claim 1 is already reproducible without downloads.
- **TCIA LIDC-IDRI** for the second half of the **uqct** claim ("deep methods give narrower regions"). The
  first half (the classical path) is already confirmed and published (2/4).
- **ISIC + DeepDRiD** for **Entropy claim 3** (medical SOTA, untouched). ISIC is fully open; DeepDRiD may need
  registration.
- **RETFound weights (1.6 GB, HF) + fundus datasets (JSIEC/BRSET/EDDFS/RIADD)** for **EviScreen**
  (`FCVPVqRLDJ`). There is an eval-only path, `reproduce_directly`, straight from the authors.

Do not chase MFIDDR/DRTiD for ProConMV: the data are not the problem, but the authors' code and annotations
(their own contribution) are not released, so a full reproduction is blocked anyway.

## Queue without GPU (candidates from `claims_anchored.json`, not started)

- **Stable Localized CP** `lSMTccAN61` (12 points): synthetic n=30, std reduction of 31.2%/16.3% (Table 2);
  plus Theorem 4.7 on the coverage interval. Directly on the conformal interest.
- **ECL** `gFPPTokv9C` (12): Theorem 3.1 gives a **necessary and sufficient** condition for calibration under
  covariate shift (checkable by construction); Theorem 3.3 on unbiasedness of the mini-batch gradient.
- **Inverse Conformal Risk Control** `lV4tqcVIyx` (10): Proposition 3, monotonicity; a clean binary test on
  their own newsvendor/LP with an exact optimum.
- **Truncated Policy Gradient** `XEbD5cYpgV` (10): Proposition 1 is an exact algebraic identity at k=0, a
  guaranteed clean verdict.
- **Deep Single-Index Frechet Regression** `SCjiMmzyFA` (10): responses are SPD matrices (covariance or
  connectivity), exact numbers on a simulation with n=1000, q=3. Closest to Kate's own field.

## Blocked (with triggers for revisiting)

- **TCSeg** `bGxMpQsIuN`: the repository is README-only (0 .py files). Trigger: code release.
- **ProConMV** `4F4Ziv1d9G`: the repository is empty and the six lesion concepts are not released. Corollary 1
  is already checked (`reproductions/proconmv/verify_corollary1.py`, 99.4%), but both challenge claims are
  empirical.
- **ORBIT** `pd6GY2x8FG`: longitudinal fundus/OCT from Shanghai General Hospital, IRB, not released.
- **SzCORE** `3vhn9kVVY5`: the key numbers are on a closed held-out set.
- **S2M-Net**: withdrawn by the authors.

---

# Reference

## Finding medical papers

`medical_neuro_papers.csv` was built from the area tag and systematically misses medicine. A scan of all 6,341
titles gives 368 medical papers, 113 of them outside those 255 (Chemistry/Physics, Causality, LLM, CV, Time
Series, Evaluation, Privacy). **uqct, PyHealth and ProConMV were never among the 255.** Search the whole
`all_papers.csv`. Data availability is not visible from the abstract (only 18 of the 368 name it): shortlist by
fit, then read the PDF for the data, then decide.

## Environment

- Python 3.11, numpy 2.4.6, torch 2.13.0+cpu. **scipy and sklearn are not installed**: the normal CDF goes
  through `torch.erf`, and the "modern classifier" is a torch MLP.
- trackio 0.31.5: `C:\Users\kondr\AppData\Local\hermes\hermes-agent\venv\Scripts\trackio.exe`.
- Windows console: always `PYTHONIOENCODING=utf-8`. PDFs through pymupdf (`fitz`).
- OpenReview: curl and the API get 403 (Cloudflare); the built-in browser gets through.
- Reproduction folders: `reproductions/confsleepnet/`, `solvable-ae/`, `calpro/`, `uqct/`, `entropy/`, `ccd/`,
  `proconmv/`, `survfd/`, `survival-eval/`, `kmarl/`, `dcpnpdp/`, `mdrcp/` (all under `reproductions/`).
- References: `notes/PAPERS.md`; at the repository root, not committed: `all_papers.csv` (6,341),
  `medical_neuro_papers.csv` (255), `papers.json`, `claims.json` (vague), `claims_anchored.json` (precise,
  what is judged).

## Working rules and the canonical logbook format

At the time: logbooks in Russian, no em or en dashes, no plus signs in the text, no files created without an
explicit request, nothing published without Kate's word, Kate's own terms kept. **Reproduce from the paper's
text rather than inventing a plausible substitute**; if the paper has several cases, run all of them. Mark
`toy` honestly and name the blocker, but mark it **outside** the verdict sentence; see Cause 1 above.

**Before every publication, a separate subagent simulates the claim run from scratch** (fresh seeds, no access
to my conclusions) and checks whether the stated verdict holds. Do not publish until the simulation has
confirmed the number.

Canonical format (mandatory, checked by the validator): read
`https://huggingface.co/datasets/ICML-2026-agent-repro/challenge/resolve/main/README.md` before building.
Slug `repro-<slug-of-the-full-title>`. Title `Reproduction: <paper title>`. One logbook per paper, one page per
claim. Page order: Index / Executive summary (pinned summary + a `## Scope & cost` table + a poster with the
literal `poster_embed.html`) / Claim 1..N / Conclusion (the reproduction-bundle artifact).
Pipeline: scaffold, then work, then validate, then publish. The scripts `scaffold.py`/`validate.py`/
`finalize_bundle.py` were in the scratchpad; the logbook tools are now in `tools/`.
