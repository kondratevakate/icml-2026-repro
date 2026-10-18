# Protocol: how the ICML 2026 agent reproduction challenge aggregated verdicts

Frozen 2026-10-18, before any logbook page is downloaded and before any analysis
below is run. Git commit and file hashes give provenance, not external
preregistration. Target: a methods paper on vote aggregation in crowd
reproduction, plus a note to the organisers. Not about medicine; independent of
the PSE9 analyses in `verdict_audit/` (which are reused only as code).

## Background (exploratory, not confirmatory)

Earlier exploratory runs on the same snapshot, not frozen, gave: inter-team
kappa about 0.23; Spearman-Brown 8-13 teams for reliability 0.7-0.8; pairwise
agreement falling with creation-time gap; an unconverged 4-class Dawid-Skene
calling the Frank-Wolfe claim verified with posterior 1.0. These motivate the
questions. None of them is a result of this protocol, and none is reused.

## Questions

Q1. How often are logbooks of the same claim near-duplicates of each other?
Q2. After collapsing near-duplicates, how many independent teams does a claim
    have, and what are the inter-team kappa and the number of teams needed?
Q3. Which agents and models do logbooks name as the producer of the work?
Q4. Do simple majority and binary Dawid-Skene agree with the organisers on the
    publicly confirmed falsifications?

## Inputs

| File | SHA-256 | Role |
|---|---|---|
| `data/verdicts_confirmation_2026-10-17/verdicts.json` | `652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526` | All verdicts (6,885 logbooks, one judge `zai-org/GLM-5.2`) |
| `data/spaces_2026-10-17/spaces.json` | `71fbe96bbc5d543a55aa69d19701fa5dbc34c2e6e0019719a154f41b53aa3c5a` | Space `createdAt` (6,677 Spaces) |
| `claims_anchored.json` | `eb3f2d878646ca5c40da121741a613c1f9e1e10c14845f373db107e74a4ef439` | Anchored claim texts |
| `verdict_audit/voters/reference_falsifications.csv` | hash in manifest | Organiser-confirmed falsifications (Q4) |
| Logbook pages | per-file hash in `pages_manifest.json` | Downloaded after Kate's approval (below) |

The scripts refuse to run if a hash differs. Paths are relative to the
repository root.

## Shared definitions (reused from `verdict_audit/analyze.py` unchanged)

* Logbook: one verdict record keyed by `space_id`. Team: the namespace before
  `/`. Paper: `orid`. Deduplication: `analyze.dedupe` (latest `judged_at` per
  team and paper). On the snapshot this keeps 6,709 logbooks (176 dropped).
* Verdicts: `analyze.category` (verified, falsified, toy, inconclusive, other).
  Decisive = verified or falsified.
* Claim identity: (paper, `analyze.norm` of the anchored claim text).
  Multi-rated claim: an anchored claim judged by at least two distinct teams
  (`analyze.multi_rated`). On the snapshot: 4,992 claims, 926 papers, 5,333
  logbooks.
* Agreement: `analyze.agreement` (Fleiss-type kappa, pooled marginals, paper
  cluster bootstrap, B = 2000, seed 20261017).

## Download (requires Kate's approval; not done at freeze)

For each of the 5,333 logbooks in multi-rated claims: `snapshot_download` of
the Space at revision `sha` from `verdicts.json`, restricted to `logbook.json`,
`pages/*.md` and `pages/**/*.md`. Files go to `verdict_audit/voters/data/`
(git-ignored). `pages_manifest.json` records, per Space, status (ok, missing,
error), file list and SHA-256 of every file. Failures are retried once and then
reported. No page content is committed; only aggregates are.

## Q1 Near-duplicate detection

* Text per logbook: the concatenation, in sorted path order, of pages whose
  directory name starts with `claim` (case-insensitive). If a logbook has no
  such page, all pages are used and the logbook is counted as `no_claim_pages`.
  Text is normalized with `analyze.norm`.
* Features: the set of word 5-shingles. Logbooks with fewer than 50 shingles are
  `too_short`: not tested, treated as independent, counted.
* Similarity: exact Jaccard of the shingle sets. (The brief named MinHash. MinHash
  estimates this same Jaccard; at this scale the exact value is cheap, so no
  estimation error is added. Deviation recorded here.)
* Candidate pairs: logbooks of different teams on the same paper that share at
  least one multi-rated claim.
* Threshold: tau = 0.5, fixed now. Template control: Jaccard on 20,000 random
  pairs of logbooks on different papers (seed 20261017). If the 99.9th
  percentile of this control is at or above 0.5, tau becomes the smallest of
  0.6, 0.7, 0.8, 0.9 that exceeds it. If none does, Q1 and the copy-collapsed
  parts of Q2 are reported as not evaluable.
* Copy pair: Jaccard >= tau. Direction: the logbook with the earlier Space
  `createdAt` is the source. Missing or equal `createdAt`: undirected.
* Reported: tau used and control quantiles; number and share of candidate pairs
  flagged; share of multi-rated claims with at least one copy pair; exact
  verdict agreement in copy pairs versus non-copy candidate pairs (descriptive);
  Jaccard distribution summary. Sensitivity (descriptive only): tau 0.3 and 0.8;
  containment |A and B| / min(|A|, |B|) at tau instead of Jaccard.

## Q2 Effective teams, kappa, teams needed

* For each multi-rated claim, build a graph on its rating logbooks with copy
  pairs as edges. Each connected component is one independent team. Effective
  teams = number of components. The component's rating is the verdict of its
  earliest-created logbook (missing `createdAt` sorts last; ties by `space_id`).
* Reported: mean and distribution of nominal versus effective teams per claim;
  kappa (`analyze.agreement`, permutation off) on the uncollapsed and collapsed
  ratings, collapsed restricted to claims with >= 2 components; both also on the
  binary decisive outcome (from the same function).
* Teams needed (Spearman-Brown), with kappa used as single-team reliability r:
  n = R(1 - r) / (r(1 - R)) for R = 0.7 and 0.8, at the kappa point estimate and
  at both CI ends. This treats kappa as an intraclass reliability; it is an
  approximation and is labelled as such.
* Secondary (descriptive): pairwise exact agreement between logbooks of the
  same claim by `createdAt` gap (< 1 day, 1-7 days, > 7 days), with and without
  copy pairs. If agreement still falls with gap after removing copies, copying
  does not explain the gradient on its own.
* Sensitivity: exclude organiser namespace `ICML-2026-agent-repro`; exclude
  the author's namespace `kondratevakate` (see Evidence boundary).

## Q3 Agent and model mentions

* Text: all downloaded pages and the `logbook.json` title of each logbook.
* Patterns (case-insensitive, word boundaries), fixed now in `patterns.py`:
  agent tools (Claude Code, Codex, Cursor, Gemini CLI, Aider, OpenHands, Devin,
  GitHub Copilot, Cline, Windsurf, Manus, Kimi CLI, Qwen Code, OpenCode, Goose,
  Amp) and model families (Claude / Opus / Sonnet / Haiku, GPT-4 / GPT-4o /
  GPT-4.1 / GPT-5, o1 / o3 / o4-mini, Gemini, GLM, Qwen, DeepSeek, Kimi, Llama,
  Mistral, Grok).
* A model-family match is ambiguous: many papers study these models. Validation
  decides what is reported:
  - Sample A: 120 matches (60 tool, 60 model family, seed 20261017), each shown
    as a 400-character window. Coders label `producer` (names the agent or model
    that did the reproduction), `subject` (the model is studied or used as an
    experimental component), or `unclear`.
  - Sample B: 40 logbooks with no match. Coders read `index.md` and the
    executive summary and label whether a producer agent or model is named.
  - Two blind coders (Claude subagents; no access to each other, to the regex
    category, to verdicts, or to team names, which are masked). Agreement:
    Cohen's kappa. A match counts as producer only if both coders say producer.
  - A category's prevalence is reported only if its producer precision in
    sample A is at least 0.80. Otherwise only the raw match count is shown, with
    a "not validated" label. Sample B gives the miss rate.
* Reported: share of logbooks naming a producer agent or model, by category.
  No link to verdicts is tested (descriptive only).

## Q4 Aggregation on confirmed falsifications (descriptive)

* Reference set, fixed now in `reference_falsifications.csv` from the organisers'
  blog post <https://huggingface.co/blog/icml-2026-open-reproductions>
  (retrieved 2026-10-18): paging `ESa07RwpVr`, Frank-Wolfe `zrn7rRuvhW`,
  self-distillation `qA6FgH0nnZ`, three projections `XGxBnVqK63`. The mapped
  anchored claim is listed with a confidence flag; for Frank-Wolfe the blog's
  claim wording does not match any anchored text one to one, so all anchored
  claims of the four papers are reported, with the mapped claim marked.
* Majority: on decisive votes of the copy-collapsed ratings; verified if
  verified > falsified, falsified if fewer, tie otherwise; also on uncollapsed.
* Binary Dawid-Skene on decisive votes of the copy-collapsed ratings, all
  anchored claims with >= 1 decisive vote. One sensitivity and one specificity
  per team, one class prior; Beta(2, 2) pseudo-counts on every parameter;
  initialised from majority (ties 0.5); EM until the largest posterior change
  is below 1e-6 or 1,000 iterations. Convergence and iterations are reported;
  an unconverged run is reported as unconverged, not used.
* Reported per reference claim: nominal votes by category, collapsed votes,
  majority, DS posterior P(verified). No test; four claims cannot support one.
  Agreement with the organisers is stated per claim.

## Prespecified interpretation

* Copying is called "material" only if, after collapsing, the mean number of
  teams per multi-rated claim falls by at least 10% AND the collapsed kappa
  differs from the uncollapsed one by more than 0.05 in either direction.
  Otherwise the numbers are reported without that label.
* Q3 and Q4 are descriptive; no labels.

## Evidence boundary

* All verdicts come from one LLM judge. Agreement is agreement of judged
  logbooks, not of human experts.
* Text similarity finds copied text. It does not find a team that read others'
  verdicts and then ran its own experiments, choosing claims or settings from
  what it read. Kate disclosed doing this for some of her own logbooks (53 in
  the deduplicated set). Effective team counts are therefore an upper bound on
  independence; her namespace is excluded in a sensitivity run and her own
  logbooks are the only ones named in any output.
* Shared agent templates produce similar text without copying; the template
  control and the containment sensitivity address this only partly.
* Only aggregates are committed. No other team is named or ranked.

## Amendment 1 (2026-10-18, during download, before any analysis was run)

Mechanics only; no analysis choice changes.

* `snapshot_download` failed on Windows because long page slugs exceed the path
  length limit, and the first bulk run hit the Hugging Face API rate limit
  (1,000 calls per 300 s). `download.py` now lists files with
  `list_repo_files` at revision `sha` (throttled to 2.5 calls per second) and
  fetches each file from its `resolve/<sha>/<path>` URL. Each file is stored
  under `data/pages/<sha1(space_id)[:16]>/<sha1(path)[:16]>.txt`; the manifest
  maps repository paths to stored files and their SHA-256, and readers verify
  every hash. The set of files is the one prespecified: `logbook.json` and every
  `pages/**.md`.
* Some logbooks keep pages as flat files (`pages/<slug>.md`) instead of
  `pages/<slug>/page.md`. "Directory name starts with `claim`" is applied to the
  first path component under `pages/` in both layouts.
