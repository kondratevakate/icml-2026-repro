# REVIEW handoff — ICML-2026 repro benchmark (2026-08-03)

Pushed to `kondratevakate/icml-2026-repro` branch `hermes-agentic-research`
(commits `0a595cf → 1563b5f → 3979dcf`).

## What changed
1. **Repository reorganized** into a scalable layout under `experiments/run03_hermes_leaf_pilot/`:
   - `pilot_4arm/` — CAffNet `20hdQQQrA4` run on all 4 arms (reference validation).
   - `benchmark_v1_prefed/` — 18 complete pre-fed runs (logbook + score present).
   - `benchmark_v1_arm2b/` — 10 compaction runs (frozen; compaction ruled OFF).
   - `runs_incomplete/` — 45 partial runs (no logbook/_score) — staged for later finalization,
     **NOT scored, NOT published**.
   - `docs/` — consolidated handoffs + validation tables.
   - `infra/` — pipeline scripts (`score_run.py`, `make_bundles.py`, `adapt_*.py`, `snapshot_usage.py`).
   - `DECISIONS.md` — locked decisions D1–D3.

2. **Decisions locked** (`DECISIONS.md`):
   - D1: arm2 (+skills) = primary workhorse; arm1 = control; arm3 (Sakana) & arm4 (ARC)
     documented non-viable at scale (evidence in `pilot_4arm/`).
   - D2: context compaction **OFF** (CAffNet clean pilot regressed 4→3 verified).
   - D3: pre-fed claims = primary scored condition; autonomous extraction = RQ8 (open).

3. **RQ8 corrected** in the design doc: the "no paper body" failure anecdote re `uiw8P2JGbW`
   is UNVERIFIED — that paper is actually 6/6 verified. The structural gap (OpenReview
   `arxiv_id=None` → empty `paper/`) is real but does not block math claims.

4. **Trackio reproducibility enforced**: every complete run + pilot arm now has
   `logbook.md` AND a regenerable `.trackio/logbook/logbook.json` (regenerated via
   `groundtruth/local_to_trackio.py` from `logbook.md`). The Trackio bundle is the
   canonical, machine-readable artifact others reuse; `logbook.md` is the judge-readable source.

5. **`.gitignore` hygiene**: regenerable Trackio static (html/js/css/svg), paper source
   HTML/PDF, large datasets (`/data`, openml, arff), and nested `.git` clones are excluded.
   `logbook.json` + `pages/*.md` are kept (they are the reproducible source-of-truth).

## For your review
- **Structurally sound?** Is the 4-arm design + pre-fed primary + RQ8-as-open the framing
  you want for the method paper? Or should autonomous extraction (Condition B) be promoted
  to a primary axis now (requires building the `extraction-GT` layer in `corpus_all`)?
- **The 45 `runs_incomplete/`**: these are agent outputs without a finalized logbook.
  Decision needed: finalize via rewriter delegate (write logbook from `results/`), or discard
  as abandoned pilots. Do NOT score/publish until resolved.
- **Compaction**: locked OFF on one clean paper (CAffNet). TBSyYj4VV6 was confounded
  (prompt also changed), so the rule rests on a single clean data point. OK to keep OFF, or
  want a 2nd clean (prompt-held) compaction pilot before locking?
- **Trackio coverage**: complete runs + pilot arms now 100% have the bundle. The 45
  incomplete runs do not (by definition).

## Reproduce any run
```
cd groundtruth && python3 local_to_trackio.py <run_dir>   # regenerates .trackio bundle
python3 score_run.py <run_dir>                            # re-scores from logbook.md
```

## Open items (not done, awaiting direction)
- Finalize or discard the 45 incomplete runs.
- Build `extraction-GT` layer if RQ8 promoted to primary.
- 2nd clean compaction pilot (optional, before fully locking D2).
