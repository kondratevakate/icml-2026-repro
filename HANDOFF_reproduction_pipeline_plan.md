# Reproduction Pipeline Plan — arm2 + context-compaction, cross-branch + HF Space sync

**Generated:** 2026-08-02 (session: pilot CAffNet 4-arm comparison closed, full ledger scored)
**Author:** Hermes (main) + user direction
**Status:** PLAN / NOT YET EXECUTED — scope confirmed, blockers identified

---

## 1. Conclusion from what we already computed

Single LLM model across all 4 arms: `tencent/hy3:free` via `localhost:8319/v1` (local router).
The variable was **architecture**, not model. So "one model to recompute everything" =
standardize on **one architecture** for all papers; the model stays hy3:free (already $0).

Ledger (all 33 paper-runs re-scored from artifacts via `score_run.py` v2):

| Architecture | Pilot pts/10 | Rigor | Speed on hy3:free |
|---|---|---|---|
| Sakana v2 (BFTS) | 10/10 | weak (1 mutation, claims 4-5 "verified" w/o caveat) | **~7h/paper** |
| Hermes leaf (control) | 8/10 | **strong** (found claim4 falsification) | ~30-45 min |
| Hermes + K-Dense skills (arm2) | 8/10 | **honest** (claim4 split, claim5 inconclusive) | ~20-30 min |
| ARC-light | 0/10 | does not reproduce | seconds (empty) |

**Myth busted:** "Sakana is guaranteed-best" is false. Sakana scores higher but is less
rigorous; Hermes-leaf found a falsification Sakana missed. Sakana is also **non-viable at
scale** (33 papers × 7h = 231h).

**Decision (user):** standardize the recompute on **arm2 (Hermes + K-Dense skills) + context
compaction**, then sync to HF Space if local result is better than what's published there.

---

## 2. Scope (as understood)

1. Recompute **every reproduced work**, including papers living in **other branches**:
   - `master` — has 9 published ICML-2026 logbooks + 7 Codex logbooks (commits `b09d6a2`, `02df07c`)
   - `codex/medical-reproducibility-map` — medical domain maps
   - `hermes-run03-arm3-sakana-pilot` — Sakana pilot
   - `hermes-agentic-research` (current) — pilot 4-arm + batch-1/2
2. After each paper is re-reproduced locally, **compare against the published HF Space**
   (`.trackio` folders) and **update the Space if local is better** (more honest / fuller).

**Already-published Spaces (11 `.trackio` folders in repo root):**
calpro, ccd, confsleepnet, dcpnpdp, entropy, kmarl, mdrcp, solvable-ae, survfd,
survival-eval, uqct.

---

## 3. BLOCKERS (must resolve before mass run)

- **HF token IS available.** Found at `/home/kate/.cache/huggingface/token` (37 bytes).
  `huggingface_hub` is not installed in system python (only needs `pip install huggingface_hub`
  in the run venv). Space sync is UNBLOCKED — no user-provided token needed.
- **Compaction on arm2 DOES save tokens.** At 33+ papers the leaf transcript grows large;
  summarizing history before each LLM call cuts router token volume (shared 95M limit) and
  reduces empty-tool-call retries. Agreed: add compaction to arm2 (keeps executed numbers
  verbatim, compresses only discussion/prose).
- **`publish_logbooks.py` assumes a branch + `.trackio` structure** — needs the local
  re-run to also emit `.trackio/logbook/` (or an adapter) before it can push.
- **Router rate-limit (HTTP 524)** already hit batch-1 (JOyxs9ElI7, 418BWmKIzX). Mass runs
  need retry/backoff or the router will choke.

---

## 4. Pipeline design (proposed, not yet built)

For ONE paper `<orid>`:
1. `delegate_task role=leaf` + inject K-Dense `paper-claim-reproduction` skill (arm2 setup).
2. Optional compaction wrapper: summarize agent transcript history before each LLM call,
   preserve `results/claim*.json` verbatim.
3. Agent writes `verify_claim*.py`, runs them, writes `logbook.md` + `_run_meta.json`.
4. `score_run.py <dir>` → `_score.json` (artifact-grounded).
5. Compare `_score.json` to the published Space (fetch `metadata.json` + `_score.json` from
   the corresponding `.trackio` folder / HF Space).
6. If local `full_evidence_rate`/`inconclusive_rate`/verdicts are **strictly better** →
   `publish_logbooks.py <dir>` (requires HF token) to update Space.

Roll out: **test on 1 paper first** (pilot CAffNet 20hdQQQrA4 — already has local arm2 and a
Space), then batch in background (night window) only after the pipeline is verified honest.

---

## 5. What is DONE vs TODO

**DONE:**
- Pilot 20hdQQQrA4: 4 arms compared, arm1/2/3 real, arm4-light executed partially.
- Full ledger (section 6a-6e) re-scored from artifacts; table built.
- Variant B naming locked: arm1=Hermes-leaf, arm2=Hermes+skills, arm3=Sakana, arm4=ARC.

**TODO (blocked/in-progress):**
- [ ] Obtain HF token (user) → unblock Space sync.
- [ ] Build compaction wrapper for arm2 leaf loop; verify it does not lose precision.
- [ ] Build the per-paper pipeline script (run → score → compare → publish).
- [ ] Test pipeline on 1 paper (20hdQQQrA4).
- [ ] Roll out to master/codex/* branches + remaining batches (background, night).

---

## 6. Notes / caveats

- "Recompute everything in other branches" = **full corpus rerun**, not an append. Estimate:
  33+ papers × ~25 min = 14h+ pure compute (excl. router limits, excl. other-branch papers).
- Do NOT retag Sakana/BFTS as "the model" — it is an architecture, and it is the slow one.
- arm4 (ARC) excluded from recompute (does not reproduce; lightweight only).
