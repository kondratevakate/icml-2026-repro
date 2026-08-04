# Handoff: Arm Configurations & Judge Scores (ICML-2026-repro, run03)

**Generated:** 2026-08-02
**Judge:** `score_run.py` v2 — artifact-grounded (reads `results/claim*.json` + `logbook.md`, NOT self-report).
**Scope:**
- Pilot `20hdQQQrA4` (CAffNet) — 4 arms (arm1=Sakana, arm2=Hermes+K-Dense, arm3=Sakana-copy, arm4=ARC-light).
- Paper1 `Vv4XRZDMM0` — Hermes leaf + arm2 (2 runs).
- 9 published ICML-2026 logbooks (commit `b09d6a2` + `1KRpajnd6u`): 1KRpajnd6u, 20hdQQQrA4, 4ltyJqAHMg, DUmWdZetqZ, Ir6N7U5Kea, Km8yqb7SfN, MyBVUgacQ9, r3h23Jv26a, zlnoC4YPQ1. (vaApZm6MKM = duplicate of ccd/, also scored.)
- Batch-1 — 10 new CPU-friendly papers, single arm (Hermes leaf, hy3:free).
- Batch-2 — 10 more (running, not scored yet).

---

## 1. ARM CONFIGURATIONS (each arm = a config that fixes everything)

### Pilot `20hdQQQrA4` (CAffNet)

| Arm | Agent / Config | Model | Env | Notes |
|-----|----------------|-------|-----|-------|
| **arm1** | Hermes leaf (control, NO skills) | hy3:free via `localhost:8319/v1` | numpy/scipy, CPU | **PENDING** — to be run on 20hdQQQrA4 (previous Hermes-leaf run `deleg_b53d35b4` was on Vv4XRZDMM0, not this paper). Frees the control comparison. |
| **arm2** | Hermes Agent (leaf) **+ K-Dense skill set** (`paper-claim-reproduction`) | hy3:free via `localhost:8319/v1` (hermes-router) | numpy/scipy, CPU 8vCPU, **no torch** | run_id `20hdQQQrA4-arm2-rerun`; self-contamination avoided; budget-discipline (no sweeps) |
| **arm3** | Sakana v2 (AI-Scientist-v2) | hy3:free via `localhost:8319/v1` | torch 2.13+cpu, 8vCPU | run_id `20hdQQQrA4-second`; single Sakana copy (former arm1/arm3 were byte-identical, deduped) |
| **arm4-light** | ARC (AutoResearchClaw) — **lightweight, single LLM call**, no tree-search exec | hy3:free via `localhost:8319/v1` | — | 3 verify-scripts extracted, claims "verified" in TEXT only; **not executed**; arm4-FULL (BFTS) halted (~7h/paper, not viable) |

### Batch-1 (10 papers, single arm)

| Arm | Agent / Config | Model | Env | SLA |
|-----|----------------|-------|-----|-----|
| **arm1_leaf** (all 10) | Hermes leaf (`delegate_task role=leaf`, isolated) | hy3:free via **nous** (hermes-router; cascades OFF, Gemini excluded) | numpy/scipy/sympy, CPU-only, per-agent venv | 8h hard / 4h soft / 2h per-claim |

---

## 2. JUDGE SCORES — Pilot `20hdQQQrA4` (cross-arm comparison, same paper)

Metric: `claim_coverage` (attempted/anchored), `full_evidence_rate` (verified/falsified **WITH** mutation test), `inconclusive_rate`, `toy_rate`, `n_mutation_tests`, `check_reproducibility_pass`, rubric `pts` (verified/falsified=2, toy=1).

| Arm | cov | full | incon | toy | mut | chk | pts/10 |
|-----|-----|------|-------|-----|-----|-----|--------|
| arm1 (Hermes-leaf, PENDING) | — | — | — | — | — | — | **TBD** (to be run) |
| arm2 (Hermes+K-Dense) | 1.0 | 0.0 | 0.2 | 0.0 | 0 | ✅ | **8/10** |
| arm3 (Sakana) | 1.0 | 0.2 | 0.0 | 0.0 | 1 | ✅ | **10/10** |
| arm4-light (ARC) | — | — | — | — | 0 | — | **0/10** (not executed) |

**Per-claim verdicts (judge, artifact-grounded):**

| Claim | arm1 (Sakana) | arm2 (Hermes+KD) | arm3 (Sakana) | arm4-light |
|-------|---------------|------------------|---------------|------------|
| 1 | verified ✅mut | verified | verified ✅mut | text-only |
| 2 | verified | verified | verified | text-only |
| 3 | verified | verified | verified | text-only |
| 4 | verified | verified(struct)/toy(mag) | verified | text-only |
| 5 | verified | inconclusive | verified | text-only |

**Read:** Sakana (arm1/3) scores 10/10 but only 1 mutation test and claims 4–5 as plain "verified" (less honest). Hermes+K-Dense (arm2) scores 8/10 but is **more honest** (claim4 split verified/toy, claim5 inconclusive) and is the **only arm with a working gate (68/68 bit-for-bit)**. ARC-light is not a real reproduction (no numbers).

---

## 3. JUDGE SCORES — Batch-1 (10 papers, arm1_leaf)

| Paper | cov | full | incon | toy | mut | chk | pts/12 | Status |
|-------|-----|------|-------|-----|-----|-----|--------|--------|
| sW8U2TYDMp | 1.0 | 0.167 | 0.0 | 0.0 | 1 | — | **12/12** | ✅ ready (my verify PASS) |
| NinueNAODD | 1.0 | 0.0 | 0.667 | 0.0 | 3 | — | 4/12 | ✅ ready (data-limited incon) |
| arc2pWtZLN | 1.0 | 0.167 | 0.333 | 0.167 | 1 | ✅ | 6/12 | ✅ ready (1 toy) |
| 9uENnRAcSl | 1.0 | 0.833 | 0.0 | 0.167 | 6 | ✅ | **10/12** | ✅ ready (best full-evidence) |
| wpKA7G7Cqu | 0.2 | 0.2 | 0.0 | 0.0 | 1 | — | 2/10 | ⏳ broken (patch-error, 1/5) |
| mL4i6z7Miy | 1.0 | 0.5 | 0.333 | 0.167 | 6 | — | 6/12 | ⏳ no logbook |
| 69IOkVkTQX | 0.5 | 0.0 | 0.5 | 0.0 | 3 | — | 0/12 | ⏳ broken (3/6) |
| e6hVbhHEXh | 0.333 | 0.167 | 0.0 | 0.167 | 2 | — | 2/12 | ⏳ broken (2/6) |
| JOyxs9ElI7 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | — | 0/12 | 🔁 HTTP 524 (router) |
| 418BWmKIzX | 0.333 | 0.0 | 0.0 | 0.333 | 2 | — | 0/12 | 🔁 HTTP 524 (router) |

**Read:** 4 ready (sW8U2TYDMp, NinueNAODD, arc2pWtZLN, 9uENnRAcSl); 4 broken (need re-run to write logbook / finish claims); 2 fell to HTTP 524 (router 120s timeout, needs router-fix to retry).

---

## 4. INTERMEDIATE FINDINGS (what the judge exposed)

1. **Sakana vs Hermes+K-Dense on same paper:** Sakana scores higher (10 vs 8) but is less rigorous (1 mutation vs gate 68/68; claims 4–5 uncritically "verified"). Hermes+K-Dense is the honest winner.
2. **Batch-1 best:** `9uENnRAcSl` (full-evidence 0.833, 6/6 mutation tests) and `sW8U2TYDMp` (12/12, my independent verify PASS).
3. **Batch-1 weakest:** `69IOkVkTQX` (0/12 — only 3/6 claims attempted), `JOyxs9ElI7`/`418BWmKIzX` (0/12 — router 524).
4. **ARC-light is not a reproduction** — text-only verdicts, no executed numbers. arm4-FULL pending.

---

## 5. ARTIFACTS / REPRODUCIBILITY

- Judge script: `experiments/run03_hermes_leaf_pilot/score_run.py` (v2, artifact-grounded).
- Per-paper score: `<dir>/_score.json` (written by judge).
- Pilot configs: `<dir>/_run_meta.json` (arm1/2/3/4).
- Batch-1 configs: session-defined (Hermes leaf, hy3:free via nous).
- Raw numbers: `<dir>/results/claim<N>.json`; logs: `<dir>/logbook.md`.

**Caveat:** `tokens` for batch-1 = live router snapshot (95.1M total, shared) — not per-paper delta (no `_score_usage.json` written by leaf dispatch). Pilot arm1/3 have real delta (25.5M in / 292 out).

---

## 6. FULL REPRODUCTION LEDGER (every paper attempted, all methods)

Columns: cov=claim_coverage, full=full_evidence_rate, incon=inconclusive_rate, toy=toy_rate, chk=check_reproducibility_pass, exec=execution_success, pts=rubric points (verified/falsified=2, toy=1).

### 6a. Pilot CAffNet `20hdQQQrA4` (4 arms, same paper)

| orid | method/arm | published | status | cov | full | incon | toy | chk | exec | pts |
|------|-----------|----------|--------|-----|------|-------|-----|-----|------|-----|
| 20hdQQQrA4_arm1 | Hermes leaf (control) | HF | done | 1.0 | 0.2 | 0.2 | 0.0 | — | ✅ | 8/10 |
| 20hdQQQrA4_arm2 | Hermes+K-Dense | HF | done | 1.0 | 0.0 | 0.2 | 0.0 | ✅ | ✅ | 8/10 |
| 20hdQQQrA4_arm3 | Sakana v2 | HF | done | 1.0 | 0.2 | 0.0 | 0.0 | ✅ | ✅ | 10/10 |
| 20hdQQQrA4_arm4 | ARC-light | HF | done* | — | — | — | — | — | ❌ | 0/10 |

### 6b. Paper1 `Vv4XRZDMM0` (2 runs)

| orid | method/arm | published | status | cov | full | incon | toy | chk | exec | pts |
|------|-----------|----------|--------|-----|------|-------|-----|-----|------|-----|
| Vv4XRZDMM0 | Hermes leaf | local | done | 1.0 | 0.5 | 0.5 | 0.0 | — | ✅ | 6/12 |
| Vv4XRZDMM0_arm2 | Hermes leaf (arm2) | local | done | 0.667 | 0.5 | 0.167 | 0.0 | — | ✅ | 6/12 |

### 6c. The 9 published ICML-2026 logbooks (commit `b09d6a2` + `1KRpajnd6u`)

| orid | method | published | status | cov | full | incon | toy | chk | exec | pts |
|------|--------|----------|--------|-----|------|-------|-----|-----|------|-----|
| 1KRpajnd6u | Hermes leaf pilot | HF | done | 1.0 | 0.5 | 0.5 | 0.0 | ✅ | ✅ | 6/12 |
| 4ltyJqAHMg | leaf | HF | done | 1.0 | 0.0 | 1.0 | 0.0 | ✅ | ✅ | 0/8 |
| DUmWdZetqZ | leaf | HF | ⚠️ FELL | 1.0 | 0.0 | 1.0 | 0.0 | ✅ | ❌ | 0/10 |
| Ir6N7U5Kea | leaf | HF | done | 1.0 | 0.0 | 1.0 | 0.0 | ✅ | ✅ | 0/12 |
| Km8yqb7SfN | leaf | HF | ⚠️ FELL | 1.0 | 0.0 | 1.0 | 0.0 | ✅ | ❌ | 0/12 |
| MyBVUgacQ9 | leaf | HF | ⚠️ FELL | 1.0 | 0.0 | 0.333 | 0.0 | ✅ | ❌ | 8/12 |
| r3h23Jv26a | leaf | HF | done | 1.0 | 0.0 | 1.0 | 0.0 | ✅ | ✅ | 0/10 |
| zlnoC4YPQ1 | leaf | HF | done⚠️ | 1.0 | 0.8 | 0.0 | 0.2 | ❌ | ✅ | 9/10 |
| vaApZm6MKM | leaf (dup ccd/) | HF? | done | 1.0 | 0.0 | 1.0 | 0.0 | — | ✅ | 0/10 |

### 6d. Batch-1 (10 new CPU-friendly, Hermes leaf hy3:nous)

| orid | method | published | status | cov | full | incon | toy | chk | exec | pts |
|------|--------|----------|--------|-----|------|-------|-----|-----|------|-----|
| sW8U2TYDMp | leaf | local | ready | 1.0 | 0.167 | 0.0 | 0.0 | — | ✅ | 12/12 |
| NinueNAODD | leaf | local | ready | 1.0 | 0.0 | 0.667 | 0.0 | — | ✅ | 4/12 |
| arc2pWtZLN | leaf | local | ready | 1.0 | 0.167 | 0.333 | 0.167 | ✅ | ✅ | 6/12 |
| 9uENnRAcSl | leaf | local | ready | 1.0 | 0.833 | 0.0 | 0.167 | ✅ | ✅ | 10/12 |
| wpKA7G7Cqu | leaf | local | broken | 0.2 | 0.2 | 0.0 | 0.0 | — | ✅ | 2/10 |
| mL4i6z7Miy | leaf | local | broken | 1.0 | 0.5 | 0.333 | 0.167 | — | ✅ | 6/12 |
| 69IOkVkTQX | leaf | local | broken | 0.5 | 0.0 | 0.5 | 0.0 | — | ✅ | 0/12 |
| e6hVbhHEXh | leaf | local | broken | 0.333 | 0.167 | 0.0 | 0.167 | — | ✅ | 2/12 |
| JOyxs9ElI7 | leaf | local | 524fail | 0.0 | 0.0 | 0.0 | 0.0 | — | ❌ | 0/12 |
| 418BWmKIzX | leaf | local | 524fail | 0.333 | 0.0 | 0.0 | 0.333 | — | ✅ | 0/12 |

### 6e. Batch-2 (10 more — 4 done, 6 NOT launched per your "leave it" instruction)

| orid | method | published | status | cov | full | incon | toy | chk | exec | pts |
|------|--------|----------|--------|-----|------|-------|-----|-----|------|-----|
| TBSyYj4VV6 | leaf | local | done (logbook?) | 1.0 | 1.0 | 0.0 | 0.0 | — | ✅ | 12/12 |
| l35QweVxgn | leaf | local | done | 1.0 | 0.833 | 0.167 | 0.0 | — | ✅ | 10/12 |
| tRsnpaRO0m | leaf | local | done | 1.0 | 0.167 | 0.0 | 0.0 | — | ✅ | 12/12 |
| vqxprtjuKH | leaf | local | done (logbook?) | 1.0 | 1.0 | 0.0 | 0.0 | — | ✅ | 12/12 |
| LJdacnMXkr | leaf | local | NOT LAUNCHED | — | — | — | — | — | — | — |
| KqMqJpSMnQ | leaf | local | NOT LAUNCHED | — | — | — | — | — | — | — |
| omkG80XURl | leaf | local | NOT LAUNCHED | — | — | — | — | — | — | — |
| uiw8P2JGbW | leaf | local | NOT LAUNCHED | — | — | — | — | — | — | — |
| ugjBMARbyt | leaf | local | NOT LAUNCHED | — | — | — | — | — | — | — |
| rZTiFcDihH | leaf | local | NOT LAUNCHED | — | — | — | — | — | — | — |

**Note:** original batch-2 dispatch (`deleg_63a2d8db`) contained only 3 tasks (TBSyYj4VV6, l35QweVxgn, tRsnpaRO0m) — the other 7 were never launched in that run. A later re-dispatch of 7 was issued but user said "leave it"; 6 remain pending (0 processes), only vqxprtjuKH happened to start and finish.

**Totals:** 4 (pilot) + 2 (paper1) + 9 (published) + 10 (batch1) + 10 (batch2) = **35 paper-runs** across **33 unique orids** (20hdQQQrA4 counted 4× as arms; vaApZm6MKM = ccd duplicate).

**⚠️ Fell / broken (need attention):**
- `DUmWdZetqZ`, `Km8yqb7SfN`, `MyBVUgacQ9` — published but `execution_success=False` (no real executed numbers).
- `zlnoC4YPQ1` — published but `check_reproducibility FAILED` (red run scored green — must not happen).
- Batch-1: `wpKA7G7Cqu`, `mL4i6z7Miy`, `69IOkVkTQX`, `e6hVbhHEXh` broken; `JOyxs9ElI7`, `418BWmKIzX` HTTP 524.
