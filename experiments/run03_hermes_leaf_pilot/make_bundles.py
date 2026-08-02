#!/usr/bin/env python3
"""
run03 harness: build per-paper input bundles + TASK.md for the
Hermes-leaf + hy3:free control, scaled across the 45-paper run pool.

Reads:
  groundtruth/out/selected_for_runs.json   (run pool: 45 papers)
  groundtruth/data/claims_anchored.json    (orid -> [ {text,status}, ... ])
  groundtruth/data/index.json              (orid -> meta: title, or, arxiv, area)

Writes (idempotent), under experiments/run03_hermes_leaf_pilot/<orid>/:
  input_bundle.json   identical schema to run02 (orid,title,openreview,arxiv,anchored_claims)
  TASK.md             the reproduction protocol
  results/            empty dir for the agent's raw numeric outputs
  _meta.json          run-pool metadata (area, disputed, space, hw_signals)

Also writes:
  _manifest.json      status per paper (built / skipped-no-anchored-claims)
  pilot_papers.json   stratified 5-paper pilot (one per chosen area)

No git, no router, no network writes. CPU-only by construction.
"""
import json, os, textwrap, sys

REPO = "/home/kate/projects/02_academia/icml-2026-repro"
POOL = os.path.join(REPO, "groundtruth/out/selected_for_runs.json")
ANCH = os.path.join(REPO, "groundtruth/data/claims_anchored.json")
IND  = os.path.join(REPO, "groundtruth/data/index.json")
OUT  = os.path.join(REPO, "experiments/run03_hermes_leaf_pilot")

TASK_MD = """\
# TASK: reproduce the anchored claims of one ICML 2026 paper (CPU only)

You are an autonomous reproduction agent. Work in the current directory.

## Input
`input_bundle.json` contains: paper title, OpenReview URL, arXiv id, and the list of
**anchored claims**. These claims are the task specification — do not invent your own.

## What to produce
1. `plan.md` — for each claim: is it reproducible on CPU within budget? theory (analytic/symbolic),
   simulation, or requires data/GPU. A correct refusal is worth more than a fake toy substitute,
   BUT see the data rule below — most "needs data" claims are reproducible once you download it.
2. Verification scripts (`verify_claim<N>.py`) that actually run. Python 3.12, CPU only.
   A pre-built `.venv` is provided with numpy, scipy, sympy, **torch (CPU build)**,
   scikit-learn, lightning, pandas — USE IT (activate via `source .venv/bin/activate` or run
   with `.venv/bin/python`). `pip install` is also allowed for anything extra.
3. `results/claim<N>.json` — raw numeric outputs of each run, with the exact command used.
4. `logbook.md` — per claim: `verdict` in {verified, falsified, toy, inconclusive}, the number(s)
   that justify it, the exact source location in the paper (theorem/section/equation number),
   and an **"Evidence boundary"** section at the end listing what your evidence does NOT cover.

## Environment & data (read before deciding a claim is impossible)
- **torch / scikit-learn are ALREADY INSTALLED** in `.venv`. Do NOT refuse a claim just because
  it needs deep-learning or sklearn code — use the provided env.
- **Datasets are NOT pre-bundled, but fetching them is PART OF THE TASK.** If a claim needs a
  dataset (FMoW/WILDS, CIFAR, ImageNet subsets, UCI, etc.), DOWNLOAD it yourself into `./data/`
  using `wget`/`curl`, `huggingface-cli`, `torchvision.datasets`, or the paper's official script.
  Only mark a claim `inconclusive` for DATA reasons if the data is genuinely **inaccessible**
  (paywalled, behind an approval you cannot obtain, or private). A claim reproducible after a
  download MUST be attempted, not skipped.
- Keep CPU-only: do not request GPU. Prefer small subsets / few epochs to fit the time budget.

## Hard rules
- **Mutation test**: for every claim you mark `verified`, also break the mechanism deliberately
  (invert a condition, replace a component with a naive alternative) and show the result changes
  as predicted. A number without a mutation test proves correlation, not mechanism.
- **Exhaustive enumeration over seeds** when the claim space is finite. Seeds are a fallback.
- Never mark a claim `verified` on a single seed.
- If data is unavailable, the verdict is `inconclusive` with the reason — NOT a synthetic toy
  standing in for real data.
- Do NOT search for or read other people's reproduction logbooks (HuggingFace Spaces tagged
  `icml2026-repro`). Reading the paper itself and its official code is allowed.
- **Time budget (HARD — exceeding this fails the run):** per paper, from your first tool call:
  **hard stop 8h** (you MUST stop at 8h, write `logbook.md` with a verdict or honest
  `inconclusive` for every attempted claim, and exit — incomplete is fine, silent over-run
  is not); **soft target 4h** (aim to have all CPU-feasible claims done + logbook drafted,
  spend the rest only on the hardest claims); **per-claim cap 2h** (any single claim,
  especially data/GPU-required: FMoW, large downloads, model training, gets at most 2h of
  attempt — then write `inconclusive` with the exact reason + evidence boundary, NOT a toy
  substitute, NOT a fabricated number). Priority: CPU-feasible claims (theory / small
  simulations) first, data/GPU claims last. A correct refusal of an infeasible claim is
  worth more than a fake toy result. Track elapsed time in `logbook.md`.

## Finish
Print a final summary table: claim number | verdict | one-line evidence.
Also write `check_reproducibility.py` that re-asserts every number quoted in `logbook.md`
against `results/*.json` (it is the reproducibility gate for this logbook).
"""

def main():
    pool = json.load(open(POOL))
    anch = json.load(open(ANCH))
    index = json.load(open(IND))

    idx_by_orid = {p["orid"]: p for p in index.get("papers", [])}
    papers = pool["papers"]

    manifest = {}
    built = 0
    skipped = []
    for p in papers:
        o = p["orid"]
        claims = anch.get(o)
        if not claims:
            skipped.append(o)
            manifest[o] = {"status": "skipped", "reason": "no anchored claims in GT"}
            continue
        meta = idx_by_orid.get(o, {})
        bundle = {
            "orid": o,
            "title": p.get("title") or meta.get("title"),
            "openreview": meta.get("or") or f"https://openreview.net/forum?id={o}",
            "arxiv": meta.get("arxiv"),
            "anchored_claims": [c["text"] for c in claims],
        }
        d = os.path.join(OUT, o)
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
        with open(os.path.join(d, "input_bundle.json"), "w") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
        with open(os.path.join(d, "TASK.md"), "w") as f:
            f.write(TASK_MD)
        with open(os.path.join(d, "_meta.json"), "w") as f:
            json.dump({
                "area": p.get("area"),
                "disputed": p.get("disputed"),
                "space": p.get("space"),
                "hw_signals": p.get("hw_signals"),
                "n_claims": p.get("n_claims"),
            }, f, indent=2, ensure_ascii=False)
        manifest[o] = {"status": "built", "area": p.get("area"),
                       "n_claims": len(bundle["anchored_claims"])}
        built += 1

    # stratified pilot: one per chosen area, first in pool order with anchored claims
    pilot_areas = ["Probabilistic Methods", "Reinforcement Learning",
                   "Deep Learning", "Optimization", "General Machine Learning"]
    pilot = []
    for area in pilot_areas:
        for o, m in manifest.items():
            if m.get("status") == "built" and m.get("area") == area and o not in pilot:
                pilot.append(o)
                break

    with open(os.path.join(OUT, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "pilot_papers.json"), "w") as f:
        json.dump({
            "n_total_pool": len(papers),
            "n_built": built,
            "n_skipped": len(skipped),
            "skipped_orids": skipped,
            "pilot_orids": pilot,
            "pilot_areas": {o: manifest[o]["area"] for o in pilot},
        }, f, indent=2, ensure_ascii=False)

    print(f"BUILT bundles: {built}/{len(papers)}")
    print(f"SKIPPED (no anchored claims): {len(skipped)} -> {skipped}")
    print("PILOT (5, stratified by area):")
    for o in pilot:
        print(f"  {o}  [{manifest[o]['area']}]  claims={manifest[o]['n_claims']}  "
              f"{manifest[o].get('title','')[:50]}")

if __name__ == "__main__":
    main()
