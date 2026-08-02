#!/usr/bin/env python3
"""
apply_time_budget_patch.py — inject the hard time-budget SLA (agreed 2026-08-01) into every
bundle's TASK.md. Idempotent: skips bundles that already have it.

SLA (per paper): hard stop 8h, soft target 4h, per-claim cap 2h (data/GPU claims).
Replaces the old "Budget: ~75 minutes" line if present.
"""
import os, glob

OUT = os.path.dirname(os.path.abspath(__file__))

SECTION = """
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
"""


def main():
    bundles = [d for d in os.listdir(OUT)
               if os.path.isdir(os.path.join(OUT, d)) and not d.startswith((".", "_"))
               and os.path.exists(os.path.join(OUT, d, "input_bundle.json"))]
    n = 0
    for o in sorted(bundles):
        tm = os.path.join(OUT, o, "TASK.md")
        if not os.path.exists(tm):
            continue
        t = open(tm).read()
        if "Time budget (HARD" in t:
            continue
        if "~75 minutes" in t or "75 minutes wall time" in t:
            t = t.replace("- Budget: ~75 minutes wall time. Prefer finishing 2-3 claims properly over 6 claims shallowly.\n",
                          SECTION.strip() + "\n")
        elif "## Finish" in t:
            t = t.replace("## Finish", SECTION.strip() + "\n\n## Finish", 1)
        else:
            t = t.rstrip() + "\n\n" + SECTION
        open(tm, "w").write(t)
        n += 1
    print(f"DONE. patched={n} / {len(bundles)} bundles (idempotent)")


if __name__ == "__main__":
    main()
