#!/usr/bin/env python3
"""
apply_context_budget_patch.py — inject Renat's context/call budget into every bundle's
TASK.md so the leaf agent stops burning tokens on paper+code wholesale + re-runs.

Adapted from Renat's TASK_context_budget_PATCH.md (2026-08-01) to be bundle-agnostic
(the original referenced Vv4-specific paths paper/paper.txt, mdcp_run/ — generalized
here so it applies to all 42 bundles). Idempotent: skips bundles that already have it.

This is LOCAL (no router/config change, no LLM calls). Router cascade + gemini fixes
are a SEPARATE escalation (infra + paid Bill review) and NOT done here.
"""
import os, glob

OUT = os.path.dirname(os.path.abspath(__file__))

SECTION = """
## Context & call budget (HARD — violating this fails the run)
- **Read the paper ONCE.** First action: read the paper text (NOT the HTML if a .txt
  exists) and write `notes_paper.md` (<= 400 lines) containing only: claim <-> theorem/
  lemma/equation number, the exact formulas needed, hyperparameters, dataset names, and
  the reported numbers. After that, `notes_paper.md` is your ONLY source about the paper.
  Never re-read the paper file and never paste paper text into a prompt again.
- **Never load the official code wholesale.** Do not cat/read the paper's code repo as a
  whole. Use grep/search for the specific symbol you need and read at most +/-40 lines
  around the hit. Record what you learned in `notes_code.md` (<= 200 lines). That file,
  not the repo, is your reference afterwards.
- **Per-message cap:** no single prompt may exceed ~25K tokens (~100K chars) of attached
  material. If you need more, summarize to a note file first.
- **No re-derivation:** once `results/claim<N>.json` exists and `check_reproducibility.py`
  passes for that claim, the claim is DONE. Do not re-run, re-verify or "improve" it.
- **Call budget:** <= 12 LLM turns per claim, <= 60 for the whole paper. Track your turn
  count in `logbook.md`. When the budget is spent, write up what you have and STOP.
- **Long-running official scripts:** redirect stdout to `results/<name>.log` and read only
  the last 50 lines (`tail -n 50 results/<name>.log`). Do not stream their output into
  the conversation.
- Prefer one long tool call over many short ones: batch shell commands, since every turn
  resends the entire conversation.
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
        if "Context & call budget (HARD" in t:
            continue
        # insert before the "## Finish" section (last one) if present, else append
        if "## Finish" in t:
            t = t.replace("## Finish", SECTION.strip() + "\n\n## Finish", 1)
        else:
            t = t.rstrip() + "\n\n" + SECTION
        open(tm, "w").write(t)
        n += 1
    print(f"DONE. patched={n} / {len(bundles)} bundles (idempotent)")


if __name__ == "__main__":
    main()
