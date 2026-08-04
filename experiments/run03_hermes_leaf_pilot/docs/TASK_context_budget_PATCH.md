# PATCH for TASK.md — context & call budget (insert as a new section before "## Finish")

## Context budget (HARD — violating this fails the run)
- **Read the paper ONCE.** First action: read `paper/paper.txt` (NOT paper.html) and write
  `notes_paper.md` (≤ 400 lines) containing only: claim ↔ theorem/lemma/equation number,
  the exact formulas needed, hyperparameters, dataset names, and the reported numbers.
  After that, `notes_paper.md` is your ONLY source about the paper. Never re-read
  `paper/*` and never paste paper text into a prompt again.
- **Never load the official code wholesale.** Do not cat/read `mdcp_run/` or `code_mdcp/`
  as a whole (98 files, ~1.8 MB ≈ 450K tokens). Use grep/search for the specific symbol you
  need and read at most ±40 lines around the hit. Record what you learned in `notes_code.md`
  (≤ 200 lines). That file, not the repo, is your reference afterwards.
- **Per-message cap:** no single prompt may exceed ~25K tokens (~100K characters) of attached
  material. If you need more, summarize to a note file first.
- **No re-derivation:** once `results/claim<N>.json` exists and `check_reproducibility.py`
  passes for that claim, the claim is DONE. Do not re-run, re-verify or "improve" it.
- **Call budget:** ≤ 12 LLM turns per claim, ≤ 60 for the whole paper. Track your turn count
  in `logbook.md`. When the budget is spent, write up what you have and STOP.
- **Long-running official scripts:** do not stream their stdout into the conversation.
  Redirect to `results/<name>.log` and read only the last 50 lines
  (`tail -n 50 results/<name>.log`).
- Prefer one long tool call over many short ones: batch shell commands, since every turn
  resends the entire conversation.
