# notes_code.md

**No official code.** The paper (arXiv 2602.03972 / OpenReview DUmWdZetqZ) is pure theory: it
contains no experimental section, no reproducibility statement, and no link to a code repository
anywhere in the main text or appendices (searched the extracted text for "github", "code",
"implementation", "repository" — no artifact link). There was therefore nothing to grep.

Consequences recorded for the evidence boundary:
- Algorithms 3 (FC2FB), 4 (FCW2S), 5 (PE-KHN) and 6 (FC2AT) in `repro_lib.py` /
  `verify_claim*.py` are transcriptions of the paper's pseudocode as summarised in
  `notes_paper.md`, not ports of author code. A misreading of the pseudocode would not be
  detected by any of the tests.
- Two pseudocode ambiguities were resolved as follows (both stated in the paper itself):
  1. Algorithm 4: once an instance self-terminates it receives no further samples, and the
     terminated set is tracked separately (clarified by the paper in Appendix C).
  2. Algorithm 3 / 6: "run A(delta) with the budget limit B'" is implemented as force-termination
     without any output when the inner algorithm has not self-terminated by B' samples
     (Algorithm 3 line 7).
- External algorithms referenced but not defined in this paper — Fixed Budget Peace
  (Katz-Samuels et al. 2020), UniTT (Poiani et al. 2024), VD-BESTARMID (Lu et al. 2023), SHVar /
  SHAdaVar (Lalitha et al. 2023), VBR (Faella et al. 2020), OD-LinBAI (Yang & Tan 2022) — were
  NOT implemented. Their papers were not opened. Claims resting on them are marked inconclusive.
