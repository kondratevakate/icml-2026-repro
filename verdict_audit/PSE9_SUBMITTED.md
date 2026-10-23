# PSE9 submissions (9th Perspectives on Scientific Error, Copenhagen 2027)

Submitted by Kate on 2026-10-23 (deadline 2026-10-23, midnight CET) through the workshop form linked from
https://errorsin.science/. Format requested: talk (organisers consider declined talks for a poster). Notification
expected mid-November 2026. Texts below are as entered in the form; contact and fee-waiver answers are not recorded.

## Abstract 1 (voters study, `voters/`)

Author: Ekaterina Kondrateva (Hertie and Maastricht University; exact wording as entered by Kate in the form). Field:
Computer Science.

From scarcity to abundance: when AI agents reproduce every paper, who decides who is right?

For years the reproducibility crisis was a problem of scarcity: reproductions were costly, so most published claims
were never checked. Agentic reproduction removes that scarcity, and the bottleneck moves from running the checks to
deciding between them. In the ICML 2026 agent reproduction challenge, a single LLM judge scored public reproduction
logbooks claim by claim; Umoh [1] showed that the judge is stable on fixed input (Krippendorff's alpha 0.874) while
verdicts on the same claim from different logbooks agree poorly (alpha 0.229). On the public verdict snapshot (6,885
logbooks; 4,992 claims rated by at least two teams), disagreement is worst exactly where it matters to an author:
when one team reports a claim falsified, another team on the same claim also reports it falsified 30% of the time,
against 69% for verified. Pooling the verdicts erases most falsifications: simple majority and a binary Dawid-Skene
model recover one and none, respectively, of the four falsifications the organisers confirmed, and in three of the
four the judge had marked the logbook the organisers cite as falsified, only to be outvoted. Later logbooks that
report a falsification against the first logbook's verdict are rated higher quality by the judge, yet the final
plurality sides with them in only 35% of 834 such pairs. Cheap reproduction does not end the crisis; it turns it
into a problem of adjudication. We propose adjudication instead of votes for any claimed falsification, verdicts on
several axes instead of one label, success criteria fixed before launch, and independence and provenance enforced
by design.

References:
[1] Umoh EH. Who verifies the verifier? Measuring LLM-as-judge reliability across 32,688 automated research
verdicts [poster]. AWS-MLU Fall AI Teaching and Research Symposium; 2026 Sep 22; Arlington, VA. Available from:
https://github.com/SturdivantAI/who-verifies-the-verifier

## Abstract 2 (medical papers, `readiness/`, Protocol 9 and Notes 9-9 to 9-15)

Authors: Ekaterina Kondrateva (Hertie AI Institute for Brain Health, University of Tübingen; Maastricht University),
Florian Kofler (Hertie AI Institute for Brain Health, University of Tübingen; University of Zurich; Technical
University of Munich; Helmholtz Munich), Attila Simkó (Umeå University). Field: Medicine & Public Health.
The final paragraph was edited in the form by Kate after this text was drafted; the form copy is authoritative.

Well documented, rarely picked: medical AI papers under agentic reproduction

Medical machine-learning papers are often assumed to be hard to reproduce because their data are restricted and their code is not shared. We tested this across nine venue-years of CVPR, ICCV, ECCV, ICLR, ICML and NeurIPS (2025-2026): all 2,170 papers that two blind coders labelled clinical or life science against an equal random sample of other papers, with two blind LLM coders reading the full texts of 3,278. On paper, life-science papers were the better-documented group: code released 59% versus 46% (+13 points, 95% CI +10 to +17), main data openly downloadable where access was stated (+15, CI +11 to +19), variance reported 66% versus 42% (+24, CI +21 to +28). These held for each coder, without any one venue-year, and on untruncated texts.

In practice, they were settled less often. In the ICML 2026 agent reproduction challenge, attempted medical papers received a verified or falsified verdict 19 points less often than 1,821 other attempted papers (CI -28 to -10; permutation p < 0.001), mainly clinical papers. The judge was not the cause: within the same team, claims of medical and other papers were decided about equally often (-3.6 points, CI -7.8 to +0.8). Post hoc, two factors accounted for the gap. Medical papers were first taken up by teams that settled fewer of their other papers (40% versus 60%); adjusting for this team-level verdict rate removed the gap (-3.6, CI -12 to +5). Their claims were also costlier to check: blind coding of 1,083 claims found retraining or large-scale runs in 57% versus 39% (+18, CI +10 to +26). The gap in full verification, -19 points, fell to -5.6 (CI -12 to +1) after both. Stated practices also predicted poorly what reproducers obtained: a code link agreed with running the authors' code at kappa 0.22.

Crowd reproduction does not fail medical papers for lack of materials; it sends them to fewer and less decisive teams and asks costlier questions. Challenges should report who attempted each paper and what its claims cost to check, not only how many were verified.

Numbers trace to `readiness/results/p9.json`, `p9_areas.json`, `p9_adversarial.json`, `p9_extended.json`,
`p9_more.json` and `p9_claims.json`. Analyses after A9.5 are post hoc (Notes 9-9 to 9-15), each with its reading
rule frozen before computing.
