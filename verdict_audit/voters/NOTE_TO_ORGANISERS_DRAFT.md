# DRAFT, not sent: aggregating verdicts in the 2027 agent reproduction challenge

To: ICML agent reproduction challenge organisers. From: [sender name].
Status: draft for Kate's review; numbers are exploratory, from a protocol frozen
before the analysis (`verdict_audit/voters/PROTOCOL.md`, commits 7c83ea3 and
fa7dd28). All verdicts come from the one LLM judge.

Thank you for the challenge and for publishing the verdicts. We used the public
verdict snapshot (6,885 logbooks, downloaded 2026-10-17) and the logbook pages at
each judged revision to ask one question: how well do claim-level votes add up
to a reliable verdict? Four findings, then six suggestions.

## Findings

1. **Teams agree only weakly.** Among 4,992 claims rated by at least two teams
   (926 papers), Fleiss-type kappa is 0.20 (95% CI 0.18-0.23). 8.1% of these
   claims have at least one verified-versus-falsified conflict. Spearman-Brown,
   after collapsing copies (finding 2), puts the teams needed for reliability
   0.7 at about 10 (8.5-11.4 across the kappa CI), and for 0.8 at about 17. 48% of multi-rated claims have two or three teams.
2. **Copying exists but does not explain the disagreement.** 164 pairs of
   logbooks on the same paper are near-duplicates (Jaccard >= 0.5 on word
   5-shingles; random pairs from different papers reach at most 0.33, 99.9th
   percentile 0.016). The pairs touch 17.7% of multi-rated claims, and copies
   agree on 97% of their shared claims versus 52% for other pairs. Collapsing
   copies lowers the mean number of teams per claim by only 3% and changes kappa
   by -0.009. Agreement still falls with the gap between Space creation dates
   after copies are removed (0.55 within a day, 0.52 at 1-7 days, 0.49 beyond a
   week). Text similarity cannot detect teams that read others' verdicts before
   running their own experiments, so true independence is lower than this.
3. **Majority and Dawid-Skene miss the falsifications you confirmed.** For the
   four cases in your blog post, simple majority over decisive votes agrees with
   you on one (self-distillation). Paging: 2 verified votes against 1 falsified.
   Frank-Wolfe: 15 against 3. Three projections: no decisive vote at all. A
   binary Dawid-Skene model (converged, 258 teams) agrees on none. It rates the
   self-distillation claim 0.86 verified, against two falsified votes and no
   verified vote, because its estimated prior that a claim is verified is 0.87. In three
   of the four cases the judge did mark the logbook you cite as falsified, and
   aggregation outvoted it. For three projections the judge scored that logbook
   `toy`, even though its own summary notes a genuine padding bug in the paper's
   code.
4. **Logbooks do not say which agent or model produced them.** 3.0% of logbooks
   mention a coding-agent name, and two blind coders judged only 25% of sampled
   mentions to name the producer. Model-family names are almost always the object
   of study (0% producer). None of 40 logbooks without a match names a producer.

## Suggestions for 2027

1. **Pin the claim set before launch.** Publish one versioned claim list per
   paper and judge every logbook against it. In the 2026 snapshots, Spaces that
   were rejudged on unchanged content were scored against an entirely different
   claim list (`verdict_audit/PROTOCOL.md`, Amendment 2: all 238 such Spaces),
   which makes votes hard to pool.
2. **Keep votes independent until close.** Hide verdicts until the challenge
   ends and allow private Spaces until then; publish only coverage (attempts per
   claim) so teams can still go where checks are few. Ask each logbook to declare
   whether it consulted other logbooks or verdicts, and which. Count declared
   syntheses as their own category, not as independent votes.
3. **Record agent and model as structured fields** in `logbook.json`: agent tool,
   model and version, and how much a human intervened. Free text cannot recover
   this (finding 4), and without it no one can study agent effects or correlated
   errors.
4. **Adjudicate contested claims instead of counting votes.** A falsification is
   an existence claim: one checked counterexample is enough, and a majority of
   weaker "verified" runs cannot cancel it. Send every claim with at least one
   falsified vote to the adversarial re-verification you already ran, with an
   author response window, and publish it as "contested" until resolved.
5. **Separate scale from validity.** Keep the rule that a toy-scale run cannot
   confirm an empirical claim. But a demonstrated error in the authors' released
   code, or an exact counterexample to a theorem, should be eligible for
   falsification at any scale.
6. **Require artifacts that make a verdict checkable:** commit hash, seeds,
   environment, raw outputs, and the exact claim-list version. Freeze the judged
   revision (you already store its `sha`) so that a later reader sees what the
   judge saw.

Every number above is in `verdict_audit/voters/results/` (`results.json`,
`q3.json`, `posthoc_blog_logbooks.json`). The check of the blog-cited logbooks
in finding 3 is post hoc. Happy to share code and aggregates.
