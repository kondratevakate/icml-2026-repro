# Votes are the wrong primitive for falsification: aggregating verdicts in a crowd of reproduction agents

[Author name], [affiliation]

Target: Perspectives on Scientific Error 9 (PSE9), Copenhagen, 2027. The
submission format is not yet known (no PSE9 call found on 2026-10-18); this
full draft will be cut to the required length once the call is out.

DRAFT, 2026-10-18. Not peer reviewed. All numbers are traceable to files in
`verdict_audit/voters/results/` (Appendix A).

## Abstract

The ICML 2026 agent reproduction challenge produced thousands of public
reproduction logbooks, each scored claim by claim by a single LLM judge. Umoh
[1] showed that this judge is stable on fixed input (Krippendorff's alpha
0.874) while verdicts on the same claim from different logbooks agree poorly
(alpha 0.229). We ask what follows for turning many teams' verdicts into one.
On the public verdict snapshot (6,885 logbooks; 4,992 claims rated by at least
two teams), agreement is weakest exactly where it matters: when one team
reports a claim falsified, another team on the same claim also reports it
falsified 30% of the time, against 69% for verified. Simple majority and a
binary Dawid-Skene model recover one and none, respectively, of the four
falsifications the organisers confirmed; in three of the four, the judge had
marked the logbook the organisers cite as falsified, and aggregation outvoted
it. Later logbooks that report a falsification against the first logbook's
verdict are rated higher quality by the judge, yet the final plurality sides
with them in only 35% of 834 such pairs. Near-verbatim copying between
logbooks is rare and barely changes agreement, and a natural experiment on
judge latency gives no clear evidence that later teams followed published
verdicts. Which agent or model produced a logbook cannot be recovered from its
text. We propose adjudication instead of votes for any claimed falsification,
verdicts on several axes instead of one label, success criteria fixed before
launch, and independence and provenance enforced by design.

## 1 Introduction

Agentic reproduction moves the bottleneck of checking published claims from
running experiments to judging the evidence. In the ICML 2026 agent
reproduction challenge, 1,221 community members published 6,816 logbooks for
2,226 papers between 15 July and 2 August 2026, and an automated judge based
on GLM-5.2 scored 35,908 claims [2]. The organisers report that
independent teams reached opposite verdicts on the same claims for 242 papers;
35 participants formally claimed a falsification, and the organisers
re-verified every claimed falsification themselves [2].

Umoh [1] measured the reliability of these verdicts directly. On 32,688
claim-level verdicts, two readings of the same claim from different logbooks
differ 48.06% of the time (alpha 0.229), whereas the judge re-scoring a fixed
logbook disagrees with itself 8.41% of the time (alpha 0.874). Disagreement
therefore sits mostly in the logbooks, not in the judge. Umoh also documents
two properties of the judging harness: omitted claims were back-filled as
inconclusive, and the claim set was not pinned [1].

We take Umoh's result as given and ask the next question: when several teams
judge the same claim, how should their verdicts be combined, and what does
the current combination miss? Our contributions are:

1. Agreement broken down by verdict category. Agreement on falsified is about
   a third of agreement on verified, so a pooled coefficient hides the case
   that matters most for error correction.
2. A direct test of aggregation rules against the four falsifications the
   organisers confirmed, and an analysis of dissenting falsifications.
3. Three checks on independence: near-verbatim copying, a natural experiment
   on exposure to published verdicts, and recoverability of the agent or
   model behind each logbook.
4. Design proposals for the next edition, each grounded in how replication is
   aggregated in psychology, medicine and economics.

All analyses were prespecified in three protocols frozen in git before the
corresponding data were downloaded or analysed; deviations and post hoc
analyses are labelled as such.

## 2 Related work

**Agents reproducing papers.** PaperBench asks agents to replicate 20 ICML
2024 papers against rubrics co-developed with the papers' authors and grades
attempts with an LLM judge [3]. RECLAIM fixes in advance, for 100
NeurIPS 2025 papers, the result to reproduce, what counts as success and a
compute budget, and grades runs from logs and outputs rather than agents'
reports [4]. Repeated runs of a replication workflow differ in how
papers are divided into targets and in the rules used to accept evidence
[5]. These benchmarks study one agent per paper. The challenge instead
produced many independent attempts per claim, which raises an aggregation
problem that single-agent benchmarks do not have.

**Aggregating judgements.** The Dawid-Skene model estimates each observer's
error rates without access to the true label [6]. Condorcet's jury
theorem generalised to correlated votes shows that the conditions for a
majority to be right are severe for small groups [7]. Democratic voting
is biased against specialised knowledge that is not widely shared; selecting
the answer that is more popular than people predicted yields the best answer
under reasonable assumptions about voter behaviour [8]. Prediction markets predicted the outcomes of 44
psychology replications well and outperformed individual forecasts [9].

**Social influence.** In an informational cascade an individual follows
predecessors regardless of private information [10]. Mild social influence
narrows the diversity of estimates without reducing collective error [11].
Prior ratings bias later ratings, with positive herding accumulating and
negative influence being corrected [12]. When votes are cast
sequentially and previous choices are visible, majorities are more often
wrong, and on difficult tasks wrong majorities persist [13]; in centralised
networks central individuals dominate the estimate and errors become more
likely [14].

**Agreement statistics.** Kappa can be low despite high observed agreement
when marginals are imbalanced [15]; reporting agreement separately for
positive and negative decisions avoids the paradoxes [16], and the AC1
coefficient is more stable in the presence of high agreement [17].

**Disagreement as information.** Twenty-nine teams analysing one data set
reached different conclusions, and neither expertise nor prior beliefs
explained the spread [18]. Autonomous LLM analysts reproduce this
dispersion, and their results shift with the model and persona used [19].
In conference peer review, two independent NeurIPS 2021 committees disagreed
on 23% of papers [20], and half of the variation in 2014 reviewer
scores was subjective [21].

## 3 Data and methods

**Data.** We use the public verdict snapshot downloaded on 2026-10-17 (6,885
logbooks, judged up to 2026-08-03, one judge, `zai-org/GLM-5.2`). A team is
the namespace of a Space; we keep one logbook per team and paper (the latest
judged), leaving 6,709 logbooks. Claims are identified by normalised text
matched to a frozen list of anchored claim texts. 4,992 anchored claims from
926 papers were judged by at least two teams. For the 5,333 logbooks behind
these claims we downloaded the logbook pages and commit history at the judged
revision; 138 Spaces had been deleted, leaving 5,195. Two earlier snapshots
(downloaded 2026-07-22 and 2026-07-28) give earlier verdicts and publication
times.

**Prespecification.** Protocol 1 (copies, agreement, provenance, aggregation)
was frozen before any page download; Protocol 2 (exposure to published
verdicts) before commit histories were downloaded; Protocol 3 (resubmission)
before file trees were downloaded. Protocols, amendments and a post hoc note
on an estimator bias are in `verdict_audit/voters/PROTOCOL.md`. Only
aggregates are reported; no team is named.

**Agreement.** Fleiss-type kappa with pooled marginals on four verdicts
(verified, falsified, toy, inconclusive) over claims with at least two teams,
with a paper-cluster bootstrap (2,000 resamples). Gwet's AC1 and
category-specific agreement were added after the first results were seen.

**Copies and effective teams.** For each logbook we take the text of its claim
pages and compute word 5-shingles. A pair of logbooks on the same paper is a
copy pair if the Jaccard similarity is at least 0.5. Among 20,000 random pairs
of logbooks on different papers, the 99.9th percentile was 0.016 and the
maximum 0.33. Copy pairs form components per claim; each component counts as
one team, represented by its earliest-created logbook.

**Reference falsifications.** The organisers' blog names four confirmed cases
[2]: a robustness bound for learning-augmented paging, a convergence
statement about attention dynamics and Frank-Wolfe, the KL direction in a
self-distillation method, and a perplexity figure distorted by padding in a
study of attention projections. We map each to one anchored claim (the
Frank-Wolfe mapping is uncertain) and compare simple majority over decisive
votes and a binary Dawid-Skene model (one sensitivity and specificity per
team, Beta(2,2) pseudo-counts) with the organisers' verdict.

**Exposure natural experiment.** For ordered pairs of logbooks on the same
claim, a later logbook is exposed if its judged revision was committed after
the earlier logbook's verdict was published. Within strata of the same time
gap, exposure varies only through the judge's latency. We estimate a
Mantel-Haenszel risk difference in agreement and compare it with a null in
which judge latencies are permuted among logbooks judged on the same day.

**Provenance.** Regular expressions for coding-agent names and model families
were validated by two blind coders (Claude subagents, model `claude-opus-5-5`) on 120 matches and 40
logbooks without matches; a category is reported only if at least 80% of its
matches name the agent or model that produced the logbook.

## 4 Results

### 4.1 Agreement is weakest on falsifications

Over the 4,992 multi-rated claims, observed agreement is 0.503 and kappa 0.202
(95% CI 0.177-0.227), in line with Umoh's alpha of 0.229 on a different
snapshot and coefficient [1]. AC1 is 0.358. Category-specific agreement
separates the cases (Table 1): given one team's verdict, the chance that
another team on the same claim gives the same verdict is 0.687 for verified
but 0.300 for falsified. 8.1% of multi-rated claims carry at least one
verified-versus-falsified conflict. Most claims have few teams: 48% have two
or three. Treating kappa as single-team reliability, Spearman-Brown puts the
number of independent teams needed for a reliability of 0.7 at about 10
(8.5-11.4 across the kappa CI) and for 0.8 at about 17; this is an
approximation.

Table 1. Category-specific agreement on multi-rated claims.

| Verdict | Specific agreement |
|---|---|
| verified | 0.687 |
| falsified | 0.300 |
| inconclusive | 0.247 |
| toy | 0.219 |

### 4.2 Aggregation misses confirmed falsifications

Table 2 shows the four reference claims. Majority over decisive votes agrees
with the organisers on one (self-distillation). Dawid-Skene converged in 41
iterations over 4,643 claims and 258 teams; it agrees on none. It rates the
self-distillation claim 0.86 verified, against two falsified votes and no
verified vote, because its estimated prior that a claim is verified is 0.87.
For three of the four cases, the judge had marked the logbook the organisers
cite as falsified on the relevant claim (post hoc check). For the padding case
the judge scored that logbook `toy`; its own overall comment describes "a
genuine padding-label bug in the paper's code" demonstrated only at toy scale.
For Frank-Wolfe, the second logbook to start already reported the
counterexample; of the 19 that started after it, 14 said verified.

Table 2. Reference falsifications (copy-collapsed decisive votes).

| Case | Verified | Falsified | Majority | Dawid-Skene P(verified) | Cited logbook (judge) |
|---|---|---|---|---|---|
| Paging robustness bound | 2 | 1 | verified | 0.96 | falsified |
| Frank-Wolfe convergence (mapping uncertain) | 15 | 3 | verified | 1.00 | falsified |
| Self-distillation KL direction | 0 | 2 | falsified | 0.86 | falsified |
| Projection-sharing perplexity | 0 | 0 | none | not defined | toy |

### 4.3 Dissenting falsifications are rated higher but lose

There are 834 pairs in which a later logbook says falsified against a
different verdict from the first logbook on the same claim, spread over 451
claims and 252 papers (the largest claim contributes 1.1%). The judge rated
the later logbook higher quality by 0.60 on a 0-2 scale (CI 0.45-0.75). The
final plurality of all teams sided with the later falsifier in 35.5% of these
pairs, with the first logbook in 33.1%, with neither in 14.4%, and was tied in
17.0%. Across all 11,588 pairs in which a later logbook disagrees with the
first, the quality difference is near zero (0.04, CI -0.07 to 0.15). Because
the judge assigns quality and verdict in the same response, a falsification
may itself raise the quality rating; this is a signal, not proof, that the
minority was often right.

### 4.4 Dependence between teams

**Copies.** 164 of 23,197 tested pairs (0.71%) are near-duplicates. They touch
17.7% of multi-rated claims, and copies agree on 97.3% of their shared claims
against 51.5% for other pairs. Collapsing copies lowers the mean number of
teams per claim by 3.2% (5.82 to 5.64) and kappa by 0.009 (to 0.193).

**Exposure to published verdicts.** Later logbooks agreed with an earlier
verdict 49.9% of the time when it had been published and 52.2% when it had
not; the stratified difference is -0.011 (CI -0.036 to 0.016), and the
prespecified herding label is not met. After seeing this result we found that
the estimator is biased downward: within a gap stratum, exposed pairs sit
further apart in time, and agreement falls with the gap. The latency
permutation keeps this structure and centres at -0.023; against it, the
observed difference is 0.012 higher (one-sided p = 0.03). We therefore read
the result as inconclusive. The test has little leverage because the judge
usually published within about an hour (median 1.2 h; 1.9% of verdicts took
longer than a day), and it cannot see teams that read logbook pages rather
than verdicts. The first logbook of a claim does not agree with the others more
than later logbooks do (difference -0.028, CI -0.058 to 0.000).

**Provenance.** 3.0% of logbooks mention a coding-agent name, but only 25% of
sampled mentions name the agent that produced the logbook; model-family names
appear in 23.3% of logbooks and none of the sampled mentions names the
producer (coder agreement kappa 0.96). None of 40 logbooks without a match
names its producer. Neither category passes the validation gate, so the
agent and model behind a logbook cannot be recovered from its text.

## 5 Design proposals for the next edition

These proposals follow from the results but were not tested here.

1. **Adjudicate claimed falsifications instead of counting votes.** A
   falsification is an existence claim, and the data show that votes bury it
   (Sections 4.1-4.3). The organisers already re-verified every formal
   falsification [2]; this can become the rule for any claim with at
   least one falsified verdict. Systematic reviews use two independent
   assessors and resolve differences in interpretation by arbitration
   [22]; the Institute for Replication sends each report to the
   original authors before publication [23].
2. **Verdicts on several axes instead of one label.** The Reproducibility
   Project: Psychology reported several indicators of replication side by
   side rather than one verdict [24], and the Institute for Replication
   separates computational reproducibility from robustness [23]. Systematic
   reviews avoid vote counting based on statistical significance [25].
   Separate axes (code runs; full scale reached; effect within stated bounds;
   robust to reasonable changes; error found in the authors' artifacts) would
   let a demonstrated bug falsify a claim even when the run is small, as in the
   padding case.
3. **Fix the claim list and the success criterion before launch,** as RECLAIM
   does [4], ideally with the authors, as PaperBench does for rubrics
   [3]. Umoh documents that the claim set was not pinned [1].
4. **Enforce independence and provenance by design.** Hide verdicts until the
   challenge closes and publish only coverage; require structured fields for
   agent, model and human involvement; and require agent traces of every URL
   and file read. Social influence can reduce diversity without improving
   accuracy [11] and sequential visibility can entrench wrong majorities
   [13]; results of LLM analysts depend on the model used [19].
5. **Elicit more than a vote for contested claims,** for example predictions
   of other teams' verdicts [8] or market prices [9].

Resubmission deserves its own study. Our snapshots contain 1,307 pairs of
consecutive judgements that scored a different revision of the same Space;
they come from 62 teams, and the ten most active teams account for 77% of
them. When an LLM review panel returned its decision and reasons to research
agents over five rounds, the number of model-task pairs with an evasion rose
from 7 to 56 [26]. A prespecified analysis of whether verdicts move toward verified
after resubmission, with same-revision rejudgements as a control, is frozen
but not yet run.

## 6 Limitations

All verdicts come from one LLM judge, so agreement is agreement between
judged logbooks, not between expert assessments. The data are observational:
teams chose which papers to attempt. Only four falsifications are confirmed,
and they were selected by the organisers for publication, so we cannot
estimate how often aggregation misses falsifications in general. The judge's
quality rating may depend on the verdict it accompanies. The provenance coders
are language models. Copy detection finds near-verbatim text only; paraphrased
copying and reading other teams' logbooks are not measured. The exposure
estimator was biased, which we found after seeing the result. Part of the
disagreement may be legitimate analytic variation rather than error
[18]. Peer-review consistency figures [20, 21] are
context, not a benchmark on the same scale. Some logbooks are already deleted,
which limits reproduction of this study from public data.

## 7 Conclusion

Umoh showed that the judge is consistent and the logbooks are not. We add that
the disagreement is concentrated on falsifications and that majority and
Dawid-Skene aggregation discard most confirmed ones. For crowd reproduction,
a vote is the wrong primitive for falsification: a claimed counterexample
needs adjudication, and verdicts need several axes and a recorded provenance.

## Disclosure

The author took part in the challenge with 53 judged logbooks; their agent had
access to other teams' logbooks, and which logbooks were influenced cannot be
reconstructed (all 49 local trace indexes are empty). Excluding the author's
logbooks changes collapsed kappa from 0.193 to 0.191 and the exposure
difference from -0.0108 to -0.0105.

## Provenance of this study

We apply our own recommendation to this study. The analysis code, the three
protocols and this draft were written with Claude Code using the model
`claude-opus-5-5`, directed and reviewed by the author, who approved each
protocol freeze and each bulk download before it ran. The two blind coders
were Claude subagents running the same model; their prompt is published with
the code. An access log of the analysis session lists every web page fetched,
every web search, the literature-database calls and the Hugging Face API
methods used (`ACCESS_LOG.csv`). Two inputs lack a recorded origin: the list
of Space creation dates and the anchored claim texts. Both are fixed by hash.
No clean re-run from scratch has been performed; the analyses are
deterministic by construction (fixed seeds, stable hashes), but this has not
been confirmed by re-execution.

## Data and code availability

Code, protocols, aggregate results, run order and environment:
`verdict_audit/voters/` (`README.md`, `requirements.txt`). The verdict
snapshot is revision `f0c5a8b369d6d618ecc9fa060aa2e2f27f9e2602` of the public
dataset `ICML-2026-agent-repro/verdicts` (file `verdicts.json`; its LFS
SHA-256 matches our copy). Logbook pages, commit histories and file trees
were read from public Spaces at their judged revisions; page contents and
Space ids are not redistributed; aggregate digests of all downloads are
(`results/download_manifests_hashed.json`).


## References

1. Umoh EH. Who verifies the verifier? Measuring LLM-as-judge reliability across 32,688 automated research verdicts [poster]. Presented at: AWS-MLU Fall AI Teaching and Research Symposium; 2026 Sep 22; Arlington, VA. Available from: https://github.com/SturdivantAI/who-verifies-the-verifier
2. Abid A. What we learned by reproducing 2,200 papers from ICML [Internet]. Hugging Face Blog; 2026. Available from: https://huggingface.co/blog/icml-2026-open-reproductions
3. Starace G, Jaffe O, Sherburn D, Aung J, Chan JS, Maksin L, et al. PaperBench: evaluating AI's ability to replicate AI research. In: Proceedings of the 42nd International Conference on Machine Learning. Proceedings of Machine Learning Research. Vol 267. PMLR; 2025. p. 56843–73.
4. Salunkhe M, Ding H, Verma S, Kindratenko V. RECLAIM: can agents reproduce the claims of machine learning papers? [Preprint]. arXiv:2609.28850. 2026.
5. Hans A, Bilionis I. Coding-agents can replicate scientific machine learning papers [Preprint]. arXiv:2607.02134. 2026.
6. Dawid AP, Skene AM. Maximum likelihood estimation of observer error-rates using the EM algorithm. J R Stat Soc Ser C Appl Stat. 1979;28(1):20–8. doi:10.2307/2346806
7. Ladha KK. The Condorcet jury theorem, free speech, and correlated votes. Am J Pol Sci. 1992;36(3):617–34. doi:10.2307/2111584
8. Prelec D, Seung HS, McCoy J. A solution to the single-question crowd wisdom problem. Nature. 2017;541(7638):532–5. doi:10.1038/nature21054
9. Dreber A, Pfeiffer T, Almenberg J, Isaksson S, Wilson B, Chen Y, et al. Using prediction markets to estimate the reproducibility of scientific research. Proc Natl Acad Sci U S A. 2015;112(50):15343–7. doi:10.1073/pnas.1516179112
10. Bikhchandani S, Hirshleifer D, Welch I. A theory of fads, fashion, custom, and cultural change as informational cascades. J Polit Econ. 1992;100(5):992–1026. doi:10.1086/261849
11. Lorenz J, Rauhut H, Schweitzer F, Helbing D. How social influence can undermine the wisdom of crowd effect. Proc Natl Acad Sci U S A. 2011;108(22):9020–5. doi:10.1073/pnas.1008636108
12. Muchnik L, Aral S, Taylor SJ. Social influence bias: a randomized experiment. Science. 2013;341(6146):647–51. doi:10.1126/science.1240466
13. Frey V, van de Rijt A. Social influence undermines the wisdom of the crowd in sequential decision making. Manage Sci. 2021;67(7):4273–86. doi:10.1287/mnsc.2020.3713
14. Becker J, Brackbill D, Centola D. Network dynamics of social influence in the wisdom of crowds. Proc Natl Acad Sci U S A. 2017;114(26):E5070–6. doi:10.1073/pnas.1615978114
15. Feinstein AR, Cicchetti DV. High agreement but low kappa: I. The problems of two paradoxes. J Clin Epidemiol. 1990;43(6):543–9. doi:10.1016/0895-4356(90)90158-L
16. Cicchetti DV, Feinstein AR. High agreement but low kappa: II. Resolving the paradoxes. J Clin Epidemiol. 1990;43(6):551–8. doi:10.1016/0895-4356(90)90159-M
17. Gwet KL. Computing inter-rater reliability and its variance in the presence of high agreement. Br J Math Stat Psychol. 2008;61(1):29–48. doi:10.1348/000711006X126600
18. Silberzahn R, Uhlmann EL, Martin DP, Anselmi P, Aust F, Awtrey E, et al. Many analysts, one data set: making transparent how variations in analytic choices affect results. Adv Methods Pract Psychol Sci. 2018;1(3):337–56. doi:10.1177/2515245917747646
19. Bertran M, Fogliato R, Wu ZS. Many AI analysts, one dataset: navigating the agentic data science multiverse. Proc Natl Acad Sci U S A. 2026;123(29):e2606495123. doi:10.1073/pnas.2606495123
20. Beygelzimer A, Dauphin YN, Liang P, Vaughan JW. Has the machine learning review process become more arbitrary as the field has grown? The NeurIPS 2021 consistency experiment [Preprint]. arXiv:2306.03262. 2023.
21. Cortes C, Lawrence ND. Inconsistency in conference peer review: revisiting the 2014 NeurIPS experiment [Preprint]. arXiv:2109.09774. 2021.
22. Lefebvre C, Glanville J, Briscoe S, Featherstone R, Littlewood A, Metzendorf MI, et al. Chapter 4: Searching for and selecting studies. In: Cochrane Handbook for Systematic Reviews of Interventions. Version 6.5.1 [Internet]. Cochrane. Available from: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-04
23. Brodeur A, Mikola D, Cook N, Fiala L. Behind the paper: studying reproducibility at scale [Internet]. Springer Nature Research Communities; 2026. Available from: https://communities.springernature.com/amp/posts/behind-the-paper-a-new-hope-studying-reproducibility-at-scale
24. Open Science Collaboration. Estimating the reproducibility of psychological science. Science. 2015;349(6251):aac4716. doi:10.1126/science.aac4716
25. McKenzie JE, Brennan SE. Chapter 12: Synthesizing and presenting findings using other methods. In: Cochrane Handbook for Systematic Reviews of Interventions. Version 6.5 [Internet]. Cochrane. Available from: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-12
26. Huang Y, Xu Z, Ma Y, Wang W, Liu Z, Xu Z, et al. Reward hacking challenges oversight of autonomous research agents [Preprint]. arXiv:2609.28614. 2026.

## Appendix A: provenance of numbers

| Number | File | Key |
|---|---|---|
| 6,709 logbooks; 4,992 claims; 926 papers | results.json | snapshot |
| 5,333 / 5,195 / 138 logbooks | results.json | Q1_copies.logbooks_wanted, download_status |
| Template control 0.016, 0.33 | results.json | Q1_copies.control |
| 164 copy pairs; 0.71%; 17.7%; 97.3% vs 51.5% | results.json | Q1_copies.primary, verdict_agreement |
| kappa 0.202 (0.177-0.227); Po 0.503; 8.1% | results.json | Q2_effective_teams.kappa_uncollapsed |
| kappa 0.193; -0.009; 5.82 to 5.64; -3.2% | results.json | Q2_effective_teams |
| Teams needed 9.7 (8.5-11.4), 16.7 | results.json | Q2_effective_teams.teams_needed_collapsed |
| 48% of claims with two or three teams | results.json | Q2_effective_teams.nominal_teams.dist |
| AC1 0.358; specific agreement | exposure.json | exploratory.protocol1_q2_prevalence_robust |
| Reference votes, majority, Dawid-Skene | results.json | Q4_aggregation |
| Cited logbooks' verdicts | posthoc_blog_logbooks.json | all |
| Frank-Wolfe order (2nd falsified; 14 of 19) | exposure.json | H4.reference_claims.frank_wolfe |
| 834 pairs; +0.60; 35.5/33.1/14.4/17.0% | exposure.json | H4.later_falsified |
| 451 claims; 252 papers; 1.1% | review_checks.json | h4_falsified_dissent |
| 11,588 pairs; 0.04 | exposure.json | H4.all_disagreeing |
| Exposure -0.011; null -0.023; p 0.03; 49.9/52.2% | exposure.json | H1 |
| Latency median 1.2 h | exposure.json | pairs.judge_latency_hours |
| 1.9% of verdicts > 24 h | review_checks.json | judge_latency_hours |
| First-mover -0.028 | exposure.json | H3 |
| Provenance 3.0%, 25%, 23.3%, 0%, kappa 0.96, 0/40 | q3.json, results.json | categories, Q3_regex_raw |
| 1,307 events; 62 teams; 77% | review_checks.json | resubmission |
| 49 empty trace indexes | review_checks.json | local_trace_indexes |
| Author exclusion 0.191, -0.0105 | results.json, exposure.json | sensitivity.exclude_author |
