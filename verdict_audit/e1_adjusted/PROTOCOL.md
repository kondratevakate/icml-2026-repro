# Protocol 4: medical label check (E0) and the adjusted medical verdict gap (E1)

Frozen 2026-10-17 before any sample was drawn, coded or modelled. Git commit is
the provenance. Companion to `verdict_audit/PROTOCOL.md` (raw gap, discovery and
confirmation) and `verdict_audit/healthcare/PROTOCOL.md` (release audits).

## Status of the evidence

The raw medical gap has already been seen in both snapshots. No new verdict
data exist: the challenge is closed. E1 is therefore adjusted estimation on
seen data, not confirmation. It answers whether the gap survives two named
confounders: who reproduced the paper, and whether the claim is theoretical.

## Data

`data/verdicts_confirmation_2026-10-17/verdicts.json`, SHA-256
`652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526`
(6,885 logbooks, final snapshot). Deduplication: one logbook per
(reproducer, paper), latest `judged_at`, ties by `space_id`, as in protocol 1.
Medical label: `audit/public_repro_audit.csv` (703 papers), as before.
Abstracts: `papers.json` key `abstracts`.
Verdict categories: verified, falsified, toy, inconclusive.

## E0: validity of the medical label

Sample: seed 20261018; 100 papers drawn without replacement from the 703
labelled medical, and 100 from the other 5,638 papers of `papers.json`.

Coding sheet: `orid`, title, abstract, in random order, without the audit
label. Two independent coders, both Claude subagents with no access to the audit
label, this protocol's analysis section, or each other's file. Category:

* `clinical`: patients, clinical data, medical imaging, clinical decisions or
  public health are the main task or main evaluation data;
* `life_science`: biology, genomics, proteins, cells, neuroscience or drug
  discovery are the main task or main evaluation data, without clinical data;
* `no`: otherwise, including methods papers that mention medicine only as one
  example among several.

Consensus: agreement, else the first author (this session) adjudicates and
records a reason. Reported: coder agreement (Cohen's kappa over three
categories), and the PPV and NPV of the audit label against consensus
`clinical` or `life_science`, each with a Wilson 95% interval. PPV and NPV are
computed within the stratified sample, which matches how they are used below.

Use in E1: under non-differential misclassification, an observed difference
equals the true difference times (PPV + NPV - 1). The corrected estimate divides
by this factor, with a bootstrap over the E0 sample for its uncertainty. This is
a sensitivity analysis, not the primary estimate.

## E1: claim-type dictionary

A claim is `theory` if its normalized text matches, case-insensitive, the regex

`\b(theorem|lemma|proposition|corollary|proof|prove[sd]?|provabl\w*|bound(s|ed)?|guarantee\w*|converge\w*|regret|minimax|consisten(t|cy)|identifiab\w*|asymptotic\w*|sample complexity|o\(|theta\(|omega\()`

otherwise `empirical`.

Validation sample: seed 20261018; 200 unique claim texts drawn from all claims
in deduplicated logbooks. Two blind subagent coders label each `theory`
(a mathematical statement: theorem, bound, rate, guarantee, identity) or
`empirical` (experiments, method behaviour, datasets, performance). Coders do
not see the regex, verdicts or medical labels. Consensus as in E0.

Gate: Cohen's kappa between the dictionary and consensus at least 0.60. If the
gate fails, the theory covariate is dropped from all models and this is reported;
no other classifier is substituted.

## E1: reason for non-verified verdicts (secondary, exploratory)

Each `inconclusive` or `toy` verdict's evidence text receives one reason by the
first matching rule, case-insensitive:

1. `artifact_absent`: `not (been )?released|unreleased|not (publicly )?available|unavailable|absent|withheld|not provided|no (official )?(code|checkpoint|weights|data|dataset|predictions|outputs)|proprietary|restricted|access`
2. `not_attempted`: `not attempted|not (re)?run|did not run|never (ran|run)|no experiments?|blocked|not executed|no (numerical )?results`
3. `reduced_scale`: `toy|small[- ]scale|subset|reduced|synthetic|smaller|fewer`
4. `other`.

Validation: 200 evidence texts (seed 20261018) from deduplicated logbooks, double
coded blind into the same four reasons using the rule text as a written
definition but not the regex. Gate kappa at least 0.60 against consensus, else
the reasons are reported as unvalidated keyword counts only.

Analysis: share of each reason among non-verified verdicts, medical versus
other, with paper-cluster bootstrap 95% intervals (2000 resamples, seed
20261017). Descriptive only.

## E1: primary model (hypothesis H1)

Unit: claim verdict in a deduplicated logbook. Outcome Y = 1 if verified.

Linear probability model with reproducer fixed effects:

`Y ~ medical + theory + C(reproducer)`

Variance: two-way cluster-robust by paper and by reproducer. Estimand: the
within-reproducer difference in the probability of a verified verdict, medical
minus other, holding claim type fixed, in percentage points.

H1: the coefficient on `medical` is below zero. Reported: estimate, 95% CI,
two-sided p. It enters a Holm correction with the three other primary
hypotheses of the project (E3 release gap, E4 R1 sensitivity, E4 I3
equivalence); the raw p is reported now and the Holm-adjusted value later.

Decision rule fixed now: if the 95% CI includes zero, the medical gap is not a
headline claim and medicine is reported as the application domain only.

Only reproducers with both medical and other claims identify the coefficient;
their number and coverage are reported.

## E1: secondary and sensitivity analyses (reported, never substituted)

1. Conditional logistic regression stratified by reproducer, same covariates;
   odds ratio with paper-cluster bootstrap CI.
2. Without the theory covariate.
3. Excluding the organiser namespace `ICML-2026-agent-repro`.
4. Narrow medical label (`medical_neuro_papers.csv`).
5. Equal weight per paper (weights 1 / claims of that paper).
6. Separately in the discovery period (judged up to 2026-07-28T12:51:33Z) and
   after it.
7. Label-corrected estimate from E0.

## Evidence boundary

* Observational. Reproducers chose papers; adjustment removes reproducer skill
  and claim type, not paper difficulty, compute or data access.
* All coders are Claude models: E0 and the validations measure agreement among
  model coders, not expert ground truth. Kate may re-code a subset later; that
  would be reported as an additional check, not replace these codes.
* Aggregates only; no participant other than Kate is named.

## Amendment 1 (2026-10-17, before any model result was seen)

The first `models` run stopped inside the secondary analysis with a
RecursionError from statsmodels `ConditionalLogit`, whose exact conditional
likelihood recursion cannot handle strata of several hundred claims. No result
file was written and no estimate was printed. Changes:

* Secondary analysis 1 becomes a logistic regression with reproducer fixed
  effects (unconditional). With large strata the incidental-parameter bias is
  small; this is a deviation and is labelled as such.
* Its CI uses two-way cluster-robust variance (paper, reproducer) instead of a
  paper-cluster bootstrap, because 2,000 refits of a model with several hundred
  dummies are not feasible locally.
* Everything else, including the primary model, is unchanged.
