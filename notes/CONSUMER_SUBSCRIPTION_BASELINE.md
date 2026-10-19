# Best-effort consumer subscription baseline

Protocol frozen: 2026-07-29

## Research question

How accurately and cheaply can a non-specialist reproduce life-science and
medical ML papers using only a current, approximately USD 20/month,
state-of-the-art consumer AI subscription and ordinary CPU hardware?

This is a practical access baseline, not a controlled comparison of model
weights. It measures the strongest result available to an ordinary subscriber
under a fixed workflow and effort budget.

## Frozen numerical baseline

The project uses the 15 previously reported, Claude-only logbooks as its
retrospective consumer-subscription baseline:

- logbooks: `15`;
- validated points: `46`;
- mean: `46 / 15 = 3.07` points per logbook;
- median: `3` points per logbook;
- total available points across those papers: `130`;
- normalized score: `46 / 130 = 35.4%`;
- full-score logbooks: `3 / 15 = 20.0%`;
- nonzero-score logbooks: `15 / 15 = 100%`.

The paper-level observations are frozen in
`consumer_baseline_historical_claude.csv`.

Call this comparator `Claude-$20-historical`. The logbooks were produced with
consumer Claude access and no additional paid model API. They were not run
under a single prospective protocol: model versions, prompts, paper
difficulty, elapsed time, and intervention count were not held constant.
Therefore this baseline supports a practical historical comparison with the
future auto-research engine, but not a causal Claude-versus-Codex claim.

No additional Claude runs are required. `Consumer-SOTA-$20` is not reported
because no paired multi-provider experiment will be run.

## Experimental unit

One previously unseen paper is one experimental unit. A paper is unseen when:

- the agent has not processed it in an earlier run;
- the context contains no competing logbook, verdict, or reproduction;
- no code or notes produced by the other tested system are present;
- the human operator has not inspected competing solutions for that paper.

The paper, supplement, official repository, official datasets, and normal
public technical documentation are allowed. Leaderboard solutions, verdicts,
and third-party reproductions are prohibited until both systems' predictions
and artifacts have been frozen.

## Best-effort budget

For each system and paper:

- one fresh task/context;
- one fixed baseline prompt:
  `CONSUMER_BASELINE_PROMPT_V1.md` (`consumer-baseline-v1`);
- one primary implementation attempt;
- one repair pass after local preflight;
- at most 2 hours of active wall-clock work;
- at most 4 CPU-hours of execution;
- no paid API calls;
- no paid GPU;
- no output or artifact from the other tested system.

Free public data downloads and a local laptop or CPU cloud machine are allowed.
Cloud CPU cost is reported separately and is not silently counted as part of
the subscription. If a subscription usage limit stops the run, the run ends
and the limit is recorded; capacity is not replaced with paid API usage.

## Human role

The operator may:

- provide the paper, supplement, official repository, and baseline prompt;
- approve shell commands and data downloads;
- answer environment questions;
- choose between alternatives explicitly proposed by the agent;
- request the single repair pass.

The operator may not:

- write or debug the reproduction code;
- supply a claim decomposition created for that paper;
- reveal leaderboard evidence or another system's work;
- add further attempts after seeing the challenge verdict.

This boundary makes the baseline a best-effort consumer agent workflow rather
than a measurement of unrestricted human-plus-model collaboration.

## Required frozen outputs

Before publishing or reading any challenge verdict, save:

1. paper identifier and exact anchored claims;
2. system, product tier, displayed model name, mode, and client version;
3. start/end timestamps, wall time, CPU time, and external compute cost;
4. baseline prompt version and repository commit;
5. predicted verdict and points for every claim;
6. runnable reproduction artifact and environment specification;
7. local preflight result;
8. all human interventions;
9. failure reason if the run did not produce a valid logbook.

After judging, append the actual claim verdicts and score. Never rewrite the
frozen prediction.

## Outcomes

Primary outcomes:

- validated points per paper;
- validated points per active wall-clock hour;
- fraction of anchored claims receiving `verified` or `falsified`.

Secondary outcomes:

- full-logbook success rate;
- `toy` and `inconclusive` rates;
- CPU-hours and external compute cost;
- calibration error between predicted and actual points;
- autonomous completion rate;
- failure category.

Report medians and individual paper results. With a small sample, do not report
provider superiority as a general result.

## Evaluation set

Historical work in `MODEL_QUALITY_BASELINE.md` supplies the frozen
`Claude-$20-historical` comparator and the development failure taxonomy. The
future auto-research engine starts on unseen papers after this protocol's freeze
date.

Because the historical and future paper sets differ, report both:

- raw points per logbook;
- points divided by the available points for each paper.

Any later matched-paper rerun is a separate confirmatory experiment and does
not rewrite this baseline.

## Relationship to the auto-research engine

The auto-research engine must use the same paper eligibility rules, prohibited
information, CPU limits, and scoring outcomes. It receives a separate version,
prompt, and code commit.

The 15 historical Claude papers may be used to develop the engine but not to
estimate its final generalization performance. Papers first processed after the
engine version is frozen form the prospective auto-research test set.

The central comparison is:

`validated evidence quality and points / time / compute`

not leaderboard points alone.
