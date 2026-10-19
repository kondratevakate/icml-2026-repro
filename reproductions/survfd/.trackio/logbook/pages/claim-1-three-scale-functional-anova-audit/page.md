# Claim 1 repair: exact decomposition on log-hazard, hazard, and survival scales


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_survfd_c1_three_scales_20260729", "created_at": "2026-07-29T00:00:00+00:00", "title": "Claim 1 repair: exact three-scale audit"}
-->
## Claim

Section 3 states that SurvFD decomposes higher-order feature effects into
time-dependent and time-independent components and characterizes additivity
failure across log-hazard, hazard, and survival-function scales.

## Independent method

`repro_survfd_v2/audit_three_scales.py` implements functional ANOVA from
scratch on a complete `5^3 = 125` point product grid under the uniform counting
measure. It evaluates five time points and two risk-score scenarios:

1. an additive time-independent log-risk;
2. a log-risk with an explicit time-dependent pair effect.

For each scenario it decomposes all subsets of the target on:

- log-hazard, `log h(t|x)`;
- hazard, `h(t|x)`;
- survival, `S(t|x)`, using an analytic cumulative-hazard integral.

Every conditional projection and Mobius effect is computed by exact
enumeration. There is no Monte Carlo estimation and no fitted surrogate.

## Results

| Check | Result |
| --- | ---: |
| Maximum reconstruction error over all six scenario/scale cases | `8.88e-16` |
| Additive log-hazard has no higher-order effects | pass |
| Nonlinear hazard link creates higher-order effects | pass |
| Survival scale has time-dependent higher-order effects | pass |
| Explicit log-hazard pair is classified time-dependent | pass |

The scale transition is directly visible. In the additive scenario,
log-hazard pair effects are at floating-point zero (`~1.4e-15` RMS). The same
model has non-zero hazard pair effects (`0.00549` to `0.01080` RMS) and a
non-zero three-way hazard effect (`0.00228` RMS). On the survival scale, all
pair and three-way components are non-zero and vary with time.

The second scenario recovers the explicit `(x0, x1)` log-hazard pair as
time-dependent, with a maximum spread of `0.89588`, while unrelated
higher-order log-hazard effects remain at floating-point zero.

## Verdict

**VERIFIED.** The exact decomposition reconstructs every target to machine
precision, separates time-dependent from time-independent components, and
demonstrates why additivity on log-hazard need not survive the nonlinear
hazard and survival links. This synthetic Section 3 audit does not stand in for
the separate TCGA-BRCA application claim.
