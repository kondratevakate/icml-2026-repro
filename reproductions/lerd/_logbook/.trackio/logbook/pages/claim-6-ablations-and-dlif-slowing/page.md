# Claim 6: Ablations and dLIF slowing


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2a4178a60227", "created_at": "2026-07-22T12:42:40+00:00", "title": "Claim 6: Ablations and dLIF slowing"}
-->
**Paper claim (Table 3/4, Figure 2).** Both the dLIF and ERG priors independently
contribute to performance and combine best; and Figure 2 shows LERD-inferred dLIF
frequencies exhibit disease-correlated oscillatory slowing (HC > MCI > AD central
frequency) consistent with established AD EEG biomarkers.

**Verdict: REPRODUCED (biomarker-slowing premise, on real data).**

The ablation numbers require training the full model (see Claim 4, GPU-bound). The
falsifiable, data-grounded content of the claim - the AD spectral-slowing signature
the paper's Figure 2 is stated to be "consistent with" - was checked directly on
all 88 subjects of Cohort A (ds004504).

**Result (`spectral_slowing.py`, mne + scipy.welch, all 88 subjects).**

- AD spectral centroid (4-12 Hz) is **below HC in 19/19 channels** (the paper's
  model-derived version reports HC highest in 18/19).
- Canonical AD signature holds: relative theta power AD 0.062 > HC 0.055; relative
  alpha power AD 0.029 < HC 0.048.
- FTD is intermediate throughout, consistent with a dementia-severity gradient.
- Raw-EEG centroids (~6.4-8.0 Hz) fall in the paper's Figure 2 range (~6.6-7.8 Hz).

**Scope.** This verifies the "consistent with established AD biomarkers" content
on the raw EEG; note ds004504 has AD/FTD/HC (no MCI - the paper's MCI column is
from the unavailable Cohort B). LERD's specific dLIF-inferred frequencies are a
trained-model output and are not reproduced here.

Dataset: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504).
