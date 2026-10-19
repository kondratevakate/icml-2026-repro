# Claim 1: Correctly-labeled samples show consistent entropy decrease during training while mislabeled samples stay relatively high


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3a49047bfe28", "created_at": "2026-07-19T07:36:53+00:00", "title": "Claim 1: Correctly-labeled samples show consistent entropy decrease during training while mislabeled samples stay relatively high"}
-->
**Setup (toy, real data).** Small 3-conv CNN trained on a 12,000-image subset of CIFAR-100 with the real CIFAR-100N human label noise (coarse, 20 superclasses; measured noise rate **26.4%**). Cross-entropy on the *noisy* labels, Adam, 25 epochs, CPU. Each epoch we record every sample's prediction entropy and average it separately over correctly-labeled (noisy==clean) and mislabeled (noisy!=clean) samples. Code `entropy_dynamics.py`.

| epoch | 1 | 5 | 10 | 15 | 20 | 25 |
| --- | --- | --- | --- | --- | --- | --- |
| clean H | 2.64 | 2.02 | 1.56 | 0.94 | 0.39 | 0.14 |
| mislabeled H | 2.73 | 2.27 | 1.96 | 1.45 | 0.70 | 0.26 |
| gap | 0.09 | 0.25 | 0.40 | **0.51** | 0.31 | 0.12 |

**Verdict — reproduced in direction, with an honest caveat.** Mislabeled samples keep **higher** entropy than correctly-labeled ones at *every* epoch, and correctly-labeled entropy decreases consistently — the claim's mechanism (clean examples are learned before noisy ones are memorized). The separation is **largest mid-training** (gap 0.51 at epoch 15). Caveat: the paper's phrasing 'mislabeled maintain high entropy throughout' is true *relatively* but not *absolutely* — with 25 epochs the CNN eventually memorizes the noise too, so mislabeled entropy also collapses (0.26 by epoch 25). This is exactly why the paper's statistic integrates over the whole trajectory rather than using a single endpoint (see Claim 2).
