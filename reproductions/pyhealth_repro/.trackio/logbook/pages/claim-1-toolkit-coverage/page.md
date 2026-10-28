# Claim 1: toolkit coverage

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pyhealth_c1_001", "created_at": "2026-07-29T14:30:00+00:00", "title": "Claim 1: toolkit coverage"}
-->
**Anchored claim (verbatim).** "PyHealth 2.0 unifies 15+ datasets, 20+ clinical
tasks, 25+ models, and 5+ interpretability methods (Attention-Grad, GIM,
DeepLift, SHAP among them) spanning EHR, imaging, physiological signal, and
genomic modalities (Section 3; Appendices F-J)."

**Verdict -- VERIFIED.**

The independent audit parses the four overview tables from the public arXiv v2
source:

| Inventory | Paper total | Claimed minimum |
| --- | ---: | ---: |
| Datasets | 22 | 15 |
| Clinical tasks | 43 | 20 |
| Models | 28 | 25 |
| Interpretability methods | 7 | 5 |

It then parses public imports from the official `v2.0.1` release. After
excluding base classes, layers, and helper functions, the release exports 24
dataset classes, 51 task callables/classes, 40 models, and 8 interpretation
methods. Explicit dataset witnesses are `MIMIC4Dataset` (EHR),
`ChestXray14Dataset` (imaging), `SleepEDFDataset` (physiological signal), and
`ClinVarDataset` (genomics). The named interpretation witnesses are
`CheferRelevance`, `GIM`, `DeepLift`, and `ShapExplainer`.

All ten threshold and witness checks pass. As a destructive control, the test
removes all three genomic dataset exports; the modality check then fails and
the verdict flips to `FALSIFIED`.

This is a full source-level reproduction of a toolkit-inventory claim. It does
not claim that every model was trained on every dataset.
