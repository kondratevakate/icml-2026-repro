# Reproduction task — Explaining Concept Shift with Interpretable Feature Attribution

OpenReview: https://openreview.net/forum?id=wpKA7G7Cqu
Area: General Machine Learning
Anchored claims (5):

1. SGShift-KA (sparse GAM with knockoffs and absorption term) achieves AUC greater than 0.9 for identifying shifted features across semi-synthetic benchmarks, outperforming the Diff, WhyShift, and SHAP-difference baselines by 2-3x in recall (Table 2).

2. On the Diabetes Readmission dataset (73,615 samples, 33 features, split by ER vs. non-ER admission), SGShift variants detect true shifted features with recall exceeding 80-90% (Table 2).

3. Sample-efficiency experiments show SGShift's knockoff variants (SGShift-K, SGShift-KA) retain top detection performance even when the number of samples available from the shifted (target) domain is small (Figure 1).

4. On the COVID-19 Hospitalizations dataset (16,187 samples, 30 features, split pre/post-Omicron) and the SUPPORT2 dataset (9,105 samples, 64 features, split by survival vs. death outcome), SGShift's real-world validation shows an elbow-shaped performance curve indicating that only a sparse subset of features accounts for the concept shift (Figure 2).

5. SGShift variants consistently outperform the Diff, WhyShift, and SHAP baselines across both matched and mismatched model-specification configurations (Table 2).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
