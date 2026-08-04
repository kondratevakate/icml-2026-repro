# notes_paper.md — "Conditional Coverage Diagnostics for Conformal Prediction" (arXiv 2512.11779)

IMPORTANT VERSIONING FACT (established by reading both PDFs once):
- **v2 (29 May 2026)** = ICML camera-ready style; has only Tables 1–3, Figure 1 = sample-size figure,
  Section 4.1 Table 2 numbers: LightGBM L1 68.9±1.8, PartitionWise L1 33.7±1.9. No Table 4, no classification section.
- **v1** = the version the anchored claims point to: Table 2 has LightGBM (medium) L1-ERT **68.4±2.2**,
  PartitionWise **38.3±1.9**; Figure 4 = sample-size figure (Sec 4.2); Table 4 = classification ERT scores.
  All claim anchors (Table 2, Figure 4/Sec 4.2, Table 4/Sec 4.3.2) match **v1**. Notes below use v1 numbers,
  cross-checked with v2 where the text is identical.

## Definitions (Sec 3.1, identical in both versions)
- p(x) := P(Y ∈ C_α(X) | X=x); Z := 1{Y ∈ C_α(X)}; target coverage 1−α.
- Risk R_ℓ(h) := E[ℓ(h(X), Z)]. Bayes predictor h*(x) = argmin_q E[ℓ(q,Z)|X=x].
- For a **proper** loss ℓ, h*(X) = E[Z|X] = p(X). Under perfect conditional coverage p ≡ 1−α, hence the
  constant predictor 1−α is Bayes-optimal → **no classifier can beat the constant 1−α**. (Claim 1 principle.)
- ℓ-ERT := R_ℓ(1−α) − R_ℓ(p) = E_X[d_ℓ(1−α, p(X))], d_ℓ(p,q) := E_{y~q}[ℓ(p,y) − ℓ(q,y)].
- Plug-in with any h: ℓ-ERT(h) := R_ℓ(1−α) − R_ℓ(h) ≤ ℓ-ERT  (lower bound, always).

## Table 1 (Sec 3.1) — proper scores and ERT formulas
| name | ℓ(p,y) | ℓ-ERT |
|---|---|---|
| L1-ERT | sgn(p−(1−α))·((1−α) − y) | E_X[|p(X) − (1−α)|] |
| L2-ERT | (y − p)^2 | E_X[((1−α) − p(X))^2] |
| KL-ERT | −log p_y  (p_1=p, p_0=1−p) | E_X[D_KL(p(X) ‖ 1−α)] |
D_KL(p‖q) = p log(p/q) + (1−p) log((1−p)/(1−q)).

## Prop 3.1 (Sec 3.2) — representing convex losses as ERTs
f:[0,1]→R≥0 convex, f(1−α)=0, subderivative f′ with f′(1−α)=0.
ℓ_{f,f′}(p,y) := −f(p) − (y−p) f′(p) is a proper score with ℓ-ERT = E_X[f(p(X))].

## Sec 3.3 — over/under-coverage decomposition (Claim 4)
f = f₊ + f₋ with f₊(p) := f(max{p,1−α}) (over-coverage only), f₋(p) := f(min{p,1−α}) (under-coverage only).
ℓ(p,y) = ℓ₊(p,y) + ℓ₋(p,y) − ℓ(1−α,y);  ℓ-ERT = ℓ₊-ERT + ℓ₋-ERT;
ℓ₊(p,y) := ℓ(max{p,1−α}, y), ℓ₋(p,y) := ℓ(min{p,1−α}, y).

## Algorithm 1 (Claim 6) — finite-sample estimator
Input {(X_i,Z_i)}_{i=1..m}, k≥2 folds, proper score ℓ, level α, classifier method.
Partition into k folds; for each fold j: train h^(j) on complement, evaluate
ERT^(j) = (1/|I_val^(j)|) Σ_{i∈I_val} [ℓ(1−α, Z_i) − ℓ(h^(j)(X_i), Z_i)];
aggregate ERT = Σ_j (|I_j|/m) ERT^(j). Rationale: "we cannot train h on the values X_i it is evaluated on"
(avoid overfitting/misleading diagnostics). Random forest may instead use out-of-bag predictions.

## Sec 4.1 / Table 2 (Claim 2) — classifier comparison
Setup: 8 largest **regression** datasets in TabArena. Split 40% train f (MSE), 10% calib with
S(X,Y)=|Y−f(X)|, 1−α=0.9, 50% test subsampled to several sizes; 5-fold CV; averaged over 10 runs.
Reported metric: average % of the *maximum* recovered ERT across all methods and sample sizes.
v1 Table 2 (mean_sd): TabICLv1.1 71.9_1.9 / RealTabPFN-2.5 71.6_1.7 / CatBoost 68.7_2.8 /
**LightGBM(medium) 68.4_2.2** / ExtraTrees 65.9_2.4 / RandomForest 65.9_2.8 / **PartitionWise 38.3_1.9**
(L1-ERT column). Times per 1K samples [s]: LightGBM 2.6, PartitionWise 0.2. TabICL/TabPFN = GPU.

## Sec 4.2 / Figure 4 (Claim 3) — synthetic sample-efficiency
DGP: X ~ U([−1,1]^8), Y ~ N(0, σ(X1)), σ(x) = 0.5 + |x| + x².
Two set rules: (a) Standard CP with S(X,Y)=|Y| calibrated on 3,000 iid samples (marginally valid,
conditionally invalid); (b) Oracle sets = true conditional α/2 and 1−α/2 quantiles (conditional by construction).
Metrics vs #test points on log scale (10³→10⁵), 5-fold CV for ERT. Claim text: group-based metrics
(CovGap) "extremely unaligned with their theoretical values"; "even with 5,000 points they provide nearly
identical diagnostics across these two very different scenarios"; WSC similar instability even at 50,000;
L1-ERT stabilizes very quickly.
Related Table 3 (v2, test n=1,500, α=0.1): non-conditional vs conditional —
L1-ERT 0.091_0.007 vs −0.005_0.009 (true value ≈0.098); L2-ERT 0.009_0.001 vs −0.000;
CovGap 0.016_0.004 vs 0.014_0.004; WCovGap same; WSC 0.740 vs 0.790; FSC 0.868 vs 0.881.

## Table 4 / Sec 4.3.2 (Claim 5) — classification
Datasets MNIST, FashionMNIST, CIFAR10 (small CNN: 2 conv+maxpool, 2 FC, dropout, ReLU, early stopping
when accuracy fell below 1−α), CIFAR100 (ResNet; ERT classifier reuses pretrained trunk, new final layer).
Two conformal score strategies: "cumulative" (APS-style cumulative sum of sorted softmax) and "likelihood"
(score = predicted prob of the label). Reported (mean_sd):
| Dataset | Method | L1-ERT | KL-ERT | KL₊-ERT | KL₋-ERT |
| CIFAR10 | cumulative | 0.072 | −0.017 | −0.030 | 0.012 |
| CIFAR10 | likelihood | 0.016 | 0.028 | 0.007 | 0.022 |
| CIFAR100 | cumulative | 0.041 | 0.191 | 0.016 | 0.175 |
| CIFAR100 | likelihood | 0.007 | 0.409 | 0.085 | 0.323 |
| FashionMNIST | cumulative | 0.165 | −0.260 | −0.185 | −0.075 |
| FashionMNIST | likelihood | 0.098 | −0.068 | −0.042 | −0.026 |
| MNIST | cumulative | 0.150 | −0.216 | −0.159 | −0.057 |
| MNIST | likelihood | 0.145 | −0.187 | −0.128 | −0.059 |
Text: for CIFAR100 L1-ERT and KL-ERT give opposite orderings; likelihood produces more empty sets
(conditional coverage 0), so KL₋-ERT > KL₊-ERT there.

Code repos (not yet consulted): github.com/ElSacho/covmetrics ; ElSacho/Conditional_Coverage_Estimation.
