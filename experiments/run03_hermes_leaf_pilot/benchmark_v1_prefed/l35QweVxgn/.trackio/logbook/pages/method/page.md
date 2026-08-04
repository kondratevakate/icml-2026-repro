## Method

All six anchored claims are analytic / closed-form bounds or a decomposition. They were reproduced in two complementary ways:
- **(A) Analytic:** the closed-form expressions the paper derives (Theorems 2.1, 2.2, 2.3 and the improved gap Theorem B.1) were evaluated and their scaling dependence on d, n, m, K, eta, T verified by exact ratio tests and by checking the prescribed parameter regime (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)) drives every bound to o_d(1) (poly-logarithmically).
- **(B) Empirical (kernel-regime):** the XOR-cluster data model (mutually orthogonal task means of norm 1/sqrt(d), Gaussian noise sigma=O(1/(polylog(d) sqrt(d)))) was generated and the KERNEL-REGIME closed-form forgetting object the theorems bound (Eq. after Thm 2.2: F_tr(k)=|(1/n) sum_{x_k} eta*T x_k^T (sum_{j>k} A_j) x_k|) was computed directly, confirming the qualitative behaviour (small under orthogonality, 1/sqrt(n) sample-fluctuation, breaks under non-orthogonality).

Every verified claim has a MUTATION test: perturbing the setup (breaking task-mean orthogonality, or breaking one factor of the parameter regime, or making the per-step loss non-self-bounded) makes the claimed property break or shift, as required.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Method"}\n-->
