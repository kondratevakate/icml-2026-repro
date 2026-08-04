# plan.md

Start of work: 2026-08-01 23:57 (+04). Budget: soft 4h, hard 8h, 2h/claim.

Common numeric substrate (all CPU, numpy/scipy only): finite response space
Y = {y_1..y_K} with a scalar reward vector r. For BoN over a discrete distribution with
ties broken uniformly, the exact BoN objective is
  R[pi] = sum_k r_k (F_k^N - F_{k-1}^N)  = r_max - sum_{k<K} (r_{k+1}-r_k) F_k^N   (rewards sorted ascending)
and its functional derivative (discrete analogue of Prop 4.2) is
  dR/dpi_j = - sum_{k>=j, k<K} (r_{k+1}-r_k) * N * F_k^(N-1).
Exact mirror-descent step of Alg. 1 has a closed form on the simplex:
  pi_{t+1} propto exp( ( rtilde_t + beta log pi_ref + (1/eta) log pi_t ) / (beta + 1/eta) ).

| claim | content | class | feasible on CPU? | approach |
|---|---|---|---|---|
| 1 | Eq. (1) objective; R non-linear in pi via inference-time transform; one base policy adapts to m criteria | theory/definition + simulation | YES | Build R[pi]=g(R_1,..,R_m)-beta KL with BoN transforms on finite Y. Test (a) non-linearity: R[(1-t)pi_a+t pi_b] != (1-t)R[pi_a]+tR[pi_b] for BoN N>=2, while the un-transformed E_pi[r] is exactly linear; (b) multi-criterion adaptation: one trained base pi yields, under different T_i, higher R_i than the naive base. Mutation: N=1 (BoN degenerates to identity) must restore exact linearity. |
| 2 | Alg. 1 = GRPO with reward replaced by dR/dpi at empirical pihat | theory + simulation | YES | (a) verify Prop 4.2 discrete derivative against central finite differences of R along simplex directions; (b) verify the "drop-in" statement: running the standard GRPO mirror-descent map with rtilde substituted for r maximizes the true non-linear objective; (c) verify the empirical-pihat version (M samples) converges to the population one at O(1/M). Mutation: substitute the plain reward r (standard GRPO) -> converges to a different, strictly worse fixed point of the IAMA objective. |
| 3 | Thm 5.2 exact-update bound beta*KL[pi*|pi_0]/(((L+beta)/L)^T - 1) | theory (bound certification) | YES | Estimate a valid relative-smoothness L numerically (max over random pi,pi' pairs of the smoothness violation ratio, with margin), compute pi* by an INDEPENDENT solver (L-BFGS on softmax parametrization), run exact Alg. 1 with eta=1/L, and check the bound holds at every T for many random instances (exhaustive over N, beta, K grid x seeds). Mutation: eta = 20/L (violates the theorem's step size) -> bound expected to fail / rate degrade. |
| 4 | Thm 5.3 inexact bound with additive 2(eps+delta)/beta | theory (bound certification) | YES | Inject controlled derivative noise with span-seminorm second moment <= eps and a proximal residual with <= delta, average over many runs and the theorem's random index distribution, check E[L[pi_that]-L[pi*]] <= (beta/2)KL/(((L+beta/2)/L)^T-1) + 2(eps+delta)/beta. Also check the bias floor scales like 1/beta. Mutation: remove the 2(eps+delta)/beta term (i.e. test against the exact Thm 5.2 bound) -> violated at large noise, showing the term is necessary. |
| 5 | Prop 3.1 closed-form bimodal IAMA optimum vs delta_0.5 naive optimum | theory + simulation | YES | (a) check pi*_IAMA integrates to 1 for N=2,4,8,16; (b) first-order optimality: 0.5*dR_1/dpi(y)+0.5*dR_2/dpi(y) must be constant in y on the support at pi=pi*_IAMA (beta=0, interior simplex max); (c) numerically maximize the IAMA objective on a fine grid by mirror ascent from several inits and compare to closed form; (d) naive objective optimum concentrates at y=0.5. Mutation: N=2 must give the uniform density (alpha=1) and perturbing alpha away from 1/(N-1) must break the constant-derivative condition. |
| 6 | Sec 6.2 HH-RLHF / Alpaca-7B Pareto front with BoN N=4 | data + GPU (7B RL training) | NO | Requires RLHF training of a 7B policy plus training two Qwen3-4B Bradley-Terry reward models and two Qwen-32B golden judges; no public code URL in the paper (supplementary only); far beyond CPU / 2h. Verdict will be `inconclusive` with reason. NO toy substitute.

Order of work: 5, 1, 2, 3, 4, then 6 (refusal).
