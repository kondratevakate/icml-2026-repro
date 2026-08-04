# COMPACTION.md (turn ~18)

Run: 20hdQQQrA4_arm2b (CAffNet, orid 20hdQQQrA4), arm2, Hermes + K-Dense skills,
model tencent/hy3:free via localhost:8319/v1. CPU only.

## Files
notes_paper.md (single paper read), caffnet_core.py (numpy CAffNet Eq 8/9/12 + HardNet),
verify_claim1..5.py, results/claim{1,3}.json done; claim2/claim4/claim5 running in bg.

## CLAIM 1 (Thm 3.5 bound (3+3sqrt(n_out))K) — DONE, VERIFIED
symbolic_constant_closes = true; K=1e-3, p=2, 200 seeds/config, 8 configs, 0 infeasible.
max ratio ||P*-f_t||/K [bound]:
 (1,2) 0.9971001586566164 ; (1,4) 0.9923570959997907  [6.0]
 (2,3) 1.6424769212615287 ; (2,5) 1.446907792078362   [7.242640687119286]
 (3,4) 1.6027273263822814 ; (3,7) 1.2320821394200527  [8.196152422706632]
 (4,6) 1.68252632685596   [9.0] ; (5,8) 1.686931229412946 [9.708203932499369]
overall max_ratio 1.686931229412946; 8/8 hold; also under tighter 1+sqrt(n_out).
MUTATION (argmax instead of argmin in Eq 12) — bound broken 8/8:
 18.96320931916251 / 7.846957574033642 / 112.1564423072167 / 126.41046987139862 /
 71.4357431773435 / 275.42964969616213 / 202.0805111475756 / 677.5939335688797
Pitfall fixed: first version was vacuous (target interior -> f_theta always feasible).

## CLAIM 3 (no full row rank; cardinality min(m,n_out)) — DONE, VERIFIED
2100 instances (7 configs x 3 kinds {generic, duplicate_rows, rank_deficient} x 100 seeds):
 caffnet_instances_with_no_feasible_candidate = 0
 caffnet_instances_with_constraint_violation  = 0
 caffnet_max_violation_over_all_instances     = 1.1075584893660562e-12
 hardnet_instances_undefined_due_to_rank_deficiency = 1961 / 2100
 hardnet_instances_with_constraint_violation  = 0 (of the 139 where it is defined)
 MUTATION cap k <= min(m,n_out)-1: 478 / 2100 failures (feasibility lost)
 T3_max_cardinality_equals_min_m_nout = true ; T3_gamma_count_identity_and_2m_bound = true
 (checked exhaustively m=1..10, n_out=1..5)

## CLAIM 4 (73.33% MSE reduction, Table 2) — RUNNING
verify_claim4.py --epochs 20000 --seeds 5 (paper: 50000 epochs; reduced for CPU budget).
Smoke run (300 epochs, 1 seed): NN mse 0.1306.., CAffNet-FF/-TF worse (untrained),
CAffNet viol_max = 0.0 both, NN has violations. First real-run line:
 {'method':'NN','seed':0,'test_mse':0.000729856789123326,'viol_max':0.1333959099771358,
  'viol_mean':0.0006516323185225007,'train_s':162.76476097106934}

## CLAIM 5 (safety-critical control) — RUNNING (--epochs 8 --ntrain 12 --seeds 3)
Full D.3 setup implemented (3 polytope obstacles, smooth-union CBF kappa=10, state box,
control box; m=13, n_out=2, |Gamma|=91, dt=0.1, 150 steps, PID nominal + net correction).
Smoke run (epochs=1, ntrain=6, 49 eval states):
 NN-soft: collisions 32/49, control_violations 49, arrived 0, infeasible_steps 0
 CAffNet-FF: collisions 1/49, control_violations 1, arrived 13, infeasible_steps 5
 CAffNet-FF-aposteriori: collisions 1/49, control_violations 1, arrived 10
KEY finding: CAffNet's residual collision/violation coincides with infeasible_steps>0,
i.e. steps where S(x) is EMPTY (Assumption 3.2 violated by discrete-time CBF + tight
control bounds) -> guarantee is conditional, as the paper states.
HardNet NOT reproducible here: A(x) is 13x2, not full row rank -> formula undefined.

## FINAL (turn ~30) — all runs completed
CLAIM 2 (DONE, verified): T1 max residual 1.1834977442504169e-13 (360 checks);
 T2 rank-deficient 86, min/mean spread 1.530619010628436 / 2.732280847454395,
 full-rank zero-nullspace 34; T3 obj trainable w 1.859256323957484 vs w=0 4.500871503391247,
 median gap 0.11952442114037204, max gap 26.87171091351931, frac better 0.6;
 MUTATION_w_forced_zero_is_never_better = true.
CLAIM 4 (DONE, inconclusive; 20000 epochs, 5 seeds):
 NN mse 0.001839826621879348 (std 0.0024093821342899414), viol_max_mean 0.1705633378904027, 5/5 seeds violate
 CAffNet-FF mse 0.002884547281501829 (std 0.0019290806990403236), viol 0.0, 0/5
 CAffNet-TF mse 0.0009070503751830563 (std 0.0005683786123839912), viol 0.0, 0/5
 reduction TF vs NN = 50.69913847335672 % (paper 73.33 %); FF vs NN = -56.783647284944635 %.
CLAIM 5 (DONE, inconclusive; epochs 8, ntrain 12, 3 seeds, 49 eval states):
 NN-soft collisions (24,19,18) mean 20.333333333333332, ctrl viol mean 24.0, arrived 47.666666666666664
 CAffNet-FF collisions (1,1,1) mean 1.0, ctrl viol mean 0.3333333333333333, arrived 15.666666666666666,
   infeasible_steps 4 per seed (empty S(x) -> where the collision happens)
 CAffNet-FF-aposteriori collisions mean 1.0, arrived 13.333333333333334
 HardNet not reproducible (A(x) 13x2, not full row rank).
check_reproducibility.py: ALL LOGBOOK NUMBERS RE-ASSERTED OK (exit 0).
Verdicts: 1 verified, 2 verified, 3 verified, 4 inconclusive, 5 inconclusive.
