# Diabetes safety-generalization reproduction

This CPU bundle independently audits five claims from *Safety Generalization
Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed*
(OpenReview `kSUGLBHd0T`, arXiv `2601.21094`).

Prepared score: **5/10**.

## Results

- C1 `PARTIALLY VERIFIED` (1/2): 22/22 official GlucoSim tests pass and all
  three released environments produce deterministic 24-hour CPU trajectories.
- C2 `PARTIALLY VERIFIED` (1/2): one independently reconstructed CPO checkpoint
  reproduces a -11.00 percentage-point TIR gap and +1.99 risk-index gap between
  its training patient and nine unseen patients under a corrected post-step
  metric trace.
- C3 `NOT EXECUTED` (0/2): BA-NODE, ITransformer, and NODE were not trained.
- C4 `VERIFIED` (2/2): the conditional epsilon-margin theorem holds in 320
  boundary cases; weakening the reliability event produces 40 failures.
- C5 `PARTIALLY VERIFIED` (1/2): source audit confirms the shield structure but
  finds incompatible train/load paths, broken cohort offsets, and finite
  penalties that do not implement the theorem's hard permission set.

## Run

Install the pinned official GlucoSim checkout first:

```bash
python prepare_official.py
python -m pip install -e ../official/GlucoSim
python -m pip install -r requirements.txt
```

Then run:

```bash
python -m unittest discover -s tests -v
python audit_simulator.py --output results/simulator_execution.json
python audit_theory_and_release.py \
  --glucoalg ../official/GlucoAlg \
  --output results/theory_and_release.json
python audit_generalization_gap.py \
  --checkpoint ../official/models/t1d-adolescent-cpo/checkpoints/seed0/epoch-2441.pt \
  --config ../official/models/t1d-adolescent-cpo/config/seed0/config.json \
  --output results/generalization_gap_cpo_t1d_adolescent_seed0.json
```

The seven-day checkpoint run is deterministic under evaluation seed
`20260729` but takes several minutes on CPU. It records post-step glucose.
The released evaluator asks for `info["cgm"]`, but GlucoSim omits that key and
the released code consequently falls back to pre-step glucose; this bundle
does not claim an exact replay of that fallback.

## Integrity

The publishable bundle contains no patient records, model weights, paper
source, author repositories, credentials, leaderboard data, or peer evidence.
