# SwitchCraft reproduction bundle

Independent audit bundle for *SwitchCraft: A Programmatic Framework for
Designing State-Switching Proteins* (`YtqaHnqv8c`, arXiv `2605.31236v1`).

## Local audits

```bash
python audit_schedule.py --official-repo /path/to/switchcraft \
  --output results/schedule.json
python audit_losses.py --official-repo /path/to/switchcraft \
  --output results/losses.json
```

The schedule audit parses the official source without importing Boltz or
requiring CUDA. The loss audit executes the official loss classes on seeded
CPU tensors and compares them with independent implementations of Equations 1,
2, and 5.

## GPU gate

The paper-scale functional claims require CUDA, Boltz-1 weights, the CCD
database, and fresh generated structures. No GPU result is claimed by the
local audits. The first GPU command should be a single positive-allostery
design:

```bash
python switchcraft.py --config tasks/pos_allostery.yaml \
  --num_designs 1 --recycles 0 --verbose --visualize \
  --outpath /workspace/output/pos_allostery
```

Do not claim six-primitives coverage from this smoke test. Full C3 includes
six task families and paper-scale filtering; the supplied repository contains
templates rather than exact paper configs for several complex tasks.

## Sources

- Paper: https://arxiv.org/abs/2605.31236
- Official code: https://github.com/bjing2016/switchcraft

