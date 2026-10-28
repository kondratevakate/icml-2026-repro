# Frozen forecast

Locked at `2026-07-30T12:04:00+04:00`, after inventorying the paper, source,
official repository at commit `8f4daabf0db60f4f185a1a9b791ad7d7be41a033`,
and the four synthetic PDE entrypoints. This forecast was fixed before any
independent numerical or training run.

## Forecast

**7/12 points**, plausible range **5-9/12**.

| Claim | Forecast | Pre-check basis |
| --- | ---: | --- |
| C1 loss-valley theorem | 0/2 | The headline implication appears stronger than the assumptions introduced only inside the appendix proof. |
| C2 aligned-constraint mechanics | 2/2 | Equations and a compact official implementation are public and directly testable. |
| C3 Heat benchmark | 2/2 | The benchmark is analytic, seeded, CPU-capable, and has a complete entrypoint. |
| C4 cross-PDE MLP results | 1/2 | All four analytic problems are released, but canonical five-seed budgets total many long second-order-autodiff runs. |
| C5 backbone generality | 1/2 | All three backbones are public, but the full 36-cell comparison is locally expensive. |
| C6 components and limitations | 1/2 | Core schedule and offset ablations are constructible; full optimizer/sensitivity tables lack dedicated runners. |

The main upside is a self-contained analytic benchmark suite with no external
data. The main downside is that every published value is regenerated from
training rather than supplied as a cached output, and the canonical study used
an RTX 5090.
