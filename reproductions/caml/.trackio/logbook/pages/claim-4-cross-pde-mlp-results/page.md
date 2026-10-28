# Claim 4: cross-PDE MLP results


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a140556b6b7f", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 4: cross-PDE MLP results"}
-->
**UNSUPPORTED - 0/2.** Heat, NS, and Helmholtz entrypoints contain
five paper/code setting mismatches, and no empirical output is cached. The
four-PDE table cannot be regenerated canonically from the release as-is.


---
<!-- trackio-cell
{"type": "code", "id": "cell_8b10a1ec2365", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 4: cross-PDE MLP results evidence", "language": "python"}
-->
````output
{
  "script_assignments": {
    "heat": {
      "backbone": "'mlp'",
      "loss": "'caml'",
      "lr": "1e-3",
      "w_res": "1.0",
      "w_bc": "1.0",
      "t_d": "25",
      "t_r": "50",
      "min_epochs": "6000",
      "max_epochs": "20000",
      "target_l2": "1e-3"
    },
    "poisson": {
      "backbone": "'mlp'",
      "loss": "'caml'",
      "lr": "1e-3",
      "w_res": "1.0",
      "w_bc": "100.0",
      "t_d": "200",
      "t_r": "800",
      "min_epochs": "6000",
      "max_epochs": "20000",
      "target_l2": "1e-2"
    },
    "ns": {
      "backbone": "'mlp'",
      "loss": "'caml'",
      "lr": "1e-3",
      "w_res": "1.0",
      "w_bc": "100.0",
      "t_d": "25",
      "t_r": "50",
      "min_epochs": "6000",
      "max_epochs": "20000",
      "target_l2": "1e-2"
    },
    "helm": {
      "backbone": "'mlp'",
      "loss": "'caml'",
      "lr": "1e-3",
      "w_res": "1.0",
      "w_bc": "1.0",
      "t_d": "25",
      "t_r": "50",
      "min_epochs": "6000",
      "max_epochs": "20000",
      "target_l2": "1e-3"
    }
  },
  "paper_mlp_subset": {
    "heat": {
      "w_bc": "5",
      "min_epochs": "6000",
      "target_l2": "2.0e-3"
    },
    "poisson": {
      "w_bc": "100",
      "min_epochs": "6000",
      "target_l2": "1.0e-2"
    },
    "ns": {
      "w_bc": "100",
      "min_epochs": "6000",
      "target_l2": "5.0e-3"
    },
    "helm": {
      "w_bc": "10",
      "min_epochs": "4000",
      "target_l2": "1.0e-3"
    }
  },
  "mismatches": [
    {
      "benchmark": "heat",
      "parameter": "w_bc",
      "paper": "5",
      "code": "1.0"
    },
    {
      "benchmark": "heat",
      "parameter": "target_l2",
      "paper": "2.0e-3",
      "code": "1e-3"
    },
    {
      "benchmark": "ns",
      "parameter": "target_l2",
      "paper": "5.0e-3",
      "code": "1e-2"
    },
    {
      "benchmark": "helm",
      "parameter": "w_bc",
      "paper": "10",
      "code": "1.0"
    },
    {
      "benchmark": "helm",
      "parameter": "min_epochs",
      "paper": "4000",
      "code": "6000"
    }
  ],
  "mismatch_count": 5,
  "readme_claims_only_standard_torch": true,
  "undocumented_overrides_import": true,
  "baseline_loss_implementations_released": [
    "pinn",
    "caml"
  ],
  "paper_baseline_loss_count": 6,
  "cached_empirical_outputs_released": false
}
````
