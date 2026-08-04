# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_92b80e4931b8", "created_at": "2026-08-02T04:30:00+00:00", "title": "Overall findings"}
-->
**Overall findings** in per-claim pages. Verdicts follow the evidence; mutation tests confirm mechanism, not correlation; inconclusive claims state the blocker honestly.

<!-- trackio-cell
{"type": "markdown", "id": "cell_51574c47fa0a", "created_at": "2026-08-02T04:30:00+00:00", "title": "Evidence boundary"}
-->
The verification claims 1-3 are theory-driven with exhaustive grid testing (3200+ instances), mutation tests confirm mechanism dependence, and all bounds hold within tight tolerance.

Claims 4-5 involve neural network training; this run used CPU (torch-cpu) with reduced epochs/initial states vs paper's GPU training. The qualitative claim (hard constraints satisfied, baselines fail) is reproduced. MSE reduction (75.58% vs paper 73.33%) aligns closely. Full paper-scale reproduction would require GPU-backed training.

---
<!-- trackio-cell
{"type": "artifact", "id": "cell_da553571a390", "created_at": "2026-08-02T04:30:00+00:00", "title": "Reproduction bundle", "artifact": "repro-caffnet-hard-constraint-affine-neural-networks/repro-bundle:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-caffnet-hard-constraint-affine-neural-networks/repro-bundle:v0` · dataset

https://huggingface.co/buckets/kondratevakate/repro-caffnet-hard-constraint-affine-neural-networks-artifacts#repro-caffnet-hard-constraint-affine-neural-networks/repro-bundle:v0

---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_4b2db7d0b4ee", "created_at": "2026-08-02T04:30:00+00:00", "title": "Dashboard: repro-caffnet-hard-constraint-affine-neural-networks", "dashboard_project": "repro-caffnet-hard-constraint-affine-neural-networks"}
-->
**🎯 Trackio dashboard** `repro-caffnet-hard-constraint-affine-neural-networks`

trackio-local-dashboard://repro-caffnet-hard-constraint-affine-neural-networks
