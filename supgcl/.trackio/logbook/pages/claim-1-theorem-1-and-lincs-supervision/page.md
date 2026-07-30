# Claim 1: Theorem 1 and LINCS supervision


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_513c2a278721", "created_at": "2026-07-30T07:21:19+00:00", "title": "Claim 1: Theorem 1 and LINCS supervision"}
-->
**Verdict - VERIFIED (2/2).**

Theorem 1 was reconstructed from independent finite probability tables.
Across `100` seeded tables, direct joint
KL and the chain-rule decomposition agreed to maximum absolute error
`2.22e-16`.

Pinned author source independently confirms that LINCS knockdown graphs and
the knockdown-gene metadata feed `TeacherSampler`. This verifies the stated
mathematical decomposition and data mechanism, not downstream performance.


---
<!-- trackio-cell
{"type": "code", "id": "cell_743b49ef2317", "created_at": "2026-07-30T07:21:19+00:00", "title": "C1 machine-readable evidence", "language": "python"}
-->
````output
{
  "finding": "The KL decomposition holds independently, and pinned source connects LINCS knockdown graphs to the sampler.",
  "id": "C1",
  "independent_probability_tables": 100,
  "max_chain_rule_absolute_error": 2.220446049250313e-16,
  "official_code_anchors": {
    "augmentation_kl": true,
    "kd_gene_mapping": true,
    "knockdown_metadata": true,
    "lincs_graphs": true,
    "node_infonce": true,
    "teacher_sampler": true
  },
  "paper_anchors": {
    "corollary_limit": true,
    "factorization_assumption": true,
    "kl_decomposition": true,
    "uniform_limit": true
  },
  "score": 2,
  "verdict": "VERIFIED"
}
````
