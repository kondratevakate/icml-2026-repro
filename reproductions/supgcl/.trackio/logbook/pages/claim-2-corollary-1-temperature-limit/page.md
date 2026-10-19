# Claim 2: Corollary 1 temperature limit


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7fd6a9e0c57f", "created_at": "2026-07-30T07:21:19+00:00", "title": "Claim 2: Corollary 1 temperature limit"}
-->
**Verdict - VERIFIED (2/2).**

Across `25` independent similarity and
node-loss tables, increasing `tau_a` made both augmentation distributions
uniform with zero monotonicity failures. At `tau_a=100000`, the worst
objective distance to the uniformly averaged node loss was
`3.38e-07` and the worst augmentation
KL was `1.35e-10`.

The analytic softmax limit independently establishes the exact
`tau_a -> infinity` result stated by Corollary 1.


---
<!-- trackio-cell
{"type": "code", "id": "cell_747fb9dca633", "created_at": "2026-07-30T07:21:19+00:00", "title": "C2 machine-readable evidence", "language": "python"}
-->
````output
{
  "finding": "As tau_a grows, p and q become uniform, augmentation KL vanishes, and the objective reaches the uniform node-loss average.",
  "id": "C2",
  "independent_probability_tables": 25,
  "max_final_augmentation_kl": 1.3504297996357e-10,
  "max_final_distance_to_node_limit": 3.377662336212506e-07,
  "score": 2,
  "temperatures": [
    {
      "augmentation_kl": 0.49564176230434304,
      "distance_to_node_limit": 0.5146694225090077,
      "loss": 0.8325452660354015,
      "max_student_distance_to_uniform": 0.27026661566463445,
      "max_teacher_distance_to_uniform": 0.2622796472477671,
      "tau": 1.0
    },
    {
      "augmentation_kl": 0.006037238784011302,
      "distance_to_node_limit": 0.007937199243047977,
      "loss": 0.32581304276944173,
      "max_student_distance_to_uniform": 0.033556545866929716,
      "max_teacher_distance_to_uniform": 0.02761266501820475,
      "tau": 10.0
    },
    {
      "augmentation_kl": 6.103709586157246e-05,
      "distance_to_node_limit": 0.00025153580940462383,
      "loss": 0.3181273793357984,
      "max_student_distance_to_uniform": 0.0035546709300006696,
      "max_teacher_distance_to_uniform": 0.002887476136716638,
      "tau": 100.0
    },
    {
      "augmentation_kl": 6.109587334303363e-07,
      "distance_to_node_limit": 1.9666917423333263e-05,
      "loss": 0.3178955104438171,
      "max_student_distance_to_uniform": 0.00035745265769634993,
      "max_teacher_distance_to_uniform": 0.0002899998890758815,
      "tau": 1000.0
    },
    {
      "augmentation_kl": 6.110167150978165e-09,
      "distance_to_node_limit": 1.9117679654900854e-06,
      "loss": 0.31787775529435924,
      "max_student_distance_to_uniform": 3.576510880151651e-05,
      "max_teacher_distance_to_uniform": 2.90124982450235e-05,
      "tau": 10000.0
    },
    {
      "augmentation_kl": 6.110221111762557e-11,
      "distance_to_node_limit": 1.9062750239440263e-07,
      "loss": 0.31787603415389615,
      "max_student_distance_to_uniform": 3.5767092970429015e-06,
      "max_teacher_distance_to_uniform": 2.9013749041073567e-06,
      "tau": 100000.0
    }
  ],
  "uniform_convergence_monotonic_failures": 0,
  "verdict": "VERIFIED"
}
````
