#!/usr/bin/env python3
"""Build the static Trackio logbook from frozen diabetes-safety evidence."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT / ".trackio"
SPACE_ID = "kondratevakate/repro-safety-generalization-diabetes-testbed"
TITLE = (
    "Reproduction: Safety Generalization Under Distribution Shift in Safe "
    "Reinforcement Learning: A Diabetes Testbed"
)
PAGES = {
    "C1": "Claim 1: unified diabetes simulator",
    "C2": "Claim 2: safety-generalization gap",
    "C3": "Claim 3: BA-NODE prediction accuracy",
    "C4": "Claim 4: conditional safety theorem",
    "C5": "Claim 5: released predictive shield",
}


def create_logbook() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT}")
    original_cwd = Path.cwd()
    original_find = lb.find_project_dir
    try:
        os.chdir(ROOT)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = original_find
        os.chdir(original_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected logbook path: {created}")


def load_evidence() -> dict:
    return {
        "simulator": json.loads(
            (HERE / "results" / "simulator_execution.json").read_text(
                encoding="utf-8"
            )
        ),
        "generalization": json.loads(
            (
                HERE
                / "results"
                / "generalization_gap_cpo_t1d_adolescent_seed0.json"
            ).read_text(encoding="utf-8")
        ),
        "audit": json.loads(
            (HERE / "results" / "theory_and_release.json").read_text(
                encoding="utf-8"
            )
        ),
    }


def claim_text(claim_id: str, evidence: dict) -> str:
    if claim_id == "C1":
        simulator = evidence["simulator"]
        hashes = [
            item["trajectory_sha256"]
            for item in simulator["environments"].values()
        ]
        return f"""**Verdict - PARTIALLY VERIFIED (1/2).**

All 22 official GlucoSim tests pass on CPU. Independent 288-step runs execute
all three released environment IDs with 14-dimensional observations and
`MultiDiscrete([5, 5])` actions. Fixed-seed repeats are identical; the three
trajectory hashes begin `{hashes[0][:10]}`, `{hashes[1][:10]}`, and
`{hashes[2][:10]}`.

The full eight-algorithm policy benchmark was not rerun, so this claim receives
partial credit.
"""
    if claim_id == "C2":
        gap = evidence["generalization"]
        return f"""**Verdict - PARTIALLY VERIFIED IN ONE SCOPED RUN (1/2).**

A released CPO/T1D/adolescent/seed-0 checkpoint was reconstructed independently
from its config and state dictionary. One seven-day rollout gives ID TIR
`{gap["id"]["tir_percent"]:.2f}%` and risk index
`{gap["id"]["risk_index"]:.2f}`. Across patients 2-10, mean TIR is
`{gap["ood_mean_tir_percent"]:.2f}%` and risk index is
`{gap["ood_mean_risk_index"]:.2f}`: gaps of
`{gap["tir_gap_ood_minus_id"]:.2f}` percentage points and
`+{gap["risk_gap_ood_minus_id"]:.2f}`.

The metric trace records post-step glucose. The released evaluator requests
`info["cgm"]`, but GlucoSim omits that key and the released code falls back to
the pre-step observation. This is therefore a corrected-protocol mechanism
check, not an exact replay of that fallback or the paper's three-seed,
eight-algorithm aggregate.
"""
    if claim_id == "C3":
        return """**Verdict - NOT EXECUTED (0/2).**

BA-NODE, ITransformer, and NODE were not trained and their reported
MAE/FDE/RMSE values are not copied from the paper as evidence. The public
transition dataset makes a future Linux/GPU reproduction possible.
"""
    if claim_id == "C4":
        theorem = evidence["audit"]["safety_theorem"]
        return f"""**Verdict - VERIFIED AS A CONDITIONAL THEOREM (2/2).**

The epsilon-margin implication passes
`{theorem["checked_implications"]}` exhaustive boundary combinations with
maximum safety residual `{theorem["max_safety_margin_residual"]:.1f}`.
Weakening the reliability event produces
`{theorem["weakened_reliability_mutation_failures"]}` failures.

This verifies the stated implication only. It does not prove that the learned
predictor satisfies one-sided reliability in deployment.
"""
    release = evidence["audit"]["release_source"]
    penalty = evidence["audit"]["finite_penalty"]
    return f"""**Verdict - PARTIALLY VERIFIED AT SOURCE LEVEL (1/2).**

The released code contains the predictive workflow, but its runnable contracts
do not close unchanged:

- training writes `ba_node_b...`, while the shield loads `fe_b5...`;
- adult, child, and adolescent patient 001 all resolve to normalization index
  `{release["released_cohort_indices"]["adult#001"]}` instead of `0/10/20`;
- a logit penalty of `{penalty["penalty"]:.0f}` leaves unsafe-action
  probability `{penalty["unsafe_action_probability"]:.3e}` in a two-action
  counterexample, rather than hard pruning.

These findings limit reproducibility of the released shield. They do not
falsify the paper's clinical-gain tables.
"""


def copy_bundle() -> None:
    destination = PROJECT / "logbook" / "repro_diabetes_safety"
    destination.mkdir()
    for relative in (
        "README.md",
        "requirements.txt",
        "claims_list.json",
        "prepare_official.py",
        "audit_simulator.py",
        "audit_generalization_gap.py",
        "audit_theory_and_release.py",
        "results/simulator_execution.json",
        "results/generalization_gap_cpo_t1d_adolescent_seed0.json",
        "results/theory_and_release.json",
        "tests/test_audit_theory_and_release.py",
        "tests/test_generalization_components.py",
    ):
        source = HERE / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_space_readme() -> None:
    content = f"""---
title: "{TITLE}"
emoji: "\U0001FA7A"
colorFrom: red
colorTo: blue
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-kSUGLBHd0T
 - arxiv:2601.21094
---

# {TITLE}

Independent CPU reproduction. Prepared score: **5/10**.

The runnable bundle and frozen evidence are included in
[`repro_diabetes_safety/`](./repro_diabetes_safety/README.md).
"""
    (PROJECT / "logbook" / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    evidence = load_evidence()
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-kSUGLBHd0T"],
            "paper": {"arxiv_id": "2601.21094"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive_slug = lb.ensure_page(PROJECT, "Executive summary")
    claim_slugs = {
        claim_id: lb.ensure_page(PROJECT, title)
        for claim_id, title in PAGES.items()
    }
    conclusion_slug = lb.ensure_page(PROJECT, "Conclusion")

    executive = """**Outcome.** This CPU reproduction executes all three
diabetes environments, independently evaluates one released policy under
patient shift, audits the conditional safety theorem, and tests whether the
released predictive-shield contracts close.

| Claim | Result | Score |
| --- | --- | ---: |
| Unified simulator | Partial | 1/2 |
| Safety-generalization gap | Partial | 1/2 |
| BA-NODE accuracy | Not executed | 0/2 |
| Conditional theorem | Verified | 2/2 |
| Released shield | Partial | 1/2 |
| **Prepared** |  | **5/10** |

No leaderboard or peer reproduction was inspected or used as evidence.

Primary sources: [arXiv](https://arxiv.org/abs/2601.21094),
[OpenReview](https://openreview.net/forum?id=kSUGLBHd0T),
[GlucoSim](https://github.com/safe-autonomy-lab/GlucoSim), and
[GlucoAlg](https://github.com/safe-autonomy-lab/GlucoAlg).
"""
    lb.add_markdown_cell(
        PROJECT, executive_slug, executive, title="Executive summary"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    gap = evidence["generalization"]
    poster = f"""<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #cbd5e1;padding:20px;background:#fff;color:#17212b">
<h2 style="font-size:18px;margin:0 0 8px;border-bottom:3px solid #b42318;padding-bottom:8px">Diabetes safety generalization: independent CPU audit</h2>
<p style="font-size:12px">ICML 2026 | kSUGLBHd0T | arXiv 2601.21094 | Prepared 5/10</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef2f6"><th style="text-align:left;padding:6px">Evidence</th><th>Finding</th></tr>
<tr><td style="padding:6px">Simulator</td><td>22/22 tests; 3 deterministic environments</td></tr>
<tr><td style="padding:6px">OOD mechanism</td><td>TIR gap {gap["tir_gap_ood_minus_id"]:.2f} pp; risk gap +{gap["risk_gap_ood_minus_id"]:.2f}</td></tr>
<tr><td style="padding:6px">Theorem</td><td>320 boundary checks; conditional implication holds</td></tr>
<tr><td style="padding:6px">Release audit</td><td>Path, cohort-index, and soft-pruning divergences</td></tr>
</table></div>"""
    lb.add_figure_cell(
        PROJECT, executive_slug, html=poster, title="Reproduction poster"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    claim_evidence = {
        "C1": evidence["simulator"],
        "C2": evidence["generalization"],
        "C3": {"verdict": "NOT_EXECUTED", "score": 0},
        "C4": evidence["audit"]["safety_theorem"],
        "C5": {
            "release_source": evidence["audit"]["release_source"],
            "finite_penalty": evidence["audit"]["finite_penalty"],
        },
    }
    for claim_id, slug in claim_slugs.items():
        lb.add_markdown_cell(
            PROJECT, slug, claim_text(claim_id, evidence), title=PAGES[claim_id]
        )
        lb.add_code_cell(
            PROJECT,
            slug,
            output=json.dumps(claim_evidence[claim_id], indent=2, sort_keys=True),
            title=f"{claim_id} machine-readable evidence",
            language="json",
        )

    conclusion = """The released simulator and the scoped OOD mechanism are
reproducible on CPU. The paper's epsilon-margin theorem is correct under its
explicit reliability assumption. The broad dynamics and shielding tables are
not reproduced, and the released predictive-shield path requires explicit
repairs before an end-to-end rerun.

## Evidence boundary

- No paper table is treated as execution evidence.
- The CPO run is one algorithm, cohort, and training seed.
- The theorem is conditional and does not certify BA-NODE.
- Source divergences limit reproducibility but do not falsify clinical gains.
"""
    lb.add_markdown_cell(
        PROJECT, conclusion_slug, conclusion, title="Conclusion"
    )

    index = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive_slug}) |",
    ]
    index.extend(
        f"| [{PAGES[claim_id]}](#/{claim_slugs[claim_id]}) |"
        for claim_id in PAGES
    )
    index.extend([f"| [Conclusion](#/{conclusion_slug}) |", ""])
    (PROJECT / "logbook" / "pages" / "index.md").write_text(
        "\n".join(index), encoding="utf-8"
    )

    copy_bundle()
    lb.write_site_files(PROJECT)
    write_space_readme()
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
