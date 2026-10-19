#!/usr/bin/env python3
"""Build the Trackio logbook from frozen sBayFDNN audit evidence."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "claims_audit.json"
SPACE_ID = "kondratevakate/repro-sparse-bayesian-functional-deep-learning"
TITLE = (
    "Reproduction: Sparse Bayesian Deep Functional Learning with "
    "Structured Region Selection"
)
PAGES = {
    "C1": "Claim 1: spike-and-slab region selection",
    "C2": "Claim 2: Theorem 5.4 approximation bound",
    "C3": "Claim 3: Theorem 5.7 posterior contraction",
    "C4": "Claim 4: Theorem 5.9 selection consistency",
    "C5": "Claim 5: ECG performance",
    "C6": "Claim 6: Tecator performance",
}


def create_logbook() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT}")
    original_cwd = Path.cwd()
    original_find = lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = original_find
        os.chdir(original_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected logbook path: {created}")


def claim_text(claim: dict) -> str:
    cid = claim["id"]
    if cid == "C1":
        return f"""**Verdict - VERIFIED (2/2).**

An independent group-normal Bayes derivation agrees with the released plug-in
PIP at all `{claim["pip_grid_points"]}` tested column norms. Maximum absolute
error is `{claim["max_pip_absolute_error"]:.3g}`, and the analytic selection
threshold gives PIP `{claim["pip_at_analytic_threshold"]:.15f}`.

An independent clamped-knot support implementation also matches the released
basis-to-region mapping exactly. This verifies the method mechanism, not the
paper's broad empirical superiority claim.
"""
    if cid == "C2":
        ratio = claim["counterexample"]["error_lower_bound_over_rate"]
        return f"""**Verdict - FALSIFIED AS STATED (2/2).**

Theorem 5.4 quantifies over a network class with parameter bound `E_n`, but its
three stated assumptions impose no lower bound on `E_n`.

Set the strictly positive `E_n=1/n`, width one, `X(t)=1`, `beta(t)=1`, and
`g(u)=u`. The design, coefficient, and link assumptions hold. Every admissible
network output is `O(1/n)`, so its uniform error tends to one. Meanwhile the
claimed two-term rate tends to zero; on the audit grid, error/rate grows from
`{ratio[0]:.2f}` to
`{ratio[-1]:.2f}` and diverges symbolically. Adding a sufficiently large
parameter-bound condition can repair this counterexample, but that condition
is absent from the theorem as stated.
"""
    if cid == "C3":
        ratio = claim["proof_required_complexity_over_epsilon2"]
        return f"""**Verdict - INCONCLUSIVE PROOF GAP (0/2).**

Theorem 5.7 states `epsilon_n^2 <= C * complexity_n`, but its proof requires
`complexity_n <= C' * epsilon_n^2` for entropy and prior-mass bounds.

Take `complexity_n = 1/n` and `epsilon_n^2 = 1/n^2`. The written condition
holds at all six tested values, while the proof-required ratio grows from
`{ratio[0]:.0f}` to `{ratio[-1]:.0f}`. No fixed constant can supply the
missing reverse inequality. This invalidates the supplied proof under the
written condition, but it does not construct a posterior sequence satisfying
all assumptions whose contraction conclusion fails. No falsification credit
is claimed.
"""
    if cid == "C4":
        return """**Verdict - INCONCLUSIVE (0/2).**

The posterior inclusion argument controls `q_j`, but the proof then transfers
the result to MAP plug-in `hat q_j` by citing equivalence under an "appropriate
choice of prior hyperparameters." That condition is not included among
Theorem 5.9's stated assumptions and is not established in the paper.

This is a proof gap, not a standalone counterexample to the conclusion, so no
falsification credit is claimed.
"""
    if cid == "C5":
        return """**Verdict - NOT EXECUTED (0/2).**

The official release has no ECG preprocessing, split, baseline, or repeated
evaluation pipeline. The public raw dataset and paper-rendered table are not
substitutes for independent execution.
"""
    return """**Verdict - NOT EXECUTED (0/2).**

The official release has no Tecator preprocessing, split, baseline, or
repeated evaluation pipeline. No RMSE or spectral-region verdict is inferred
from the paper table or cached notebook.
"""


def copy_bundle() -> None:
    destination = PROJECT / "logbook" / "repro_sbayfdnn"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        HERE,
        destination,
        ignore=shutil.ignore_patterns(
            "build_logbook.py", "__pycache__", "*.pyc", ".pytest_cache"
        ),
    )


def write_hub_readme() -> None:
    content = """---
title: "Reproduction: Sparse Bayesian Deep Functional Learning with Structured Region Selection"
emoji: "🫀"
colorFrom: green
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-3IFIedDIoN
 - arxiv:2602.20651
---

# Sparse Bayesian Deep Functional Learning reproduction

Independent method and theorem audit. Prepared score: **4/12**.

The runnable clean-room bundle and frozen evidence are included in
[`repro_sbayfdnn/`](./repro_sbayfdnn/README.md).
"""
    (PROJECT / "logbook" / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": [
                "icml2026-repro",
                "paper-3IFIedDIoN",
                "arxiv:2602.20651",
            ],
            "paper": {"arxiv_id": "2602.20651"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive_slug = lb.ensure_page(PROJECT, "Executive summary")
    claim_slugs = {
        cid: lb.ensure_page(PROJECT, title) for cid, title in PAGES.items()
    }
    conclusion_slug = lb.ensure_page(PROJECT, "Conclusion")

    executive = f"""**Outcome.** This independent CPU/source audit prepares
**{evidence["prepared_score"]}/{evidence["maximum_score"]}**. The released
spike-and-slab selection mechanism is verified. Theorem 5.4 is falsified as
stated by its unconstrained parameter bound. Theorems 5.7 and 5.9 have proof
gaps but remain inconclusive, and the unreleased ECG/Tecator pipelines remain
unexecuted.

| Item | Value |
| --- | --- |
| Prepared score | 4/12 |
| GPU | None |
| Medical data | None |
| Independent tests | 5 passing |
| Author commit | `{evidence["provenance"]["official_code_commit"]}` |

No leaderboard or peer reproduction was inspected or used.

Primary sources: [arXiv](https://arxiv.org/abs/2602.20651),
[OpenReview](https://openreview.net/forum?id=3IFIedDIoN), and
[official code](https://github.com/mengyunwu2020/sBayFDNN).
"""
    lb.add_markdown_cell(
        PROJECT, executive_slug, executive, title="Executive summary"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #c9d2d8;padding:20px;background:#fff;color:#17212b">
<h2 style="font-size:18px;margin:0 0 8px;border-bottom:3px solid #176b5b;padding-bottom:8px">sBayFDNN: independent method and theorem audit</h2>
<p style="font-size:12px">ICML 2026 | 3IFIedDIoN | arXiv 2602.20651 | Prepared 4/12</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#edf3f1"><th style="text-align:left;padding:6px">Evidence</th><th>Finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">PIP and regions</td><td>1,201 Bayes checks; exact interval mapping</td><td style="color:#176b5b;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Theorem 5.4</td><td>Unconstrained E_n admits shrinking-network counterexample</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Theorem 5.7</td><td>Statement inequality is opposite proof requirement</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">Theorem 5.9</td><td>Unstated posterior-to-MAP premise</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">ECG / Tecator</td><td>Release pipelines absent</td><td>NOT EXECUTED</td></tr>
</table></div>"""
    lb.add_figure_cell(
        PROJECT, executive_slug, html=poster, title="Reproduction poster"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(
        PROJECT, cell_id, pinned=True, page=executive_slug
    )

    for cid, slug in claim_slugs.items():
        claim = next(item for item in evidence["claims"] if item["id"] == cid)
        lb.add_markdown_cell(
            PROJECT, slug, claim_text(claim), title=PAGES[cid]
        )
        lb.add_code_cell(
            PROJECT,
            slug,
            output=json.dumps(claim, indent=2, sort_keys=True),
            title=f"{cid} machine-readable evidence",
            code_paths=["repro_sbayfdnn/audit_claims.py"],
            language="python",
        )

    conclusion = """The released method mechanics support the declared
spike-and-slab region-selection construction. The approximation theorem is
false as written because it does not constrain the parameter bound of the
network class. The posterior-contraction proof uses the reverse of its written
rate condition, and the selection-consistency proof needs an additional
posterior-to-MAP premise; both are conservatively left inconclusive.

The ECG and Tecator results remain outside this reproduction because their
exact release pipelines are absent. Paper tables and cached author notebook
outputs are not treated as independent evidence.
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
        f"| [{PAGES[cid]}](#/{claim_slugs[cid]}) |" for cid in PAGES
    )
    index.extend([f"| [Conclusion](#/{conclusion_slug}) |", ""])
    (PROJECT / "logbook" / "pages" / "index.md").write_text(
        "\n".join(index), encoding="utf-8"
    )
    copy_bundle()
    write_hub_readme()
    lb.write_site_files(PROJECT)
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
