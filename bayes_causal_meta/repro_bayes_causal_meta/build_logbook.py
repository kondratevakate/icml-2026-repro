#!/usr/bin/env python3
"""Build the static logbook for the Bayesian causal meta-learning audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "audit.json"
SPACE_ID = "kondratevakate/repro-bayes-causal-meta-release-audit"
TITLE = "Reproduction: Bayesian Causal Meta-Learning"
TITLES = {
    "C1": "Claim 1: embedding-conditional prior",
    "C2": "Claim 2: prior-risk continuity",
    "C3": "Claim 3: negative-transfer theorem",
    "C4": "Claim 4: synthetic task-shift evidence",
    "C5": "Claim 5: clinical cross-disease evidence",
}


def create_logbook() -> None:
    old_cwd, old_find = Path.cwd(), lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = old_find
        os.chdir(old_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected project: {created}")


def add_page(project: Path, slug: str, title: str, markdown: str, evidence: object) -> None:
    lb.add_markdown_cell(project, slug, markdown, title=title)
    lb.add_code_cell(
        project,
        slug,
        output=json.dumps(evidence, ensure_ascii=False, indent=2),
        title=f"{title} evidence",
        code_paths=["repro_bayes_causal_meta/audit_bayes_causal_meta.py"],
        language="python",
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-k76ll7aQyE", "clinical-meta-learning"],
            "paper": {"arxiv_id": "2602.19788"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)
    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    prior = data["released_prior"]
    proposition = data["paper_proposition"]
    theorem = data["theorem_constant"]
    smoke = data["toy_smoke"]
    inventory = data["release_inventory"]
    summary = f"""**Outcome.** The release receives **5/10 points**, matching
the frozen forecast. Proposition 4 is sound for the paper's linear Gaussian
prior, but the released normalized prior is discontinuous at zero and
Appendix Theorem 5 drops the condition number of `W`.

| Independent check | Result |
| --- | --- |
| Released prior jump at zero | {prior["jump_norm_at_zero"]:.1f} |
| Paper Gaussian-TV bounds hold | {proposition["all_bounds_hold"]} |
| Theorem-constant counterexample | {theorem["counterexample"]} |
| Same-seed validation AUROC range | {smoke["same_seed_validation_auroc_range"]:.4f} |
| UK Biobank data released | {inventory["has_ukbb_data"]} |
| Prepared score | 5/10 |

Primary sources: [arXiv](https://arxiv.org/abs/2602.19788) and
[official code](https://github.com/lottamakinen/causal-meta-learning). No
leaderboard or third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)
    poster = """<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #bbc7d1;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #2563eb;padding-bottom:8px">Bayesian causal meta-learning: independent audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Prior</td><td>Release normalizes every nonzero Wz, creating scale invariance and a jump at zero</td></tr><tr><td>Proposition 4</td><td>Gaussian-TV and error-decomposition bounds verified for the paper model</td></tr><tr><td>Theorem 5</td><td>Appendix constant omits cond(W); explicit 2D counterexample</td></tr><tr><td>Empirics</td><td>Toy training and BALD smoke work; UK Biobank artifacts absent</td></tr></table><p><b>Prepared 5/10</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT, pages["C1"], TITLES["C1"],
        """**PARTIAL - 1/2.** The release adds an embedding-dependent offset
to global parameters, but normalizes every non-zero update to a fixed norm.
The resulting map is scale-invariant away from zero and discontinuous at
zero, unlike `theta+Wz` in the paper.""",
        prior,
    )
    add_page(
        PROJECT, pages["C2"], TITLES["C2"],
        """**VERIFIED - 2/2.** Exact equal-covariance Gaussian total
variation stays below the stated Pinsker bound at five distances. The
expert/causal/OOD decomposition follows directly from the triangle
inequality. This verification applies to the paper's linear prior.""",
        proposition,
    )
    add_page(
        PROJECT, pages["C3"], TITLES["C3"],
        """**PARTIAL - 1/2.** A condition-number-aware constant can recover
the sufficient condition. Appendix A.2 instead sets `C=kappa0/kappa`,
dropping `cond(W)`; a diagonal 2D example satisfies the stated norm
condition while failing the parameter-space inequality used by the proof.""",
        theorem,
    )
    add_page(
        PROJECT, pages["C4"], TITLES["C4"],
        """**PARTIAL - 1/2.** Adaptive toy training and BALD expert inference
execute on CPU after creating the output directory and enabling UTF-8.
Three runs with the same declared seed span 0.0745 validation AUROC because
the entrypoint does not seed training RNGs. This is not the canonical
30-run experiment.""",
        smoke,
    )
    add_page(
        PROJECT, pages["C5"], TITLES["C5"],
        """**UNSUPPORTED - 0/2.** The restricted UK Biobank cohort, clinical
predictions, checkpoints, and ten-seed result tables are not released.""",
        inventory,
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        """The package supports **5/10** points. The paper's prior-risk
continuity proposition is sound as a result about a linear Gaussian prior,
but that prior is not what the release implements. The synthetic paths are
executable with minor packaging workarounds; the clinical claims remain
non-reproducible without restricted UK Biobank data and cached outputs.""",
        title="Conclusion",
    )

    root = PROJECT / "logbook" / "pages"
    lines = [
        f"# {TITLE}", "", lb.TOC_HEADING, "", lb.TOC_HEADER, lb.TOC_SEP,
        f"| [Executive summary](#/{executive}) |",
    ]
    lines += [f"| [{TITLES[key]}](#/{slug}) |" for key, slug in pages.items()]
    lines += [f"| [Conclusion](#/{conclusion}) |", ""]
    (root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    lb.write_site_files(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

