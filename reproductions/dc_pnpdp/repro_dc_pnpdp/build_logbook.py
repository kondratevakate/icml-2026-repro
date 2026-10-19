#!/usr/bin/env python3
"""Build the static Trackio logbook for the DC-PnPDP release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "audit.json"
SPACE_ID = "kondratevakate/repro-dc-pnpdp-release-audit"
TITLE = "Reproduction: Dual-Coupled PnP Diffusion"
TITLES = {
    "C1": "Claim 1: dual fixed-point optimality",
    "C2": "Claim 2: loose-PnP bias",
    "C3": "Claim 3: Spectral Homogenization",
    "C4": "Claim 4: CT reconstruction gains",
    "C5": "Claim 5: MRI reconstruction breadth",
    "C6": "Claim 6: components and efficiency",
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
        code_paths=["repro_dc_pnpdp/audit_dc_pnpdp.py"],
        language="python",
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-jEBkuuETjr", "medical-imaging"],
            "paper": {"arxiv_id": "2602.23214"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)
    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    fixed = data["fixed_points"]
    penalty = data["released_cg_penalty"]
    sh = data["spectral_homogenization"]
    inventory = data["release_inventory"]
    summary = f"""**Outcome.** The DC-PnPDP release receives **5/12 points**,
below the frozen forecast of 6/12. The effective-weight dual fixed-point
theorem is algebraically sound, but the released CT example removes the ADMM
penalty and the repository contains no MRI implementation or cached metrics.

| Independent check | Result |
| --- | --- |
| Dual objective residual | {fixed["dual"]["original_objective_residual"]:.2e} |
| Loose original-objective residual | {fixed["loose"]["original_objective_residual"]:.5f} |
| Zero-penalty anchor difference | {penalty["zero_penalty_anchor_difference"]:.1e} |
| SH PSD relative RMSE | {sh["effective_psd_relative_rmse_to_white"]:.4f} |
| MRI implementation released | {inventory["has_mri_implementation"]} |
| Prepared score | 5/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2602.23214) and
[official code](https://github.com/duchenhe/DC-PnPDP). No leaderboard or
third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)
    poster = """<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #b8c4cc;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #0f766e;padding-bottom:8px">DC-PnPDP: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Theory</td><td>Dual theorem verifies with lambda=rho gamma; loose theorem has scaling and interpretation gaps</td></tr><tr><td>Release</td><td>Example w_tik=0 versus paper LACT value 1e-5; no MRI path</td></tr><tr><td>SH</td><td>Finite mechanism, but paper/code sampling differs and native-order smoothing breaks FFT symmetry</td></tr><tr><td>Empirics</td><td>No data, reconstructions, metric outputs, ablation outputs, or timing artifacts</td></tr></table><p><b>Prepared 5/12</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT, pages["C1"], TITLES["C1"],
        """**VERIFIED - 2/2.** An independent scalar quadratic satisfies
consensus, both deterministic ADMM update equations, and the original
objective's stationarity condition with effective weight `lambda=rho*gamma`
to numerical precision.""",
        fixed["dual"],
    )
    add_page(
        PROJECT, pages["C2"], TITLES["C2"],
        """**PARTIAL - 1/2.** The loose fixed-point equation is reproduced,
but an exact proximal example shows it is stationarity of a Moreau-envelope
objective, not the original regularized objective. The main theorem,
appendix statement, and proof use incompatible scalings.""",
        {"loose": fixed["loose"], "scaling": inventory["theorem_scaling_inconsistency"]},
    )
    add_page(
        PROJECT, pages["C3"], TITLES["C3"],
        """**PARTIAL - 1/2.** Official SH is finite and shape preserving.
Paper and code use different random-spectrum constructions; zero-padded
smoothing in native FFT order breaks Hermitian symmetry before the code
discards the inverse FFT's imaginary component.""",
        sh,
    )
    add_page(
        PROJECT, pages["C4"], TITLES["C4"],
        """**UNSUPPORTED - 0/2.** No CT data or cached metrics are released.
The quickstart runs DiffPIR with `w_tik=0`, while the paper reports positive
`rho=1e-5/sigma_t^2` for LACT. At zero penalty the exact released x-update
loses dependence on the consensus anchor.""",
        penalty,
    )
    add_page(
        PROJECT, pages["C5"], TITLES["C5"],
        """**UNSUPPORTED - 0/2.** The paper reports fastMRI brain and knee,
but the pinned repository contains no MRI implementation, configuration, or
output.""",
        inventory,
    )
    add_page(
        PROJECT, pages["C6"], TITLES["C6"],
        """**PARTIAL - 1/2.** Both dual and SH components are present and all
released Python files compile. The ablation and A100 runtime measurements are
not cached and cannot be regenerated from the local CPU-only setup.""",
        inventory,
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        """DC-PnPDP supports **5/12** points. Its deterministic dual
fixed-point algebra is sound under the proximal assumption and effective
weight, but the empirical release does not encode the paper's MRI path and
its documented CT command does not use the positive ADMM penalty stated in
the paper. Canonical image claims remain unsupported by released outputs.""",
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

