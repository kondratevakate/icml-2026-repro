#!/usr/bin/env python3
"""Build the static Trackio logbook for the CAML release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "audit.json"
SPACE_ID = "kondratevakate/repro-caml-release-audit"
TITLE = "Reproduction: Mitigating Gradient Pathology with Aligned Constraint"
TITLES = {
    "C1": "Claim 1: loss-valley theorem",
    "C2": "Claim 2: aligned-constraint mechanics",
    "C3": "Claim 3: Heat benchmark",
    "C4": "Claim 4: cross-PDE MLP results",
    "C5": "Claim 5: backbone generality",
    "C6": "Claim 6: components and limitations",
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
        code_paths=["repro_caml/audit_caml.py"],
        language="python",
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-Fisw2kc7EY", "pinn"],
            "paper": {"arxiv_id": "2605.25001"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)
    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    theorem = data["theorem_counterexample"]
    method = data["aligned_constraint"]
    conform = data["release_conformance"]
    mini = data["mini_heat"]
    summary = f"""**Outcome.** The CAML release receives
**{data["prepared_score"]}/{data["score_denominator"]} points**, below the
frozen forecast of 7/12. The aligned-offset equation and delay schedule match
the official implementation, but the headline theorem is false as stated and
five released benchmark settings differ from the paper.

| Independent check | Result |
| --- | --- |
| Non-unique operator has connected solution set | {theorem["solution_set_connected"]} |
| Closed-form offset absolute difference | {method["absolute_difference"]:.1e} |
| Degenerate zero-coefficient offset finite | {data["degenerate_offset"]["is_finite"]} |
| Paper/code benchmark-setting mismatches | {conform["mismatch_count"]} |
| Mini Heat runtime | {mini["wall_seconds"]:.2f} s for {mini["epochs"]} epochs |
| Prepared score | {data["prepared_score"]}/{data["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2605.25001) and
[official code](https://github.com/YichenLuo-0/CAML). No leaderboard or
third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)
    poster = f"""<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #bbc7d1;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #7c3aed;padding-bottom:8px">CAML: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Theorem</td><td>False as stated: N(u)=u², f=1 has two isolated disconnected solutions</td></tr><tr><td>Method</td><td>Offset and delay schedule exactly conform; zero-denominator case returns NaN</td></tr><tr><td>Empirics</td><td>Five paper/code settings differ; no cached outputs or full baseline suite</td></tr><tr><td>Runtime</td><td>Official Heat path executes after adding undocumented overrides dependency</td></tr></table><p><b>Prepared {data["prepared_score"]}/{data["score_denominator"]}</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    add_page(
        PROJECT, pages["C1"], TITLES["C1"],
        """**UNSUPPORTED - 0/2.** Non-uniqueness alone does not imply a
connected or non-isolated solution set. `N(u)=u²`, `f=1` is an explicit
counterexample. The appendix proof uses additional nonlinear regularity,
kernel, connectedness, and representability assumptions.""",
        theorem,
    )
    add_page(
        PROJECT, pages["C2"], TITLES["C2"],
        """**VERIFIED - 2/2.** The official linear offset equals an
independent closed-form calculation exactly, is detached, and the delay/ramp
schedule matches all boundary cases. With zero zeroth-order coefficients, the
unguarded denominator produces NaN.""",
        {"aligned_constraint": method, "degenerate": data["degenerate_offset"]},
    )
    add_page(
        PROJECT, pages["C3"], TITLES["C3"],
        """**PARTIAL - 1/2.** A finite 10-epoch Heat smoke run executes after
installing the undocumented `overrides` dependency. The code uses boundary
weight 1 and stop threshold 1e-3, versus 5 and 2e-3 in the paper; therefore it
is not a canonical five-seed reproduction.""",
        mini,
    )
    add_page(
        PROJECT, pages["C4"], TITLES["C4"],
        """**UNSUPPORTED - 0/2.** Heat, NS, and Helmholtz entrypoints contain
five paper/code setting mismatches, and no empirical output is cached. The
four-PDE table cannot be regenerated canonically from the release as-is.""",
        conform,
    )
    add_page(
        PROJECT, pages["C5"], TITLES["C5"],
        """**UNSUPPORTED - 0/2.** Three backbone implementations exist, but
entrypoints are hardcoded to MLP and do not apply the appendix's
backbone-specific configurations. Four of six baseline loss methods are
absent.""",
        conform,
    )
    add_page(
        PROJECT, pages["C6"], TITLES["C6"],
        """**PARTIAL - 1/2.** Offset and gate component mechanics are
independently checked. No ablation, schedule-sensitivity, optimizer, benign
failure-case, or overhead runner is released.""",
        {"aligned_constraint": method, "limits": data["limits"]},
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""CAML supports **{data["prepared_score"]}/12** points. Its compact
method code is equation-conformant for nondegenerate aligned constraints, but
the theorem needs materially stronger hypotheses and the empirical release
does not encode the paper's own benchmark settings. A canonical table rerun
would first require authors to identify which parameter set produced the
reported values.""",
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
