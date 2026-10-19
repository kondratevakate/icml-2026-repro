#!/usr/bin/env python3
"""Build the static Trackio logbook for the GlobalHealthAtlas audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "artifact_audit.json"
SPACE_ID = "kondratevakate/repro-globalhealthatlas-release-audit"
TITLE = "Reproduction: GlobalHealthAtlas release audit"
TITLES = {
    "C1": "Claim 1: corpus scale and coverage",
    "C2": "Claim 2: composition and quality control",
    "C3": "Claim 3: evaluator validity and stability",
    "C4": "Claim 4: multilingual benchmark",
    "C5": "Claim 5: SFT and transfer",
    "C6": "Claim 6: robustness and leakage",
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
        code_paths=["repro_globalhealthatlas/audit_artifacts.py"],
        language="python",
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-02Nq73lCe6", "public-health"],
            "paper": {"arxiv_id": "2602.00491"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    pages = {key: lb.ensure_page(PROJECT, title) for key, title in TITLES.items()}
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    bench = data["benchmark_reconstruction"]
    transfer = data["transfer_reconstruction"]
    adapters = data["adapters"]
    executive_text = f"""**Outcome.** The official GlobalHealthAtlas release
receives **{data["prepared_score"]}/{data["score_denominator"]} points**, equal
to the frozen forecast. The cached benchmark is unusually auditable: weighted
aggregation of 16 detailed model exports reconstructs all corresponding main
table values. The public corpus, expert gold annotations, and repeated
evaluator runs are not among the linked artifacts.

| Item | Independent result |
| --- | --- |
| GitHub revision | `{data["artifact_availability"]["github_revision"]}` |
| Released result CSVs | {data["artifact_availability"]["result_csv_count"]} |
| Main-table models reconstructed | {bench["rows_matching_paper_rounding"]}/{bench["models_compared"]} |
| Transfer cells matching paper rounding | {transfer["rows_matching_paper_rounding"]}/{transfer["rows"]} |
| Evaluator adapter SHA matches Hub | {adapters["evaluator"]["hub_checksum_matches"]} |
| Public-Model adapter SHA matches Hub | {adapters["public_model"]["hub_checksum_matches"]} |
| Prepared score | {data["prepared_score"]}/{data["score_denominator"]} |

Primary sources: [arXiv](https://arxiv.org/abs/2602.00491),
[OpenReview](https://openreview.net/forum?id=02Nq73lCe6),
[official repository](https://github.com/Jan8217/GlobalHealthAtlas),
[Public-Evaluator](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Evaluator),
and [Public-Model](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Model).
No leaderboard or third-party verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, executive_text, title="Executive summary")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)
    poster = f"""<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #c8d2dc;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #2563a6;padding-bottom:8px">GlobalHealthAtlas: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Corpus</td><td>Not released under the linked official artifacts</td></tr><tr><td>Main benchmark</td><td>{bench["models_compared"]}/{bench["models_compared"]} detailed exports reconstruct paper rounding</td></tr><tr><td>Transfer</td><td>11/12 cells match; Qwen-8B GPQA discrepancy</td></tr><tr><td>Adapters</td><td>Both safetensors payloads match Hub LFS SHA-256</td></tr></table><p><b>Prepared {data["prepared_score"]}/{data["score_denominator"]}</b>. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(PROJECT, executive, html=poster, title="Reproduction poster")
    cell = lb.last_cell_id(PROJECT, page=executive)
    if cell:
        lb.set_cell_pinned(PROJECT, cell, pinned=True, page=executive)

    availability = data["artifact_availability"]
    add_page(
        PROJECT,
        pages["C1"],
        TITLES["C1"],
        """**PARTIAL — 1/2.** The detailed test exports exercise the 15-domain,
17-language schema and support the reported test-set scale. The linked release
does not include the 280,210-row corpus or the source-document inventory, so
total corpus scale and evidence grounding cannot be independently counted.""",
        availability,
    )
    add_page(
        PROJECT,
        pages["C2"],
        TITLES["C2"],
        """**PARTIAL — 1/2.** Aggregate exports preserve task and difficulty
breakdowns, and the paper's totals are internally consistent. Row-level corpus
data, 14,010 expert quality-audit rows, deduplication results, and quality-gate
decisions are absent.""",
        {"availability": availability, "limits": data["limits"]},
    )
    add_page(
        PROJECT,
        pages["C3"],
        TITLES["C3"],
        f"""**UNSUPPORTED — 0/2.** The evaluator adapter is an intact
{adapters["evaluator"]["tensor_count"]}-tensor safetensors payload and matches
Hub LFS SHA-256. Artifact integrity does not reproduce ICC 0.9735 or stability:
the 100 expert/evaluator pairs and ten-run matrices were not released. The
published training script also names a psychology dataset and one epoch, not a
faithful paper recipe.""",
        {"adapter": adapters["evaluator"], "training": data["training_metadata"]},
    )
    add_page(
        PROJECT,
        pages["C4"],
        TITLES["C4"],
        f"""**SUPPORTED — 2/2.** Independent count-weighted aggregation of
{bench["models_compared"]} detailed per-model CSVs reconstructs all matching
overall, QA, and SC paper values within displayed rounding. Maximum absolute
error is `{bench["max_score_abs_error"]:.6f}`.""",
        bench,
    )
    add_page(
        PROJECT,
        pages["C5"],
        TITLES["C5"],
        f"""**PARTIAL — 1/2.** All 15 incremental-SFT rows recompute to
`{data["incremental_reconstruction"]["max_abs_error"]:.2e}` maximum error.
Eleven of 12 transfer cells match; Qwen-8B base GPQA recomputes to 5.2034
rather than paper 5.342, and the final Qwen-14B SFT rows are mislabeled as
base in the CSV.""",
        {
            "incremental": data["incremental_reconstruction"],
            "transfer": transfer,
        },
    )
    add_page(
        PROJECT,
        pages["C6"],
        TITLES["C6"],
        f"""**PARTIAL — 1/2.** All
{data["robustness_reconstruction"]["rows"]} robustness cells exactly match
Table 11; all 10-gram leakage rates are zero. The exports omit raw questions,
perturbations, predictions, and seeds, so end-to-end regeneration is not
possible from the release.""",
        {
            "robustness": data["robustness_reconstruction"],
            "leakage": data["leakage_reconstruction"],
        },
    )
    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        f"""The release supports **{data["prepared_score"]}/12** points. It is
strongest at cached-result transparency: the main benchmark and most secondary
tables are independently reconstructable from released aggregates. Its
decisive boundary is upstream provenance: the corpus, expert validation rows,
evaluator stability runs, raw predictions, perturbations, and seeds are not
publicly linked.""",
        title="Conclusion",
    )

    root = PROJECT / "logbook" / "pages"
    lines = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive}) |",
    ]
    lines += [f"| [{TITLES[key]}](#/{slug}) |" for key, slug in pages.items()]
    lines += [f"| [Conclusion](#/{conclusion}) |", ""]
    (root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    lb.write_site_files(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
