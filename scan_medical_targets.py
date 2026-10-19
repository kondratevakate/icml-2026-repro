"""Rank ICML 2026 medical/relevant papers for reproduction throughput.

This is a conservative lexical triage, not evidence that code or data are
available.  Its purpose is to reduce 6,341 papers to a small PDF-review queue.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPERS_PATH = ROOT / "papers.json"
CLAIMS_PATH = ROOT / "claims_anchored.json"
CSV_PATH = ROOT / "medical_throughput_shortlist.csv"
MD_PATH = ROOT / "MEDICAL_THROUGHPUT_SHORTLIST.md"


MEDICAL = {
    "clinical", "medicine", "medical", "health", "healthcare", "patient",
    "hospital", "disease", "diagnosis", "diagnostic", "treatment", "therapy",
    "drug", "pharma", "pharmaceutical", "biomedical", "biological", "biology",
    "genomic", "genome", "gene", "protein", "peptide", "molecule", "molecular",
    "single-cell", "transcriptomic", "histology", "pathology", "radiology",
    "imaging", "mri", "ct", "pet", "fundus", "retina", "brain", "neuro",
    "eeg", "ecg", "sleep", "survival", "censoring", "cox", "ehr",
    "electronic health", "wearable", "physiological", "physiology", "biosignal",
    "biophysical", "neurodegenerative", "alzheimer", "cancer", "tumor", "lesion",
    "surgery", "clinical trial", "public health", "epidemiolog", "omics",
}

# Generic ML words such as "neural", "cell", and "imaging" are deliberately
# absent: they match a large fraction of non-medical ICML papers. Areas provide
# the neuro backstop; outside those areas, require one of these domain signals.
OUTSIDE_AREA_MEDICAL = {
    "clinical", "medicine", "medical", "health", "healthcare", "patient", "hospital",
    "disease", "diagnosis", "diagnostic", "treatment", "therapy", "drug", "pharma",
    "biomedical", "genomic", "genome", "gene", "protein", "peptide", "molecule",
    "single-cell", "transcriptomic", "histology", "pathology", "radiology", "mri",
    "pet", "fundus", "retina", "eeg", "ecg", "sleep", "survival", "censoring",
    "cox", "ehr", "wearable", "physiological", "biophysical", "alzheimer", "cancer",
    "tumor", "lesion", "surgery", "clinical trial", "public health", "epidemiolog", "omics",
}

DIRECT_MEDICAL = {
    "clinical", "medicine", "medical", "health", "healthcare", "patient",
    "hospital", "disease", "diagnosis", "treatment", "therapy", "radiology",
    "pathology", "histology", "mri", " ct ", "pet", "fundus", "retina", "eeg",
    "ecg", "sleep", "survival", "censoring", "cox", "ehr", "wearable", "cancer",
    "tumor", "lesion", "alzheimer", "neurodegenerative", "clinical trial",
}

OPEN_SIGNALS = {
    "code is available", "code and data", "open-source", "open source",
    "publicly available", "public dataset", "public datasets", "github.com",
    "huggingface.co", "zenodo", "openneuro", "physionet", "tc ia", "tcia",
    "mimic-iii", "mimic-iv", "uk biobank", "adni", "brats", "isic", "ptb-xl",
    "sleep-edf", "brainweb", "lidc-idri", "fastmri",
}

PRIVATE_SIGNALS = {
    "private dataset", "proprietary", "in-house", "our hospital", "retrospective cohort",
    "not publicly available", "upon request", "restricted access", "clinical site",
    "institutional review board", "irb", "electronic medical records",
}

THEORY_SIGNALS = {
    "theorem", "corollary", "proposition", "lemma", "proof", "guarantee",
    "convergence", "closed-form", "closed form", "identifiability", "bound",
}

LOW_COST_SIGNALS = {
    "simulation", "synthetic", "toy", "analytical", "analytic", "tabular",
    "benchmark", "ablation", "evaluation", "diagnostic", "calibration",
}

GPU_HEAVY_SIGNALS = {
    "foundation model", "large language model", "llm", "diffusion", "3d",
    "whole-slide", "whole slide", "transformer", "multi-modal", "multimodal",
    "pretrain", "pre-training", "generative", "video", "billion", "large-scale",
}

BROAD_SIGNALS = {
    "state-of-the-art", "state of the art", "sota", "across", "extensive experiments",
    "multiple datasets", "various datasets", "large-scale", "real-world",
}


def phrases_present(text: str, phrases: set[str]) -> list[str]:
    return sorted(phrase for phrase in phrases if phrase in text)


def keywords_present(text: str, keywords: set[str]) -> list[str]:
    hits = []
    for word in keywords:
        if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text):
            hits.append(word)
    return sorted(hits)


def is_medical_relevant(paper: dict, text: str) -> bool:
    if paper.get("sub") == "Health Medicine" or paper.get("sub") == "Neuroscience Cognitive Science":
        return True
    return bool(keywords_present(text, OUTSIDE_AREA_MEDICAL))


def classify(paper: dict, abstract: str, claims: list[dict]) -> dict:
    title = paper["title"]
    text = f"{title} {abstract}".lower()
    direct = keywords_present(text, DIRECT_MEDICAL)
    medical = keywords_present(text, MEDICAL)
    open_hits = phrases_present(text, OPEN_SIGNALS)
    private_hits = phrases_present(text, PRIVATE_SIGNALS)
    theory_hits = phrases_present(text, THEORY_SIGNALS)
    low_cost_hits = phrases_present(text, LOW_COST_SIGNALS)
    gpu_hits = phrases_present(text, GPU_HEAVY_SIGNALS)
    broad_hits = phrases_present(text, BROAD_SIGNALS)
    claim_text = " ".join(item.get("text", "") for item in claims).lower()
    claim_theory = phrases_present(claim_text, THEORY_SIGNALS)
    claim_low_cost = phrases_present(claim_text, LOW_COST_SIGNALS)

    # The score estimates initial triage priority, rather than expected judge points.
    # A score must never be interpreted as a predicted leaderboard result.
    score = 0
    score += min(8, len(direct) * 2 + len(medical))
    score += min(8, len(claims) * 2)
    score += min(4, len(open_hits) * 2)
    score += min(5, (len(theory_hits) + len(claim_theory)) * 2)
    score += min(3, len(low_cost_hits) + len(claim_low_cost))
    score -= min(10, len(private_hits) * 5)
    score -= min(7, len(gpu_hits))
    score -= min(5, len(broad_hits))

    if private_hits:
        bucket = "blocked/private-review"
    elif score >= 15 and len(claims) >= 3:
        bucket = "A: PDF/code review now"
    elif score >= 10:
        bucket = "B: feasibility probe"
    else:
        bucket = "C: defer"

    potential = len(claims) * 2
    tags = []
    if open_hits:
        tags.append("open-signal")
    if theory_hits or claim_theory:
        tags.append("theory")
    if low_cost_hits or claim_low_cost:
        tags.append("simulation/eval")
    if gpu_hits:
        tags.append("gpu-risk")
    if broad_hits:
        tags.append("broad-claim")
    if private_hits:
        tags.append("private-risk")

    return {
        "orid": paper["orid"], "title": title, "area": paper.get("area", ""),
        "subarea": paper.get("sub", ""), "arxiv": paper.get("arxiv", ""),
        "openreview": paper.get("or", ""), "n_claims": len(claims),
        "max_points": potential, "triage_score": score,
        "bucket": bucket, "tags": ";".join(tags),
        "medical_hits": ";".join(medical), "open_signals": ";".join(open_hits),
        "private_signals": ";".join(private_hits), "theory_signals": ";".join(sorted(set(theory_hits + claim_theory))),
        "gpu_risks": ";".join(gpu_hits), "broad_risks": ";".join(broad_hits),
        "abstract": abstract,
        "claim_preview": " | ".join(item.get("text", "") for item in claims[:2]),
    }


def main() -> None:
    papers_data = json.loads(PAPERS_PATH.read_text(encoding="utf-8"))
    claims_data = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    abstracts = papers_data["abstracts"]
    rows = []
    for paper in papers_data["papers"]:
        abstract = abstracts.get(paper["orid"], "")
        text = f"{paper['title']} {abstract}".lower()
        if not is_medical_relevant(paper, text):
            continue
        rows.append(classify(paper, abstract, claims_data.get(paper["orid"], [])))

    rows.sort(key=lambda row: (-row["triage_score"], -row["max_points"], row["title"]))
    fields = list(rows[0])
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["bucket"] for row in rows)
    immediate = [row for row in rows if row["bucket"] == "A: PDF/code review now" and not row["private_signals"]][:40]
    lines = [
        "# Medical Throughput Shortlist",
        "",
        "Generated by `scan_medical_targets.py` from the local ICML paper and anchored-claim snapshots.",
        "This is lexical triage only: verify data access, repository state, and exact claims in the PDF before starting a logbook.",
        "It deliberately does not predict points or hours. Those require a claim-level PDF and code review.",
        "",
        f"- Medical/relevant candidates found: {len(rows)}",
        f"- A queue: {counts['A: PDF/code review now']}; B queue: {counts['B: feasibility probe']}; blocked/private review: {counts['blocked/private-review']}",
        "- Go/no-go after PDF review: at least 6 expected points, 1.5 expected points/hour, and one full non-private verification or falsification path.",
        "",
        "## First PDF/code-review queue",
        "",
        "| Rank | Paper | Claims | Max | Triage | Why now |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(immediate, 1):
        why = row["tags"] or "medical fit"
        title = row["title"].replace("|", "\\|")
        lines.append(
            f"| {index} | [{title}]({row['openreview']}) | {row['n_claims']} | {row['max_points']} | "
            f"{row['triage_score']} | {why} |"
        )
    lines.extend([
        "",
        "## Operating rule",
        "",
        "Do not create a reproduction folder from this ranking alone. For each A candidate, inspect the PDF and repository, then record actual data access, runnable entrypoint, exact claim-to-table mapping, required hardware, and a preflight score. Downgrade any paper whose central claim is a broad SOTA table, private cohort, or unavailable checkpoint.",
        "",
        "The full machine-readable queue, including risk signals and first two anchored claims, is in `medical_throughput_shortlist.csv`.",
    ])
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {CSV_PATH.name}")
    print(f"Wrote {len(immediate)} immediate-review rows to {MD_PATH.name}")


if __name__ == "__main__":
    main()
