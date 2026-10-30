"""Protocol 9, Note 9-21 (post hoc, descriptive): biosecurity topics among the frame's life-science papers.

Title-only keyword screen; lexicon frozen 2026-10-29 before any count was computed. The frame has no abstracts,
so counts are a lower bound and a screening label, not a coded variable.

Usage: python posthoc_p9_biosec.py   -> results/p9_biosec.json
"""
import collections
import json
import re
import sys
from pathlib import Path

TOPICS = {
    "pathogens_infection": r"\b(pathogen\w*|virus\w*|viral|virolog\w*|bacteri\w*|microb\w*|antimicrobial|antibiotic\w*|"
                           r"infect\w*|sars-cov-2|covid\w*|influenza|hiv|malaria|tuberculosis|phage\w*|toxin\w*|fung\w*)\b",
    "protein_design": r"\b(protein\w*|peptide\w*|enzyme\w*|antibod\w*|binder\w*|epitope\w*|folding|inverse folding|de novo design)\b",
    "dna_genome_models": r"\b(dna|rna|genom\w*|gene|genes|genetic\w*|sequence design|nucleotide\w*|codon\w*|crispr|"
                         r"gene editing|mrna|single-cell|transcriptom\w*|omics)\b",
    "epidemics_surveillance": r"\b(epidemi\w*|pandemic\w*|outbreak\w*|surveillance|disease spread|contact tracing|"
                              r"wastewater|forecast\w* (cases|infections))\b",
    "countermeasures_trials": r"\b(vaccin\w*|drug discovery|molecul\w*|ligand\w*|docking|clinical trial\w*|"
                              r"countermeasure\w*|therapeutic\w*)\b",
    "biosafety_dual_use": r"\b(biosecurity|biosafety|dual[- ]use|bioweapon\w*|biorisk\w*|hazard\w*)\b",
}
COMPILED = {k: re.compile(v, re.I) for k, v in TOPICS.items()}


def main(frame_path, out_path):
    rows = json.load(open(frame_path, encoding="utf-8"))
    ls = [r for r in rows if r["group"] == "life_science"]
    out, by_topic, by_venue = [], collections.Counter(), collections.defaultdict(collections.Counter)
    any_hit = 0
    for r in ls:
        hits = [k for k, rx in COMPILED.items() if rx.search(r["title"])]
        out.append({**r, "biosec_topics": hits})
        if hits:
            any_hit += 1
        for k in hits:
            by_topic[k] += 1
            by_venue[r["stratum"]][k] += 1
    venue_any = collections.Counter(r["stratum"] for r in out if r["biosec_topics"])
    venue_all = collections.Counter(r["stratum"] for r in out)
    label_any = collections.Counter(r["label"] for r in out if r["biosec_topics"])
    summary = {
        "n_life_science": len(ls),
        "n_any_topic": any_hit,
        "by_topic": dict(by_topic),
        "by_label_any": dict(label_any),
        "by_venue_any": {v: [venue_any[v], venue_all[v]] for v in sorted(venue_all)},
        "lexicon": TOPICS,
        "frozen": "2026-10-29",
    }
    json.dump({"summary": summary, "papers": out}, open(out_path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1)[:3000])


HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    main(HERE / "results" / "p9_frame.json", HERE / "results" / "p9_biosec.json")
