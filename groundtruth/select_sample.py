#!/usr/bin/env python3
"""Select a stratified sample of ICML 2026 papers for the AI-Scientist benchmark.

Inclusion:
  - medical / life-science relevance (title+abstract keyword screen)
  - has >=1 judged logbook (silver reference exists)  -> ground truth available
  - no GPU requirement signal (CPU-feasible)

Strata: theory | simulation | open_data | (restricted/gpu -> excluded here by design)
Output: groundtruth/out/sample20.json + sample20.md
"""
import json, os, re, random, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")

MED = r"(medical|clinical|patient|health|disease|diagnos|hospital|EHR|ICU|biomed|genomic|protein|cell|neuro|brain|MRI|EEG|ECG|fMRI|cancer|tumor|drug|omics|survival|epidemi|physiolog|sleep|mortality|radiolog|patholog|biolog|molecul|RNA|clinic)"
GPU = r"(GPU|A100|H100|pretrain|pre-train|large[- ]scale training|billion[- ]parameter|foundation model|LLM|diffusion model|transformer training|fine-tun|ImageNet|epochs on)"
THEORY = r"(theorem|proof|we prove|bound|convergence|lemma|proposition|asymptotic|minimax|identifiab|guarantee)"
SIM = r"(simulation|synthetic data|simulated|monte carlo|toy model|generative process|numerical experiment)"
OPEN = r"(MIMIC|PhysioNet|UK Biobank|OpenNeuro|TCGA|ADNI|eICU|public dataset|publicly available|open dataset|benchmark dataset|HCP|ABIDE)"


def main(n_per_stratum=7, seed=20260730):
    best = json.load(open(os.path.join(OUT, "best_solutions.json")))
    anchored = json.load(open(os.path.join(DATA, "claims_anchored.json")))

    pool = []
    for orid, b in best.items():
        claim_text = " ".join(c.get("text", "") for c in anchored.get(orid, []))
        text = b["title"] + " " + b.get("area", "") + " " + b.get("sub", "") + " " + claim_text
        if not re.search(MED, text, re.I):
            continue
        if re.search(GPU, text, re.I):
            continue
        if b["n_anchored_claims"] == 0:
            continue
        if re.search(THEORY, text, re.I):
            stratum = "theory"
        elif re.search(SIM, text, re.I):
            stratum = "simulation"
        elif re.search(OPEN, text, re.I):
            stratum = "open_data"
        else:
            stratum = "other"
        pool.append({**b, "stratum": stratum, "claim_text": claim_text[:900]})

    counts = collections.Counter(x["stratum"] for x in pool)
    print("eligible pool:", len(pool), dict(counts))

    rng = random.Random(seed)
    sample = []
    for s in ["theory", "simulation", "open_data", "other"]:
        cand = [x for x in pool if x["stratum"] == s]
        # prefer papers where the silver reference is strong but incomplete: informative targets
        cand.sort(key=lambda x: (-x["best_points"], x["n_solutions"]))
        top = cand[: max(3 * n_per_stratum, 10)]
        rng.shuffle(top)
        sample += top[:n_per_stratum]
    sample = sample[:20]

    json.dump(sample, open(os.path.join(OUT, "sample20.json"), "w"), indent=1)
    print("sampled:", len(sample), dict(collections.Counter(x["stratum"] for x in sample)))
    for x in sample:
        print(f"  [{x['stratum']:10}] {x['orid']}  {x['best_points']}/{x['best_max']}  "
              f"anchored={x['n_anchored_claims']}  {x['title'][:70]}")
    return sample


if __name__ == "__main__":
    main()
