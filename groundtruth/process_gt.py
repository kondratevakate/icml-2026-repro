#!/usr/bin/env python3
"""Process-grounded GT: extract solution STRUCTURE from reproduction logbooks
(codex trackio pages + hermes logbook.md), join with judge verdicts, and build a
PARTIAL ORDER of solutions per claim (dominance + minimal-sufficient Pareto set).

Unlike soft-GT (which only stores verdict distributions), this captures HOW a
claim was reproduced: mutation tests, seed counts, compute cost, evidence layers.
A model is then scored on whether its solution sits high in the partial order
(minimally sufficient) rather than merely guessing the label.

Mapping logbook -> orid: codex pages embed 'openreview.net/forum?id=<orid>'.
"""
import os, re, json, csv, glob, collections, argparse

GT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(GT, "out")
AUD = os.path.join(GT, "audit_snapshot")
CODEX_ROOT = "/tmp/logbooks_codex"  # extracted git archive of codex branch
CATS = ["verified", "falsified", "toy", "inconclusive"]

MUT_RE = re.compile(r"\b(mutat\w*|broken|invert\w*|swap\w*|ablati\w*|perturb\w*|counter[- ]?example|negativ\w* transfer)\b", re.I)
SEED_RE = re.compile(r"\b(\d+)\s*(?:seeds?|runs?|reps?|repetitions?|trials?)\b", re.I)
TIME_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|min|h|hour|cpu[- ]?hours?|gpu[- ]?hours?)\b", re.I)
CLAIM_RE = re.compile(r"claim[-\s]?(\d+)", re.I)


def find_orid_in_pages(pages_dir):
    """Return orid from any page containing openreview forum?id=."""
    for f in glob.glob(os.path.join(pages_dir, "*.md")):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        m = re.search(r"forum\?id=([A-Za-z0-9]+)", txt)
        if m:
            return m.group(1)
    return None


def extract_signals(text):
    """Heuristic structural signals of a reproduction solution."""
    has_mut = bool(MUT_RE.search(text))
    seeds = [int(x) for x in SEED_RE.findall(text)]
    n_seeds = max(seeds) if seeds else None
    times = []
    for v, unit in TIME_RE.findall(text):
        try:
            val = float(v)
            val = val / 60.0 if unit.startswith("min") else val
            val = val * 3600.0 if unit.startswith(("h", "hour", "cpu", "gpu")) else val
            times.append(val)
        except ValueError:
            pass
    cost = max(times) if times else None  # seconds; rough proxy for compute
    # evidence layers mentioned
    layers = {
        "mechanics": bool(re.search(r"\b(mechanic\w*|forward|backward|gradient|loss)\b", text, re.I)),
        "theory": bool(re.search(r"\b(theorem|lemma|proof|bound|inequalit\w*)\b", text, re.I)),
        "empirics": bool(re.search(r"\b(empiric\w*|experiment\w*|simulat\w*|dataset|real[- ]?data)\b", text, re.I)),
        "numbers": bool(re.search(r"\b\d+\.\d+\b", text)),
    }
    return {"has_mutation_test": has_mut, "n_seeds": n_seeds, "compute_cost_s": cost,
            "evidence_layers": layers, "n_numbers": len(re.findall(r"\b\d+\.\d+\b", text))}


def load_codex_solutions():
    """orid -> list of per-claim solution structures from codex trackio pages."""
    sol = collections.defaultdict(dict)
    if not os.path.isdir(CODEX_ROOT):
        return sol
    for d in os.listdir(CODEX_ROOT):
        pages = os.path.join(CODEX_ROOT, d, ".trackio", "logbook", "pages")
        if not os.path.isdir(pages):
            continue
        orid = find_orid_in_pages(pages)
        if not orid:
            continue
        # claim pages are directories named claim-N-.../page.md
        for claim_dir in glob.glob(os.path.join(pages, "claim-*")):
            if not os.path.isdir(claim_dir):
                continue
            m = CLAIM_RE.search(os.path.basename(claim_dir))
            if not m:
                continue
            cidx = int(m.group(1)) - 1
            f = os.path.join(claim_dir, "page.md")
            if not os.path.isfile(f):
                continue
            txt = open(f, encoding="utf-8", errors="ignore").read()
            sol[orid][cidx] = extract_signals(txt)
    return sol


def load_hermes_solutions():
    """orid -> per-claim from current-branch logbook.md (run02-style)."""
    sol = collections.defaultdict(dict)
    for lb in glob.glob(os.path.join(GT, "..", "experiments", "*", "logbook.md")):
        txt = open(lb, encoding="utf-8", errors="ignore").read()
        m = re.search(r"OpenReview\s*`([A-Za-z0-9]+)`", txt)
        orid = m.group(1) if m else None
        if not orid:
            continue
        # split by '## Claim N' sections
        parts = re.split(r"\n##\s*Claim\s*(\d+)", txt)
        for i in range(1, len(parts), 2):
            cidx = int(parts[i]) - 1
            body = parts[i + 1]
            sol[orid][cidx] = extract_signals(body)
    return sol


def dominance(a, b):
    """Does solution a dominate b? Strict Pareto: a is STRICTLY better on at
    least one axis (proves more OR costs less) while not worse on the other.
    'proves' approximated by evidence_layers superset + mutation test."""
    la, lb = a["evidence_layers"], b["evidence_layers"]
    proves_ge = (a["has_mutation_test"] >= b["has_mutation_test"]) and all(la[k] >= lb[k] for k in la)
    proves_gt = (a["has_mutation_test"] > b["has_mutation_test"]) or any(la[k] > lb[k] for k in la)
    cost_le = (a["compute_cost_s"] or 1e9) <= (b["compute_cost_s"] or 1e9)
    cost_lt = (a["compute_cost_s"] or 1e9) < (b["compute_cost_s"] or 1e9)
    # strict: strictly better on proves, or equal proves but strictly cheaper
    return (proves_gt and cost_le) or (proves_ge and cost_lt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "medical"], default="all")
    args = ap.parse_args()

    sols = json.load(open(os.path.join(OUT, "solutions_by_paper.json")))
    rows = list(csv.DictReader(open(os.path.join(AUD, "public_repro_audit.csv"))))
    medical = {r["orid"] for r in rows}
    area_of = {r["orid"]: r.get("area", "") for r in rows}

    codex = load_codex_solutions()
    hermes = load_hermes_solutions()

    papers = []
    n_with_structure = 0
    for orid, es in sols.items():
        if args.mode == "medical" and orid not in medical:
            continue
        if not es:
            continue
        n = int(es[0]["n_claims"])
        if n == 0:
            continue
        # judge verdicts (aligned)
        vecs = [e["verdicts"] for e in es if len(e["verdicts"]) == n]
        if not vecs:
            continue
        struct_by_idx = {}
        for src in (codex, hermes):
            if orid in src:
                struct_by_idx.update(src[orid])
        if not struct_by_idx:
            continue
        n_with_structure += 1

        # build per-claim partial order
        claims = []
        for k in range(n):
            vs = [v[k] for v in vecs]
            cnt = collections.Counter(vs)
            top, topn = cnt.most_common(1)[0]
            judge = top if (len(vs) >= 3 and topn / len(vs) >= 0.6) else "disputed"
            sig = struct_by_idx.get(k)
            claims.append({
                "claim": k,
                "judge_verdict": judge,
                "judge_dist": {c: round(cnt[c] / len(vs), 4) for c in CATS},
                "structure": sig,  # None if no logbook for this claim
            })

        # partial order across available structured solutions for this paper
        # (each structured claim is one 'solution node'; dominance by evidence)
        structured = [c for c in claims if c["structure"]]
        edges = []
        for i, a in enumerate(structured):
            for jj, b in enumerate(structured):
                if i != jj and dominance(a["structure"], b["structure"]):
                    edges.append((a["claim"], b["claim"]))
        # minimal sufficient = nodes with no incoming dominance edge (Pareto front)
        dominated = {e[1] for e in edges}
        minimal = [c["claim"] for c in structured if c["claim"] not in dominated]

        papers.append({
            "orid": orid,
            "area": area_of.get(orid, ""),
            "n_claims": n,
            "claims": claims,
            "dominance_edges": edges,
            "minimal_sufficient_claims": minimal,
            "n_structured": len(structured),
        })

    summary = {
        "papers_with_logbook_structure": n_with_structure,
        "total_papers_in_mode": sum(1 for o in sols if (args.mode != "medical" or o in medical)),
        "claims_with_structure": sum(p["n_structured"] for p in papers),
        "claims_judged": sum(p["n_claims"] for p in papers),
        "structure_coverage": round(sum(p["n_structured"] for p in papers) /
                                    sum(p["n_claims"] for p in papers), 4) if papers else 0,
        "mutation_tests_found": sum(1 for p in papers for c in p["claims"]
                                    if c["structure"] and c["structure"]["has_mutation_test"]),
        "papers_with_minimal_sufficient": sum(1 for p in papers if p["minimal_sufficient_claims"]),
    }
    out = {"summary": summary, "papers": papers}
    op = os.path.join(OUT, f"process_gt_{args.mode}.json")
    json.dump(out, open(op, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", op)


if __name__ == "__main__":
    main()
