#!/usr/bin/env python3
"""Level B: second-model (hy3 via hermes-router) validity scoring of reproduction
solutions. For each structured claim from process_gt, ask hy3 whether the solution
REALLY reproduced the claim or merely wrote a convincing logbook. Produces a
validity_score (0-1) that feeds the partial-order dominance check.

This is what Kate asked for: score the SOLUTION, not the guessed label.
A model that guessed 'verified' but ran no mutation test should score low here
even if its label matches the judge.

Endpoint: hermes-router at http://localhost:8319/v1 (reads PROXY_API_KEYS from its
.env). model = 'hy3' (router maps to tencent/hy3:free via Nous). No direct Nous
key needed. Fails clearly if router is down.
"""
import os, re, json, glob, sys, time, urllib.request, urllib.error

GT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(GT, "out")
ROUTER = "http://localhost:8319/v1"
MODEL = "hy3"

SYSTEM = ("You are a strict reproducibility auditor. Given a reproduction logbook "
          "claim section and the judge's verdict, decide whether the solution "
          "REALLY reproduced the claim (ran the experiment, mutation/ablation test, "
          "reported concrete numbers) or merely WROTE A CONVINCING NARRATIVE. "
          "Respond ONLY with JSON: {\"validity\": <float 0-1>, \"reason\": \"<one line>\"}.")

PROMPT = """JUDGE VERDICT: {verdict}
JUDGE DISTRIBUTION: {dist}

REPRODUCTION LOGBOOK CLAIM SECTION:
---
{text}
---

Does this solution actually reproduce the claim, or is it a convincing narrative
without real evidence (no mutation/ablation test, no concrete numbers, no run)?
Score validity 0.0 (pure narrative) to 1.0 (fully reproduced with evidence).
Return JSON only."""


def router_key():
    p = "/home/kate/.local/share/hermes-router/.env"
    if not os.path.isfile(p):
        return None
    for line in open(p):
        m = re.match(r"^PROXY_API_KEYS=(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return None


def _build_orid_index(codex_root="/tmp/logbooks_codex"):
    """Map orid -> pages dir (scan once)."""
    idx = {}
    if not os.path.isdir(codex_root):
        return idx
    for d in os.listdir(codex_root):
        pages = os.path.join(codex_root, d, ".trackio", "logbook", "pages")
        if not os.path.isdir(pages):
            continue
        for f in glob.glob(os.path.join(pages, "*.md")):
            t = open(f, encoding="utf-8", errors="ignore").read()
            m = re.search(r"forum\?id=([A-Za-z0-9]+)", t)
            if m:
                idx[m.group(1)] = pages
                break
    return idx


_ORID_INDEX = None


def load_claim_text(orid, claim_idx, codex_root="/tmp/logbooks_codex"):
    """Re-read the original claim-page markdown for context (uses cached index)."""
    global _ORID_INDEX
    if _ORID_INDEX is None:
        _ORID_INDEX = _build_orid_index(codex_root)
    pages = _ORID_INDEX.get(orid)
    if not pages:
        return ""
    matches = glob.glob(os.path.join(pages, f"claim-{claim_idx + 1}-*"))
    if matches and os.path.isdir(matches[0]):
        pf = os.path.join(matches[0], "page.md")
        if os.path.isfile(pf):
            return open(pf, encoding="utf-8", errors="ignore").read()
    return ""


def ask_hy3(key, verdict, dist, text):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(verdict=verdict, dist=dist, text=text[:6000])},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        ROUTER + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        resp = json.loads(r.read())
        content = resp["choices"][0]["message"]["content"]
        # extract JSON from content
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            return json.loads(m.group(0))
        return {"validity": None, "reason": "no-json:" + content[:80]}
    except urllib.error.HTTPError as e:
        return {"validity": None, "reason": f"HTTP {e.code}"}
    except Exception as e:
        return {"validity": None, "reason": str(e)[:80]}


def main():
    key = router_key()
    if not key:
        sys.exit("ERROR: hermes-router PROXY_API_KEYS not found at "
                 "/home/kate/.local/share/hermes-router/.env — cannot call hy3.")
    # quick health (router serves /health at root, not under /v1)
    try:
        urllib.request.urlopen(ROUTER.replace("/v1", "") + "/health", timeout=10)
    except Exception as e:
        sys.exit(f"ERROR: hermes-router not reachable: {e}")

    src = json.load(open(os.path.join(OUT, "process_gt_medical.json")))
    # resume: if a prior levelB run exists, reuse its validity scores
    prior_path = os.path.join(OUT, "process_gt_levelB_medical.json")
    prior = json.load(open(prior_path)) if os.path.isfile(prior_path) else None
    if prior:
        for pp in prior["papers"]:
            sp = next((x for x in src["papers"] if x["orid"] == pp["orid"]), None)
            if not sp:
                continue
            for c in pp["claims"]:
                if c.get("validity") is not None:
                    sc = next((x for x in sp["claims"] if x["claim"] == c["claim"]), None)
                    if sc:
                        sc["validity"] = c["validity"]
                        sc["validity_reason"] = c.get("validity_reason", "")
    papers = src["papers"]
    scored = 0
    for p in papers:
        for c in p["claims"]:
            if not c["structure"]:
                continue
            if c.get("validity") is not None:  # resume: already scored
                continue
            text = load_claim_text(p["orid"], c["claim"])
            if not text:
                c["validity"] = None
                c["validity_reason"] = "no-logbook-text"
                continue
            res = ask_hy3(key, c["judge_verdict"], c["judge_dist"], text)
            c["validity"] = res.get("validity")
            c["validity_reason"] = res.get("reason", "")[:200]
            scored += 1
            print(f"  {p['orid']}#{c['claim']} validity={c['validity']} ({c['validity_reason']})")
            time.sleep(0.5)  # be gentle on free tier

    # recompute minimal-sufficient with validity as an extra dominance axis
    for p in papers:
        structured = [c for c in p["claims"] if c["structure"]]
        edges = []
        for a in structured:
            for b in structured:
                if a is b:
                    continue
                va = a.get("validity") or 0
                vb = b.get("validity") or 0
                la = a["structure"]["evidence_layers"]
                lb = b["structure"]["evidence_layers"]
                proves_ge = (a["structure"]["has_mutation_test"] >= b["structure"]["has_mutation_test"]) \
                    and all(la[k] >= lb[k] for k in la)
                proves_gt = (a["structure"]["has_mutation_test"] > b["structure"]["has_mutation_test"]) \
                    or any(la[k] > lb[k] for k in la)
                cost_le = (a["structure"]["compute_cost_s"] or 1e9) <= (b["structure"]["compute_cost_s"] or 1e9)
                cost_lt = (a["structure"]["compute_cost_s"] or 1e9) < (b["structure"]["compute_cost_s"] or 1e9)
                val_ge = va >= vb
                val_gt = va > vb
                # strict Pareto: strictly better on >=1 axis, not worse on others
                if ((proves_gt and cost_le and val_ge) or (proves_ge and cost_lt and val_ge)
                        or (proves_ge and cost_le and val_gt)):
                    edges.append([a["claim"], b["claim"]])
        dom = {e[1] for e in edges}
        p["dominance_edges_validity"] = edges
        p["minimal_sufficient_validity"] = [c["claim"] for c in structured if c["claim"] not in dom]

    out = {"summary": {**src["summary"],
                       "levelB_scored_claims": scored,
                       "levelB_validity_present": sum(1 for p in papers for c in p["claims"]
                                                      if c.get("validity") is not None)},
           "papers": papers}
    op = os.path.join(OUT, "process_gt_levelB_medical.json")
    json.dump(out, open(op, "w"), indent=1)
    print("wrote", op, "| scored", scored)


if __name__ == "__main__":
    main()
