"""AI-scientist papers, Amendment 1: candidate sheets, coder agreement, adjudication and the validated list.

Usage: python code.py build | score | build_adj | merge | run
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screen import P8, RE as NARROW  # noqa: E402

WORK, CODES, RES = HERE / "work", HERE / "codes", HERE / "results"
SEED, BATCH = 20261024, 20
WIDE_A = re.compile(r"\bagent(?:s|ic)?\b|\bllms?\b|large language models?", re.I)
WIDE_B = re.compile(
    r"scientific discovery|scientific research|research (?:idea|ideation|paper|pipeline|process|workflow|task)s?|"
    r"hypothes(?:is|es) generation|automat\w* (?:science|scientific|research|discovery|experiment|peer review|review)|"
    r"ai scientist|co-?scientist|scientist|research agent|"
    r"reproduc\w* (?:of )?(?:the )?(?:ml |research |scientific )?(?:papers|results|experiments)|replicat\w* (?:papers|studies)|"
    r"self-driving lab|autonomous (?:lab|experiment)|peer review|paper writing|(?:write|writing) (?:scientific )?papers|"
    r"novelty|literature review|survey generation|paperbench|mle-bench|re-bench|scienceagentbench|core-bench|"
    r"mlagentbench|researchbench|discoverybench|ml engineering|machine learning engineering|kaggle", re.I)
LABELS = ("include", "exclude", "unclear")
CATS = ("ai_scientist", "research_step", "reproduction", "benchmark", "lab", "")


def build() -> None:
    if (WORK / "keys.json").exists():
        raise SystemExit("keys exist; candidates are drawn once")
    units = json.loads((P8.DATA / "papers.json").read_text(encoding="utf-8"))
    seen, cand = {}, []
    for u in sorted(units, key=lambda u: (u["stratum"], u["forum_id"])):
        text = f"{u['title']} {u['abstract']}"
        narrow = any(r.search(text) for r in NARROW.values())
        if not (narrow or (WIDE_A.search(text) and WIDE_B.search(text))):
            continue
        k = (u["stratum"], " ".join(u["title"].lower().split()))
        if k in seen:
            seen[k]["duplicates"].append(u["forum_id"])
            continue
        seen[k] = dict(stratum=u["stratum"], paper_id=u["forum_id"], title=" ".join(u["title"].split()),
                       abstract=" ".join(u["abstract"].split()), narrow=narrow, duplicates=[])
        cand.append(seen[k])
    random.Random(SEED).shuffle(cand)
    keys = {f"A{n:04d}": c for n, c in enumerate(cand)}
    WORK.mkdir(exist_ok=True)
    (WORK / "keys.json").write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
    items = sorted(keys)
    for b, start in enumerate(range(0, len(items), BATCH)):
        for role in ("as1", "as2"):
            with (WORK / f"{role}_batch_{b:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["item_id", "title", "abstract"])
                w.writeheader()
                for i in items[start:start + BATCH]:
                    w.writerow(dict(item_id=i, title=keys[i]["title"], abstract=keys[i]["abstract"]))
    print(len(keys), "candidates;", sum(c["narrow"] for c in cand), "from the screen;", -(-len(items) // BATCH),
          "batches per coder")


def read(role):
    out = {}
    for f in sorted(CODES.glob(f"{role}_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lab, cat = (r.get("label") or "").strip(), (r.get("category") or "").strip()
                if lab not in LABELS or cat not in CATS or (lab == "include") != bool(cat):
                    raise SystemExit(f"{f.name}: invalid row {r['item_id']}: {lab} {cat}")
                out[r["item_id"]] = (lab, cat)
    return out


def score() -> dict:
    from analyze_p8 import kappa
    keys = json.loads((WORK / "keys.json").read_text(encoding="utf-8"))
    a, b = read("as1"), read("as2")
    s = sorted(set(a) & set(b))
    res = dict(candidates=len(keys), coded=len(s),
               kappa_label=kappa([a[i][0] for i in s], [b[i][0] for i in s], LABELS),
               disagreements=sum(a[i] != b[i] for i in s))
    print(json.dumps(res, indent=1))
    return res


def build_adj() -> None:
    keys = json.loads((WORK / "keys.json").read_text(encoding="utf-8"))
    a, b = read("as1"), read("as2")
    if set(a) != set(keys) or set(b) != set(keys):
        raise SystemExit("coding incomplete")
    rng = random.Random(SEED)
    todo = [i for i in sorted(keys) if a[i] != b[i]]
    with (WORK / "adj_sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["item_id", "answer_a", "answer_b", "title", "abstract"])
        w.writeheader()
        for i in todo:
            x, y = ("/".join(a[i]).rstrip("/"), "/".join(b[i]).rstrip("/"))
            x, y = (x, y) if rng.random() < 0.5 else (y, x)
            w.writerow(dict(item_id=i, answer_a=x, answer_b=y, title=keys[i]["title"], abstract=keys[i]["abstract"]))
    print(len(todo), "disagreements")


def run() -> dict:
    sys.path.insert(0, str(P8.HERE))
    keys = json.loads((WORK / "keys.json").read_text(encoding="utf-8"))
    a, b = read("as1"), read("as2")
    adj = {}
    if (CODES / "adjudication.csv").exists():
        with (CODES / "adjudication.csv").open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lab, cat = r["label"].strip(), (r.get("category") or "").strip()
                if lab not in LABELS or cat not in CATS or not r.get("reason", "").strip():
                    raise SystemExit(f"bad adjudication {r}")
                adj[r["item_id"]] = (lab, cat)
    final = {}
    for i in keys:
        if a[i] == b[i]:
            final[i] = a[i]
        elif i in adj:
            final[i] = adj[i]
        else:
            raise SystemExit(f"not adjudicated: {i}")
    cv = P8.claim_verdicts()
    rows = []
    for i, k in sorted(keys.items(), key=lambda kv: (kv[1]["stratum"], kv[1]["paper_id"])):
        lab, cat = final[i]
        att = dec = ""
        if k["stratum"] == "icml-2026":
            ids = [k["paper_id"]] + k["duplicates"]
            cl = next((cv[x] for x in ids if x in cv), None)
            att = "yes" if cl is not None else "no"
            dec = ("yes" if any(c["verdict"] in P8.DECISIVE for c in cl) else "no") if cl is not None else ""
        rows.append(dict(stratum=k["stratum"], paper_id=k["paper_id"], title=k["title"], label=lab, category=cat,
                         keyword_screen="yes" if k["narrow"] else "no", attempted=att, decided=dec))
    RES.mkdir(exist_ok=True)
    with (RES / "validated.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    inc = [r for r in rows if r["label"] == "include"]
    narrow = [r for r in rows if r["keyword_screen"] == "yes"]
    icml = [r for r in inc if r["stratum"] == "icml-2026"]
    summary = dict(score=score(), candidates=len(rows), included=len(inc), unclear=sum(r["label"] == "unclear" for r in rows),
                   by_venue_year=dict(sorted(Counter(r["stratum"] for r in inc).items())),
                   by_category=dict(Counter(r["category"] for r in inc)),
                   keyword_screen=dict(matches=len(narrow), included=sum(r["label"] == "include" for r in narrow),
                                       precision=sum(r["label"] == "include" for r in narrow) / len(narrow),
                                       validated_found_by_screen=sum(r["keyword_screen"] == "yes" for r in inc)),
                   icml_2026=dict(included=len(icml), attempted=sum(r["attempted"] == "yes" for r in icml),
                                  decided=sum(r["decided"] == "yes" for r in icml)))
    (RES / "validated_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    {"build": build, "score": score, "build_adj": build_adj,
     "run": lambda: print(json.dumps(run(), indent=1))}[sys.argv[1]]()
