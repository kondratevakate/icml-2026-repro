"""Protocol 4, amendment 2: H1 on the protocol 6 validated labels."""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_e1 as E  # noqa: E402

F = HERE.parent / "falsifiability"


def labels():
    a = {r["item"]: r["label"] for r in E.read(F / "codes_A.csv")}
    b = {r["item"]: r["label"] for r in E.read(F / "codes_B.csv")}
    adj = {r["item"]: r["label"] for r in E.read(F / "adjudication.csv")}
    sheet = {r["item"]: r["orid"] for r in E.read(F / "sheets/candidates.csv")}
    cons = {sheet[i]: (a[i] if a[i] == b[i] else adj[i]) for i in a}
    return cons, set(sheet.values())


def main():
    out = HERE / "results_e1_validated.json"
    if out.exists():
        sys.exit("exists; refusing to overwrite")
    cons, cand = labels()
    broad = E.A.load_orids(E.ROOT / "audit/public_repro_audit.csv")
    df = E.frame()
    df["kate"] = (df["reproducer"].str.lower() == "kondratevakate").astype(int)
    other = ~df["paper"].isin(cand | broad)
    res = {}
    for exposure in ("clinical", "life_science"):
        sel = df[other | df["paper"].map(lambda o: cons.get(o) == exposure)].copy()
        sel["exposure"] = sel["paper"].map(lambda o: int(cons.get(o) == exposure))
        res[exposure] = dict(
            without_kate=E.lpm(sel[sel["kate"] == 0], med_col="exposure", theory=True),
            with_kate=E.lpm(sel, med_col="exposure", theory=True))
    c = res["clinical"]["without_kate"]["ci"]
    res["decision_clinical_not_judged_more_severely"] = bool(c[0] <= 0 <= c[1])
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
