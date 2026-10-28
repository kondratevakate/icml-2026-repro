"""POST HOC (not in the frozen protocol): judge verdicts on the logbooks the organisers' blog cites
for each confirmed falsification. Team names are not written; papers are keyed by short name."""
import hashlib
import json

import common as C

CITED = {  # SHA-256 of the Space ids the blog cites (https://huggingface.co/blog/icml-2026-open-reproductions,
           # retrieved 2026-10-18); hashed so names are not spelled out here (they are public in that blog post)
    "paging": "dced11c2cbe00d4f6d2417315edb265c967eadd17ba3b1b53b07bdd475f5cd71",
    "frank_wolfe": "d83077dc238d9bd7e8b5cc0951f78b06cf14e4e437f02accafbd50282a3458c7",
    "self_distillation": "395c33ea49679db261bb87c0b7e42a80f060d40e19b8e0b101a5ff74c66b2098",
    "three_projections": "9ed96a13461709704b406dd1e0a00cf8d2e77df18ccc478585cda090f19b5262",
}


def main() -> None:
    C.check_inputs()
    verdicts = C.A.load_json(C.ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json")
    _, _, raw, _, _ = C.load()
    out = {}
    by_hash = {hashlib.sha256(k.encode()).hexdigest(): k for k in verdicts}
    for short, digest in CITED.items():
        r = verdicts[by_hash[digest]]
        index = {C.A.norm(c["text"]): i for i, c in enumerate(raw[r["orid"]])}
        out[short] = dict(orid=r["orid"], judged_at=r["judged_at"], verdicts_by_anchored_index={
            str(index.get(C.A.norm(c["claim"]), f"unanchored_{j}")): C.A.category(c["verdict"])
            for j, c in enumerate(r["claims"])})
    out["note"] = "post hoc; descriptive; not a prespecified analysis"
    (C.HERE / "results" / "posthoc_blog_logbooks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
