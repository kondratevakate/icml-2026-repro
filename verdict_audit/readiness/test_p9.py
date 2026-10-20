import fetch_p9 as F

ATOM = """<feed><entry><id>http://arxiv.org/abs/2501.01234v2</id><published>2025-01-03T00:00:00Z</published>
<title>Deterministic Image-to-Image Translation via
  Denoising Brownian Bridge Models</title></entry>
<entry><id>http://arxiv.org/abs/1901.00001v1</id><published>2019-01-01T00:00:00Z</published>
<title>Deterministic Image-to-Image Translation via Denoising Brownian Bridge Models</title></entry></feed>"""


def test_norm_and_parse_atom():
    assert F.norm("Image-to-Image: A &amp; B — Café") == "image to image a b cafe"
    e = F.parse_atom(ATOM)
    assert [x["id"] for x in e] == ["2501.01234", "1901.00001"] and e[0]["year"] == 2025


def test_pick_rules():
    title = "Deterministic Image-to-Image Translation via Denoising Brownian Bridge Models"
    entries = F.parse_atom(ATOM)
    assert F.pick(entries, title, 2025)[:2] == ("ok", "2501.01234")  # the 2019 entry fails the year rule
    twin = entries + [dict(id="2502.09999", title=entries[0]["title"], year=2025)]
    assert F.pick(twin, title, 2025)[0] == "ambiguous"
    assert F.pick(entries, "A completely different paper", 2025)[0] == "no_match"


def test_parse_pmlr_and_cvf():
    pmlr = ('<div class="paper"><p class="title">T</p><p class="links">[<a href="https://raw.githubusercontent.com/'
            'mlresearch/v306/main/assets/x26a/x26a.pdf">Download PDF</a>][<a href="https://openreview.net/forum?id=YDz-tz_1">'
            'OpenReview</a>]</p></div>')
    assert F.parse_pmlr(pmlr, "https://proceedings.mlr.press/v306/") == {
        "YDz-tz_1": "https://raw.githubusercontent.com/mlresearch/v306/main/assets/x26a/x26a.pdf"}
    cvf = '<dt class="ptitle"><br><a href="/content/CVPR2025/html/X_CVPR_2025_paper.html">Deep Thing: A Study</a></dt>'
    assert F.parse_cvf(cvf, "https://openaccess.thecvf.com/CVPR2025?day=all") == {
        "deep thing a study": "https://openaccess.thecvf.com/content/CVPR2025/papers/X_CVPR_2025_paper.pdf"}


def test_key_is_path_safe():
    assert F.key_of(dict(stratum="cvpr-2025", forum_id="src:10379")) == "cvpr-2025__src-10379"


def test_wave_holds_partial_batches_until_final(tmp_path, monkeypatch):
    import json
    import build_p9_sheets as S
    keys = {f"P{i:04d}": dict(key=f"k{i}", title=f"t{i}", coder2=i < 3) for i in range(45)}
    (tmp_path / "keys.json").write_text(json.dumps(keys))
    (tmp_path / "batches.json").write_text(json.dumps(dict(coder1=[], coder2=[])))
    monkeypatch.setattr(S, "KEYS", tmp_path / "keys.json")
    monkeypatch.setattr(S, "BATCHES", tmp_path / "batches.json")
    monkeypatch.setattr(S, "SHEETS", tmp_path / "sheets")
    monkeypatch.setattr(S, "excerpts", lambda: {f"k{i}": dict(truncated=False, excerpt="x") for i in range(45)})
    S.wave()
    b = json.loads((tmp_path / "batches.json").read_text())
    assert [len(x["items"]) for x in b["coder1"]] == [20, 20] and b["coder2"] == []
    S.wave(final=True)
    b = json.loads((tmp_path / "batches.json").read_text())
    assert [len(x["items"]) for x in b["coder1"]] == [20, 20, 5] and [len(x["items"]) for x in b["coder2"]] == [3]
    assert len({i for x in b["coder1"] for i in x["items"]}) == 45


def test_p9_label_set_matches_the_coder_prompt():
    import build_p9_sheets as S
    prompt = (S.CODES / "CODER_PROMPT.md").read_text(encoding="utf-8")
    header = "item_id," + ",".join(S.FIELDS)
    assert header in prompt
    for field, labels in S.LABELS.items():
        for lab in labels:
            assert lab in prompt, (field, lab)
