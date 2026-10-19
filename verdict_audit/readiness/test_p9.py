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
