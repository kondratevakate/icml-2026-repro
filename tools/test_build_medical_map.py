"""Checks on build_medical_map.py: the page's arithmetic and its geometry.

Each test here corresponds to a defect that reached a rendered page: a signed
axis printing "-0" at the origin, an in-block label running past the viewBox,
`unclear` counted as a negative, and variance levels drawn from coder 1's raw
sheet instead of the adjudicated codes. Run from this directory:

    python -m pytest tools/test_build_medical_map.py        # from the repo root
    python -m pytest test_build_medical_map.py              # from tools/
"""
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_medical_map as B  # noqa: E402

# Space Mono is monospaced with a 600/1000 advance; Space Grotesk is
# proportional, so its classes get a loose upper bound instead.
ADVANCE = {
    "m-sub": (11, 0.60), "m-tick": (10, 0.60), "m-val": (14, 0.60),
    "m-val-sm": (12, 0.60), "m-grp": (11, 0.60), "m-band-n": (10, 0.60),
    "m-block-v": (11, 0.72), "m-block-n": (30, 0.60), "m-block-n-sm": (19, 0.60),
    "m-row": (13, 0.62), "m-field": (13, 0.62),
}


def panels():
    """Every panel on the page, built from the committed result files."""
    rows, keys = B.final_rows()
    den = B.coded_counts(rows, keys)
    venues = B.coded_by_venue(rows, keys)
    axes = [dict(label=label,
                 rows=[dict(label="life-science", **den[f]["life_science"]),
                       dict(label="other", **den[f]["other"])])
            for f, label in (("code", "Code released"),
                             ("main_data_access", "Main data openly downloadable"),
                             ("weights_release", "Weights released"),
                             ("variance", "Variance reported"))]
    a95 = B.load("p9.json")["A9_5"]
    matched = [dict(label="Life-science / medical", sub="attempted at ICML 2026",
                    decided=a95["cells"]["life_science"]["decided"],
                    n=a95["cells"]["life_science"]["n"]),
               dict(label="Other papers, same venue", sub="attempted at ICML 2026",
                    decided=a95["cells"]["other"]["decided"],
                    n=a95["cells"]["other"]["n"])]
    return {"venues": B.venue_blocks(venues),
            "levels": B.paired_bars(axes),
            "waffle": B.waffle(matched)}


class AxisTests(unittest.TestCase):
    def test_origin_tick_is_a_bare_zero(self):
        # "+0" and "-0" are not quantities; the origin gets one label.
        self.assertEqual(B.signed_tick(0.0), "0")
        self.assertEqual(B.signed_tick(-0.0), "0")
        self.assertEqual(B.signed_tick(0.001), "0")

    def test_signed_ticks_keep_their_sign_away_from_the_origin(self):
        self.assertEqual(B.signed_tick(0.13), "+13")
        self.assertEqual(B.signed_tick(-0.19), "-19")

    def test_no_signed_zero_reaches_the_rendered_page(self):
        svg = B.waterfall([dict(label="x", sub="y", rd=-0.19, ci=[-0.28, -0.09])], -0.30, 0.05)
        self.assertNotIn(">+0<", svg)
        self.assertNotIn(">-0<", svg)
        self.assertIn(">0<", svg)


class CodedCountTests(unittest.TestCase):
    """`unclear` is no signal, so it leaves the denominator.

    Painting a no-signal cell as a negative is the error A9.4 measures, and it
    is what the retired keyword layer did to 306 of its 355 papers.
    """

    def setUp(self):
        self.rows, self.keys = B.coder1_rows()

    def test_unclear_is_excluded_from_both_numerator_and_denominator(self):
        keys = {"A": {"group": "other", "stratum": "icml-2026"},
                "B": {"group": "other", "stratum": "icml-2026"},
                "C": {"group": "other", "stratum": "icml-2026"}}
        rows = {"A": {"code": "released"}, "B": {"code": "no"}, "C": {"code": "unclear"}}
        out = B.coded_counts(rows, keys)["code"]["other"]
        self.assertEqual((out["num"], out["den"]), (1, 2))
        self.assertAlmostEqual(out["share"], 0.5)

    def test_promised_code_is_not_released_code(self):
        keys = {"A": {"group": "life_science", "stratum": "icml-2026"}}
        out = B.coded_counts({"A": {"code": "promised"}}, keys)["code"]["life_science"]
        self.assertEqual((out["num"], out["den"]), (0, 1))

    def test_empty_group_does_not_divide_by_zero(self):
        keys = {"A": {"group": "other", "stratum": "icml-2026"}}
        out = B.coded_counts({"A": {"code": "released"}}, keys)["code"]["life_science"]
        self.assertEqual((out["num"], out["den"], out["share"]), (0, 0, 0.0))

    def test_page_levels_are_the_analysis_shares(self):
        # The levels panel once showed variance at 75.1% vs 50.7% from coder 1's
        # raw sheet while p9.json, after adjudication, has 66.4% vs 42.1%.
        rows, keys = B.final_rows()
        den = B.coded_counts(rows, keys)
        a93 = B.load("p9.json")["A9_3"]
        for field in B.POSITIVE:
            for group in ("life_science", "other"):
                self.assertEqual(den[field][group]["den"], a93[field][group]["n_known"], (field, group))
                self.assertAlmostEqual(den[field][group]["share"], a93[field][group]["share"], 12, (field, group))

    def test_committed_sheet_has_a_denominator_per_field_and_group(self):
        den = B.coded_counts(self.rows, self.keys)
        for field, cells in den.items():
            for group in ("life_science", "other"):
                self.assertGreater(cells[group]["den"], 0, field)
                self.assertLessEqual(cells[group]["num"], cells[group]["den"], field)

    def test_venue_split_sums_to_the_coded_corpus(self):
        venues = B.coded_by_venue(self.rows, self.keys)
        coded = B.load("p9.json")["coded"]
        self.assertEqual(sum(v["n"] for v in venues),
                         coded["life_science"] + coded["other"])
        self.assertEqual(sum(v["life_science"] for v in venues), coded["life_science"])
        self.assertEqual([v["n"] for v in venues],
                         sorted((v["n"] for v in venues), reverse=True))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.panels = panels()

    def test_every_panel_is_well_formed(self):
        for name, svg in self.panels.items():
            with self.subTest(panel=name):
                ET.fromstring(svg)  # this module's own output, not external input

    def test_no_shape_falls_outside_its_viewbox(self):
        for name, svg in self.panels.items():
            root = ET.fromstring(svg)
            _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
            for el in root.iter():
                if el.tag.split("}")[-1] != "rect":
                    continue
                with self.subTest(panel=name):
                    self.assertLessEqual(float(el.get("x")) + float(el.get("width")), vw + 1)
                    self.assertLessEqual(float(el.get("y")) + float(el.get("height")), vh + 1)

    def test_no_label_runs_past_its_viewbox(self):
        """A label wider than the box it sits in is how the venue blocks broke."""
        for name, svg in self.panels.items():
            root = ET.fromstring(svg)
            _, _, vw, vh = (float(v) for v in root.get("viewBox").split())
            for el in root.iter():
                if el.tag.split("}")[-1] != "text":
                    continue
                size, ratio = ADVANCE[el.get("class")]
                width = len(el.text or "") * size * ratio
                x, anchor = float(el.get("x")), el.get("text-anchor", "start")
                left = x if anchor == "start" else (x - width if anchor == "end"
                                                    else x - width / 2)
                with self.subTest(panel=name, text=el.text):
                    self.assertGreaterEqual(left, -1)
                    self.assertLessEqual(left + width, vw + 1)
                    self.assertLessEqual(float(el.get("y")), vh)

    def test_venue_block_area_tracks_the_paper_count(self):
        root = ET.fromstring(self.panels["venues"])
        cells = []
        for r in root.iter():
            if r.tag.split("}")[-1] != "rect":
                continue
            n = int(next(iter(r)).text.split(":")[1].split()[0].replace(",", ""))
            cells.append((n, float(r.get("width")) * float(r.get("height"))))
        total_n = sum(n for n, _ in cells)
        total_a = sum(a for _, a in cells)
        for n, a in cells:
            self.assertAlmostEqual(n / total_n, a / total_a, delta=0.02)

    def test_waffle_is_one_square_per_paper_with_the_settled_ones_first(self):
        a95 = B.load("p9.json")["A9_5"]["cells"]
        squares = re.findall(r'class="m-(yes|no)"', self.panels["waffle"])
        self.assertEqual(len(squares), a95["life_science"]["n"] + a95["other"]["n"])
        expected = sum(round(c["decided"] * c["n"]) for c in a95.values())
        self.assertEqual(squares.count("yes"), expected)
        # Sorted within each of the two groups: yes… no… yes… no… -> 3 changes.
        changes = sum(1 for a, b in zip(squares, squares[1:]) if a != b)
        self.assertEqual(changes, 3)

    def test_paired_bars_print_the_crude_difference_of_the_shares(self):
        pairs = re.findall(r'class="m-field"[^>]*>([^<]+)</text>'
                           r'<text class="m-val"[^>]*>([^<]+)</text>', self.panels["levels"])
        self.assertEqual(len(pairs), 4)
        rows, keys = B.final_rows()
        den = B.coded_counts(rows, keys)
        by_label = {"Code released": "code",
                    "Main data openly downloadable": "main_data_access",
                    "Weights released": "weights_release",
                    "Variance reported": "variance"}
        for label, shown in pairs:
            field = den[by_label[label]]
            crude = field["life_science"]["share"] - field["other"]["share"]
            self.assertEqual(shown.replace("−", "-"),
                             B.pts(crude).replace("−", "-"), label)


class PageTests(unittest.TestCase):
    def test_build_is_deterministic(self):
        self.assertEqual(B.build(), B.build())

    def test_every_provenance_file_exists(self):
        page = B.build()
        for name in re.findall(r"<td><code>(\w+\.json)</code></td>", page):
            path = (B.RES / name) if (B.RES / name).exists() else (B.ROOT / "audit" / name)
            self.assertTrue(path.exists(), name)

    def test_page_carries_no_client_side_data_blob(self):
        page = B.build()
        self.assertNotIn("<script", page)
        self.assertNotIn("base64", page)


if __name__ == "__main__":
    unittest.main()
