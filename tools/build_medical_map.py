"""Build docs/medical-reproducibility.html from the frozen Protocol 9 results.

Every number on the page is read from a result file at build time; nothing is
typed into the template. The page is static SVG (no client-side data blob), so a
reader can diff the HTML against the JSON and a reviewer can regenerate it.

Inputs (all tracked):
  verdict_audit/readiness/results/p9.json             labels L9.1-L9.3, gate, coded, A9_4, A9_5
  verdict_audit/readiness/results/p9_adversarial.json F1 (code, coder 1 full coverage), F2 permutation
  verdict_audit/readiness/results/p9_extended.json    units, E1 (team strength)
  verdict_audit/readiness/results/p9_claims.json      score, K1 (check cost), K3 (decomposition)
  verdict_audit/readiness/results/readiness.json      R_A3 cells (decided share of the other group)
  audit/map_label_check.json                          retired keyword layer vs the coded label
  verdict_audit/readiness/codes/p9/                    final codes per field, venue, group (analyze_p9.final_codes:
                                                       coder 1 above the gate, adjudicated variance and compute)

Usage: python tools/build_medical_map.py   -> docs/medical-reproducibility.html
"""
from __future__ import annotations

import csv
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "verdict_audit" / "readiness" / "results"
CODES = ROOT / "verdict_audit" / "readiness" / "codes" / "p9"
KEYS = ROOT / "verdict_audit" / "readiness" / "data" / "p9" / "keys.json"
OUT = ROOT / "docs" / "medical-reproducibility.html"

# Positive category per field, from the Protocol 9 codebook (build_p9_sheets.LABELS)
# and the A9.3 indicator list in verdict_audit/readiness/PROTOCOL.md.
POSITIVE = {"code": "released", "main_data_access": "open", "weights_release": "yes", "variance": "yes"}


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def coder1_rows() -> tuple[dict, dict]:
    """Coder 1's sheet (full coverage) and the item keys it is joined on."""
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    rows = {}
    for path in sorted(glob.glob(str(CODES / "p9_coder1_batch_*.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows[r["item_id"]] = r
    return rows, keys


def final_rows() -> tuple[dict, dict]:
    """The final codes the analysis uses (Note 9-5, 9-6): coder 1 for fields above the agreement gate, and for
    variance and compute coder 1 where both coders agree, else the adjudicated label. Coder 1's raw sheet would
    show variance before adjudication."""
    import sys
    sys.path.insert(0, str(ROOT / "verdict_audit" / "readiness"))
    from analyze_p9 import final_codes
    return final_codes(), json.loads(KEYS.read_text(encoding="utf-8"))


def coded_counts(rows: dict, keys: dict) -> dict:
    """Per field and group: how many items were coded, and how many positively.

    `unclear` is not evidence either way, so it leaves the denominator rather
    than counting against the field. Painting a no-signal cell as a negative is
    the error A9.4 measures.
    """
    out = {}
    for field, positive in POSITIVE.items():
        den, num = Counter(), Counter()
        for item, r in rows.items():
            value = (r.get(field) or "").strip()
            if value and value != "unclear":
                group = keys[item]["group"]
                den[group] += 1
                if value == positive:
                    num[group] += 1
        out[field] = {g: dict(num=num[g], den=den[g],
                              share=num[g] / den[g] if den[g] else 0.0)
                      for g in ("life_science", "other")}
        out[field]["n"] = den["life_science"] + den["other"]
    return out


def coded_by_venue(rows: dict, keys: dict) -> list[dict]:
    """Coded full texts per venue, largest first. Strata are `<venue>-<year>`."""
    seen = Counter()
    for item in rows:
        seen[(keys[item]["stratum"].split("-")[0], keys[item]["group"])] += 1
    out = [dict(venue=v, life_science=seen[(v, "life_science")], other=seen[(v, "other")],
                n=seen[(v, "life_science")] + seen[(v, "other")])
           for v in sorted({v for v, _ in seen})]
    return sorted(out, key=lambda d: -d["n"])


# ---------------------------------------------------------------- SVG helpers

def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pts(v: float) -> str:
    """Signed percentage points, one decimal."""
    return f"{v * 100:+.1f}"


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def svg_open(uid: str, width: int, height: int, label: str) -> str:
    return (f'<svg class="m-svg" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{esc(label)}" preserveAspectRatio="xMinYMin meet">')


def signed_tick(v: float) -> str:
    """Axis label in points; a bare zero at the origin, never "+0" or "-0".

    Rounding first covers both the float drift of an accumulating tick loop and
    a step fine enough that a real tick falls under half a point.
    """
    points = round(v * 100)
    return "0" if points == 0 else f"{points:+d}"


def axis(x0: int, x1: int, y: int, lo: float, hi: float, step: float, fmt) -> str:
    """Horizontal tick axis with a rule per tick."""
    out, v = "", lo
    while v <= hi + 1e-9:
        x = x0 + (x1 - x0) * (v - lo) / (hi - lo)
        cls = "m-zero" if abs(v) < 1e-9 else "m-grid"
        out += f'<line class="{cls}" x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{y}"/>'
        out += f'<text class="m-tick" x="{x:.1f}" y="20" text-anchor="middle">{fmt(v)}</text>'
        v += step
    return out


def diff_rows(rows: list[dict], lo: float, hi: float, step: float) -> str:
    """Rate differences with CI whiskers: one bar per row, anchored at zero."""
    x0, x1, top, rh = 300, 760, 40, 50
    height = top + len(rows) * rh + 14
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)  # noqa: E731
    out = svg_open("diff", 900, height, "Rate differences, life-science minus other")
    out += axis(x0, x1, top + len(rows) * rh, lo, hi, step, signed_tick)
    for i, r in enumerate(rows):
        cy = top + i * rh + rh / 2 - 4
        a, b = sx(min(0, r["rd"])), sx(max(0, r["rd"]))
        out += f'<text class="m-row" x="0" y="{cy - 2:.0f}">{esc(r["label"])}</text>'
        out += f'<text class="m-sub" x="0" y="{cy + 13:.0f}">{esc(r["sub"])}</text>'
        out += f'<rect class="m-bar" x="{a:.1f}" y="{cy - 9:.1f}" width="{max(1, b - a):.1f}" height="12"/>'
        out += (f'<line class="m-ci" x1="{sx(r["ci"][0]):.1f}" y1="{cy - 3:.1f}" '
                f'x2="{sx(r["ci"][1]):.1f}" y2="{cy - 3:.1f}"/>')
        for e in r["ci"]:
            out += f'<line class="m-ci" x1="{sx(e):.1f}" y1="{cy - 9:.1f}" x2="{sx(e):.1f}" y2="{cy + 3:.1f}"/>'
        out += f'<text class="m-val" x="775" y="{cy - 1:.0f}">{pts(r["rd"])}</text>'
        out += (f'<text class="m-sub" x="775" y="{cy + 13:.0f}">'
                f'[{pts(r["ci"][0])}, {pts(r["ci"][1])}]</text>')
    return out + "</svg>"


def group_rows(rows: list[dict], title: str, note: str = "") -> str:
    """Two-group share comparison on a 0-100% axis."""
    x0, x1, top, rh = 300, 760, 44, 52
    height = top + len(rows) * rh + (26 if note else 12)
    sx = lambda v: x0 + (x1 - x0) * v  # noqa: E731
    out = svg_open("grp", 900, height, title)
    out += axis(x0, x1, top + len(rows) * rh, 0, 1, 0.25, lambda v: f"{v * 100:.0f}%")
    for i, r in enumerate(rows):
        cy = top + i * rh + rh / 2 - 6
        fill = "m-bar" if r["medical"] else "m-bar-alt"
        out += f'<text class="m-row" x="0" y="{cy - 1:.0f}">{esc(r["label"])}</text>'
        out += f'<text class="m-sub" x="0" y="{cy + 14:.0f}">{esc(r["sub"])}</text>'
        out += f'<rect class="m-track" x="{x0}" y="{cy - 11:.1f}" width="{x1 - x0}" height="20"/>'
        out += (f'<rect class="{fill}" x="{x0}" y="{cy - 11:.1f}" '
                f'width="{max(1, sx(r["value"]) - x0):.1f}" height="20"/>')
        out += f'<text class="m-val" x="775" y="{cy + 3:.0f}">{pct(r["value"])}</text>'
    if note:
        out += f'<text class="m-sub" x="300" y="{height - 4}">{esc(note)}</text>'
    return out + "</svg>"


def waterfall(steps: list[dict], lo: float, hi: float) -> str:
    """The gap and what is left of it after each adjustment."""
    x0, x1, top, rh = 300, 760, 40, 56
    height = top + len(steps) * rh + 14
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)  # noqa: E731
    out = svg_open("wf", 900, height, "Settlement gap before and after adjustment")
    out += axis(x0, x1, top + len(steps) * rh, lo, hi, 0.05, signed_tick)
    for i, s in enumerate(steps):
        cy = top + i * rh + rh / 2 - 6
        a, b = sx(min(0, s["rd"])), sx(max(0, s["rd"]))
        out += f'<text class="m-row" x="0" y="{cy - 1:.0f}">{esc(s["label"])}</text>'
        out += f'<text class="m-sub" x="0" y="{cy + 14:.0f}">{esc(s["sub"])}</text>'
        out += f'<rect class="m-bar" x="{a:.1f}" y="{cy - 10:.1f}" width="{max(1, b - a):.1f}" height="14"/>'
        out += (f'<line class="m-ci" x1="{sx(s["ci"][0]):.1f}" y1="{cy - 3:.1f}" '
                f'x2="{sx(s["ci"][1]):.1f}" y2="{cy - 3:.1f}"/>')
        for e in s["ci"]:
            out += f'<line class="m-ci" x1="{sx(e):.1f}" y1="{cy - 10:.1f}" x2="{sx(e):.1f}" y2="{cy + 4:.1f}"/>'
        out += f'<text class="m-val" x="775" y="{cy - 1:.0f}">{pts(s["rd"])}</text>'
        out += (f'<text class="m-sub" x="775" y="{cy + 14:.0f}">'
                f'[{pts(s["ci"][0])}, {pts(s["ci"][1])}]</text>')
    return out + "</svg>"


def composition(groups: list[dict], bands: list[tuple[str, str]], boundary: int) -> str:
    """Stacked composition of claim check cost; shade encodes cost, not category."""
    x0, x1, top, rh = 300, 760, 44, 70
    height = top + len(groups) * rh + 22
    out = svg_open("cmp", 900, height, "Claim check cost composition")
    for i, g in enumerate(groups):
        cy, x = top + i * rh, x0
        total = sum(g["counts"].get(k, 0) for k, _ in bands)
        out += f'<text class="m-row" x="0" y="{cy + 14:.0f}">{esc(g["label"])}</text>'
        out += f'<text class="m-sub" x="0" y="{cy + 29:.0f}">{esc(g["sub"])}</text>'
        for j, (key, _) in enumerate(bands):
            n = g["counts"].get(key, 0)
            if not n:
                continue
            w = (x1 - x0) * n / total
            out += (f'<rect class="m-band m-band-{j}" x="{x:.1f}" y="{cy:.1f}" '
                    f'width="{w:.1f}" height="26"><title>{esc(key)}: {n} claims</title></rect>')
            if w > 26:
                out += (f'<text class="m-band-n" x="{x + w / 2:.1f}" y="{cy + 42:.0f}" '
                        f'text-anchor="middle">{n}</text>')
            x += w
        edge = x0 + (x1 - x0) * sum(g["counts"].get(k, 0) for k, _ in bands[:boundary]) / total
        out += f'<line class="m-split" x1="{edge:.1f}" y1="{cy - 5:.1f}" x2="{edge:.1f}" y2="{cy + 31:.1f}"/>'
        out += f'<text class="m-val" x="775" y="{cy + 17:.0f}">{pct(edge_share(g, bands, boundary))}</text>'
    out += f'<text class="m-sub" x="300" y="{height - 2}">left of the rule: retraining or large-scale runs</text>'
    return out + "</svg>"


def edge_share(g: dict, bands: list[tuple[str, str]], boundary: int) -> float:
    total = sum(g["counts"].get(k, 0) for k, _ in bands)
    return sum(g["counts"].get(k, 0) for k, _ in bands[:boundary]) / total


def kappa_rows(rows: list[dict]) -> str:
    """Agreement between what a paper states and what a reproducer obtained."""
    x0, x1, top, rh = 300, 700, 40, 46
    height = top + len(rows) * rh + 14
    sx = lambda v: x0 + (x1 - x0) * v  # noqa: E731
    out = svg_open("kap", 900, height, "Stated artifact versus obtained artifact, Cohen's kappa")
    out += axis(x0, x1, top + len(rows) * rh, 0, 1, 0.2, lambda v: f"{v:.1f}")
    for i, r in enumerate(rows):
        cy = top + i * rh + rh / 2 - 4
        out += f'<text class="m-row" x="0" y="{cy - 1:.0f}">{esc(r["label"])}</text>'
        out += f'<text class="m-sub" x="0" y="{cy + 13:.0f}">{esc(r["sub"])}</text>'
        out += (f'<rect class="m-bar" x="{x0}" y="{cy - 9:.1f}" '
                f'width="{max(1, sx(r["kappa"]) - x0):.1f}" height="12"/>')
        out += f'<text class="m-val" x="715" y="{cy + 2:.0f}">{r["kappa"]:.2f}</text>'
    return out + "</svg>"


def venue_block(x: float, y: float, w: float, h: float, v: dict, big: bool) -> str:
    n_cls = "m-block-n" if big else "m-block-n-sm"
    y_count, y_split = (58, 78) if big else (43, 61)
    split = (f'{v["life_science"]:,} life-sci  ·  {v["other"]:,} other' if big
             else f'{v["life_science"]:,}  ·  {v["other"]:,}')
    return (f'<rect class="m-block" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}">'
            f'<title>{esc(v["venue"])}: {v["n"]:,} full texts coded</title></rect>'
            f'<text class="m-block-v" x="{x + 12:.1f}" y="{y + 23:.0f}">{esc(v["venue"].upper())}</text>'
            f'<text class="{n_cls}" x="{x + 12:.1f}" y="{y + y_count:.0f}">{v["n"]:,}</text>'
            f'<text class="m-sub" x="{x + 12:.1f}" y="{y + y_split:.0f}">{esc(split)}</text>')


def venue_blocks(venues: list[dict], cutoff: float = 0.10) -> str:
    """Coded corpus by venue; block area is the number of full texts read.

    Venues holding at least `cutoff` of the corpus get a full-height column and
    the rest share one right-hand column, so a small venue keeps enough area to
    be labelled from the inside instead of pointed at from outside.
    """
    width, height, gap = 880, 240, 6
    total = sum(v["n"] for v in venues)
    big = [v for v in venues if v["n"] / total >= cutoff]
    small = [v for v in venues if v["n"] / total < cutoff]
    avail = width - gap * (len(big) + (1 if small else 0) - 1)
    out = svg_open("ven", 900, height,
                   "Coded full texts by venue; block area is the paper count")
    x = 0.0
    for v in big:
        w = avail * v["n"] / total
        out += venue_block(x, 0, w, height, v, big=True)
        x += w + gap
    if small:
        sub_total = sum(v["n"] for v in small)
        w, y = avail * sub_total / total, 0.0
        avail_h = height - gap * (len(small) - 1)
        for v in small:
            h = avail_h * v["n"] / sub_total
            out += venue_block(x, y, w, h, v, big=False)
            y += h + gap
    return out + "</svg>"


def paired_bars(fields: list[dict]) -> str:
    """Observed share per documentation axis, both groups, with the raw counts."""
    x0, x1, top, fh = 300, 700, 46, 78
    height = top + len(fields) * fh + 6
    out = svg_open("pair", 900, height, "Observed positive share per documentation axis")
    out += axis(x0, x1, top + len(fields) * fh - 24, 0, 1, 0.25, lambda v: f"{v * 100:.0f}%")
    for i, f in enumerate(fields):
        y = top + i * fh
        crude = f["rows"][0]["share"] - f["rows"][1]["share"]
        out += f'<text class="m-field" x="0" y="{y - 9:.0f}">{esc(f["label"])}</text>'
        out += f'<text class="m-val" x="880" y="{y - 9:.0f}" text-anchor="end">{pts(crude)}</text>'
        for j, r in enumerate(f["rows"]):
            by = y + j * 26
            out += (f'<text class="m-grp" x="290" y="{by + 14:.0f}" text-anchor="end">'
                    f'{esc(r["label"])}</text>')
            out += f'<rect class="m-track" x="{x0}" y="{by:.0f}" width="{x1 - x0}" height="19"/>'
            out += (f'<rect class="{"m-bar" if j == 0 else "m-bar-alt"}" x="{x0}" y="{by:.0f}" '
                    f'width="{max(1, (x1 - x0) * r["share"]):.1f}" height="19"/>')
            out += f'<text class="m-val-sm" x="{x1 + 12}" y="{by + 14:.0f}">{pct(r["share"])}</text>'
            out += (f'<text class="m-sub" x="{x1 + 66}" y="{by + 14:.0f}">'
                    f'{r["num"]:,} of {r["den"]:,}</text>')
    return out + "</svg>"


def waffle(groups: list[dict], per_row: int = 32) -> str:
    """One square per attempted paper, settled ones first.

    Grouping the settled squares at the start of the reading order makes the
    comparison a difference of two block lengths, so the figure still reads
    without discriminating the two hues.
    """
    x0, cell, gap, top, lead = 300, 11, 3, 36, 30
    pitch = cell + gap
    for g in groups:
        g["rows"] = -(-g["n"] // per_row)
        g["decided_n"] = round(g["decided"] * g["n"])
    height = top + sum(g["rows"] * pitch + lead for g in groups) + 6
    out = svg_open("waf", 900, height, "Attempted papers and whether any team reached a verdict")
    y = float(top)
    for g in groups:
        out += f'<text class="m-row" x="0" y="{y + 12:.0f}">{esc(g["label"])}</text>'
        out += f'<text class="m-sub" x="0" y="{y + 27:.0f}">{esc(g["sub"])}</text>'
        for k in range(g["n"]):
            cls = "m-yes" if k < g["decided_n"] else "m-no"
            out += (f'<rect class="{cls}" x="{x0 + (k % per_row) * pitch:.0f}" '
                    f'y="{y + (k // per_row) * pitch:.0f}" width="{cell}" height="{cell}"/>')
        out += (f'<text class="m-val" x="880" y="{y + 14:.0f}" text-anchor="end">'
                f'{pct(g["decided"])}</text>')
        out += (f'<text class="m-sub" x="880" y="{y + 29:.0f}" text-anchor="end">'
                f'{g["decided_n"]} of {g["n"]} settled</text>')
        y += g["rows"] * pitch + lead
    return out + "</svg>"


# ---------------------------------------------------------------- page

CSS = """
:root{
  --bg:0 0% 100%; --fg:0 0% 0%; --muted:0 0% 45%; --rule:0 0% 0%;
  --success:142 72% 36%; --warning:32 95% 44%; --destructive:0 84% 60%;
  --shade-0:0 0% 0%; --shade-1:0 0% 35%; --shade-2:0 0% 62%; --shade-3:0 0% 84%;
  --shadow:5px 5px 0px 0px #000;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:0 0% 0%; --fg:0 0% 100%; --muted:0 0% 65%; --rule:0 0% 100%;
    --success:142 60% 52%; --warning:32 90% 58%; --destructive:0 72% 62%;
    --shade-0:0 0% 100%; --shade-1:0 0% 72%; --shade-2:0 0% 46%; --shade-3:0 0% 24%;
    --shadow:5px 5px 0px 0px #fff;
  }
}
*{box-sizing:border-box}
body{margin:0;background:hsl(var(--bg));color:hsl(var(--fg));
  font-family:'Space Grotesk',ui-sans-serif,system-ui,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased;}
main{max-width:1100px;margin:0 auto;padding:40px 20px 72px;}
.eyebrow{font-family:'Space Mono',ui-monospace,monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:hsl(var(--muted));margin:0 0 18px;}
h1{font-size:40px;line-height:1.05;font-weight:700;letter-spacing:-.02em;margin:0 0 18px;}
@media (min-width:768px){h1{font-size:60px}}
.lead{font-size:17px;line-height:1.6;max-width:62ch;margin:0 0 26px;}
.lead b{font-weight:600}
h2{font-family:'Lora',ui-serif,Georgia,serif;font-size:22px;font-weight:700;margin:0 0 6px;}
@media (min-width:768px){h2{font-size:26px}}
.step{font-family:'Space Mono',ui-monospace,monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:hsl(var(--muted));margin:0 0 4px;}
p.note{font-size:14px;line-height:1.6;color:hsl(var(--muted));max-width:70ch;margin:0 0 18px;}
p.read{font-size:14px;line-height:1.6;max-width:72ch;margin:14px 0 0;
  border-left:2px solid hsl(var(--rule));padding-left:12px;}
section{border-top:1px solid hsl(var(--rule));padding:32px 0 10px;}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;border:0;padding:0 0 34px;}
@media (max-width:860px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:460px){.kpis{grid-template-columns:1fr}}
.kpi{border:1px solid hsl(var(--rule));box-shadow:var(--shadow);padding:14px 14px 12px;}
.kpi .k{font-family:'Space Mono',ui-monospace,monospace;font-size:28px;font-weight:700;
  line-height:1.1;display:block;}
.kpi .t{font-size:12px;line-height:1.45;color:hsl(var(--muted));display:block;margin-top:6px;}
.scope{display:flex;flex-wrap:wrap;gap:0;margin:0 0 34px;border:1px solid hsl(var(--rule));}
.scope div{flex:1 1 200px;padding:12px 14px;border-right:1px solid hsl(var(--rule));}
.scope div:last-child{border-right:0}
.scope dt{font-family:'Space Mono',ui-monospace,monospace;font-size:10px;letter-spacing:.12em;
  text-transform:uppercase;color:hsl(var(--muted));margin:0 0 4px;}
.scope dd{margin:0;font-size:14px;line-height:1.4;}
.chip{display:inline-block;font-family:'Space Mono',ui-monospace,monospace;font-size:10px;
  letter-spacing:.08em;text-transform:uppercase;border:1px solid currentColor;padding:2px 6px;
  margin:0 0 0 8px;vertical-align:2px;}
.chip-ok{color:hsl(var(--success))}
.chip-warn{color:hsl(var(--warning))}
.chip-bad{color:hsl(var(--destructive))}
.m-svg{display:block;width:100%;height:auto;overflow:visible;margin:4px 0 0;}
.m-row{fill:hsl(var(--fg));font-size:13px;font-weight:500}
.m-sub{fill:hsl(var(--muted));font-size:11px;font-family:'Space Mono',ui-monospace,monospace}
.m-tick{fill:hsl(var(--muted));font-size:10px;font-family:'Space Mono',ui-monospace,monospace}
.m-val{fill:hsl(var(--fg));font-size:14px;font-weight:700;font-family:'Space Mono',ui-monospace,monospace}
.m-bar{fill:hsl(var(--shade-0))}
.m-bar-alt{fill:hsl(var(--shade-2))}
.m-track{fill:none;stroke:hsl(var(--rule));stroke-width:1}
.m-ci{stroke:hsl(var(--rule));stroke-width:1.5}
.m-grid{stroke:hsl(var(--rule));stroke-width:1;stroke-opacity:.16}
.m-zero{stroke:hsl(var(--rule));stroke-width:1.5}
.m-split{stroke:hsl(var(--rule));stroke-width:2}
.m-band-0{fill:hsl(var(--shade-0))}
.m-band-1{fill:hsl(var(--shade-1))}
.m-band-2{fill:hsl(var(--shade-2))}
.m-band-3{fill:hsl(var(--shade-3))}
.m-band-4{fill:none;stroke:hsl(var(--rule));stroke-width:1}
.m-band-n{fill:hsl(var(--muted));font-size:10px;font-family:'Space Mono',ui-monospace,monospace}
.m-block{fill:hsl(var(--shade-3));stroke:hsl(var(--rule));stroke-width:1}
.m-block-v{fill:hsl(var(--fg));font-size:11px;letter-spacing:.12em;font-family:'Space Mono',ui-monospace,monospace}
.m-block-n{fill:hsl(var(--fg));font-size:30px;font-weight:700;font-family:'Space Mono',ui-monospace,monospace}
.m-block-n-sm{fill:hsl(var(--fg));font-size:19px;font-weight:700;font-family:'Space Mono',ui-monospace,monospace}
.m-field{fill:hsl(var(--fg));font-size:13px;font-weight:600}
.m-grp{fill:hsl(var(--muted));font-size:11px;font-family:'Space Mono',ui-monospace,monospace}
.m-val-sm{fill:hsl(var(--fg));font-size:12px;font-weight:700;font-family:'Space Mono',ui-monospace,monospace}
.m-yes{fill:hsl(var(--success))}
.m-no{fill:hsl(var(--destructive))}
.lg-yes{background:hsl(var(--success))}
.lg-no{background:hsl(var(--destructive))}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:14px 0 0;font-size:11px;
  font-family:'Space Mono',ui-monospace,monospace;color:hsl(var(--muted));}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:14px;height:10px;display:inline-block;border:1px solid hsl(var(--rule));}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px;}
th,td{border:1px solid hsl(var(--rule));padding:7px 9px;text-align:left;vertical-align:top;}
th{font-family:'Space Mono',ui-monospace,monospace;font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;font-weight:400;color:hsl(var(--muted));}
td code{font-family:'Space Mono',ui-monospace,monospace;font-size:11px;}
footer{border-top:1px solid hsl(var(--rule));margin-top:34px;padding-top:18px;font-size:12px;
  line-height:1.6;color:hsl(var(--muted));}
footer a{color:inherit}
"""


def build() -> str:
    p9, adv = load("p9.json"), load("p9_adversarial.json")
    ext, clm = load("p9_extended.json"), load("p9_claims.json")
    rdy = load("readiness.json")
    check = json.loads((ROOT / "audit" / "map_label_check.json").read_text(encoding="utf-8"))
    rows1, keys = final_rows()
    den = coded_counts(rows1, keys)
    venues = coded_by_venue(rows1, keys)

    coded = p9["coded"]
    n_coded = coded["life_science"] + coded["other"]
    gate = p9["gate"]
    code_eff = adv["F1"]["code"]["coder1_only"]
    labels = p9["labels"]

    # --- 1. documentation -------------------------------------------------
    doc = [
        dict(label="Code released", rd=code_eff["rd"], ci=code_eff["ci"],
             sub=f"{den['code']['n']} coded  ·  gate met" if gate["code"] else f"{den['code']['n']} coded"),
        dict(label="Main data openly downloadable", rd=labels["L9.1"]["rd"], ci=labels["L9.1"]["ci"],
             sub=f"{den['main_data_access']['n']} coded  ·  gate met"),
        dict(label="Weights released", rd=labels["L9.2"]["rd"], ci=labels["L9.2"]["ci"],
             sub=f"{den['weights_release']['n']} coded  ·  gate met"),
        dict(label="Variance reported", rd=labels["L9.3"]["rd"], ci=labels["L9.3"]["ci"],
             sub=f"{den['variance']['n']} coded  ·  adjudicated"),
    ]

    axes = [
        dict(label=label, rows=[dict(label="life-science", **den[field]["life_science"]),
                                dict(label="other", **den[field]["other"])])
        for field, label in (("code", "Code released"),
                             ("main_data_access", "Main data openly downloadable"),
                             ("weights_release", "Weights released"),
                             ("variance", "Variance reported"))
    ]

    # --- 2. settlement ----------------------------------------------------
    units = ext["units"]
    med_decided = p9["A9_5"]["cells"]["life_science"]["decided"]
    cells = rdy["R_A3"]["protocol7_label"]["cells"]
    oth_decided = ((cells["other_code1"]["n"] * cells["other_code1"]["decided"]
                    + cells["other_code0"]["n"] * cells["other_code0"]["decided"])
                   / (cells["other_code1"]["n"] + cells["other_code0"]["n"]))
    decided_gap = ext["E1"]["decided"]["unadjusted"]
    perm_p = adv["F2"]["permutation"]["p"]
    settle = [
        dict(label="Life-science / medical", value=med_decided, medical=True,
             sub=f"{units['medical']} attempted papers"),
        dict(label="All other papers", value=oth_decided, medical=False,
             sub=f"{units['other']:,} attempted papers"),
    ]

    a95 = p9["A9_5"]
    matched = [
        dict(label="Life-science / medical", decided=a95["cells"]["life_science"]["decided"],
             n=a95["cells"]["life_science"]["n"], sub="attempted at ICML 2026"),
        dict(label="Other papers, same venue", decided=a95["cells"]["other"]["decided"],
             n=a95["cells"]["other"]["n"], sub="attempted at ICML 2026"),
    ]

    # --- 3. where the gap goes -------------------------------------------
    k3 = clm["K3"]
    ladder = [
        dict(label="Full-verification gap", rd=k3["unadjusted"]["rd"], ci=k3["unadjusted"]["ci"],
             sub="medical minus other, as observed"),
        dict(label="after first team's settle rate", rd=k3["strength"]["rd"], ci=k3["strength"]["ci"],
             sub="who picked the paper up first"),
        dict(label="after that and claim check cost", rd=k3["strength_and_cost"]["rd"],
             ci=k3["strength_and_cost"]["ci"], sub="what the claims cost to check"),
    ]

    # --- 4. the two factors ----------------------------------------------
    strength = ext["E1"]["first_team_strength"]
    teams = [
        dict(label="Teams that took a medical paper first", value=strength["mean"][0], medical=True,
             sub=f"{strength['n'][0]} papers  ·  rate on their other papers"),
        dict(label="Teams that took another paper first", value=strength["mean"][1], medical=False,
             sub=f"{strength['n'][1]:,} papers  ·  rate on their other papers"),
    ]
    k1, score = clm["K1"], clm["score"]
    bands = [("retrain", "retraining"), ("large_scale", "large-scale run"),
             ("inference_only", "inference only"), ("analytic", "analytic"), ("unclear", "unclear")]
    comp = [
        dict(label="Medical claims", sub=f"{clm['papers']['medical']} papers",
             counts=k1["check_cost_distribution"]["medical"]),
        dict(label="Other claims", sub=f"{clm['papers']['other']} papers",
             counts=k1["check_cost_distribution"]["other"]),
    ]

    # --- 5. stated versus obtained ---------------------------------------
    a94 = p9["A9_4"]
    kappas = [
        dict(label="Code link  →  authors' code actually run",
             kappa=a94["code_vs_code_used"]["kappa"], sub=f"{a94['code_vs_code_used']['n']} adjudicated pairs"),
        dict(label="Stated data access  →  data actually obtained",
             kappa=a94["main_data_access_vs_data_obtained"]["kappa"],
             sub=f"{a94['main_data_access_vs_data_obtained']['n']} adjudicated pairs"),
        dict(label="Weights released  →  weights actually used",
             kappa=a94["weights_release_vs_weights_used"]["kappa"],
             sub=f"{a94['weights_release_vs_weights_used']['n']} adjudicated pairs"),
        dict(label="NeurIPS checklist item 5  →  code released",
             kappa=a94["neurips_checklist_item5_vs_code"]["kappa"],
             sub=f"{a94['neurips_checklist_item5_vs_code']['n']} papers"),
    ]

    # --- provenance -------------------------------------------------------
    prov = [
        ("Code released, +pts", pts(code_eff["rd"]), "p9_adversarial.json", "F1.code.coder1_only"),
        ("Main data open, +pts", pts(labels["L9.1"]["rd"]), "p9.json", "labels.L9.1"),
        ("Weights released, +pts", pts(labels["L9.2"]["rd"]), "p9.json", "labels.L9.2"),
        ("Variance reported, +pts", pts(labels["L9.3"]["rd"]), "p9.json", "labels.L9.3"),
        ("Attempted papers", f"{units['medical']} / {units['other']:,}", "p9_extended.json", "units"),
        ("Matched set, decided", f"{pct(a95['cells']['life_science']['decided'])} / "
         f"{pct(a95['cells']['other']['decided'])}", "p9.json", "A9_5.cells"),
        ("Matched-set gap", pts(a95["unadjusted"]["rd"]), "p9.json", "A9_5.unadjusted"),
        ("Decided, medical", pct(med_decided), "p9.json", "A9_5.cells.life_science.decided"),
        ("Decided, other", pct(oth_decided), "readiness.json", "R_A3.protocol7_label.cells"),
        ("Settlement gap", pts(decided_gap["rd"]), "p9_extended.json", "E1.decided.unadjusted"),
        ("Permutation p", f"{perm_p}", "p9_adversarial.json", "F2.permutation.p"),
        ("Gap after team strength", pts(k3["strength"]["rd"]), "p9_claims.json", "K3.strength"),
        ("Gap after both", pts(k3["strength_and_cost"]["rd"]), "p9_claims.json", "K3.strength_and_cost"),
        ("First-team settle rate", f"{pct(strength['mean'][0])} / {pct(strength['mean'][1])}",
         "p9_extended.json", "E1.first_team_strength.mean"),
        ("Costly claims", f"{pct(k1['costly']['medical'])} / {pct(k1['costly']['other'])}",
         "p9_claims.json", "K1.costly"),
        ("Claims double-coded", f"{score['items']:,}", "p9_claims.json", "score.items"),
        ("Code link vs code run, kappa", f"{a94['code_vs_code_used']['kappa']:.2f}", "p9.json",
         "A9_4.code_vs_code_used.kappa"),
        ("Retired keyword layer, kappa", f"{check['medical_set']['kappa']:.2f}", "map_label_check.json",
         "medical_set.kappa"),
    ]
    rows = "".join(f"<tr><td>{esc(a)}</td><td><code>{esc(b)}</code></td>"
                   f"<td><code>{esc(c)}</code></td><td><code>{esc(d)}</code></td></tr>" for a, b, c, d in prov)

    legend_cost = "".join(
        f'<span><i class="lg-{i}"></i>{esc(name)}</span>' for i, (_, name) in enumerate(bands))
    swatches = "".join(f".lg-{i}{{background:hsl(var(--shade-{min(i, 3)}))}}" for i in range(4)) + \
               ".lg-4{background:none}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Well documented, rarely picked</title>
<meta name="description" content="Protocol 9 / PSE9 result: medical AI papers are the better-documented
 group on paper and the less-settled group in practice, and what accounts for the difference."/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Lora:wght@600;700&family=Space+Mono:wght@400;700&display=swap"/>
<style>{CSS}{swatches}</style>
</head>
<body>
<main>

<p class="eyebrow">Protocol 9 · PSE9 Copenhagen 2027 · ICML 2026 agent reproduction challenge</p>
<h1>Well documented,<br/>rarely picked</h1>
<p class="lead">Medical machine-learning papers are assumed to be hard to reproduce because their data are
restricted and their code is not shared. Across nine venue-years they were the <b>better-documented</b> group on
every axis measured. In the challenge they were <b>settled less often</b> anyway. Crowd reproduction does not fail
medical papers for lack of materials: it sends them to fewer and less decisive teams, and asks costlier questions.</p>

<dl class="scope">
  <div><dt>Frame</dt><dd>CVPR, ICCV, ECCV, ICLR, ICML, NeurIPS · 2025–2026</dd></div>
  <div><dt>Labelling</dt><dd>Two blind coders, clinical / life-science, plus an equal random sample of others</dd></div>
  <div><dt>Full texts coded</dt><dd>{n_coded:,} — {coded['life_science']:,} life-science, {coded['other']:,} other</dd></div>
  <div><dt>Reproduction evidence</dt><dd>Public verdict snapshot of the ICML 2026 challenge</dd></div>
</dl>

<section>
<p class="step">Scope</p>
<h2>What was read</h2>
<p class="note">Every block is one venue; its area is the number of full texts coded there. Both coders worked
blind on the clinical / life-science label and on an equal random sample of other papers from the same
venue-year.</p>
{venue_blocks(venues)}
</section>

<div class="kpis">
  <div class="kpi"><span class="k">{n_coded:,}</span><span class="t">full texts coded by blind coders (coder 2 on a random 25%)</span></div>
  <div class="kpi"><span class="k">{pts(code_eff['rd'])}</span><span class="t">points more code released, life-science minus other</span></div>
  <div class="kpi"><span class="k">{pts(decided_gap['rd'])}</span><span class="t">points less often settled once attempted</span></div>
  <div class="kpi"><span class="k">{pts(k3['strength_and_cost']['rd'])}</span><span class="t">points of that gap left after two factors</span></div>
</div>

<section>
<p class="step">Step 1 — on paper</p>
<h2>Every documentation axis favours the medical papers<span class="chip chip-ok">prediction reversed</span></h2>
<p class="note">Rate differences, life-science minus other, Mantel–Haenszel over the nine venue-years, with 95%
confidence intervals. The protocol pre-registered the opposite direction for all three labelled fields
(L9.1–L9.3: life-science papers report open data, release weights and report variance <i>less</i> often).
None of those labels was assigned, because each difference came out positive.</p>
{diff_rows(doc, 0, 0.40, 0.10)}
<p class="note" style="margin:26px 0 0">The same four axes as levels rather than differences: the share of
coded papers on each side that met the field, and the count behind it, from the final codes the analysis uses
(adjudicated for variance). The difference printed on the right of each axis is the crude one: it is close to,
but not the same as, the venue-year–stratified estimate above.</p>
{paired_bars(axes)}
<p class="read">Read it as: nothing here supports the premise that medical papers withhold their materials.
Variance did not meet the agreement gate, so coder 2 coded it for every paper and all {p9['adjudicated']:,}
disagreements on variance and compute were adjudicated before these numbers were computed.</p>
</section>

<section>
<p class="step">Step 2 — in practice</p>
<h2>The same papers are settled less often<span class="chip chip-bad">gap −19 pts</span></h2>
<p class="note">Attempted papers in the ICML 2026 challenge, by whether any team reached a verified or
falsified verdict. Permutation p = {perm_p}.</p>
{group_rows(settle, "Decided share by group",
            f"gap {pts(decided_gap['rd'])} points, 95% CI "
            f"[{pts(decided_gap['ci'][0])}, {pts(decided_gap['ci'][1])}]")}
<p class="note" style="margin:26px 0 0">The same question on the matched set, one square per attempted
paper: the {a95['icml2026_coded']} coded ICML 2026 papers only, so both groups come from one venue and one
leaderboard snapshot. Settled squares are grouped first, so the comparison is the length of the two blocks.
Gap {pts(a95['unadjusted']['rd'])} points, 95% CI [{pts(a95['unadjusted']['ci'][0])},
{pts(a95['unadjusted']['ci'][1])}] — the bars above compare the same {units['medical']} medical papers against
every other attempted paper in the challenge, which is why the two gaps differ.</p>
{waffle(matched)}
<div class="legend"><span><i class="lg-yes"></i>verified or falsified</span><span><i class="lg-no"></i>no
verdict reached</span></div>
<p class="read">Read it as: being attempted is not the problem and the judge is not the problem — within the
same team, medical and other claims are decided about equally often. Something about <i>which</i> teams and
<i>which</i> claims is.</p>
</section>

<section>
<p class="step">Step 3 — where the gap goes</p>
<h2>Two factors absorb it<span class="chip chip-ok">CI reaches zero</span></h2>
<p class="note">The full-verification gap, then the same gap adjusted for the first team's settle rate on its
other papers, then for that and the claims' check cost. Post-hoc; each reading rule frozen before computing.</p>
{waterfall(ladder, -0.30, 0.05)}
<p class="read">Read it as: once you know who picked the paper up and what its claims cost to check, the
medical penalty is no longer distinguishable from zero.</p>
</section>

<section>
<p class="step">Step 4 — the two factors</p>
<h2>Weaker first teams, costlier claims</h2>
<p class="note">First: how often the team that took a paper first settled its <i>other</i> papers. Then: how
the papers' claims distribute over check cost, from two blind coders on {score['items']:,} claims
(kappa {score['agreement']['check_cost']['kappa']:.2f}).</p>
{group_rows(teams, "First team's settle rate on its other papers",
            f"difference {pts(strength['diff'])} points, 95% CI "
            f"[{pts(strength['ci'][0])}, {pts(strength['ci'][1])}]")}
{composition(comp, bands, 2)}
<div class="legend">{legend_cost}</div>
<p class="read">Read it as: medical papers went first to teams that settled 40% of what else they touched,
against 60% elsewhere, and {pct(k1['costly']['medical'])} of their claims needed retraining or a large-scale
run, against {pct(k1['costly']['other'])} ({pts(k1['costly']['diff'])} points, 95% CI
[{pts(k1['costly']['ci'][0])}, {pts(k1['costly']['ci'][1])}]).</p>
</section>

<section>
<p class="step">Step 5 — the caveat on any artifact map</p>
<h2>What a paper states barely predicts what a reproducer gets<span class="chip chip-warn">kappa 0.13–0.55</span></h2>
<p class="note">Agreement between the coded artifact statement and what the reproduction logbook reports
having actually used, on adjudicated pairs.</p>
{kappa_rows(kappas)}
<p class="read">Read it as: a declared code link is close to uninformative about whether anyone ran that code.
Any figure that paints a stated artifact signal as reproduction evidence is painting the wrong variable — which
is why the keyword-selected layer this page used to show was retired.</p>
</section>

<section>
<p class="step">Provenance</p>
<h2>Every number on this page, and the file it comes from</h2>
<p class="note">Paths are relative to <code>verdict_audit/readiness/results/</code>, except
<code>map_label_check.json</code> (<code>audit/</code>). Regenerate with
<code>python tools/build_medical_map.py</code>.</p>
<table>
<thead><tr><th>Quantity</th><th>Value</th><th>File</th><th>Key</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</section>

<footer>
<p><b>What changed.</b> Until this build the page showed a keyword-selected set of
{check['map_papers']} “medical” ICML 2026 papers with a July leaderboard snapshot. Checked against the
two-blind-coder label for the same venue ({check['protocol7_life_science']} life-science papers), that set
agreed at kappa {check['medical_set']['kappa']:.2f}: {check['medical_set']['map_only']} of its papers are not
life-science and {check['medical_set']['reference_only']} life-science papers were missing, and its artifact
signals agreed with the coded reference at kappa
{check['fields_on_shared_papers']['code']['kappa']:.2f} (code) and
{check['fields_on_shared_papers']['full_text_subset']['restricted_vs_full_text']['kappa']:.2f}
(restricted data). The check is in <code>audit/map_label_check.json</code>
(<code>tools/check_map_against_labels.py</code>). This page is built only from the Protocol 9 frame.</p>
<p>Analyses after A9.5 are post hoc, each with its reading rule frozen before computing.
Source: <a href="https://github.com/kondratevakate/icml-2026-repro">kondratevakate/icml-2026-repro</a>.</p>
</footer>

</main>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
