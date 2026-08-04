"""rewrite_logbook_format.py — normalize a v1 logbook.md so trackio navigation
exposes one page per claim, without changing any verdict or evidence.

Strategy (content-preserving):
1. Locate the verdict table (header row containing 'Claim' and 'Verdict').
   Build claim_id -> verdict map (C1 / Claim 1 / 1 -> verdict).
2. Detect claim subsections. A claim subsection is any heading line whose text
   starts with a claim marker: 'C<n>', 'Claim <n>', or '<n>.' (at h2/h3/h4).
   These become top-level '## Claim <n> — <title>' pages.
3. If the logbook already has '## Claim <n>' headings, leave it untouched
   (already in the right shape).
4. Emit normalized markdown: keep all original sections, but promote detected
   claim subsections to '## Claim <n> — <title>' and prepend the verdict line.

Run: python3 rewrite_logbook_format.py <logbook.md>  -> prints normalized md
(or --inplace to overwrite). Read-only by default.
"""
import re, sys, argparse


def extract_verdicts(md):
    """Return {claim_num: verdict_word} from the verdict table, if present."""
    verdicts = {}
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if re.search(r"\|\s*#\s*\|.*claim.*\|\s*verdict", ln, re.I):
            # following rows are data
            for row in lines[i+1:]:
                if not row.strip().startswith("|"):
                    if row.strip().startswith("|") is False and row.strip():
                        break
                    continue
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if len(cells) < 3:
                    continue
                cid, cname, cverd = cells[0], cells[1], cells[2]
                m = re.match(r"^(?:C?(\d+)|(\d+))\b", cid.strip())
                if not m:
                    continue
                num = m.group(1) or m.group(2)
                v = re.sub(r"[*_`]", "", cverd).strip().lower()
                verdicts[num] = v
            break
    return verdicts


def claim_heading_re():
    return re.compile(r"^(#{2,4})\s+(?:C?(\d+)|claim\s+(\d+)|(\d+)\.)\b\s*[-–:]?\s*(.*)$", re.I)


def rewrite(md):
    verdicts = extract_verdicts(md)
    # already normalized?
    if any(re.match(r"^##\s+claim\s+\d+", ln, re.I) for ln in md.splitlines()):
        return md, False
    out = []
    seen = set()
    for ln in md.splitlines():
        m = claim_heading_re().match(ln)
        if m:
            hashes, c1, c2, c3, rest = m.groups()
            num = c1 or c2 or c3
            if num in seen:
                out.append(ln)  # duplicate id, keep as-is
                continue
            seen.add(num)
            title = rest.strip(" -–:") or f"Claim {num}"
            # rest may already contain a leading dash/colon from the source heading
            title = re.sub(r"^[\s\-–:]+", "", title)
            verdict = verdicts.get(num)
            new_head = f"## Claim {num} — {title}"
            out.append(new_head)
            if verdict:
                out.append("")
                out.append(f"**Verdict:** {verdict}")
                out.append("")
            continue
        out.append(ln)
    return "\n".join(out), True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--inplace", action="store_true")
    a = ap.parse_args()
    src = open(a.md).read()
    new, changed = rewrite(src)
    if a.inplace:
        open(a.md, "w").write(new)
        print(f"rewrote {a.md} (changed={changed})")
    else:
        print(new)
        print(f"\n--- changed={changed} ---", file=sys.stderr)
