"""Build a public reproducibility audit snapshot for medical/life-science papers.

The audit intentionally separates two evidence layers:

1. Independent reproduction evidence from the ICML reproduction leaderboard.
2. Author artifact signals found in local challenge metadata/abstracts.

The output is a dated snapshot, not a final manual verdict. Use GITHUB_TOKEN for
deeper GitHub repository checks when auditing many repositories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "audit"

SPACES_URL = (
    "https://huggingface.co/api/spaces?filter=icml2026-repro"
    "&expand[]=tags&expand[]=sha&sort=createdAt&direction=-1&limit=500"
)
VERDICTS_URL = (
    "https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts"
    "/resolve/main/verdicts.json"
)

URL_RE = re.compile(r"https?://[^\s\)\]\}>\"']+")
TRAILING_PUNCT = ".,;:)]}\"'"

WEIGHT_EXTENSIONS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".pkl", ".joblib",
}
WEIGHT_WORDS = {
    "checkpoint", "checkpoints", "pretrained", "pre-trained", "weights",
    "model weights", "model-weight", "torch_save",
}
DATA_WORDS = {
    "dataset", "datasets", "data", "download", "zenodo", "figshare", "osf",
    "physionet", "openneuro", "tcia", "mimic", "adni", "brats", "isic",
    "sleep-edf", "ptb-xl", "brainweb", "fastmri",
}
PRIVATE_WORDS = {
    "private dataset", "proprietary", "in-house", "our hospital",
    "retrospective cohort", "not publicly available", "upon request",
    "restricted access", "clinical site", "institutional review board", "irb",
    "electronic medical records",
}
GPU_WORDS = {
    "gpu", "cuda", "a100", "h100", "v100", "l4", "rtx", "nvidia",
    "diffusion", "foundation model", "large language model", "llm", "3d",
    "whole-slide", "whole slide", "pretrain", "pre-training",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def request_json(url: str, headers: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "codex-medical-repro-audit",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp), dict(resp.headers)


def request_status(url: str) -> tuple[int | None, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "codex-medical-repro-audit"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.geturl()
    except urllib.error.HTTPError as exc:
        # Some hosts disallow HEAD. Retry a tiny GET before calling it dead.
        if exc.code in {403, 405}:
            try:
                req2 = urllib.request.Request(url, headers={"User-Agent": "codex-medical-repro-audit"})
                with urllib.request.urlopen(req2, timeout=20) as resp:
                    return resp.status, resp.geturl()
            except Exception:
                return exc.code, url
        return exc.code, url
    except Exception:
        return None, url


def next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in str(link_header).split(","):
        if 'rel="next"' in part or "rel=next" in part:
            start = part.find("<")
            end = part.find(">")
            if start != -1 and end != -1:
                return part[start + 1 : end]
    return None


def header_value(headers: dict[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def fetch_logbook_spaces() -> list[dict[str, Any]]:
    spaces: list[dict[str, Any]] = []
    url: str | None = SPACES_URL
    seen: set[str] = set()
    while url and url not in seen:
        seen.add(url)
        page, headers = request_json(url)
        if isinstance(page, list) and page:
            spaces.extend(page)
            url = next_link(header_value(headers, "link"))
        else:
            url = None
    return spaces


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in URL_RE.findall(text or ""):
        url = match.rstrip(TRAILING_PUNCT)
        if url not in urls:
            urls.append(url)
    return urls


def github_repo_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return ""
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return ""
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if owner.lower() in {"features", "topics", "orgs", "marketplace"}:
        return ""
    return f"{owner}/{repo}"


def hf_kind_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        return ""
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return "hf"
    if parts[0] in {"datasets", "spaces", "models"}:
        return f"hf_{parts[0]}"
    return "hf_model_or_profile"


def contains_any(text: str, words: set[str]) -> bool:
    low = (text or "").lower()
    return any(word in low for word in words)


def classify_data_access(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(k, "") or "") for k in ("title", "abstract", "tags", "private_signals", "open_signals")).lower()
    if row.get("private_signals") or contains_any(text, PRIVATE_WORDS):
        return "private_or_restricted_signal"
    if row.get("open_signals") or contains_any(text, DATA_WORDS):
        return "open_or_public_signal"
    if "simulation/eval" in text or "synthetic" in text or "simulated" in text:
        return "simulation_or_synthetic"
    if row.get("theory_signals"):
        return "theory_or_no_external_data"
    return "unknown"


def data_signal_flags(row: dict[str, Any]) -> dict[str, bool]:
    bucket = classify_data_access(row)
    return {
        "open_data_signal": bucket == "open_or_public_signal",
        "simulation_or_theory_signal": bucket in {"simulation_or_synthetic", "theory_or_no_external_data"},
        "private_restricted_signal": bucket == "private_or_restricted_signal",
    }


def claim_points(verdict: str) -> int:
    value = str(verdict or "").lower()
    if value in {"verified", "falsified"}:
        return 2
    if value == "toy":
        return 1
    return 0


def score_claims(claims: list[dict[str, Any]] | None) -> Counter:
    counts: Counter = Counter()
    for claim in claims or []:
        verdict = str(claim.get("verdict", "")).lower()
        if verdict in {"verified", "falsified", "toy"}:
            counts[verdict] += 1
        else:
            counts["zero"] += 1
        counts["points"] += claim_points(verdict)
    counts["claims"] = len(claims or [])
    return counts


def load_candidates() -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for path in [ROOT / "medical_neuro_papers.csv", ROOT / "medical_throughput_shortlist.csv"]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                orid = row.get("orid")
                if not orid:
                    continue
                base = candidates.get(orid, {})
                source = ";".join(x for x in [base.get("candidate_source", ""), path.name] if x)
                candidates[orid] = {**base, **row, "candidate_source": source}
    return candidates


def build_leaderboard_layer(
    candidates: dict[str, dict[str, Any]],
    spaces: list[dict[str, Any]],
    verdicts: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    logbook_rows: list[dict[str, Any]] = []
    lc_orid = {orid.lower(): orid for orid in candidates}

    for space in spaces:
        tags = space.get("tags") or []
        paper_id = ""
        for tag in tags:
            raw = str(tag)
            if raw.lower().startswith("paper-"):
                paper_id = lc_orid.get(raw[6:].lower(), raw[6:])
                break
        if paper_id not in candidates:
            continue

        space_id = space.get("id") or ""
        verdict = verdicts.get(space_id, {})
        claims = verdict.get("claims") if isinstance(verdict, dict) else None
        counts = score_claims(claims if isinstance(claims, list) else None)
        user = space_id.split("/")[0] if "/" in space_id else space_id
        row = {
            "orid": paper_id,
            "title": candidates[paper_id].get("title", ""),
            "user": user,
            "space": space_id,
            "space_url": f"https://huggingface.co/spaces/{space_id}" if space_id else "",
            "judged": bool(claims),
            "points": counts["points"],
            "claims_judged": counts["claims"],
            "claims_verified": counts["verified"],
            "claims_falsified": counts["falsified"],
            "claims_toy": counts["toy"],
            "claims_zero": counts["zero"],
            "created_at": space.get("createdAt", ""),
            "judged_at": verdict.get("judged_at", "") if isinstance(verdict, dict) else "",
        }
        by_paper[paper_id].append(row)
        logbook_rows.append(row)
    return by_paper, logbook_rows


@dataclass
class GithubAudit:
    repo: str
    status: str = ""
    stars: int | str = ""
    forks: int | str = ""
    pushed_at: str = ""
    default_branch: str = ""
    archived: bool | str = ""
    repo_size_kb: int | str = ""
    has_weights: bool | str = ""
    has_data_files: bool | str = ""
    has_requirements: bool | str = ""
    evidence: str = ""


def github_api(path: str, token: str) -> tuple[Any | None, int | None]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com{path}"
    try:
        data, _ = request_json(url, headers=headers)
        return data, 200
    except urllib.error.HTTPError as exc:
        return None, exc.code
    except Exception:
        return None, None


def audit_github_repo(repo: str, token: str, deep: bool) -> GithubAudit:
    audit = GithubAudit(repo=repo, evidence=f"https://github.com/{repo}")
    data, status = github_api(f"/repos/{repo}", token)
    if not isinstance(data, dict):
        audit.status = f"github_api_{status or 'error'}"
        return audit

    audit.status = "ok"
    audit.stars = data.get("stargazers_count", "")
    audit.forks = data.get("forks_count", "")
    audit.pushed_at = data.get("pushed_at", "")
    audit.default_branch = data.get("default_branch", "")
    audit.archived = data.get("archived", "")
    audit.repo_size_kb = data.get("size", "")
    if not deep or not audit.default_branch:
        return audit

    tree, tree_status = github_api(
        f"/repos/{repo}/git/trees/{urllib.parse.quote(str(audit.default_branch))}?recursive=1",
        token,
    )
    if not isinstance(tree, dict) or not isinstance(tree.get("tree"), list):
        audit.status = f"repo_ok_tree_{tree_status or 'error'}"
        return audit

    paths = [str(item.get("path", "")).lower() for item in tree["tree"]]
    audit.has_weights = any(Path(path).suffix in WEIGHT_EXTENSIONS or contains_any(path, WEIGHT_WORDS) for path in paths)
    audit.has_data_files = any(
        path.startswith("data/")
        or "/data/" in path
        or Path(path).suffix in {".csv", ".tsv", ".parquet", ".h5", ".hdf5", ".npz"}
        for path in paths
    )
    audit.has_requirements = any(
        Path(path).name in {"requirements.txt", "environment.yml", "pyproject.toml", "dockerfile", "docker-compose.yml"}
        for path in paths
    )
    return audit


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(n: int, d: int) -> float:
    return round((100.0 * n / d), 1) if d else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--live-link-checks", action="store_true", help="Run live HTTP status checks for extracted artifact links.")
    parser.add_argument("--github-metadata", action="store_true", help="Query GitHub repository metadata through the GitHub API.")
    parser.add_argument("--github-limit", type=int, default=0, help="Maximum repositories to query for metadata; 0 means all when authenticated, 50 otherwise.")
    parser.add_argument("--deep-github", action="store_true", help="Inspect GitHub repository trees for weights/data/requirements.")
    parser.add_argument("--deep-limit", type=int, default=50, help="Maximum repos to inspect deeply without a GitHub token.")
    parser.add_argument("--no-live-link-checks", action="store_true", help="Skip HTTP status checks for artifact links.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checked_at = now_iso()
    candidates = load_candidates()
    if not candidates:
        raise SystemExit("No medical candidate CSV files found.")

    print(f"Loaded {len(candidates)} medical/life-science candidates", file=sys.stderr)
    spaces = fetch_logbook_spaces()
    verdicts, _ = request_json(VERDICTS_URL)
    if not isinstance(verdicts, dict):
        verdicts = {}
    print(f"Fetched {len(spaces)} HF Spaces and {len(verdicts)} verdict entries", file=sys.stderr)

    by_paper, logbook_rows = build_leaderboard_layer(candidates, spaces, verdicts)

    link_rows: list[dict[str, Any]] = []
    repo_order: list[str] = []
    for orid, row in candidates.items():
        text = " ".join(str(v or "") for v in row.values())
        for url in extract_urls(text):
            if "openreview.net/forum" in url:
                continue
            repo = github_repo_from_url(url)
            hf_kind = hf_kind_from_url(url)
            kind = "github_repo" if repo else hf_kind or "external_url"
            status: int | None = None
            final_url = url
            if args.live_link_checks and not args.no_live_link_checks:
                status, final_url = request_status(url)
                time.sleep(0.03)
            link_rows.append({
                "checked_at": checked_at,
                "orid": orid,
                "title": row.get("title", ""),
                "kind": kind,
                "url": url,
                "final_url": final_url,
                "http_status": status if status is not None else "",
                "github_repo": repo,
                "hf_kind": hf_kind,
            })
            if repo and repo not in repo_order:
                repo_order.append(repo)

    token = os.environ.get("GITHUB_TOKEN", "")
    deep = args.deep_github
    github_limit = args.github_limit or (len(repo_order) if token else 50)
    deep_limit = len(repo_order) if token else max(0, args.deep_limit)
    github_audits: dict[str, GithubAudit] = {}
    if args.github_metadata or args.deep_github:
        for index, repo in enumerate(repo_order[:github_limit]):
            repo_deep = deep and index < deep_limit
            github_audits[repo] = audit_github_repo(repo, token, repo_deep)
            time.sleep(0.1)
        for repo in repo_order[github_limit:]:
            github_audits[repo] = GithubAudit(repo=repo, status="not_checked", evidence=f"https://github.com/{repo}")
    else:
        for repo in repo_order:
            github_audits[repo] = GithubAudit(repo=repo, status="not_checked", evidence=f"https://github.com/{repo}")

    rows: list[dict[str, Any]] = []
    for orid, row in sorted(candidates.items(), key=lambda item: (item[1].get("area", ""), item[1].get("title", ""))):
        logs = by_paper.get(orid, [])
        positive = [item for item in logs if int(item["points"]) > 0]
        best = max([int(item["points"]) for item in logs], default=0)
        verified = sum(int(item["claims_verified"]) for item in logs)
        falsified = sum(int(item["claims_falsified"]) for item in logs)
        toy = sum(int(item["claims_toy"]) for item in logs)
        paper_links = [item for item in link_rows if item["orid"] == orid]
        repos = sorted({item["github_repo"] for item in paper_links if item["github_repo"]})
        repo_audits = [github_audits[repo] for repo in repos if repo in github_audits]
        text = " ".join(str(row.get(k, "") or "") for k in row)
        has_weight_signal = contains_any(text, WEIGHT_WORDS) or any(a.has_weights is True for a in repo_audits)
        data_flags = data_signal_flags(row)
        has_dataset_signal = data_flags["open_data_signal"] or any(a.has_data_files is True for a in repo_audits)
        gpu_signal = bool(row.get("gpu_risks")) or contains_any(text, GPU_WORDS) or "gpu-risk" in str(row.get("tags", ""))
        rows.append({
            "checked_at": checked_at,
            "orid": orid,
            "title": row.get("title", ""),
            "area": row.get("area", ""),
            "subarea": row.get("sub", row.get("subarea", "")),
            "n_claims": row.get("n_claims", ""),
            "max_points": row.get("max_pts", row.get("max_points", "")),
            "data_access_signal": classify_data_access(row),
            "gpu_required_signal": gpu_signal,
            "author_url_count": len(paper_links),
            "github_repo_count": len(repos),
            "github_repos": ";".join(repos),
            "hf_url_count": sum(1 for item in paper_links if item["hf_kind"]),
            "repo_live_ok": any(a.status.startswith("ok") or a.status.startswith("repo_ok") for a in repo_audits),
            "weights_signal": has_weight_signal,
            "open_dataset_signal": has_dataset_signal,
            "simulation_or_theory_data_signal": data_flags["simulation_or_theory_signal"],
            "private_restricted_data_signal": data_flags["private_restricted_signal"],
            "requirements_signal": any(a.has_requirements is True for a in repo_audits),
            "leaderboard_logbooks": len(logs),
            "leaderboard_judged_logbooks": sum(1 for item in logs if item["judged"]),
            "leaderboard_positive_logbooks": len(positive),
            "leaderboard_best_points": best,
            "leaderboard_verified_claim_events": verified,
            "leaderboard_falsified_claim_events": falsified,
            "leaderboard_toy_claim_events": toy,
            "leaderboard_top_users": ";".join(item["user"] for item in sorted(positive, key=lambda x: (-int(x["points"]), x["user"]))[:5]),
            "openreview": row.get("openreview", row.get("or", "")),
            "arxiv": row.get("arxiv", ""),
        })

    fields = list(rows[0])
    write_csv(out_dir / "public_repro_audit.csv", rows, fields)
    write_csv(out_dir / "leaderboard_medical_logbooks.csv", logbook_rows)
    write_csv(out_dir / "artifact_links.csv", link_rows)
    write_csv(out_dir / "github_repo_audit.csv", [audit.__dict__ for audit in github_audits.values()])

    total = len(rows)
    summary = {
        "checked_at": checked_at,
        "candidate_count": total,
        "leaderboard_source": "HF Spaces tagged icml2026-repro plus public verdicts.json",
        "artifact_source": "local challenge metadata and abstracts; live URL checks where possible",
        "github_repos_extracted": len(repo_order),
        "github_repos_metadata_checked": sum(1 for audit in github_audits.values() if audit.status != "not_checked"),
        "github_repos_metadata_unchecked": sum(1 for audit in github_audits.values() if audit.status == "not_checked"),
        "github_repos_live_ok": sum(1 for audit in github_audits.values() if audit.status.startswith("ok") or audit.status.startswith("repo_ok")),
        "with_author_github_repo": sum(1 for row in rows if int(row["github_repo_count"]) > 0),
        "with_live_github_repo_ok": sum(1 for row in rows if row["repo_live_ok"]),
        "with_hf_url": sum(1 for row in rows if int(row["hf_url_count"]) > 0),
        "with_weights_signal": sum(1 for row in rows if row["weights_signal"]),
        "with_open_dataset_signal": sum(1 for row in rows if row["open_dataset_signal"]),
        "with_simulation_or_theory_data_signal": sum(1 for row in rows if row["simulation_or_theory_data_signal"]),
        "with_private_restricted_data_signal": sum(1 for row in rows if row["private_restricted_data_signal"]),
        "with_gpu_required_signal": sum(1 for row in rows if row["gpu_required_signal"]),
        "with_leaderboard_attempt": sum(1 for row in rows if int(row["leaderboard_logbooks"]) > 0),
        "with_leaderboard_positive": sum(1 for row in rows if int(row["leaderboard_positive_logbooks"]) > 0),
        "with_verified_or_falsified_claim": sum(
            1 for row in rows
            if int(row["leaderboard_verified_claim_events"]) + int(row["leaderboard_falsified_claim_events"]) > 0
        ),
        "with_falsified_claim": sum(1 for row in rows if int(row["leaderboard_falsified_claim_events"]) > 0),
        "claim_events": {
            "verified": sum(int(row["leaderboard_verified_claim_events"]) for row in rows),
            "falsified": sum(int(row["leaderboard_falsified_claim_events"]) for row in rows),
            "toy": sum(int(row["leaderboard_toy_claim_events"]) for row in rows),
        },
        "percentages": {},
    }
    for key, value in list(summary.items()):
        if key.startswith("with_") and isinstance(value, int):
            summary["percentages"][key] = pct(value, total)

    with (out_dir / "public_repro_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    md = [
        "# Public Medical Reproducibility Audit",
        "",
        f"Snapshot: `{checked_at}`.",
        "",
        "This audit separates independent reproduction evidence from author-artifact signals.",
        "Leaderboard evidence comes from Hugging Face Spaces tagged `icml2026-repro` and the public verdicts dataset.",
        "Author-artifact evidence is extracted from local ICML challenge metadata and abstracts, with live URL checks where possible.",
        "",
        "## Headline Counts",
        "",
        f"- Medical/life-science candidates: {total}",
        f"- Author GitHub repository signal: {summary['with_author_github_repo']} ({summary['percentages']['with_author_github_repo']}%)",
        f"- Live GitHub repository OK: {summary['with_live_github_repo_ok']} ({summary['percentages']['with_live_github_repo_ok']}%)",
        f"- HF URL signal: {summary['with_hf_url']} ({summary['percentages']['with_hf_url']}%)",
        f"- Weights/checkpoint signal: {summary['with_weights_signal']} ({summary['percentages']['with_weights_signal']}%)",
        f"- Open/public dataset signal: {summary['with_open_dataset_signal']} ({summary['percentages']['with_open_dataset_signal']}%)",
        f"- Simulation/theory no-external-data signal: {summary['with_simulation_or_theory_data_signal']} ({summary['percentages']['with_simulation_or_theory_data_signal']}%)",
        f"- Private/restricted data signal: {summary['with_private_restricted_data_signal']} ({summary['percentages']['with_private_restricted_data_signal']}%)",
        f"- GPU-required/risk signal: {summary['with_gpu_required_signal']} ({summary['percentages']['with_gpu_required_signal']}%)",
        f"- Leaderboard attempts: {summary['with_leaderboard_attempt']} ({summary['percentages']['with_leaderboard_attempt']}%)",
        f"- Positive leaderboard evidence: {summary['with_leaderboard_positive']} ({summary['percentages']['with_leaderboard_positive']}%)",
        f"- At least one verified/falsified claim event: {summary['with_verified_or_falsified_claim']} ({summary['percentages']['with_verified_or_falsified_claim']}%)",
        f"- At least one falsified claim event: {summary['with_falsified_claim']} ({summary['percentages']['with_falsified_claim']}%)",
        "",
        "## Output Files",
        "",
        "- `public_repro_audit.csv`: one row per candidate paper.",
        "- `leaderboard_medical_logbooks.csv`: one row per relevant leaderboard Space/logbook.",
        "- `artifact_links.csv`: extracted non-OpenReview artifact URLs and live status checks.",
        "- `github_repo_audit.csv`: GitHub repository metadata; tree-level fields require `--deep-github`.",
        "- `public_repro_audit_summary.json`: machine-readable headline counts.",
        "",
        "## Caveats",
        "",
        "- `repo_signal`, `dataset_signal`, `weights_signal`, and `gpu_required_signal` are audit signals, not final manual verdicts.",
        "- Leaderboard points may come from full reproduction, falsification, or toy-scale evidence, and may not cover all claims in the paper.",
        "- GitHub live metadata is rate-limited without authentication; use `GITHUB_TOKEN` plus `--github-metadata --deep-github` for a complete repository/weights pass.",
        "- Dataset size and GPU-hours require a second pass over repositories, READMEs, paper PDFs, and/or published logbook evidence.",
    ]
    (out_dir / "PUBLIC_MEDICAL_REPRO_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
