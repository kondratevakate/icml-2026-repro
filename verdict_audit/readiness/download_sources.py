"""Download the Protocol 7 sources once into data/ and record their hashes (PROTOCOL.md, Sources)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VENUES = [("iclr", 2025), ("iclr", 2026), ("neurips", 2024), ("neurips", 2025), ("neurips", 2026), ("icml", 2025),
          ("icml", 2026), ("iccv", 2025), ("cvpr", 2024), ("cvpr", 2025), ("eccv", 2024), ("eccv", 2026)]
HOSTS = {"iclr": "iclr.cc", "neurips": "neurips.cc", "icml": "icml.cc", "iccv": "iccv.thecvf.com",
         "cvpr": "cvpr.thecvf.com", "eccv": "eccv.ecva.net"}
RECLAIM_REV = "59099a8ececf954665dff90a41ad8541e3b21291"
HEADERS = {"User-Agent": "Mozilla/5.0 (research audit of public conference metadata)"}
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
        return r.read()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    mpath = HERE / "results" / "sources_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"files": {}}
    manifest.setdefault("downloaded_at_by_file", {})
    for venue, year in VENUES:
        url = f"https://{HOSTS[venue]}/static/virtual/data/{venue}-{year}-orals-posters.json"
        out = DATA / f"{venue}-{year}.json"
        if not out.exists():  # never re-download: keeps frozen hashes stable
            out.write_bytes(fetch(url))
            manifest["downloaded_at_by_file"][out.name] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["files"][out.name] = dict(source=url, sha256=sha256(out))
    out = DATA / "reclaim_eval_100.jsonl"
    if not out.exists():
        out.write_bytes(Path(hf_hub_download("Mithilss/reclaim", "eval_100.jsonl", repo_type="dataset", revision=RECLAIM_REV)).read_bytes())
    manifest["files"][out.name] = dict(source=f"hf://datasets/Mithilss/reclaim@{RECLAIM_REV}/eval_100.jsonl",
                                       sha256=sha256(out))
    if not manifest.get("reclaim_repo_head"):
        head = subprocess.check_output(["git", "ls-remote", "https://github.com/mithils3/reclaim", "HEAD"], text=True).split()
        manifest["reclaim_repo_head"] = head[0] if head else None
    ids = [json.loads(line)["custom_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (DATA / "reclaim_arxiv_abstracts.json").exists():
        (HERE / "results" / "sources_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({k: v["sha256"][:12] for k, v in manifest["files"].items()}, indent=1))
        return
    root = ET.fromstring(fetch("https://export.arxiv.org/api/query?max_results=200&id_list=" + ",".join(ids)))
    abstracts = {}
    for e in root.findall("a:entry", ATOM):
        aid = e.find("a:id", ATOM).text.split("/abs/")[-1].split("v")[0]
        abstracts[aid] = dict(title=" ".join(e.find("a:title", ATOM).text.split()),
                              abstract=" ".join(e.find("a:summary", ATOM).text.split()))
    out = DATA / "reclaim_arxiv_abstracts.json"
    out.write_text(json.dumps(abstracts, indent=1), encoding="utf-8")
    manifest["files"][out.name] = dict(source="arXiv API id_list", sha256=sha256(out), n=len(abstracts),
                                       missing=sorted(set(ids) - set(abstracts)))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sources_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: (v.get("sha256", "")[:12], v.get("n"), v.get("missing")) for k, v in manifest["files"].items()}, indent=1),
          "repo head", manifest["reclaim_repo_head"])


if __name__ == "__main__":
    main()
