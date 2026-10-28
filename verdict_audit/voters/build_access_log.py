"""Build ACCESS_LOG.csv from a Claude Code session transcript (JSONL).

Records only external access and agent launches: web fetches and searches (URL or query),
literature-database calls, Hugging Face API methods used in shell commands (counted, not listed
per Space), and subagent launches with the model they ran on. No command text, file contents or
local paths are written. Usage: python build_access_log.py <session.jsonl> [<subagents dir>]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = re.compile(r"https?://[^\s'\"<>)\]]+")
HF_METHODS = ("list_repo_files", "list_repo_commits", "list_repo_tree", "get_paths_info", "snapshot_download",
              "cat_file", "list_spaces", "list_datasets", "list_models", "resolve/")
XML_NAMESPACE_HOSTS = ("www.w3.org", "arxiv.org")  # Atom namespaces in arXiv API parsing code
PUBLIC_HOSTS = ("api.crossref.org", "api.openalex.org", "export.arxiv.org", "eutils.ncbi.nlm.nih.gov",
                "api.semanticscholar.org", "dblp.org")


def entries(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tool_uses(rec):
    msg = rec.get("message") or {}
    for part in msg.get("content") or [] if isinstance(msg.get("content"), list) else []:
        if isinstance(part, dict) and part.get("type") == "tool_use":
            yield part


def main() -> None:
    transcript = Path(sys.argv[1])
    sub_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    rows, hf_calls, db_calls = [], Counter(), Counter()
    session_models = Counter()
    for rec in entries(transcript):
        ts = rec.get("timestamp", "")
        model = (rec.get("message") or {}).get("model")
        if model:
            session_models[model] += 1
        for tu in tool_uses(rec):
            name, inp = tu.get("name", ""), tu.get("input") or {}
            if name == "WebFetch":
                rows.append([ts, "web_fetch", inp.get("url", "")])
            elif name == "WebSearch":
                rows.append([ts, "web_search", inp.get("query", "")])
            elif name.endswith("consensus__search"):
                rows.append([ts, "consensus_search", inp.get("query", "")])
            elif name == "Agent":
                rows.append([ts, "subagent_launch", f"{inp.get('description', '')} (model param: {inp.get('model') or 'not set'})"])
            elif name in ("Bash", "PowerShell"):
                cmd = inp.get("command", "")
                for u in URL.findall(cmd):
                    host = u.split("/")[2]
                    if host in PUBLIC_HOSTS:
                        db_calls[host] += 1
                    elif host.endswith("huggingface.co"):
                        hf_calls["huggingface.co URL in command"] += 1
                    elif host not in XML_NAMESPACE_HOSTS:
                        rows.append([ts, "url_in_shell_command (may be text written to a file, not an access)", u])
                for m in HF_METHODS:
                    if m in cmd:
                        hf_calls[m] += 1
    sub_models = Counter()
    if sub_dir and sub_dir.exists():
        for f in sorted(sub_dir.glob("*.jsonl")):
            for rec in entries(f):
                m = (rec.get("message") or {}).get("model")
                if m:
                    sub_models[(f.stem, m)] += 1
    out = HERE / "ACCESS_LOG.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_utc", "kind", "target"])
        w.writerows(sorted(rows))
        for host, n in sorted(db_calls.items()):
            w.writerow(["", "literature_api_commands", f"{host}: {n} shell commands"])
        for m, n in sorted(hf_calls.items()):
            w.writerow(["", "huggingface_api_in_commands", f"{m}: {n} shell commands (bulk downloads are counted in the manifests)"])
        for m, n in sorted(session_models.items()):
            w.writerow(["", "session_model", f"{m}: {n} assistant messages"])
        for (stem, m), n in sorted(sub_models.items()):
            w.writerow(["", "subagent_model", f"{stem}: {m} ({n} messages)"])
    print(f"{len(rows)} rows; db {dict(db_calls)}; hf {dict(hf_calls)}; session models {dict(session_models)}; "
          f"subagent models {dict(Counter(m for (_, m) in sub_models))} in {len({s for s, _ in sub_models})} transcripts")


if __name__ == "__main__":
    main()
