#!/usr/bin/env python3
"""build_space_map.py — scan all owner Spaces, extract paper-<orid> tags from README,
and write groundtruth/space_map.json. Prevents duplicate Spaces in publish_local.py.

Usage: python3 build_space_map.py
"""
import json, os, re, argparse
from huggingface_hub import HfApi

OWNER = "kondratevakate"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "space_map.json")


def main():
    api = HfApi()
    mp = {}
    # keep existing non-paper keys (e.g. _comment)
    try:
        old = json.load(open(OUT))
        for k, v in old.items():
            if k.startswith("_"):
                mp[k] = v
    except Exception:
        mp["_comment"] = "Maps paper orid -> HF Space id. Ensures we EDIT existing spaces, never create duplicates."
    n = 0
    for s in api.list_spaces(author=OWNER):
        sid = s.id
        try:
            r = api.hf_hub_download(repo_id=sid, repo_type="space",
                                   filename="README.md", local_files_only=False)
            txt = open(r).read()
            m = re.search(r"paper-([A-Za-z0-9]{10})", txt)
            if m:
                orid = m.group(1)
                mp[orid] = sid
                n += 1
                print(f"  {orid} -> {sid}")
        except Exception:
            continue
    json.dump(mp, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}: {n} paper-tagged spaces mapped")


if __name__ == "__main__":
    main()
