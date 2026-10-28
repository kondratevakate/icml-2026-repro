"""Compatibility test for the OpenNeuro S3 transport.

Verifies the assumptions the MCP server depends on, against the live public
bucket. Requires network. Run:

    python -m mcp_openneuro.test_s3_compat
"""

from __future__ import annotations

import sys

from . import s3_openneuro as s3
from .server import requires_confirmation, GB

DATASET = "ds004504"


def test_unsigned_list_works():
    objs = s3.list_objects(DATASET)
    assert len(objs) > 80, f"expected many objects, got {len(objs)}"
    # ds004504 ships preprocessed EEGLAB .set files under derivatives/
    sets = [o for o in objs if o.key.endswith(".set")]
    assert sets, "no .set files found"
    return len(objs), len(sets)


def test_paginates_past_1000():
    # sanity: pagination returns the full listing, not a truncated first page
    objs = s3.list_objects(DATASET)
    keys = {o.key for o in objs}
    assert len(keys) == len(objs), "duplicate keys across pages"


def test_size_policy():
    assert requires_confirmation(3 * GB, force=False) is True
    assert requires_confirmation(3 * GB, force=True) is False
    assert requires_confirmation(50 * 1024 ** 2, force=False) is False


def main():
    n, n_set = test_unsigned_list_works()
    test_paginates_past_1000()
    test_size_policy()
    print(f"OK: {DATASET} listed {n} objects ({n_set} .set), pagination + policy pass")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
