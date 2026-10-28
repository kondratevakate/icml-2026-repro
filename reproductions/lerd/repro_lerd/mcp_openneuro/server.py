"""MCP server exposing OpenNeuro public datasets over anonymous S3.

Tools:
  * openneuro_list      inspect a dataset (object count, total size) without downloading
  * openneuro_download  fetch one object or a whole prefix to local disk

Run (stdio transport):
    python -m mcp_openneuro.server
or point an MCP client at this module.

Reference for the design (see HANDOFF): NIH Imaging Data Commons ships an MCP
around the ``idc-index`` package; the same shape applies here, with an
OpenNeuro-specific S3 transport instead of IDC's index.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from . import s3_openneuro as s3

mcp = FastMCP("openneuro")

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Download-size policy (Kate's open question #2).
#
# Anonymous S3 has no server-side cost ceiling, so the client must decide when a
# download is large enough to require an explicit go-ahead. Default: refuse any
# transfer above 2 GB unless the caller passes force=True. ds004504 derivatives
# are roughly 2-3 GB, so the default forces a conscious choice for the full set
# while letting single subjects (25-55 MB each) through silently.
# ---------------------------------------------------------------------------
CONFIRM_OVER_BYTES = 2 * GB


def requires_confirmation(total_bytes: int, force: bool) -> bool:
    """Return True if the transfer must be blocked pending explicit confirmation."""
    return total_bytes > CONFIRM_OVER_BYTES and not force


@mcp.tool()
def openneuro_list(dataset_id: str, subprefix: str = "") -> dict:
    """List objects under an OpenNeuro dataset prefix.

    Args:
        dataset_id: e.g. "ds004504".
        subprefix: optional path inside the dataset, e.g. "derivatives".
    Returns object count, total bytes, and up to 50 sample keys.
    """
    objs = s3.list_objects(dataset_id, subprefix)
    total = s3.total_size(objs)
    return {
        "dataset_id": dataset_id,
        "subprefix": subprefix,
        "count": len(objs),
        "total_bytes": total,
        "total_gb": round(total / GB, 3),
        "sample_keys": [o.key for o in objs[:50]],
    }


@mcp.tool()
def openneuro_download(
    dataset_id: str,
    dest_dir: str,
    key: str = "",
    subprefix: str = "",
    force: bool = False,
) -> dict:
    """Download one object (``key``) or an entire prefix (``subprefix``).

    Files are written under ``dest_dir`` preserving the S3 key layout. Transfers
    above the size threshold are refused unless ``force=True`` (see
    requires_confirmation).
    """
    if key:
        objs = [o for o in s3.list_objects(dataset_id) if o.key == key]
        if not objs:
            return {"error": f"key not found: {key}"}
    else:
        objs = s3.list_objects(dataset_id, subprefix)
        if not objs:
            return {"error": f"no objects under {dataset_id}/{subprefix}"}

    total = s3.total_size(objs)
    if requires_confirmation(total, force):
        return {
            "status": "confirmation_required",
            "count": len(objs),
            "total_gb": round(total / GB, 3),
            "hint": "re-call with force=True to proceed",
        }

    written = 0
    for o in objs:
        dest = os.path.join(dest_dir, o.key)
        written += s3.download_object(o.key, dest)
    return {
        "status": "downloaded",
        "count": len(objs),
        "bytes": written,
        "dest_dir": dest_dir,
    }


if __name__ == "__main__":
    mcp.run()
