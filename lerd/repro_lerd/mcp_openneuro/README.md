# mcp_openneuro

Minimal MCP server exposing OpenNeuro public datasets over anonymous S3.
Built as the download layer for the LERD reproduction (Cohort A = `ds004504`).

## Tools

| Tool | Purpose |
|---|---|
| `openneuro_list(dataset_id, subprefix="")` | object count + total size, no download |
| `openneuro_download(dataset_id, dest_dir, key="", subprefix="", force=False)` | fetch one object or a whole prefix |

## Transport

Anonymous (unsigned) `boto3` against `s3://openneuro.org/`. No AWS credentials,
no extra deps beyond `boto3` (already installed). `s3fs` is intentionally not
used to keep the dependency surface at zero-beyond-boto3.

## Download-size policy

`server.CONFIRM_OVER_BYTES` (default 2 GB) gates large transfers: anything above
it is refused unless the caller passes `force=True`. Single subjects (25-55 MB)
pass silently; the full `ds004504/derivatives` set (about 2.3 GB) requires an
explicit `force=True`. Adjust the constant to change the threshold.

## Test

```bash
python -m mcp_openneuro.test_s3_compat
```

Live-network compatibility check: unsigned listing, pagination past 1000, and
the size policy. Last run: 359 objects, 176 `.set` files.

## Wire into a client

`.mcp.json` entry:

```json
{
  "mcpServers": {
    "openneuro": {
      "command": "python",
      "args": ["-m", "mcp_openneuro.server"],
      "cwd": "C:/Projects/02_academia/icml-repro/lerd"
    }
  }
}
```
