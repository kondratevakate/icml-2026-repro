"""Split the blind coding sheets into batches of at most 250 items (Protocol 7, Amendment 1)."""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHEETS = HERE / "data" / "sheets"
SIZE = 250


def main() -> None:
    for role in ("coder1", "coder2"):
        with (SHEETS / f"{role}_all.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for b in range(0, len(rows), SIZE):
            with (SHEETS / f"{role}_batch_{b // SIZE:02d}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["item_id", "title", "abstract"])
                w.writeheader()
                w.writerows(rows[b:b + SIZE])
        print(role, len(rows), "items,", (len(rows) + SIZE - 1) // SIZE, "batches")


if __name__ == "__main__":
    main()
