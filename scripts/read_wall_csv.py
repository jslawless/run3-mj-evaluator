#!/usr/bin/env python3
"""Read the wall.csv produced by wall_print.sh and exit.

Stub: loads the rows so the data is available for downstream work,
then does nothing and exits.
"""
import csv
import sys


def main(path: str) -> None:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # TODO: do something with `rows` (columns: wall, events
    #       -- plus `job` if the filename column is enabled in wall_print.sh).
    _ = rows


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "wall.csv"
    main(path)
    sys.exit(0)
