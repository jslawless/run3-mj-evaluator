#!/usr/bin/env python3
"""split_fileset.py - Split every file in a coffea-style fileset into chunks on EOS.

Reads the same fileset JSON format submit_evaluator.py consumes, splits each
ROOT file's events tree into chunks of at most --events-per-chunk entries,
uploads the chunks to a different EOS folder with xrdcp, and writes a new
fileset JSON listing the chunk files (full root:// URLs) so it can be fed
straight back into submit_evaluator.py.

    python split_fileset.py \\
        -i fileset.json \\
        -o /store/user/you/stitched_chunks \\
        --events-per-chunk 100000 \\
        --out-json fileset_chunks.json

Non-tree objects (cutflow, version, meta, ...) are copied into chunk 0 ONLY,
so hadd-ing the chunks (or the evaluated outputs) reproduces the original
counts instead of multiplying them by the number of chunks. Files with no
events tree at all (the slimmer's cutflow-only outputs) are copied through
unsplit.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import uproot


def xrdcp(src, dest):
    cmd = ["xrdcp", "-f", src, dest]
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"xrdcp failed (exit {result.returncode}): {src} -> {dest}")


def copy_aux_objects(in_file, out_file, tree_name):
    """Copy non-tree objects (cutflow, version, meta, ...) between files.

    Anything uproot cannot round-trip (exotic classes) is skipped with a
    warning; the evaluator treats cutflow as optional so a skip is not fatal.
    """
    for key in in_file.keys(cycle=False):
        if key == tree_name:
            continue
        obj = in_file[key]
        try:
            if hasattr(obj, "to_boost"):  # histograms (TH1 etc.)
                out_file[key] = obj.to_boost()
            elif hasattr(obj, "num_entries"):  # aux tree (e.g. the slimmer's 'meta')
                arrays = obj.arrays()
                out_file[key] = {f: arrays[f] for f in arrays.fields}
            else:
                out_file[key] = obj
        except Exception as e:
            print(f"  WARNING: could not copy '{key}' ({type(obj).__name__}): {e}")


def split_file(filepath, tree_name, eos_dir, redirector, events_per_chunk, workdir):
    """Split one ROOT file into chunks on EOS; return [(url, tree_name), ...]."""
    basename = os.path.basename(filepath)
    stem = basename[:-len(".root")] if basename.endswith(".root") else basename
    dest_dir = f"{redirector}{eos_dir}"

    with uproot.open(filepath) as in_file:
        if tree_name not in in_file:
            # Cutflow-only file (slimmer slice where nothing passed): copy through.
            print(f"  no '{tree_name}' tree - copying through unsplit")
            dest = f"{dest_dir}/{basename}"
            xrdcp(filepath, dest)
            return [(dest, tree_name)]

        tree = in_file[tree_name]
        n = tree.num_entries
        nchunks = max(1, math.ceil(n / events_per_chunk))
        print(f"  {n} entries -> {nchunks} chunk(s) of <= {events_per_chunk}")

        out = []
        for i in range(nchunks):
            start = i * events_per_chunk
            stop = min(n, start + events_per_chunk)
            chunk_name = f"{stem}_chunk{i}.root"
            local = os.path.join(workdir, chunk_name)

            arrays = tree.arrays(entry_start=start, entry_stop=stop)
            with uproot.recreate(local) as out_file:
                out_file[tree_name] = {f: arrays[f] for f in arrays.fields}
                if i == 0:
                    copy_aux_objects(in_file, out_file, tree_name)

            dest = f"{dest_dir}/{chunk_name}"
            xrdcp(local, dest)
            os.remove(local)
            out.append((dest, tree_name))

        return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--inFile", required=True,
                        help="Coffea-style fileset JSON (same format as submit_evaluator.py)")
    parser.add_argument("-o", "--eosoutdir", required=True,
                        help="EOS output dir for the chunks, bare /store/... path")
    parser.add_argument("--out-json", required=True,
                        help="Path for the new fileset JSON listing the chunk files")
    parser.add_argument("--events-per-chunk", type=int, default=100_000,
                        help="Max events per chunk")
    parser.add_argument("--tree", default="events",
                        help="Fallback tree name when the fileset does not name one")
    parser.add_argument("--redirector", default="root://cmseos.fnal.gov/",
                        help="Redirector prepended to the bare EOS output path")
    args = parser.parse_args()

    # Accept a full root://... output dir too; strip so the prefix never doubles.
    args.eosoutdir = re.sub(r"^root://[^/]+/+", "/", args.eosoutdir)

    try:
        with open(args.inFile) as f:
            fileset = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Fileset not found: {args.inFile}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {args.inFile}: {e}")

    new_fileset = {}
    with tempfile.TemporaryDirectory(prefix="split_fileset_") as workdir:
        for dataset, data in fileset.items():
            print(f"\n{dataset}:")
            new_files = {}
            for filepath, tree in data["files"].items():
                tree_name = tree if tree else args.tree
                print(f" {filepath}")
                for url, t in split_file(
                    filepath, tree_name, args.eosoutdir, args.redirector,
                    args.events_per_chunk, workdir,
                ):
                    new_files[url] = t
            new_fileset[dataset] = {"files": new_files}

    with open(args.out_json, "w") as f:
        json.dump(new_fileset, f, indent=4)

    nfiles = sum(len(d["files"]) for d in new_fileset.values())
    print(f"\nWrote {args.out_json}: {len(new_fileset)} dataset(s), {nfiles} chunk file(s)")


if __name__ == "__main__":
    main()
