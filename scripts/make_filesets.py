#!/usr/bin/env python3
"""make_filesets.py - Build per-dataset coffea fileset JSONs from an EOS directory
of run3-mj-slimmer output, ready to feed scripts/run_all.sh.

The slimmer writes its output as <eos_base>/<dataset>/slimmed_*.root (one
subdirectory per dataset; see the slimmer's run_all.sh, which submits with
-o <base>/<dataset>). This script lists each dataset's .root files over EOS and
writes one <dataset>.json into the output folder.

run_all.sh then submits one evaluator job per *.json, using the JSON *filename*
as the dataset / log dir / output subdir name; submit_evaluator.py names the
per-job .sh/output from the *internal* dataset key. So each JSON holds exactly
one dataset whose key equals the JSON filename stem -- both stay in sync.

Fileset JSON format (coffea-style, consumed by submit_evaluator.py):
    {
        "<dataset>": {
            "files": {
                "root://cmseos.fnal.gov//store/.../slimmed_xxx.root": "events",
                ...
            }
        }
    }

This is self-contained: it does the EOS listing and the fileset JSON
construction in one step (no separate filelist .txt stage).

Usage:
    python scripts/make_filesets.py <eos_path> [-o filesets]
    python scripts/make_filesets.py /store/user/jlawless/slimmed -o filesets

Then:
    source scripts/run_all.sh filesets <evaluator-wheel> <eos-outdir>
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

# The slimmer's outputs live in personal/group EOS (/store/user/..., /store/
# group/...) and must be streamed through the EOS redirector. XCache
# (root://xcache/) only serves the global CMS namespace and yields
# "[FATAL] Invalid address" for these files.
DEFAULT_REDIRECTOR = "root://cmseos.fnal.gov/"


def eos_ls(path, ls_cmd):
    """Return the entry names under an EOS directory by running `ls_cmd <path>`.

    `ls_cmd` may be a multi-word command (e.g. 'eos root://cmseos.fnal.gov ls');
    it is tokenized and the path appended. Note the interactive `eosls` is
    usually a shell *function* wrapping `eos ... ls`, which cannot be invoked
    from a subprocess -- so we call the `eos` binary directly.
    """
    cmd = shlex.split(ls_cmd) + [path]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit(
            f"'{cmd[0]}' not found on PATH. ('eosls' is often a shell function "
            "wrapping 'eos root://cmseos.fnal.gov ls' and can't be called from a "
            "subprocess.) Run this where the EOS client is available (e.g. "
            "cmslpc), or pass --ls-cmd."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"`{' '.join(cmd)}` failed:\n{e.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def to_xrd(eos_dir, fname, redirector):
    """Prefix a /store path with the XRootD redirector for WAN streaming.

    XRootD needs a DOUBLE slash between the host and an absolute path:
    'root://host//store/...'. A single slash makes 'store/...' a relative path,
    which the server rejects ('Locating relative path ... is disallowed').
    """
    full = f"{eos_dir.rstrip('/')}/{fname}"
    if full.startswith("root://"):
        return full
    host = redirector.rstrip("/")          # e.g. root://cmseos.fnal.gov
    return f"{host}//{full.lstrip('/')}"   # -> root://cmseos.fnal.gov//store/...


def discover_datasets(eos_path, ls_cmd):
    """Map dataset name -> EOS directory.

    Standard layout: <eos_path> holds one subdirectory per dataset. Falls back
    to treating <eos_path> itself as a single dataset if it contains .root
    files directly.
    """
    entries = eos_ls(eos_path, ls_cmd)
    subdirs = [e for e in entries if not e.endswith(".root")]
    has_root = any(e.endswith(".root") for e in entries)

    if subdirs:
        return {d: f"{eos_path}/{d}" for d in subdirs}
    if has_root:
        return {os.path.basename(eos_path): eos_path}
    sys.exit(f"No dataset subdirectories or .root files found under {eos_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build per-dataset coffea fileset JSONs from a slimmer EOS "
                    "output directory, ready for scripts/run_all.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "eos_path",
        help="EOS dir of slimmer output (one subdir per dataset, or .root "
             "files directly for a single dataset).",
    )
    parser.add_argument(
        "-o", "--output-dir", default="filesets",
        help="Folder for the per-dataset JSONs (consumed by run_all.sh).",
    )
    parser.add_argument(
        "--redirector", default=DEFAULT_REDIRECTOR,
        help="XRootD redirector prefix applied to /store paths.",
    )
    parser.add_argument(
        "--tree", default="events",
        help="Tree name recorded for every file (slimmer output is 'events').",
    )
    parser.add_argument(
        "--ls-cmd", default="eos root://cmseos.fnal.gov ls",
        help="Command used to list an EOS directory. Defaults to the 'eos' "
             "binary, since the interactive 'eosls' is a shell function that "
             "cannot be invoked from a subprocess.",
    )
    args = parser.parse_args()

    eos_path = args.eos_path.rstrip("/")
    datasets = discover_datasets(eos_path, args.ls_cmd)
    os.makedirs(args.output_dir, exist_ok=True)

    n_written = 0
    for name, dsdir in sorted(datasets.items()):
        files = [f for f in eos_ls(dsdir, args.ls_cmd) if f.endswith(".root")]
        if not files:
            print(f"  skip (no .root files): {dsdir}", file=sys.stderr)
            continue
        fileset = {
            name: {
                "files": {
                    to_xrd(dsdir, f, args.redirector): args.tree for f in files
                }
            }
        }
        out = os.path.join(args.output_dir, f"{name}.json")
        with open(out, "w") as fh:
            json.dump(fileset, fh, indent=4)
        print(f"  {name}: {len(files)} files -> {out}")
        n_written += 1

    print(f"\nWrote {n_written} dataset JSON(s) to {args.output_dir}/")
    print("Feed it to run_all.sh, e.g.:")
    print(f"  source scripts/run_all.sh {args.output_dir} <evaluator-wheel> <eos-outdir>")


if __name__ == "__main__":
    main()
