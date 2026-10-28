#!/usr/bin/env python3
"""resubmit_jobs.py - kill and resubmit the jobs listed by list_jobs.py.

    python scripts/list_jobs.py --min-hours 1 > slow_jobs.txt
    # edit slow_jobs.txt: delete any job you want to leave alone
    python scripts/resubmit_jobs.py slow_jobs.txt --dry-run   # show what it would do
    python scripts/resubmit_jobs.py slow_jobs.txt

For each listed job: ``condor_rm -name <schedd> <job id>``, then resubmit that
one job from its original ``submit.sub``, with the queue list replaced by just
its name. Resubmission runs from the job's own submit directory (its Iwd),
because submit.sub's paths - the per-job .sh, the wheel, the models, the
logs - are relative to where it was first submitted; so this script works
from any directory.

Nothing is killed unless it can be resubmitted: every job's submit.sub is
checked before the first ``condor_rm``, and a job whose file is missing is
skipped and reported. The resubmitted job keeps its name, so it writes the same
output file, and the job's ``xrdcp -f`` overwrites cleanly. A killed job
never reached its xrdcp, so nothing partial is on EOS.

Reads the file named on the command line, or stdin with '-'.
"""

import argparse
import os
import subprocess
import sys


def parse(stream):
    """[(schedd, job_id, name, iwd, submit_file)] from list_jobs.py output."""
    jobs = []
    for n, line in enumerate(stream, 1):
        line = line.split("\t#")[0].rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            sys.exit(f"line {n}: expected 8 tab-separated columns from list_jobs.py, "
                     f"got {len(parts)}: {line!r}")
        schedd, job_id, _status, _rt, _rth, name, iwd, submit = parts[:8]
        jobs.append((schedd, job_id, name, iwd, submit))
    return jobs


def single_job_submit(submit_file, name):
    """submit.sub with its queue statement replaced by this one job."""
    lines = []
    with open(submit_file) as f:
        for line in f:
            if line.lstrip().lower().startswith("queue"):
                break
            lines.append(line)
    return "".join(lines) + f"\nname = {name}\nqueue\n"


def run(cmd, dry, **kw):
    print("  $ " + " ".join(cmd) + (f"   (in {kw['cwd']})" if "cwd" in kw else ""))
    if dry:
        return True
    res = subprocess.run(cmd, text=True, capture_output=True, **kw)
    out = (res.stdout + res.stderr).strip()
    if out:
        print("    " + out.replace("\n", "\n    "))
    return res.returncode == 0


def main():
    p = argparse.ArgumentParser(description="Kill and resubmit the jobs listed by "
                                "list_jobs.py.")
    p.add_argument("jobfile", help="output of list_jobs.py ('-' for stdin)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    args = p.parse_args()

    stream = sys.stdin if args.jobfile == "-" else open(args.jobfile)
    jobs = parse(stream)
    if not jobs:
        sys.exit("No jobs listed.")

    # Check everything before killing anything.
    ready, skipped = [], []
    for job in jobs:
        (ready if os.path.isfile(job[4]) else skipped).append(job)
    for schedd, job_id, name, _iwd, submit in skipped:
        print(f"SKIP {job_id} ({name}): {submit} not found - left running", file=sys.stderr)

    n_ok = 0
    for schedd, job_id, name, iwd, submit in ready:
        print(f"[{job_id} @ {schedd}] {name}")
        if not run(["condor_rm", "-name", schedd, job_id], args.dry_run):
            print("  condor_rm failed - not resubmitting", file=sys.stderr)
            continue
        text = single_job_submit(submit, name)
        if args.dry_run:
            print(f"  $ condor_submit -   (in {iwd}; queue replaced by name = {name})")
            n_ok += 1
            continue
        res = subprocess.run(["condor_submit", "-"], input=text, text=True,
                             capture_output=True, cwd=iwd)
        out = (res.stdout + res.stderr).strip()
        print("    " + out.replace("\n", "\n    "))
        if res.returncode == 0:
            n_ok += 1
        else:
            print(f"  RESUBMIT FAILED for {name} - it is killed but not requeued",
                  file=sys.stderr)

    verb = "would be" if args.dry_run else "were"
    print(f"\n{n_ok} of {len(jobs)} job(s) {verb} killed and resubmitted"
          + (f"; {len(skipped)} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
