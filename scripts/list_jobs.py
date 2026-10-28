#!/usr/bin/env python3
"""list_jobs.py - list your condor jobs, one per line, for resubmit_jobs.py.

Writes a tab-separated table to stdout, ready to redirect into a file, edit
(delete the lines you want to keep running), and hand to
``scripts/resubmit_jobs.py``:

    python scripts/list_jobs.py --min-hours 1 > slow_jobs.txt
    python scripts/resubmit_jobs.py slow_jobs.txt

Columns: schedd, job id, status, run time (s), run time, job name, submit
directory (the job's Iwd), submit file. The schedd comes from each job's
GlobalJobId, so jobs spread over lpcschedd4/5/6 are each killed on the right
one. The job name is the per-job ``<name>.sh`` that submit_evaluator.py writes
(executable = ``<logdir>/$(name).sh``), and the submit file is the
``submit.sub`` beside it. Lines starting with '#' are comments.

Run time is time since the job last started, and 0 for jobs that are not
running.
"""

import argparse
import getpass
import os
import subprocess
import sys

STATUS = {1: "idle", 2: "running", 3: "removed", 4: "completed", 5: "held",
          6: "transferring", 7: "suspended"}

#: condor_q -af attributes, in output order. The run-time expression is
#: evaluated by condor, so it is the run time at the moment of the query.
ATTRS = ["GlobalJobId", "JobStatus", "Iwd", "Cmd",
         "ifThenElse(JobStatus == 2, time() - JobCurrentStartDate, 0)"]

HEADER = ["schedd", "job_id", "status", "runtime_s", "runtime", "name", "iwd", "submit_file"]


def fmt_runtime(seconds):
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def query(user):
    """Rows of (schedd, job_id, status, runtime_s, name, iwd, submit_file)."""
    cmd = ["condor_q", user, "-af:t"] + ATTRS
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except FileNotFoundError:
        sys.exit("condor_q not found - run this on a condor submit node (cmslpc).")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"condor_q failed:\n{exc.stderr}")

    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        # Only the job lines: the LPC wrapper can interleave schedd banners.
        if len(parts) != len(ATTRS) or "#" not in parts[0]:
            continue
        gid, status, iwd, exe, runtime = parts
        schedd, job_id = gid.split("#")[:2]
        exe_path = exe if os.path.isabs(exe) else os.path.join(iwd, exe)
        name = os.path.basename(exe_path)
        name = name[:-3] if name.endswith(".sh") else name
        submit = os.path.join(os.path.dirname(exe_path), "submit.sub")
        try:
            runtime_s = int(float(runtime))
        except ValueError:
            runtime_s = 0
        rows.append((schedd, job_id, STATUS.get(int(status), status), runtime_s,
                     name, iwd, submit))
    return rows


def main():
    p = argparse.ArgumentParser(description="List condor jobs as a table for "
                                "resubmit_jobs.py.")
    p.add_argument("--user", default=getpass.getuser(),
                   help="job owner (default: %(default)s)")
    p.add_argument("--min-hours", type=float, default=0.0,
                   help="only jobs that have been running at least this long "
                   "(implies --status running when > 0)")
    p.add_argument("--status", default="all",
                   choices=["all", "running", "idle", "held"],
                   help="only jobs in this state (default: %(default)s)")
    p.add_argument("--match", default=None,
                   help="only jobs whose name contains this string")
    args = p.parse_args()

    rows = query(args.user)
    if args.min_hours > 0:
        rows = [r for r in rows if r[2] == "running" and r[3] >= args.min_hours * 3600]
    if args.status != "all":
        rows = [r for r in rows if r[2] == args.status]
    if args.match:
        rows = [r for r in rows if args.match in r[4]]
    rows.sort(key=lambda r: -r[3])

    print("# " + "\t".join(HEADER))
    for schedd, job_id, status, runtime_s, name, iwd, submit in rows:
        flag = "" if os.path.isfile(submit) else "\t# no submit.sub - cannot resubmit"
        print("\t".join([schedd, job_id, status, str(runtime_s), fmt_runtime(runtime_s),
                         name, iwd, submit]) + flag)
    print(f"# {len(rows)} job(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
