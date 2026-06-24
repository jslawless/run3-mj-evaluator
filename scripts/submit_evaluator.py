#!/usr/bin/env python3
"""submit_evaluator.py - Submit run3-mj-evaluator jobs to HTCondor.

Reads a coffea-style fileset JSON of *slimmed* ROOT files (produced by
run3-mj-slimmer), splits files into per-job groups, and writes condor
submission files. Each job installs run3-mj-evaluator from a pre-built wheel,
runs the ONNX model(s) on its assigned files, and copies the output to EOS.

Build the wheel before submitting:
    pip wheel /path/to/run3-mj-evaluator -w .

Submit:
    python submit_evaluator.py \\
        -i fileset.json \\
        -o /store/user/you/evaluated \\
        --config config.json \\
        --wheel run3_mj_evaluator-1.0.0-py3-none-any.whl

-o / --eosoutdir is a BARE EOS path (e.g. /store/user/you/evaluated); the job
script adds the root://cmseos.fnal.gov/ redirector automatically. A full
root://host//store/... URL is also accepted - the leading redirector is stripped
so it is never doubled in the xrdcp destination.

Fileset JSON format (coffea-style):
    {
        "dataset_name": {
            "files": {
                "/path/to/slimmed_file.root": "events",
                ...
            }
        }
    }

Unlike the slimmer, the evaluator config references ONNX model files. Those
are not part of the wheel, so this script reads the config, ships every model
file (and any ".onnx.data" weight sidecar) alongside the job, and writes a
rewritten config whose model paths are the bare basenames -- because condor
flattens transfer_input_files into the job's working directory.
"""

import os
import re
import argparse
import json


def configure_batch(logdir, names, transfer, eosoutdir, cpu, queue, ram):
    return f"""\
universe                = vanilla
executable              = {logdir}/$(name).sh
arguments               = $(ClusterId)$(ProcId)
output                  = {logdir}/log_$(ClusterId)_$(name).out
error                   = {logdir}/log_$(ClusterId)_$(name).err
log                     = {logdir}/log_$(ClusterId)_$(name).log
Should_Transfer_Files   = YES
transfer_input_files    = {transfer}
# The job stages its own output to EOS via xrdcp, so condor must transfer
# NOTHING back. Without this, condor's default returns every leftover top-level
# sandbox file to the submit dir -- e.g. an evaluated_*.root from a job that
# died before its xrdcp -- which looks like "output went local instead of EOS".
transfer_output_files   = ""
RequestCPUs             = {cpu}
+JobFlavour             = {queue}
request_memory          = {ram}
use_x509userproxy       = true

queue name from (
{names}
)
"""


EXECUTABLE_TEMPLATE = """\
#!/usr/bin/env bash
echo "Starting job on " `date`
echo "Running on: `uname -a`"
echo "System software: `cat /etc/redhat-release`"
workarea=$PWD
echo
echo "Work Area: $workarea"
ls
echo

## The evaluator wheel needs Python >=3.8, but the bare worker / default
## apptainer container can hand us a python3 as old as 3.6 (el8/sl7), making
## pip refuse the wheel ("requires Python '>=3.8'") and leaving the
## run3-mj-evaluator entry point uninstalled. Source the cvmfs LCG view
## matching this node's OS *only* for its modern python3, then build an
## ISOLATED venv from it and pip-install the wheel WITH its PyPI deps
## (onnxruntime etc.). Unlike the slimmer there is no coffea/correctionlib
## here, so PyPI wheels are safe - and PYTHONPATH is unset so the view's own
## uproot/awkward do not shadow the pip-installed versions.
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8
LCG_BASE=/cvmfs/sft.cern.ch/lcg/views/LCG_106
osmaj=$(rpm -E %{{rhel}} 2>/dev/null || echo 9)
LCG_VIEW=$(ls "$LCG_BASE"/x86_64-el${{osmaj}}-gcc*-opt/setup.sh 2>/dev/null | sort -V | tail -1)
if [ -z "$LCG_VIEW" ] || [ ! -r "$LCG_VIEW" ]; then
  # Last resort: newest gcc for any arch this node can run.
  LCG_VIEW=$(ls "$LCG_BASE"/x86_64-el*-gcc*-opt/setup.sh 2>/dev/null | sort -V | tail -1)
fi
echo "Node OS major: $osmaj"
echo "Sourcing LCG view: $LCG_VIEW"
source "$LCG_VIEW"
export LC_ALL=C.UTF-8 LANG=C.UTF-8 LC_CTYPE=C.UTF-8   # re-assert: the view may reset LC_*
unset PYTHONPATH
echo "Base python: $(python3 --version)"

## Set up Python virtual environment and install run3-mj-evaluator
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet {WHEEL}

## Run
echo
# Abort (non-zero exit, nothing uploaded) if the evaluator fails, so partial
# outputs are never xrdcp'd to EOS.
set -e
{RUN_COMMANDS}
set +e
echo "what directory am I in?"
pwd
echo "List all root files = "
ls *.root 2>/dev/null || echo "  (no .root output produced)"
echo "List all files"
ls -alh
echo "*******************************************"
OUTDIR=root://cmseos.fnal.gov/{EOSOUTDIR}
echo "xrdcp output for condor to "
"""

EXECUTABLE_TEMPLATE2 = """\
echo $OUTDIR
# Fail loudly (non-zero exit) if the evaluator delivered no output, instead of
# the old confusing "xrdcp ... no such file" when the glob is empty.
shopt -s nullglob
root_files=( evaluated_*.root )
if [[ ${#root_files[@]} -eq 0 ]]; then
  echo "ERROR: evaluator produced no evaluated_*.root output - nothing to deliver to EOS." >&2
  exit 1
fi
for FILE in "${root_files[@]}"
do
  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"
  xrdcp -f "${FILE}" "${OUTDIR}/${FILE}" 2>&1
  XRDEXIT=$?
  if [[ $XRDEXIT -ne 0 ]]; then
    echo "ERROR: xrdcp of ${FILE} failed (exit ${XRDEXIT}); output NOT delivered." >&2
    rm -f -- "${FILE}"   # worker scratch only
    exit $XRDEXIT
  fi
  rm -f -- "${FILE}"     # worker scratch only
done

echo
echo "Ending job on " `date`
"""


def collect_model_files(config, config_path):
    """Return (model_files, basename_config) for the evaluator config.

    Every model 'path' is resolved (relative to CWD, then to the config dir),
    its existence checked, and any sibling '<path>.data' weight sidecar picked
    up. A copy of the config is returned with each model 'path' replaced by its
    basename so it resolves inside the flat condor working directory.
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    model_files = []
    rewritten = json.loads(json.dumps(config))  # deep copy

    for i, m in enumerate(rewritten.get("models", [])):
        raw = m["path"]
        candidates = [raw, os.path.join(config_dir, raw)]
        resolved = next((c for c in candidates if os.path.isfile(c)), None)
        if resolved is None:
            raise SystemExit(
                f"Model {i} path not found: '{raw}' "
                f"(looked in CWD and {config_dir})"
            )
        model_files.append(resolved)

        # ONNX models above 2 GB store weights in a '<model>.onnx.data' sidecar
        # that must sit next to the model with the same basename.
        sidecar = resolved + ".data"
        if os.path.isfile(sidecar):
            model_files.append(sidecar)

        m["path"] = os.path.basename(resolved)

    return model_files, rewritten


class Fileset:
    def __init__(self, args):
        self.infile = args.inFile
        self.nf_per_job = args.nfPerJob
        self.eosoutdir = args.eosoutdir
        self.logdir = args.logdir
        self.fileset = {}
        self.jobs = []

        self._read()
        self._split()
        os.makedirs(self.logdir, exist_ok=True)

    def _read(self):
        try:
            with open(self.infile) as f:
                self.fileset = json.load(f)
        except FileNotFoundError:
            raise SystemExit(f"Fileset not found: {self.infile}")
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSON in {self.infile}: {e}")

    def _split(self):
        print(f"\nDatasets: {len(self.fileset)}")
        total = 0
        for k, (dataset, data) in enumerate(self.fileset.items()):
            files = list(data["files"].items())  # [(path, tree_name), ...]
            n = self.nf_per_job
            subjobs = [files[i:i + n] for i in range(0, len(files), n)]
            self.jobs.append((dataset, subjobs))
            print(f"  {k + 1}: {dataset}  →  {len(files)} files  →  {len(subjobs)} jobs")
            total += len(subjobs)
        print(f"\n  Total: {total} jobs\n")


class Batch:
    def __init__(self, jobs, args):
        self.jobs = jobs
        self.eosoutdir = args.eosoutdir
        self.logdir = args.logdir
        self.cpu = args.cpu
        self.queue = args.queue
        self.ram = args.memory
        self.config = args.config
        self.wheel = args.wheel
        self.default_tree = args.tree

        # Ship the models referenced by the config and write a config whose
        # model paths are basenames (condor flattens transfer_input_files).
        with open(self.config) as f:
            cfg = json.load(f)
        self.model_files, rewritten_cfg = collect_model_files(cfg, self.config)
        self.job_config = os.path.join(self.logdir, os.path.basename(self.config))
        os.makedirs(self.logdir, exist_ok=True)
        with open(self.job_config, "w") as f:
            json.dump(rewritten_cfg, f, indent=4)
        print(f"Models shipped: {[os.path.basename(m) for m in self.model_files]}")
        print(f"Job config:     {self.job_config}\n")

        self._write_jobs()
        self._write_submit()

    def _write_jobs(self):
        wheel_basename = os.path.basename(self.wheel)
        config_basename = os.path.basename(self.config)
        for dataset, subjobs in self.jobs:
            single = (len(subjobs) == 1)
            for i, files in enumerate(subjobs):
                name = dataset if single else f"{dataset}_{i}"
                run_cmds = []
                for filepath, tree in files:
                    tree_name = tree if tree else self.default_tree
                    basename = os.path.basename(filepath)
                    # Mirror the slimmed input name, swapping the leading
                    # "slimmed" for "evaluated":
                    #   slimmed_<dataset>_<tail>.root -> evaluated_<dataset>_<tail>.root
                    if basename.startswith("slimmed_"):
                        output = "evaluated_" + basename[len("slimmed_"):]
                    else:
                        output = "evaluated_" + basename
                    run_cmds.append(
                        f"run3-mj-evaluator {filepath} {output} {config_basename}"
                        f" --tree {tree_name}"
                    )
                exe = EXECUTABLE_TEMPLATE.format(
                    WHEEL=wheel_basename,
                    RUN_COMMANDS="\n".join(run_cmds),
                    EOSOUTDIR=self.eosoutdir,
                )
                exe = exe + EXECUTABLE_TEMPLATE2
                path = f"{self.logdir}/{name}.sh"
                with open(path, "w") as f:
                    f.write(exe)
                os.chmod(path, 0o755)

    def _write_submit(self):
        names = ""
        for dataset, subjobs in self.jobs:
            single = (len(subjobs) == 1)
            for i in range(len(subjobs)):
                name = dataset if single else f"{dataset}_{i}"
                names += f"\t{name}\n"

        transfer = ",".join([self.wheel, self.job_config] + self.model_files)
        config = configure_batch(
            logdir=self.logdir,
            names=names.strip(),
            transfer=transfer,
            eosoutdir=self.eosoutdir,
            cpu=self.cpu,
            queue=self.queue,
            ram=self.ram,
        )
        with open(f"{self.logdir}/submit.sub", "w") as f:
            f.write(config)

    def submit(self, execute):
        if execute:
            os.system(f"condor_submit {self.logdir}/submit.sub")
            print()
            print("Your jobs are here:")
            os.system("condor_q")
            print()
        else:
            print()
            print(f"To submit:       condor_submit {self.logdir}/submit.sub")
            print("To check status: condor_q")
            print("To see jobs:     condor_q -nobatch")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit run3-mj-evaluator jobs to HTCondor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--inFile",   required=True,  help="Coffea-style fileset JSON of slimmed files")
    parser.add_argument("-o", "--eosoutdir",   required=True,  help="EOS output dir as a bare /store/... path (cmseos redirector added automatically)")
    parser.add_argument("--config",         required=True,  help="run3-mj-evaluator config JSON")
    parser.add_argument("--wheel",          required=True,  help="Pre-built run3-mj-evaluator .whl file")
    parser.add_argument("-n", "--nfPerJob", type=int, default=1, help="Files per job")
    parser.add_argument("--tree",   default="events",   help="Fallback input tree name (overridden by fileset JSON)")
    parser.add_argument("--logdir", default="batch",    help="Directory for condor log/sh files")
    parser.add_argument("--cpu",    type=int, default=1, help="CPUs per job")
    parser.add_argument("--queue",  default="tomorrow", help="HTCondor JobFlavour")
    parser.add_argument("--memory", default="4GB",      help="Memory per job")
    parser.add_argument("--exec",   action="store_true", help="Submit jobs immediately after writing")

    args = parser.parse_args()

    # The job template (EXECUTABLE_TEMPLATE) already prepends the cmseos.fnal.gov
    # redirector to the output dir, so accept a bare /store/... path. If a full
    # root://host//store/... URL is given, strip the leading redirector to avoid
    # a doubled prefix in the xrdcp destination.
    args.eosoutdir = re.sub(r"^root://[^/]+/+", "/", args.eosoutdir)

    fileset = Fileset(args)
    batch = Batch(fileset.jobs, args)
    batch.submit(args.exec)
