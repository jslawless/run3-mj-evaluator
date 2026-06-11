#!/bin/bash
# Submit run3-mj-evaluator over fileset JSON(s).
#
# Usage:
#   source scripts/run_all.sh <filelists-dir-or-json> <evaluator-wheel> <eos-outdir>
#
# The first argument may be either a DIRECTORY of fileset JSONs (every *.json in
# it is submitted) or a SINGLE fileset JSON (only that one is submitted).
#
# e.g. source scripts/run_all.sh filelists \
#          run3_mj_evaluator-1.0.0-py3-none-any.whl \
#          /store/user/you/evaluated
#
# Each fileset gets its own log directory and condor submission. The log dir
# is named "<dataset>_log" at the repo root (the dataset is the fileset
# filename with its .json extension stripped) -- not nested inside the
# fileset directory.

# Collect the fileset(s): a directory -> all *.json in it; a single file -> just it.
filesets=()
if [ -d "$1" ]; then
    while IFS= read -r f; do filesets+=("$f"); done \
        < <(find "$1" -maxdepth 1 -name '*.json' | sort)
elif [ -f "$1" ]; then
    filesets=("$1")
fi
if [ ${#filesets[@]} -eq 0 ]; then
    echo "ERROR: no .json fileset(s) for '$1' (pass a directory of JSONs or a single JSON)" >&2
fi

for i in "${filesets[@]}"; do
    filename=$(basename "$i")
    dataset="${filename%.*}"
    logdir="${dataset}_log"

    python scripts/submit_evaluator.py -i "$i" -o "$3/${dataset}" --config config/config.json --wheel "$2" --logdir "$logdir"
    condor_submit "${logdir}/submit.sub"
    sleep 2
done
