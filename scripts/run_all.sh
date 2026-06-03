#!/bin/bash
# Submit run3-mj-evaluator over every fileset JSON in a directory.
#
# Usage:
#   source scripts/run_all.sh <filelists-dir> <evaluator-wheel> <eos-outdir>
#
# e.g. source scripts/run_all.sh filelists \
#          run3_mj_evaluator-1.0.0-py3-none-any.whl \
#          /store/user/you/evaluated
#
# Each fileset gets its own log directory and condor submission. The log dir
# is named "<dataset>_log" at the repo root (the dataset is the fileset
# filename with its .json extension stripped) -- not nested inside the
# fileset directory.
for i in "$1"/*; do
    filename=$(basename "$i")
    dataset="${filename%.*}"
    logdir="${dataset}_log"

    python scripts/submit_evaluator.py -i "$i" -o "$3/${dataset}" --config config/config.json --wheel "$2" --logdir "$logdir"
    condor_submit "${logdir}/submit.sub"
    sleep 2
done
