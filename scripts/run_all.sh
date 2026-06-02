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
# Each fileset gets its own log directory and condor submission.
for i in "$1"/*; do
    filename=$(basename "$i")
    IFS='.' read -ra arrIN <<< "$filename"

    python scripts/submit_evaluator.py -i "$i" -o $3/${arrIN[0]} --config config/config.json --wheel $2 --logdir ${i}_log
    condor_submit ${i}_log/submit.sub
    sleep 2
done
