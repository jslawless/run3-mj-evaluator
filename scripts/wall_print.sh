#!/usr/bin/env bash
# Emit a CSV of wall time and events processed for the job .out files in $1.
# Output is plain CSV (header + one row per job) so it can be read directly
# by python's csv module or pandas.read_csv.
# Usage: wall_print.sh <dir-of-.out-files> > wall.csv

# Header. Uncomment the "job," variant (and the matching printf below) to
# include the filename column.
#echo "job,wall,events"
echo "wall,events"

for f in "$1"/*.out; do
	awk -v job="$(basename "$f")" '
	# Strip thousands-separator commas so the values don't break the CSV.
	/Total wall time:/  { t = $NF; gsub(/,/, "", t) }
	/events processed/  { e = $2;  gsub(/,/, "", e) }
	END {
		# Skip blank entries: only print rows where both values were found.
		if (t != "" && e != "") {
			#printf "%s,%s,%s\n", job, t, e
			printf "%s,%s\n", t, e
		}
	}' "$f"
done
