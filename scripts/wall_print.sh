for f in $1/*.out; do
	awk -v job="$(basename "$f")" '\
	/Total wall time:/  {t=$NF} \
	/events processed/  {e=$2} \
	END {printf "%-45s wall=%-10s events=%s\
	", job, t, e } \
	' "$f";
	echo
done
