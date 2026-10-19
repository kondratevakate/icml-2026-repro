#!/bin/bash
# usage: run_container.sh <logname> <command...>
# Runs a command in the isolated container (no network, 8 GB, 4 CPUs, only DIR mounted),
# appends output to work/logs/<logname>.log and the wall time to work/time_log.csv.
DIR="C:/Projects/02_academia/studies/icml/icml-repro-working/verdict_audit/readiness/pilot/icml-2026__SRGpmDQjJM"
LOGNAME="$1"; shift
mkdir -p "$DIR/work/logs"
[ -f "$DIR/work/time_log.csv" ] || echo "start,seconds,command" > "$DIR/work/time_log.csv"
startiso=$(date -Iseconds); start=$(date +%s)
MSYS_NO_PATHCONV=1 docker run --rm --network none --memory 8g --cpus 4 -v "$DIR:/work" -w /work p9pilot-srgpmdqjjm "$@" 2>&1 | tee -a "$DIR/work/logs/$LOGNAME.log"
rc=${PIPESTATUS[0]}
end=$(date +%s)
echo "$startiso,$((end-start)),\"$* (exit $rc, log $LOGNAME)\"" >> "$DIR/work/time_log.csv"
exit $rc
