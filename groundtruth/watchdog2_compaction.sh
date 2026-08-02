#!/bin/bash
# watchdog2: wait for arm2b logbook (from deleg_e195facf), score + compare vs original arm2.
set +e
RUN=/home/kate/projects/02_academia/icml-2026-repro/experiments/run03_hermes_leaf_pilot
REPO=/home/kate/projects/02_academia/icml-2026-repro
PY=/home/kate/.venvs/neurofm_py311/bin/python
DEC=$RUN/COMPACTION_DECISION.txt
echo "watchdog2 start $(date)" > "$DEC"

for i in $(seq 1 40); do
  if [ -f "$RUN/20hdQQQrA4_arm2b/logbook.md" ]; then
    echo "logbook found at $(date)" >> "$DEC"
    cd "$REPO" && $PY experiments/run03_hermes_leaf_pilot/score_run.py 20hdQQQrA4_arm2b >> "$DEC" 2>&1
    echo "=== compare vs original arm2 ===" >> "$DEC"
    $PY groundtruth/compare_runs.py 20hdQQQrA4_arm2 20hdQQQrA4_arm2b >> "$DEC" 2>&1
    MISM=$(grep -c "✗" "$DEC" 2>/dev/null || echo 0)
    if [ "$MISM" = "0" ]; then
      echo "DECISION: COMPACTION_OK -> launch Bundle 2 on arm2+compaction" >> "$DEC"
    else
      echo "DECISION: COMPACTION_WORSE -> launch Bundle 2 on original arm2 (no compaction)" >> "$DEC"
    fi
    echo "DONE $(date)" >> "$DEC"
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT: logbook not found in 40 min" >> "$DEC"
exit 1
