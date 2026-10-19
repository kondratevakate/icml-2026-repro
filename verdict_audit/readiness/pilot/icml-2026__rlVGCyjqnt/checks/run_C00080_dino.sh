#!/bin/bash
cd /work
for v in time_only channel_only both; do
  timeout 1500 python -W ignore checks/check_C00080_dino.py --variant $v --out work/logs/C00080_dino_$v.json > work/logs/C00080_dino_$v.log 2>&1 &
done
wait
tail -n 3 work/logs/C00080_dino_*.log
