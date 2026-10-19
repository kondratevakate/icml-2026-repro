#!/bin/bash
# runs the 3 seeds as 3 parallel single-thread processes inside one container
cd /work
python -W ignore checks/check_C00080.py --seeds 0 --out work/logs/C00080_results_seed0.json > work/logs/C00080_seed0.log 2>&1 &
python -W ignore checks/check_C00080.py --seeds 1 --skip_a --out work/logs/C00080_results_seed1.json > work/logs/C00080_seed1.log 2>&1 &
python -W ignore checks/check_C00080.py --seeds 2 --skip_a --out work/logs/C00080_results_seed2.json > work/logs/C00080_seed2.log 2>&1 &
wait
python -W ignore checks/decide_C00080.py
