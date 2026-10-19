#!/bin/bash
set -e
PY=.venv/Scripts/python.exe
echo "### SUPPORT ###"
$PY reproduce.py --data SUPPORT --family N-MTLR --n-exp 5 --out runs/support_nmtlr.json
echo "### semi-SUPPORT ###"
$PY reproduce.py --data semi-SUPPORT --family N-MTLR --n-exp 5 --out runs/semisupport_nmtlr.json
echo "### ALL DONE ###"
