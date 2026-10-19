"""Run an UNMODIFIED-LOGIC copy of a repository script: only the hard-coded cluster data directory string is replaced
by the synthetic-data directory, and the offline fastText stand-in is installed.  Usage:
  python run_patched.py SCRIPT.py DATA_DIR [script args...]"""
import os, re, sys, runpy
sys.path.insert(0, "/work/checks")
sys.path.insert(0, "/work/work/repo/RoVTL")
import fasttext_stub
fasttext_stub.install()
script, data_dir = sys.argv[1], sys.argv[2]
src = open(script).read()
src = src.replace("/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny", data_dir)
patched = "/tmp/patched_" + os.path.basename(script)
open(patched, "w").write(src)
sys.argv = [patched] + sys.argv[3:]
runpy.run_path(patched, run_name="__main__")
