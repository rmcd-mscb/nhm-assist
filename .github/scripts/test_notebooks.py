import argparse
import os
import pathlib as pl
import subprocess
import sys

from pywatershed.utils.utils import timer

repo_dir = pl.Path("../../").resolve()
scripts_dir = repo_dir / "notebooks" / "scripts"

all_notebook_scripts = set(scripts_dir.glob("*.py"))
# Add notebooks here as needed
scripts_to_not_test = set([scripts_dir / "add_pois_to_parameters.py"])
scripts_to_test = sorted(all_notebook_scripts - scripts_to_not_test)

def run_cmd(cmd):
    print(f"Running command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo_dir)  # , shell=True)
    assert proc.returncode == 0, f"Error running command: {' '.join(cmd)}"

@timer
def run_script(script_name):
    # Since we are stripping the metadata on pre-commit, we have to restore
    # this much (empty) metadata before jupytext execute
    cmd = ("python", f"{script_name}")
    run_cmd(cmd)

if __name__ == "__main__":
    failed_list = []
    for script in scripts_to_test:
        print(f"Testing notebook: {script}")
        try:
            run_script(script)
        except AssertionError:
            failed_list += [script]

    # <<
    if len(failed_list):
        msg = f"The following notebooks failed: {failed_list}"
        raise ValueError(msg)
