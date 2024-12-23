import argparse
import os
import pathlib as pl
import subprocess
import sys

from pywatershed.utils.utils import timer

repo_dir = pl.Path("../../").resolve()

notebooks_to_test = [
    "0a_Workspace_setup.ipynb",
    "0b_Create_poi_files.ipynb",
    "0c_Create_Streamflow_Observations_v3.ipynb",
    "1a_Framework_Inspection.ipynb",
    "1b_Parameter_Visualization.ipynb",
    "2a_preprocessing_for_pyWatershed.ipynb",
    "2b_Run_NHMx_pyWatershed.ipynb",
    "3a_HRU_Output_Visualization.ipynb",
    "3b_Streamflow_Output_Visualization.ipynb",
]


def run_cmd(cmd):
    print(f"Running command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo_dir)
    assert proc.returncode == 0, f"Error running command: {' '.join(cmd)}"


@timer
def run_notebook(nb_name):
    cmd = ("jupytext", "--execute", f"{nb_name}")
    run_cmd(cmd)


if __name__ == "__main__":
    failed_list = []
    for nb in notebooks_to_test:
        print(f"Testing notebook: {nb}")
        try:
            run_notebook(nb)
        except AssertionError:
            failed_list += [nb]

    # <<
    if len(failed_list):
        print(f"The following notebooks failed: {failed_list}")
