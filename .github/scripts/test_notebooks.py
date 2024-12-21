import argparse
import os
import pathlib as pl
import subprocess
import sys

repo_dir = pl.Path("../../").resolve()

notebooks_to_test = [
    "0a_Workspace_setup.ipynb",
    "0b_Create_poi_files.ipynb",
    # "0c_Create_Streamflow_Observations_v3.ipynb",
    # "1a_Framework_Inspection.ipynb",
    # "1b_Parameter_Visualization.ipynb",
    # "2a_preprocessing_for_pyWatershed.ipynb",
    # "2b_Run_NHMx_pyWatershed.ipynb",
    # "3a_HRU_Output_Visualization.ipynb",
    # "3b_Streamflow_Output_Visualization.ipynb",
]


def run_cmd(cmd):
    print(f"Running command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo_dir)
    assert proc.returncode == 0, f"Error running command: {' '.join(cmd)}"


def run_notebook(nb_name):
    cmd = ("jupytext", "--execute", f"{nb_name}")
    run_cmd(cmd)


def clean_notebook(nb_name):
    cmd = (
        "jupyter",
        "nbconvert",
        "--ClearOutputPreprocessor.enabled=True",
        "--ClearMetadataPreprocessor.enabled=True",
        "--ClearMetadataPreprocessor." + "preserve_nb_metadata_mask={('kernelspec')}",
        "--inplace",
        f"{nb_name}",
    )
    run_cmd(cmd)


def run_script(nb_name: str):
    nb_path = repo_dir / nb_name
    assert nb_path.exists(), f"no {nb_path=}, {os.getcwd()=}"
    py_script_path = nb_path.with_suffix(".py")
    py_script_name = str(py_script_path)
    cmd = ("jupytext", "--output", f"{py_script_name}", f"{str(nb_path)}")
    run_cmd(cmd)

    cmd = ("ipython", f"{py_script_name}")
    run_cmd(cmd)

    py_script_path.unlink()


if __name__ == "__main__":
    for nb in notebooks_to_test:

        # print(f"Testing notebook as script: {nb}")
        # run_script(nb)

        print(f"Testing notebook: {nb}")
        run_notebook(nb)
