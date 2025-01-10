#!/usr/bin/env python
import argparse
import pathlib as pl
from pprint import pprint
import requests

all_domains_dir = pl.Path("./domain_data")

domain_names_dict = {
    "willamette_river": "20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test",
}

domain_files_dict = {
    "20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test": [
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/bandit.cfg",  # noqa: E501
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/bandit.log",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/cbh.nc",
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/control.default",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/control.default.bandit",
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/humidity.day",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/myparam.param",
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/myparam_with_0.param",
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/precip.day",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/sf_data",
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/tmax.day",
        # "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/tmin.day",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_layers.gpkg",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nhru.cpg",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nhru.dbf",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nhru.prj",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nhru.shp",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nhru.shx",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_npoigages.cpg",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_npoigages.dbf",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_npoigages.prj",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_npoigages.shp",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_npoigages.shx",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nsegment.cpg",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nsegment.dbf",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nsegment.prj",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nsegment.shp",
        "https://usgs.osn.mghpcc.org/hytest/tutorials/NHM-Assist/20240524_v1.1_gm_byHWobs_williamette_river_NHMAssist_test/GIS/model_nsegment.shx",
    ],
}


def parse_args():
    """Parse the arguments.

    Args: None.
    Returns: The known arguments.
    """
    desc = "Retrieve and domain data from the cloud."
    parser = argparse.ArgumentParser(
        description=desc,
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Help on calling pull_domain_data.py.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional, string name of the domain you'd like. If not supplied "
            "or not available, a list of available domain names is printed."
        ),
        type=str,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If informative messages are to be suppressed.",
        default=False,
    )
    # parser.set_defaults(verbose=True)

    known, unknown = parser.parse_known_args()

    if len(unknown):
        msg = f"Unknown arguments supplied: {unknown}"
        raise ValueError(msg)

    return known


def pull_domain(name: str = None, verbose: bool = True):
    """Pull domain data from public storage URLs.

    Args:
      name: Optional, string name of the domain you'd like. If not supplied
        or not available, a list of available domain names is printed.
      verbose: If informative messages are to be printed.

    Examples:
      >>> from domain_data import pull_domain
      >>> pull_domain("williamette_river")

    """
    if name not in domain_names_dict.keys():
        msg = (
            f"No such asset in domain dictionary: {name}.\n"
            f"Please select from:\n{list(domain_names_dict.keys())}"
        )
        print(msg)
        return

    domain_url_base = domain_names_dict[name]
    domain_url_list = domain_files_dict[domain_url_base]
    domain_dir = all_domains_dir / name
    if not domain_dir.exists():
        domain_dir.mkdir(parents=True)

    for url in domain_url_list:
        file = domain_dir / url.split(domain_url_base)[-1][1:]
        if file.exists():
            if verbose:
                print(f"File {file} exists, skipping.")
            continue

        if verbose:
            print(f"Pulling {file}")
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        rr = requests.get(url)
        with open(file, "wb") as ff:
            ff.write(rr.content)

    # <<
    if verbose:
        print(f"Contents of {domain_dir}:")
        pprint([str(pp) for pp in domain_dir.glob("**/*")])

    return


if __name__ == "__main__":
    args = parse_args()
    pull_domain(name=args.name, verbose=not (args.quiet))
