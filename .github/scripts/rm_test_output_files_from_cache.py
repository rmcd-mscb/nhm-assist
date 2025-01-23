import argparse
import pathlib as pl
import sys

sys.path.append("../..")

from pull_domain import all_domains_dir, domain_files_dict, domain_names_dict


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

    known, unknown = parser.parse_known_args()

    if len(unknown):
        msg = f"Unknown arguments supplied: {unknown}"
        raise ValueError(msg)

    return known


def rm_test_output_files_from_cache(name: str = None, verbose: bool = True):
    """Pull domain data from public storage URLs.

    Args:
      name: Optional, string name of the domain you'd like. If not supplied
        or not available, a list of available domain names is printed.
      verbose: Currently unused, If informative messages are to be printed.

    Examples:
      >>> from rm_test_output_files_from_cache import rm_test_output_files_from_cache
      >>> rm_test_output_files_from_cache("willamette_river")

    """
    domain_dir = all_domains_dir / name
    domain_url_base = domain_names_dict[name]
    domain_url_list = domain_files_dict[domain_url_base]

    if not domain_dir.exists():
        "Requested domain directory does not exist: {domain_dir}"
        warn(msg)
        return

    def get_files_to_rm():
        domain_all_files_glob = {str(ff) for ff in sorted(domain_dir.glob("**/*"))}

        domain_all_files_pulled = {
            str(domain_dir / url.split(domain_url_base)[-1][1:])
            for url in domain_url_list
        }

        strs_to_rm = list(domain_all_files_glob - domain_all_files_pulled)
        paths_to_rm = [pl.Path(ss) for ss in strs_to_rm]
        files_to_rm = [pp for pp in paths_to_rm if not pp.is_dir()]
        return files_to_rm

    files_to_rm = get_files_to_rm()

    for ff in files_to_rm:
        # these exist because of the glob
        ff.unlink()

    files_to_rm = get_files_to_rm()
    assert len(files_to_rm) == 0

    return


if __name__ == "__main__":
    args = parse_args()
    rm_test_output_files_from_cache(name=args.name, verbose=not (args.quiet))
