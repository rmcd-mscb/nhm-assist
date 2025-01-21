import pathlib as pl

repo_dir = pl.Path("../../").resolve()

nc_files_path = repo_dir / "domain_data/willamette_river/notebook_output_files/nc_files"

files_to_rm = [
    nc_files_path / "owrd_cache.nc",
    nc_files_path / "sf_efc.nc",
]


def del_path(path: pl.Path):
    if path.exists():
        print(f"deleting path: {path}")
        path.unlink()


if __name__ == "__main__":
    for ff in files_to_rm:
        del_path(ff)
