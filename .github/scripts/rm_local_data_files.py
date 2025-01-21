import pathlib as pl
import shutil

repo_dir = pl.Path("../../").resolve()

nc_files_path = repo_dir / "domain_data/willamette_river/notebook_output_files/nc_files"
folium_maps_paths = (
    repo_dir / "domain_data/willamette_river/notebook_output_files/Folium_maps"
)

files_to_rm = [
    nc_files_path / "owrd_cache.nc",
    nc_files_path / "sf_efc.nc",
]


dirst_to_rm = [
    folium_maps_paths,
]


def del_path(path: pl.Path):
    if path.exists():
        print(f"deleting path: {path}")
        path.unlink()


def del_dir(path: pl.Path):
    if path.exists():
        print(f"deleting direectory: {path}")
        shutil.rmtree(path)


if __name__ == "__main__":
    for ff in files_to_rm:
        del_path(ff)

    for dd in dirs_to_rm:
        del_dir(dd)
