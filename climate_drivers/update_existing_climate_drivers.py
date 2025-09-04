# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os
import pathlib as pl
import warnings

warnings.filterwarnings("ignore")
from rich.console import Console

con = Console()
from rich import pretty

pretty.install()
import jupyter_black

jupyter_black.load()
# Find and set the "nhm-assist" root directory
root_dir = pl.Path(os.getcwd().rsplit("nhm-assist", 1)[0] + "nhm-assist")
sys.path.append(str(root_dir))
from nhm_helpers.nhm_hydrofabric import make_hf_map_elements
from nhm_helpers.map_template import make_hf_map
from nhm_helpers.nhm_assist_utilities import load_subdomain_config

config = load_subdomain_config(root_dir)
con.print(config)


# %% [markdown]
# # Updating Climate Drivers with gdptools
#
# The Willamette River modeling domain currently uses climate data from 1979–2022. The gridMET dataset, which is updated daily with a one-day lag, allows us to regularly update our climate drivers with the latest available data.
#
# ## Introduction
#
# This notebook introduces tools and workflows for updating climate drivers in the current modeling domain. It covers:
#
# 1. An overview of the [`gdptools`](https://gdptools.readthedocs.io/en/develop/) package, which spatially interpolates gridded climate data to the polygonal modeling domain (HRUs) using areal-intersection weights.
# 2. A workflow for updating climate drivers using:
#    - **`gdptools`** ([repo](https://code.usgs.gov/wma/nhgf/toolsteam/gdptools))
#    - **`pyPRMS`** ([repo](https://github.com/DOI-USGS/pyPRMS/tree/development/docs))
#
# ## `gdptools` Package
#
# `gdptools` is a Python package for spatially interpolating gridded data to a polygonal fabric using areal weighting. It is used here to interpolate [gridMET climate data](https://www.climatologylab.org/gridmet.html) to the Willamette River modeling domain. `gdptools` was also used to create the original climate drivers for this domain.
#
# While `gdptools` provides the initial spatial interpolation, further post-processing is sometimes required:
# 1. Renaming variables and dimensions for compatibility with PRMS, pyPRMS, and pyWatershed.
# 2. Filling missing data, if the gridded dataset does not fully overlap the modeling domain. For the Willamette River domain, gridMET coverage is complete, but we include the filling step for completeness.
#
# ### Source and Target Data
#
# - **Source data:** gridMET gridded climate data.
# - **Target data:** The modeling domain, defined in `0_workspace_setup.ipynb` as `./domain_data/willamette_river/model_layers.gpkg`.
#
# `gdptools` provides an interface to the ClimateR-Catalog, a collection of gridded climate datasets and metadata. gridMET is included in this catalog.
#
# ### Working with the ClimateR-Catalog
#
# The [ClimateR-Catalog](https://github.com/mikejohnson51/climateR-catalogs) contains metadata for gridded climate datasets, including URLs, variable names, and coordinate information. We use the latest Parquet catalog file from [this release](https://github.com/mikejohnson51/climateR-catalogs/releases/download/June-2024/catalog.parquet), read it into a pandas DataFrame, and filter for the gridMET dataset. We then create a dictionary mapping variables of interest to their catalog entries.
#
# `gdptools` provides the `ClimRCatData` data class, which we use in the workflow below. The steps are:
# 1. Read the catalog into a pandas DataFrame.
# 2. Search for the relevant gridMET data.
# 3. Use the `ClimRCatData` class to access the data.

# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr

import hvplot.xarray
import hvplot.pandas
import hvplot.xarray

from gdptools import WeightGen
from gdptools import AggGen
from gdptools import ClimRCatData

from pathlib import Path

# %% [markdown]
# ## Inspect the existing climate driver file
# In this section, we will read the existing climate driver file and inspect its contents. The file is located at `./domain_data/willamette_river/cbh.nc`. We will use the `xarray` library to read the NetCDF file and display its structure.
#
# Note the existing time bounds used in the Willamette River modeling domain are from 1979-2022. We will update these bounds to include the latest available data from gridMET. Also note the data variable names, `tmax`, `tmin`, `prcp`, their units and dimension names, particularly `nhru`, as we will have to rename and convert our initial data processed by `gdptools` in our post-processing steps below.

# %%
existing_ds = xr.open_dataset(config["model_dir"] / "cbh.nc")
existing_ds

# %% [markdown]
# ## 1. Read the ClimateR-Catalog into a pandas DataFrame and parameterize the `ClimRCatData` class, which we use to represent our source data.
#
# The ClimateR-Catalog is a collection of gridded climate datasets and associated metadata. We will read the latest catalog file into a pandas DataFrame for further processing. In addition to the parquet file, there is also a JSON file available and for first time users it can be useful to open the file in a text editor that supports [JSON](https://github.com/mikejohnson51/climateR-catalogs/releases/download/June-2024/catalog.json) formatting to see the structure of the catalog.  Also the gdptools documentation has a [table](https://gdptools.readthedocs.io/en/develop/#example-catalog-datasets) of common datasets available in the catalog.  

# %%
climrcat_url = "https://github.com/mikejohnson51/climateR-catalogs/releases/download/June-2024/catalog.parquet"
climrcat_df = pd.read_parquet(climrcat_url)
climrcat_df

# %% [markdown]
# ### Pandas DataFrame Query Language
#
# Pandas provides a powerful query language that allows us to filter and manipulate data in a DataFrame. We can use the `query()` method to filter rows based on specific conditions. For example, if we want to filter the DataFrame for rows where the `id` column is equal to `gridmet`, we can use the following syntax where `@` is used to reference variables in the query string:

# %%
_id = "gridmet"
gridmet_df = climrcat_df.query("id == @ _id")
gridmet_df

# %% [markdown]
# ### Further Processing of the ClimateR-Catalog Data for use with `gdptools` `ClimRCatData`
#
# Once we have filtered the DataFrame for the gridMET dataset, we can create a dictionary that maps variable names to their corresponding catalog entries. This will allow us to easily access the data for each variable when using `gdptools`.
#

# %%
# Create a dictionary of climateR-catalog values for each variable
tvars = ["tmmn", "tmmx", "pr"]
cat_params = [
    gridmet_df.query("id == @ _id & variable == @ _var").to_dict(orient="records")[0]
    for _var in tvars
]

cat_dict = dict(zip(tvars, cat_params))

# Output an example of the cat_param.json entry for "aet".
cat_dict.get("tmmn")

# %% [markdown]
# ### Read in the target data file and inspect its contents
#
# The `ClimRCatData` class requires a target data file that defines the polygonal modeling domain (HRUs). We will read the target data file located at `./domain_data/willamette_river/model_layers.gpkg` and inspect its contents. This file contains the geometry of the HRUs, which will be used for spatial interpolation of the climate data.  In addition, we need the column header used to identify the HRU geometry, in this case `model_hru_idx`.

# %%
target_gdf = gpd.read_file(config["model_dir"] / "GIS/model_layers.gpkg", layer="nhru")
target_gdf

# %% [markdown]
# ### Parameterize the `ClimRCatData` class
#
# We use this data class to further parameterize WeightGen and AggGen classes in `gdptools` for generating areal weights and aggregating data, respectively. Set the period the time bounds for the data we want to process, in this case we will we will update the existing data through then of 2024.

# %%
user_data = ClimRCatData(
    cat_dict=cat_dict,
    f_feature=target_gdf,
    id_feature="model_hru_idx",
    period=["2023-01-01", "2024-12-31"],
)

# %% [markdown]
# ## 2. Generate Areal Weights with `gdptools`
#
# The Areal Weights Generator (`WeightGen`) in `gdptools` is used to create areal weights for spatial interpolation of gridded climate data to the polygonal modeling domain (HRUs). The weights are calculated based on the intersection of the gridded data with the HRU geometries. The weights are a table representing the target column header id, the gridded data cell ids (i and j indexes), and the normalized areal weights for each cell within each HRU.  As long as the source and target data are the same used in generating the weights, the weights can be reused for subsequent updates to the climate drivers.
#
# > Note: The `calculate_weights` method returns the weights as a pandas DataFrame, and also saves the weights to a CSV file for later use. 

# %%
# Eddie moved this down and included it with the function below.
# if weights_file.exists():
#     wghts = pd.read_csv(weights_file, index_col=0)

# wghts

# %% [markdown]
# A simple check on the generated weights is to group the weights by the target column header id and sum the weights. The sum should equal 1 for each target id, indicating that the weights are normalized.  Those target geometries with weights that sum to less than 1 indicate that the gridded data does not fully cover the HRU geometry, and we will need to fill those gaps in the post-processing step..

# %%
gdptools_path = Path(config["model_dir"] / "gdptools")
if not gdptools_path.exists():
    gdptools_path.mkdir(parents=True)
weights_file = gdptools_path / "gridmet_Wn_wghts.csv"
if not weights_file.exists():
    wght_gen = WeightGen(
        user_data=user_data,
        method="serial",
        output_file=weights_file,
        weight_gen_crs=5070,
    )

    wghts = wght_gen.calculate_weights()
else:
    wghts = pd.read_csv(weights_file, index_col=0)

# %%
# Check on the weights generated for HRUs; should sum to 1
sum_wghts = wghts.groupby("model_hru_idx").sum().reset_index()
sum_wghts

# %% [markdown]
# ### Inspect the generated weights
#
# Here we provide a quick inspection of the generated weights to ensure there are no HRUs with weights that sum to less than 1. If there are, we will need to fill those gaps in the post-processing step.
#
# > Note: In this case there are no HRUs with weights that sum to less than 1, indicating that the gridded data fully covers the HRU geometries.

# %%
# Define your tolerance (atol=absolute, rtol=relative)
tolerance = 1e-6

# Boolean mask: which values are NOT close to 1 (outside tolerance)
not_close_to_1 = ~np.isclose(sum_wghts["wght"], 1.0, atol=tolerance)

print(sum_wghts[not_close_to_1])

# %% [markdown]
# ## 3. Aggregate the Climate Data with `gdptools`

# %%
agg_out_path = Path(config["model_dir"] / "gdptools")
agg_gen = AggGen(
    user_data=user_data,
    stat_method="masked_mean",
    agg_engine="serial",
    agg_writer="netcdf",
    weights=str(Path(config["model_dir"] / "gdptools/gridmet_Wn_wghts.csv")),
    out_path=agg_out_path,
    file_prefix="cbh_2024_temp",
)
ngdf, ds_out = agg_gen.calculate_agg()

# %%
new_climate_ds = xr.open_dataset(config["model_dir"] / "gdptools/cbh_2024_temp.nc")
new_climate_ds

# %% [markdown]
# ## 4. Post-Process the Aggregated Climate Data
#
# The aggregated climate data is returned as an xarray Dataset, which we will then post-process to ensure compatibility with PRMS, pyPRMS, and pyWatershed. This includes renaming variables and dimensions, converting units if necessary, and filling any missing data.

# %%
ds_new = new_climate_ds.rename(
    {
        "daily_minimum_temperature": "tmin",
        "daily_maximum_temperature": "tmax",
        "precipitation_amount": "prcp",
        "model_hru_idx": "hruid",
    }
)
ds_new = ds_new.swap_dims({"hruid": "nhru"})
ds_new = ds_new.rename({"hruid": "nhru"})
ds_new["nhru"].attrs["long_name"] = "local model Hydrologic Response Unit ID (HRU)"
ds_new

# %%
# # Eddie patch but could be totally incorrecto
# ds_new["lat"].attrs["units"] = "degree"
# ds_new["lon"].attrs["units"] = "degree"

# %%
import pint_xarray

# 1. Quantify the dataset: attaches Pint units to each variable
quantified = ds_new.pint.quantify({"lat": None, "lon": None})

# 2. Perform unit conversions in-place on a copy, maintaining Dataset structure
#    Use assign() so you do not have to break out variables
quantified_converted = quantified.assign(
    tmin=quantified["tmin"].pint.to("degF"),
    tmax=quantified["tmax"].pint.to("degF"),
    prcp=quantified["prcp"].pint.to(
        "inch"
    ),  # "inch" or "inches" depending on your registry
)

# 3. Dequantify: converts Pint Quantities back to vanilla xarray, puts units in .attrs
dequantified = quantified_converted.pint.dequantify()

# 4. Save to NetCDF
dequantified.to_netcdf(config["model_dir"] / "gdptools/cbh_2024.nc")

# 5. (Optional) Inspect units to verify
for v in dequantified.data_vars:
    print(f"{v}: {dequantified[v].attrs.get('units', 'No units attribute')}")

# %%
update_climate_ds = xr.open_dataset(config["model_dir"] / "gdptools/cbh_2024.nc")
update_climate_ds

# %% [markdown]
# ## 5. Merge the Post-Processed Data with the Existing Climate Driver File
#
# Finally, we will merge the post-processed climate data with the existing climate driver file. This will update the climate drivers with the latest available data from gridMET, while preserving the existing data structure and metadata.

# %%
import dask

# %%
import dask
from dask.distributed import Client
import xarray as xr

# Start a local Dask cluster (will use all cores by default)
client = Client()
client

# %%
# Adjust the chunk size as appropriate for your available RAM and data
chunks = {"nhru": 50}
existing_ds = xr.open_dataset(config["model_dir"] / "cbh.nc")
# existing_ds = existing_ds.drop_vars(["nhm_id"])
update_climate_ds = xr.open_dataset(config["model_dir"] / "gdptools/cbh_2024.nc")
existing_ds = existing_ds.chunk(chunks)
update_climate_ds = update_climate_ds.chunk(chunks)

# %%
new_climate_ds = xr.concat(
    [existing_ds, update_climate_ds],
    dim="time",
)
lat_nhru = new_climate_ds["lat"].isel(time=0)
lon_nhru = new_climate_ds["lon"].isel(time=0)
nhmid_nhru = new_climate_ds["nhm_id"].isel(time=0)
new_climate_ds = new_climate_ds.drop_vars(["lat", "lon", "nhm_id"], errors="ignore")
new_climate_ds = new_climate_ds.assign_coords(
    lat=(("nhru",), lat_nhru.values), lon=(("nhru",), lon_nhru.values)
)
new_climate_ds = new_climate_ds.set_coords(["lat", "lon"])
new_climate_ds = new_climate_ds.assign(nhm_id=("nhru", nhmid_nhru.values))
new_climate_ds = new_climate_ds.chunk({"time": 365, "nhru": 50})

# %%
# Check if the 'crs' variable exists, and make it a dimensionless scalar
if "crs" in new_climate_ds.variables:
    crs_attrs = new_climate_ds["crs"].attrs
    # Use .isel(nhru=0) and squeeze to remove the dimension
    crs_scalar = new_climate_ds["crs"].isel(time=0).squeeze()
    new_climate_ds["crs"] = xr.DataArray(crs_scalar.data, attrs=crs_attrs)
new_climate_ds

# %%
new_climate_ds.to_netcdf(config["model_dir"] / "cbh_updated.nc", mode="w")

# %%
