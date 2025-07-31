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
import pathlib as pl
import warnings
from rich.console import Console

con = Console()
import os

root_folder = "nhm-assist"
root_dir = pl.Path(os.getcwd().rsplit(root_folder, 1)[0] + root_folder)
print(root_dir)
sys.path.append(str(root_dir))
from nhm_helpers.nhm_assist_utilities import load_subdomain_config


# %%
from nhm_helpers.nhm_hydrofabric import make_hf_map_elements
from nhm_helpers.map_template import make_par_map
from nhm_helpers.nhm_assist_utilities import (
    make_plots_par_vals,
    create_append_gages_to_param_file,
    make_myparam_addl_gages_param_file,
)
from nhm_helpers.nhm_helpers import *
from ipywidgets import widgets
from IPython.display import display

# Import Notebook Packages
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import warnings
from collections.abc import KeysView
import networkx as nx
from pyPRMS import ParameterFile
from pyPRMS.metadata.metadata import MetaData
from rich import pretty

pretty.install()
warnings.filterwarnings("ignore")

# %%
from nhm_helpers.nhm_assist_utilities import load_subdomain_config

(
    Folium_maps_dir,
    model_dir,
    param_filename,
    gages_file,
    default_gages_file,
    nwis_gages_file,
    output_netcdf_filename,
    NHM_dir,
    out_dir,
    notebook_output_dir,
    Folium_maps_dir,
    html_maps_dir,
    html_plots_dir,
    nc_files_dir,
    subdomain,
    GIS_format,
    param_file,
    control_file_name,
    nwis_gage_nobs_min,
    nhru_nmonths_params,
    nhru_params,
    selected_output_variables,
    water_years,
    workspace_txt,
) = load_subdomain_config(root_dir)

# %%
(
    hru_gdf,
    hru_txt,
    hru_cal_level_txt,
    seg_gdf,
    seg_txt,
    nwis_gages_aoi,
    poi_df,
    gages_df,
    gages_txt,
    gages_txt_nb2,
    HW_basins_gdf,
    HW_basins,
) = make_hf_map_elements(
    root_dir,
    model_dir,
    GIS_format,
    param_filename,
    control_file_name,
    nwis_gages_file,
    gages_file,
    default_gages_file,
    nhru_params,
    nhru_nmonths_params,
    nwis_gage_nobs_min,
)
con.print(
    f"{workspace_txt}\n",
    f"\n{gages_txt}{seg_txt}{hru_txt}",
    f"\n     {hru_cal_level_txt}\n",
    f"\n{gages_txt_nb2}",
)

# %% [markdown]
# ## Run the cell below makes a .csv file that is used to select additional gages to append the paramter file.

# %%
create_append_gages_to_param_file(
    gages_df,
    seg_gdf,
    poi_df,
    model_dir,
)

# %% [markdown]
# ## Run the cell below to add the gages listed in the additional gages to append .csv file to the parameter file.

# %%
make_myparam_addl_gages_param_file(
    model_dir,
    param_filename,
)

# %% [markdown]
# ## To view thhe model with the new parameter file, update the `param_file` in [0_workspace_setup](./0_workspace_setup.ipynb). We strongly recommend renaming the new parameter file, delete the `notebook_output_files` folder in the model directory and delete the `output` folder in the model directory. Then, rerun all notebooks.

# %%
