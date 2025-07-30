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

# sys.path.append("../")
import pathlib as pl
import os

root_folder = "nhm_pest_ies"
root_dir = pl.Path(os.getcwd().rsplit(root_folder, 1)[0] + root_folder)
print(root_dir)
sys.path.append(str(root_dir))

import warnings
from rich.console import Console

con = Console()
import jupyter_black

jupyter_black.load()

# %%
from nhm_helpers.output_plots import create_streamflow_plot
from nhm_helpers.nhm_hydrofabric import make_hf_map_elements
from nhm_helpers.map_template import make_streamflow_map
from nhm_helpers.nhm_output_visualization import retrieve_hru_output_info
from ipywidgets import widgets
from IPython.display import display

poi_id_sel = None
crs = 4326

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

# %% [markdown]
# ## Introduction
# The purpose of this notebook is to assist in the evaluation of NHM subdomain model simulated streamflow. First, the notebook creates a map of gage locations color coded by Kling-Gupta efficiency (KGE; [2009](https://www.sciencedirect.com/science/article/abs/pii/S0022169409004843)) value. Gage locations are overlays in a map of NHM headwater basins (HWs) that are color coded to calibration type: yellow indicates HWs calibrated with statistical streamflow targets at the HW outlet; green indicates HWs that were further calibrated with streamflow observations at selected gage locations.
# Next, a gage may be selected and a plot created that shows a time-series of simulated and observed streamflow at daily, monthly, and annual time steps. A flow exceedance curve and table of summary statistics is also provided in the plot for streamflow evaluation purposes. Additionally, maps and plots produced are saved for use outside of notebooks as HTML files in the `html_maps` and `html_plots` folders in the `notebook_output` folder.
#
# The cell below reads the NHM subdomain model hydrofabric elements for mapping purposes using `make_hf_map_elements()`, retrieves `pywatershed` output file information for mapping and plotting using `retrieve_hru_output_info()`, and writes general NHM subdomain model run and hydrofabric information.

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
    NHM_dir,
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

plot_start_date, plot_end_date, year_list, output_var_list = retrieve_hru_output_info(
    out_dir,
    water_years,
)
con.print(
    f"{workspace_txt}\n",
    f"\n{gages_txt}{seg_txt}{hru_txt}",
    f"\n     {hru_cal_level_txt}",
)

# %% [markdown]
# ## Create an interactive map to evaluate streamflow at gages (pois)
# The following cell creates a map of the NHM subdomain model hydrofabric elements and displays monthly Kling-Gupta efficiency (KGE) values for parameter file gages as red (KGE<0.5), yellow (0.5<=KGE<0.7), and green (KGE>=0.7). The map is interactive, made with [folium](https://python-visualization.github.io/folium/v0.18.0/index.html) (see [README](./README.md) for basic map interactive functionality). Use the mouse to left-click on gage icons and display the gage id and gage name. Use the mouse to left-click on HRUs to display the headwater basin (HW) id. HW groupings of HRUs are important when interpreting parameter values and output variables. HRU parameters were grouped by HW and adjusted using a scaling factor during the byHW and byHWobs parts of NHM calibration ([Hay and others, 2023](https://pubs.usgs.gov/tm/06/b10/tm6b10.pdf)). During the byHW calibration part, statistical flow targets at HW outlets were used, and during the subsequent byHWobs part, streamflow obervations were used for selected HWs.

# %%
map_file = make_streamflow_map(
    root_dir,
    out_dir,
    plot_start_date,
    plot_end_date,
    water_years,
    hru_gdf,
    poi_df,
    poi_id_sel,
    seg_gdf,
    html_maps_dir,
    subdomain,
    HW_basins_gdf,
    HW_basins,
    output_netcdf_filename,
)

# %%
poi_df

# %% [markdown]
# ## Select a gage to evaluate simulated and observed streamflow time-series and streamflow statistics
# <font size=4> &#x270D;<font color='green'>**Enter Information:**</font> **Run the cell below. In the resulting drop-down box, select a gage. A plot will be created to evaluate observed and simulated streamflow at the selected gage.** </font><br>
# <font size=3> If no gage is selected (default), the first gage listed in the parameter file will be used. The drop-down box selection can be changed and additional plots will be displayed and exported (html files) to<br> `<NHM subdomain model folder>/notebook_output_files/html_plots`.

# %%
if poi_id_sel is None:
    poi_id_sel = poi_df.poi_id.tolist()[0]

v = widgets.Combobox(
    # value='John',
    placeholder="(optional) Enter Gage ID here",
    options=poi_df.poi_id.tolist(),
    value=poi_id_sel,
    description="Plot Gage:",
    ensure_option=True,
    disabled=False,
)


def on_change(change):
    global poi_id_sel, fig
    if change["type"] == "change" and change["name"] == "value":
        poi_id_sel = v.value


v.observe(on_change)
display(v)

# %% [markdown]
# <font size=4> &#x1F6D1;If a new selection is made above,</font><br>
# <font color='green' size = '3'>**select this cell**</font>, then select <font color='green'>**Run Selected Cell and All Below**</font> from the Run menu in the toolbar.

# %% [markdown]
# ## Make an interactive plot of simulated and observed streamflow and table of streamflow statistics for selected gage
# This plot is interactive, and made with [plotly](https://plotly.com/) (see [README](./README.md) for basic plot interactive functionality), and shows subplots of simulated and observed streamflow at daily, monthly, and annual timesteps for the selected gage. Note: the date that appears in popup window when hovering over plotted data reflects the last day of the timestep displayed in the plot. A flow exceedance curve and table of summary statistics is also provided in the plot for streamflow evaluation purposes. 

# %%
plot_file_path = create_streamflow_plot(
    poi_id_sel,
    plot_start_date,
    plot_end_date,
    water_years,
    html_plots_dir,
    output_netcdf_filename,
    out_dir,
    subdomain,
)
