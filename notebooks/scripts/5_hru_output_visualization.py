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
from ipywidgets import widgets
from IPython.display import display
from nhm_helpers.map_template import make_var_map
from nhm_helpers.nhm_hydrofabric import make_hf_map_elements
from nhm_helpers.nhm_output_visualization import retrieve_hru_output_info
from ipywidgets import VBox
from nhm_helpers.output_plots import plot_colors
from nhm_helpers.output_plots import (
    var_colors_dict,
    leg_only_dict,
    make_plot_var_for_hrus_in_poi_basin,
    oopla,
)

poi_id_sel = None

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
# This notebook maps the selected HRU output variable's values and displays in a new browser tab. The list of output variables are user-specified (`selected_output_variables` in [notebook 0](.\0_Workspace_setup.ipynb)). A gage id can be selected and two plots created. The first plot will show a time-series of all HRU values in the selected gage's catchment. The second plot will show a time-series of all HRU output variables averaged for the selected gage's catchment. Both plots will be displayed in a new browser tab. Additionally, maps and plots produced are saved for use outside of notebooks as .html files in the `html_maps` and `html_plots` folders in the `notebook_output` folder.
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

# Retrieve pywatershed output file information for mapping and plotting
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
# <font size=4> &#x270D;<font color='green'>**Enter Information:**</font> **Run the cell below. In the resulting drop-down boxes, select an HRU **output variable** and a **year** value to display in the map. You may also select a gage plots (optional). &#x270D;<font color='green'>**</font><br>
# <font size = '3'> The default is set to **recharge**, **mean_annual** with no gage selected, and the map will zoom to the model extent. If a gage number is selected here, the map will zoom to that location. Plots will also be rendered for the selected gage. If no gage is selected (default), the first gage listed in the parameter file will be used. All drop-down box selections can be changed and additional maps and plots will be created. Maps will be displayed and exported (HTML files) to `<NHM subdomain model folder>/notebook_output_files/html_maps`. Plots will be displayed and exported (HTML files) to `<NHM subdomain model folder>/notebook_output_files/html_plots`.

# %%
# Create selection widgets
##### Varibale selection widget
output_var_sel = output_var_list[
    8
]  # Set a default value so that the notebook will run without selection

style_date = {"description_width": "initial"}
style_var = {"description_width": "initial"}

v = widgets.Dropdown(
    options=output_var_list,
    value=output_var_sel,
    description="Output variable for map and plot:",
    layout=widgets.Layout(width="35%"),
    style=style_var,
)


def on_change(change):
    global output_var_sel, sel_flag
    if change["type"] == "change" and change["name"] == "value":
        output_var_sel = v.value
        # sel_flag = True


v.observe(on_change)
# display(v)

##### Year selection widget
list_of_years = year_list.copy()
list_of_years.append(
    "mean_annual"
)  # Append the mean annual so that the default will not be a year
sel_year = list_of_years[
    -1
]  # Set a default value so that the notebook will run without selection

yr = widgets.Dropdown(
    options=list_of_years,
    value=list_of_years[-1],
    description="Time step (year) for map:",
    layout=widgets.Layout(width="35%"),
    style=style_var,
)


def on_change(change):
    global sel_year  # Have to set the var as global so that it is carried outside of the fucntion to the notebook
    if change["type"] == "change" and change["name"] == "value":
        sel_year = yr.value


yr.observe(on_change)

# #################################################
v2 = widgets.Combobox(
    # value=poi_df.poi_id.tolist()[0],
    placeholder="(optional) Enter gage id",
    options=poi_df.poi_id.tolist(),
    description="Zoom to and plot gage:",
    ensure_option=True,
    disabled=False,
    layout=widgets.Layout(width="35%"),
    style=style_var,
)


def on_change2(change):
    global poi_id_sel, fig
    if change["type"] == "change" and change["name"] == "value":
        poi_id_sel = v2.value


v2.observe(on_change2)

display(VBox([v, yr, v2]))

# %% [markdown]
# <font size=4> &#x1F6D1;If new selections are made above,</font><br>
# <font size = '3'><font color='green'>**select this cell**</font>, then select <font color='green'>**Run Selected Cell and All Below**</font> from the Run menu in the toolbar.

# %% [markdown]
# ## Make interactive map of selected output variable values in the NHM subdomain
# The following cell creates an interactive folium.map that displays an annual values of all HRUs in the NHM subdomain for a selected output variable. Additionally, variable values for each HRU and additional HRU information can be viewed by left-clicking on HRUs. Maps produced are saved for use outside of notebooks as .html files in `./"subdomain model"/notebook_output/html_maps`.

# %%
# Make map

# This is for testing only; can comment out in user version
# if poi_id_sel is None:
#     poi_id_sel = poi_df.poi_id.tolist()[0]

map_file = make_var_map(
    root_dir,
    out_dir,
    output_var_sel,
    plot_start_date,
    plot_end_date,
    water_years,
    hru_gdf,
    poi_df,
    poi_id_sel,
    seg_gdf,
    html_maps_dir,
    year_list,
    sel_year,
    Folium_maps_dir,
    HW_basins,
    subdomain,
)

# %% [markdown]
# ## Create an interactive time-series plot to evaluate an NHM output variable for poi basin.
# The following cell creates a plot for the output variable selected in this notebook that shows a cumulative value for the selected poi basin at the annual, monthly, and daily time steps. Individual poi basin HRU contributions can also be compared to the poi basin values. The plot is interactive and can be saved using the widgets in the upper right-hand corner of the plot. Plots produced are saved for use outside of notebooks as .html files in `./"subdomain model"/notebook_output/html_plots`.

# %%
# This is for testing only; can comment out in user version
if poi_id_sel is None:
    poi_id_sel = poi_df.poi_id.tolist()[0]

fig_hru_sum_vars_for_poi_filename = make_plot_var_for_hrus_in_poi_basin(
    out_dir,
    param_filename,
    water_years,
    hru_gdf,
    poi_df,
    output_var_sel,
    poi_id_sel,
    plot_start_date,
    plot_end_date,
    plot_colors,
    subdomain,
    html_plots_dir,
)

# %% [markdown]
# ## Create an interactive plot to evaluate all NHM ouput variables as average fluxes (cubic feet/sec) for selected gage catchment.
# The following cell creates a plot of all listed output variables' average fluxes (cubic feet/sec) for the selected gage catchment at the annual, monthly, and daily time steps. Individual output variables can be added and removed. The plot is interactive and can be saved using the widgets in the upper right-hand corner of the plot. Plots produced are saved for use outside of notebooks as .html files in `./"subdomain model"/notebook_output/html_plots`.

# %%
fig_var_fluxes_for_poi_filename = oopla(
    out_dir,
    param_filename,
    water_years,
    hru_gdf,
    poi_df,
    output_var_list,
    output_var_sel,
    poi_id_sel,
    plot_start_date,
    plot_end_date,
    plot_colors,
    var_colors_dict,
    leg_only_dict,
    subdomain,
    html_plots_dir,
)

# %%
