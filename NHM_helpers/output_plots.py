# Import Notebook Packages
import warnings
from urllib import request
from urllib.request import urlopen
from urllib.error import HTTPError

import re
from io import StringIO
import os

# os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import xarray as xr
import pandas as pd
import pathlib as pl
import numpy as np
import pyogrio

import netCDF4

import ipyleaflet

import branca
import branca.colormap as cm

import folium
from folium import Circle, Marker
from folium import plugins
from folium.features import DivIcon
from folium.plugins import MarkerCluster
from ipywidgets import widgets

from ipyleaflet import Map, GeoJSON

# PyPRMS needs
from pyPRMS import Dimensions
from pyPRMS.metadata.metadata import MetaData
from pyPRMS import ControlFile
from pyPRMS import Parameters
from pyPRMS import ParameterFile
from pyPRMS.prms_helpers import get_file_iter, cond_check
from pyPRMS.constants import (
    DIMENSIONS_HDR,
    PARAMETERS_HDR,
    VAR_DELIM,
    PTYPE_TO_PRMS_TYPE,
    PTYPE_TO_DTYPE,
)
from pyPRMS.Exceptions_custom import ParameterExistsError, ParameterNotValidError
import networkx as nx
from collections.abc import KeysView

import pywatershed as pws

from rich.console import Console
from rich.progress import track
from rich.progress import Progress
from rich import pretty

pretty.install()
con = Console()

warnings.filterwarnings("ignore")

#### Adds:
import matplotlib as mplib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import ipyleaflet
from ipyleaflet import Map, GeoJSON

from folium import Choropleth
from folium.plugins import BeautifyIcon

import branca
import branca.colormap as cm

import plotly.graph_objects as go
import plotly
import plotly.subplots
import plotly.express as px

import dataretrieval.nwis as nwis

from NHM_helpers.NHM_helpers import (
    hrus_by_poi,
    hrus_by_seg,
    subset_stream_network,
    create_poi_group,
)
from NHM_helpers.map_template import *
from NHM_helpers.NHM_Assist_utilities import make_plots_par_vals

from NHM_helpers.NHM_output_visualization import (
    retrieve_hru_output_info,
    create_sum_var_dataarrays,
    create_mean_var_dataarrays,
    create_sum_var_annual_gdf,
    create_sum_var_annual_df,
    create_sum_var_monthly_df,
    create_var_daily_df,
    create_var_ts_for_poi_basin_df,
)
from NHM_helpers.output_plots import * #create_poi_group

import webbrowser

# HRU color list
import random

plot_colors = [
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgrey",
    "darkgreen",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dodgerblue",
    "firebrick",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "goldenrod",
    "green",
    "greenyellow",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "lavender",
    "lawngreen",
    "lime",
    "limegreen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "moccasin",
    "navy",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royalblue",
    "rebeccapurple",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "yellowgreen",
]
# random.shuffle(plot_colors)
random.Random(4).shuffle(plot_colors)
# plot_colors

# Dictionaries for plots
var_colors_dict = {
    "hru_actet": "red",
    "recharge": "brown",
    "net_rain": "blue",
    "net_snow": "lightblue",
    "net_ppt": "darkblue",
    "sroff": "lightgreen",
    "sroff_vol": "lightgreen",
    "ssres_flow": "green",
    "ssres_flow_vol": "green",
    "gwres_flow": "chocolate",
    "gwres_flow_vol": "chocolate",
    "gwres_sink": "black",
    "snowmelt": "mediumpurple",
    "gwres_stor": "darkgreen",
    "gwres_stor_change": "darkgreen",
    "ssres_stor": "green",
    "unused_potet": "orange",
}

# 'legendonly'
leg_only_dict = {
    "hru_actet": "legendonly",
    "recharge": "legendonly",
    "net_rain": "legendonly",
    "net_snow": "legendonly",
    "net_ppt": True,
    "sroff": True,
    "sroff_vol": True,
    "ssres_flow": "legendonly",
    "ssres_flow_vol": "legendonly",
    "gwres_flow": True,
    "gwres_flow_vol": True,
    "gwres_sink": "legendonly",
    "snowmelt": "legendonly",
    "gwres_stor": "legendonly",
    "gwres_stor_change": "legendonly",
    "ssres_stor": "legendonly",
    "unused_potet": "legendonly",
}

def make_plot_var_for_hrus_in_poi_basin(
    out_dir,
    param_filename,
    water_years,
    hru_gdf,
    poi_df,
    output_var_sel,
    var_units,
    poi_id_sel,
    plot_start_date,
    plot_end_date,
    plot_colors,
    subbasin,
    html_plots_dir,
):
    #plot_file = f"{html_plots_dir}/{output_var_sel}_for_{poi_id_sel}_plot.html"
    plot_file_path = pl.Path(html_plots_dir / f"{output_var_sel}_for_{poi_id_sel}_plot.html").resolve()
    
    if plot_file_path.exists():
        webbrowser.open(f"{plot_file_path}", new=2)
    else:
        var_daily, sum_var_monthly, sum_var_annual, var_units, var_desc = (
            create_sum_var_dataarrays(
                out_dir,
                output_var_sel,
                plot_start_date,
                plot_end_date,
                water_years,
            )
        )
        
        hru_gdf, hru_poi_dict = create_poi_group(hru_gdf, poi_df, param_filename)
        
        if poi_id_sel:
            hru_list = hru_poi_dict[
                poi_id_sel
            ]  # returns a list of all upstream contributing hrus
    
            fig = plotly.subplots.make_subplots(
                rows=3,
                cols=1,
                shared_xaxes="columns",
                # shared_yaxes = 'columns',
                start_cell="top-left",
                vertical_spacing=0.1,
                y_title=f"{output_var_sel}, {var_units}",
                subplot_titles=[
                    "Annual",
                    "Monthly",
                    "Daily",
                ],
                specs=[
                    [{"type": "scatter"}],
                    [{"type": "scatter"}],
                    [{"type": "scatter"}],
                ],
            )
            fig.update_layout(
                title_text=f'The NHM {subbasin} domain {output_var_sel} for poi basin<br> {poi_id_sel}, {poi_df.loc[poi_df.poi_id == poi_id_sel, "poi_name"].values[0]}',  #
                width=900,
                height=700,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=10.0),
                # legend_tracegroupgap = 5,
                font=dict(family="Arial", size=14, color="#7f7f7f"),  # font color
                paper_bgcolor="linen",
                plot_bgcolor="white",
            )
    
            fig.update_layout(
                title_automargin=True,
                title_font_color="black",
                title_font_size=20,
                title_x=0.5,
                title_y=0.945,
                title_xref="container",
                title_xanchor="center",
            )
    
            # fig.update_xaxes(range = [daily_plot_df.index[0], daily_plot_df.index[-1]])
    
            fig.update_layout(font_color="black")
            fig.update_layout(legend={"title": "Extent"})
    
            fig.update_xaxes(ticks="inside", tickwidth=2, tickcolor="black", ticklen=10)
            fig.update_yaxes(ticks="inside", tickwidth=2, tickcolor="black", ticklen=10)
    
            fig.update_xaxes(
                showline=True, linewidth=2, linecolor="black", gridcolor="lightgrey"
            )
            fig.update_yaxes(
                showline=True, linewidth=2, linecolor="black", gridcolor="lightgrey"
            )
    
            fig.update_traces(hovertemplate=None)
            fig.update_layout(hovermode="x unified")
            fig.update_layout(
                hoverlabel=dict(
                    bgcolor="linen",
                    font_size=13,
                    font_family="Rockwell",
                )
            )
    
            ######################################################
    
            annual_fig = go.Figure()
    
            # subset to selcted gage one gage
            sum_var_annual_df = create_sum_var_annual_df(
                hru_gdf,
                poi_df,
                param_filename,
                plot_start_date,
                plot_end_date,
                sum_var_annual,
                output_var_sel,
            )
            
            df_basin = sum_var_annual_df.loc[
                sum_var_annual_df["nhm_id"].isin(hru_poi_dict[poi_id_sel])
            ]
            df_basin.set_index(
                ["time", "nhm_id"], inplace=True, drop=True
            )  # resets the index to that new value and type
    
            # Calculate basin volume average output_var value from basin HRUs
            df_basin_plot1 = df_basin.groupby(level="time").sum()
            df_basin_plot1["output_var"] = (
                df_basin_plot1["vol_inch_acres"] / df_basin_plot1["hru_area"]
            )
    
            annual_fig.add_trace(
                go.Scatter(
                    x=df_basin_plot1.index,
                    y=(df_basin_plot1.output_var).ravel().tolist(),
                    mode="lines",
                    name=f"poi basin",
                    showlegend=True,
                    legendgroup="poi_basin",
                    # marker=dict(color='lightblue'),
                    line=dict(
                        color="lightblue",
                        width=5,
                        # dash='dot'
                    ),
                )
            )
    
            for (
                value
            ) in (
                hru_list
            ):  # I fixed this below to read right from the xarray, need to fix the hru_list as well.
                hru_id_sel = value
                color_sel = hru_list.index(hru_id_sel)
                ds_sub = (
                    sum_var_annual.where((sum_var_annual.nhm_id == hru_id_sel), drop=True)
                ).sel(
                    time=slice(plot_start_date, plot_end_date)
                )  # have to fix to subset to new date range
                annual_fig.add_trace(
                    go.Scatter(
                        x=ds_sub.time,
                        y=(ds_sub.values).ravel().tolist(),
                        mode="lines",
                        name=f"HRU {hru_id_sel}",
                        showlegend=True,
                        visible="legendonly",
                        legendgroup=hru_id_sel,
                        # marker=dict(color=plot_colors[color_sel]),
                        line=dict(
                            color=plot_colors[color_sel],
                            width=2,
                            # dash='dot'
                        ),
                    )
                )
    
            monthly_fig = go.Figure()
    
            # subset to selcted gage one gage
    
            sum_var_monthly_df = create_sum_var_monthly_df(
               hru_gdf,
               poi_df,
               param_filename,
               output_var_sel,
               plot_start_date,
               plot_end_date,
               sum_var_monthly,
           )
    
            
            df_basin = sum_var_monthly_df.loc[
                sum_var_monthly_df["nhm_id"].isin(hru_poi_dict[poi_id_sel])
            ]
            df_basin.set_index(
                ["time", "nhm_id"], inplace=True, drop=True
            )  # resets the index to that new value and type
    
            # Calculate basin recharge from individual HRU contributions for plotting
            df_basin_plot1 = df_basin.groupby(level="time").sum()
            df_basin_plot1["output_var"] = (
                df_basin_plot1["vol_inch_acres"] / df_basin_plot1["hru_area"]
            )
    
            monthly_fig.add_trace(
                go.Scatter(
                    x=df_basin_plot1.index,
                    y=(df_basin_plot1.output_var).ravel().tolist(),
                    mode="lines",
                    name=poi_id_sel,
                    showlegend=False,
                    legendgroup="poi_basin",
                    # marker=dict(color='lightblue'),
                    line=dict(
                        color="lightblue",
                        width=5,
                        # dash='dot'
                    ),
                )
            )
    
            for (
                value
            ) in (
                hru_list
            ):  # I fixed this below to read right from the xarray, need to fix the hru_list as well.
                hru_id_sel = value
                color_sel = hru_list.index(hru_id_sel)
                ds_sub = (
                    sum_var_monthly.where((sum_var_monthly.nhm_id == hru_id_sel), drop=True)
                ).sel(time=slice(plot_start_date, plot_end_date))
                monthly_fig.add_trace(
                    go.Scatter(
                        x=ds_sub.time,
                        y=(ds_sub.values).ravel().tolist(),
                        mode="lines",
                        name=hru_id_sel,
                        showlegend=False,
                        visible="legendonly",
                        legendgroup=hru_id_sel,
                        # marker=dict(color=plot_colors[color_sel]),
                        line=dict(
                            color=plot_colors[color_sel],
                            width=2,
                            # dash='dot'
                        ),
                    )
                )
            daily_fig = go.Figure()
    
            # subset to selcted gage one gage
    
            var_daily_df = create_var_daily_df(
                hru_gdf,
                poi_df,
                param_filename,
                output_var_sel,
                plot_start_date,
                plot_end_date,
                var_daily,
            )
    
            df_basin = var_daily_df.loc[
                var_daily_df["nhm_id"].isin(hru_poi_dict[poi_id_sel])
            ]
            df_basin.set_index(
                ["time", "nhm_id"], inplace=True, drop=True
            )  # resets the index to that new value and type
    
            # Calculate basin recharge from individual HRU contributions for plotting
            df_basin_plot1 = df_basin.groupby(level="time").sum()
            df_basin_plot1["output_var"] = (
                df_basin_plot1["vol_inch_acres"] / df_basin_plot1["hru_area"]
            )
    
            daily_fig.add_trace(
                go.Scatter(
                    x=df_basin_plot1.index,
                    y=(df_basin_plot1.output_var).ravel().tolist(),
                    mode="lines",
                    name=poi_id_sel,
                    showlegend=False,
                    legendgroup="poi_basin",
                    # marker=dict(color='lightblue'),
                    line=dict(
                        color="lightblue",
                        width=5,
                        # dash='dot'
                    ),
                )
            )
    
            for (
                value
            ) in (
                hru_list
            ):  # I fixed this below to read right from the xarray, need to fix the hru_list as well.
                hru_id_sel = value
                color_sel = hru_list.index(hru_id_sel)
                ds_sub = (var_daily.where((var_daily.nhm_id == hru_id_sel), drop=True)).sel(
                    time=slice(plot_start_date, plot_end_date)
                )
                daily_fig.add_trace(
                    go.Scatter(
                        x=ds_sub.time,
                        y=(ds_sub.values).ravel().tolist(),
                        mode="lines",
                        name=hru_id_sel,
                        showlegend=False,
                        visible="legendonly",
                        legendgroup=hru_id_sel,
                        # marker=dict(color=plot_colors[color_sel]),
                        line=dict(
                            color=plot_colors[color_sel],
                            width=2,
                            # dash='dot'
                        ),
                    )
                )
    
            for t in annual_fig.data:
                fig.append_trace(t, row=1, col=1)
    
            for t in monthly_fig.data:
                fig.append_trace(t, row=2, col=1)
    
            for t in daily_fig.data:
                fig.append_trace(t, row=3, col=1)
    
            plotly.offline.plot(fig, filename=f"{plot_file_path}")

        return plot_file_path

def oopla(
    out_dir,
    param_filename,
    water_years,
    hru_gdf,
    poi_df,
    output_var_list,
    output_var_sel,
    var_units,
    poi_id_sel,
    plot_start_date,
    plot_end_date,
    plot_colors,
    var_colors_dict,
    leg_only_dict,
    subbasin,
    html_plots_dir,
):
    """
    Make figure of three plots...

    First figure setup.

    """
    plot_file_path = pl.Path(html_plots_dir / f"water_budget_fluxes_for_{poi_id_sel}_plot.html").resolve()
    
    if plot_file_path.exists():
        webbrowser.open(f"{plot_file_path}", new=2)
    else:

        hru_gdf, hru_poi_dict = create_poi_group(hru_gdf, poi_df, param_filename)
    
        fig = plotly.subplots.make_subplots(
            rows=3,
            cols=1,
            shared_xaxes="columns",
            # shared_yaxes = 'columns',
            start_cell="top-left",
            vertical_spacing=0.1,
            y_title="Water flux, cubic-feet per second",
            subplot_titles=[
                "Annual mean",
                "Monthly mean",
                "Daily",
            ],
            specs=[
                [{"type": "scatter"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}],
            ],
        )
        fig.update_layout(
            title_text=f'The NHM {subbasin} domain water budget flux rates for <br> {poi_id_sel}, {poi_df.loc[poi_df.poi_id == poi_id_sel, "poi_name"].values[0]}',  #
            width=900,
            height=700,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=10.0),
            # legend_tracegroupgap = 5,
            font=dict(family="Arial", size=14, color="#7f7f7f"),  # font color
            paper_bgcolor="linen",
            plot_bgcolor="white",
        )
    
        fig.update_layout(
            title_automargin=True,
            title_font_color="black",
            title_font_size=20,
            title_x=0.5,
            title_y=0.945,
            title_xref="container",
            title_xanchor="center",
        )
    
        # fig.update_xaxes(range = [daily_plot_df.index[0], daily_plot_df.index[-1]])
    
        fig.update_layout(font_color="black")
        fig.update_layout(
            legend={"title": "NHM output variable"}
        )  # <--- add only this line
    
        fig.update_xaxes(ticks="inside", tickwidth=2, tickcolor="black", ticklen=10)
        fig.update_yaxes(ticks="inside", tickwidth=2, tickcolor="black", ticklen=10)
    
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="black", gridcolor="lightgrey"
        )
        fig.update_yaxes(
            showline=True, linewidth=2, linecolor="black", gridcolor="lightgrey"
        )
    
        fig.update_traces(hovertemplate=None)
    
        fig.update_layout(hovermode="x unified")
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="linen",
                font_size=13,
                font_family="Rockwell",
            )
        )
    
        daily_fig = go.Figure()
        annual_fig = go.Figure()
        monthly_fig = go.Figure()
    
        """
        Nww to plot the data
    
        """
        for var in output_var_list:
            color_sel = var_colors_dict[var]
            leg_only = leg_only_dict[var]
    
            var_daily, mean_var_monthly, mean_var_annual, var_units, var_desc = (
                create_mean_var_dataarrays(
                    out_dir,
                    var,
                    plot_start_date,
                    plot_end_date,
                    water_years,
                )
            )
    
            df_basin_plot1_annual = create_var_ts_for_poi_basin_df(
                mean_var_annual,
                var,
                hru_gdf,
                hru_poi_dict,
                poi_id_sel,
            )
    
            df_basin_plot1_monthly = create_var_ts_for_poi_basin_df(
                mean_var_monthly,
                var,
                hru_gdf,
                hru_poi_dict,
                poi_id_sel,
            )
    
            df_basin_plot1_daily = create_var_ts_for_poi_basin_df(
                var_daily,
                var,
                hru_gdf,
                hru_poi_dict,
                poi_id_sel,
            )
    
            annual_fig.add_trace(
                go.Scatter(
                    x=df_basin_plot1_annual.index,
                    y=(df_basin_plot1_annual.vol_cfs).ravel().tolist(),
                    # x=year_list,
                    # y= (gdf.loc[gdf.nhm_id == hru_id_sel, year_list].values).ravel().tolist(),
                    mode="lines",
                    name=var,
                    visible=leg_only,
                    showlegend=True,
                    legendgroup=var,
                    # marker=dict(color='lightblue'),
                    line_shape="vh",
                    line=dict(
                        color=color_sel,
                        width=2,
                        # dash='dot'
                    ),
                )
            )
    
            monthly_fig.add_trace(
                go.Scatter(
                    x=df_basin_plot1_monthly.index,
                    y=(df_basin_plot1_monthly.vol_cfs).ravel().tolist(),
                    # x=year_list,
                    # y= (gdf.loc[gdf.nhm_id == hru_id_sel, year_list].values).ravel().tolist(),
                    mode="lines",
                    name=var,
                    visible=leg_only,
                    showlegend=False,
                    legendgroup=var,
                    # marker=dict(color='lightblue'),
                    line_shape="vh",
                    line=dict(
                        color=color_sel,
                        width=2,
                        # dash='dot'
                    ),
                )
            )
    
            daily_fig.add_trace(
                go.Scatter(
                    x=df_basin_plot1_daily.index,
                    y=(df_basin_plot1_daily.vol_cfs).ravel().tolist(),
                    # x=year_list,
                    # y= (gdf.loc[gdf.nhm_id == hru_id_sel, year_list].values).ravel().tolist(),
                    mode="lines",
                    name=var,
                    visible=leg_only,
                    showlegend=False,
                    legendgroup=var,
                    # marker=dict(color='lightblue'),
                    line_shape="vh",
                    line=dict(
                        color=color_sel,
                        width=2,
                        # dash='dot'
                    ),
                )
            )
    
        for t in annual_fig.data:
            fig.append_trace(t, row=1, col=1)
    
        for t in monthly_fig.data:
            fig.append_trace(t, row=2, col=1)
    
        for t in daily_fig.data:
            fig.append_trace(t, row=3, col=1)
    
        plotly.offline.plot(fig, filename=f"{plot_file_path}")
    
    return plot_file_path