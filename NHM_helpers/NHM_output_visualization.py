# Import Notebook Packages
import warnings
from urllib import request
from urllib.request import urlopen
from urllib.error import HTTPError

import re
from io import StringIO
import os
import re
import sys
from collections import OrderedDict

os.environ["USE_PYGEOS"] = "0"

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

import datetime as dt
#from datetime import datetime

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

from NHM_helpers.efc import *
from NHM_helpers import NHM_helpers as helpers
from NHM_helpers.NHM_Assist_utilities import *
from NHM_helpers.NHM_hydrofabric import *

import jupyter_black

jupyter_black.load()


def retrieve_hru_output_info(out_dir, water_years):

    with xr.open_dataset(out_dir / "net_ppt.nc") as model_output:

        if water_years:
            october_days = model_output.sel(time=model_output.time.dt.month == 10)
            october_firsts = october_days.sel(time=october_days.time.dt.day == 1)
            plot_start_date = pd.to_datetime(october_firsts.time.values[0]).strftime(
                "%Y-%m-%d"
            )

            september_days = model_output.sel(time=model_output.time.dt.month == 9)
            september_lasts = september_days.sel(time=september_days.time.dt.day == 30)
            plot_end_date = pd.to_datetime(september_lasts.time.values[-1]).strftime(
                "%Y-%m-%d"
            )

            output = model_output.sel(time=slice(plot_start_date, plot_end_date))

            year_list = list(set(((output.time.dt.year).values).ravel().tolist()))
            year_list.remove(
                year_list[0]
            )  # Note: remove first year from the list to show available WY's in the data
            year_list = [str(x) for x in year_list]

        else:
            january_days = model_output.sel(time=model_output.time.dt.month == 1)
            january_firsts = january_days.sel(time=january_days.time.dt.day == 1)
            plot_start_date = pd.to_datetime(january_firsts.time.values[0]).strftime(
                "%Y-%m-%d"
            )

            december_days = model_output.sel(time=model_output.time.dt.month == 12)
            december_lasts = december_days.sel(time=december_days.time.dt.day == 31)
            plot_end_date = pd.to_datetime(december_lasts.time.values[-1]).strftime(
                "%Y-%m-%d"
            )

            output = model_output.sel(time=slice(plot_start_date, plot_end_date))

            year_list = list(set(((output.time.dt.year).values).ravel().tolist()))
            year_list = [str(x) for x in year_list]

    
    """
    Make alist of all the output variable files.
    """
    output_varfile_list = set([vv.stem for vv in out_dir.glob("*.nc")])
    
    """
    Remove postprocessed custom variables. These will be used in notebook 6.
    """
    output_varfile_list = list(
            output_varfile_list - {"seg_outflow", "hru_streamflow_out"})

    """ 
    Make a variable list for just those output variables dimensioned by nhm_id (hru).
    """
    output_var_list =[]
    for var in output_varfile_list:
        with xr.open_dataset(out_dir / f"{var}.nc") as output:
            if (output[var].dims == ('time', 'nhm_id')) & (output[var].units == "inches"):
                output_var_list.append(var)
            else:
                pass
            del output
    
    del model_output

    
    # for var in output_varfile_list:
    #     with xr.open_dataset(out_dir / f"{var}.nc") as output:
    #         if output[var].units == "inches":
    #             output_var_list.append(var)
    #         else:
    #             print(f"{var.dims}")
    #         del output

    return plot_start_date, plot_end_date, year_list, output_var_list


def create_sum_var_dataarrays(
    out_dir,
    output_var_sel,
    plot_start_date,
    plot_end_date,
    water_years,
):
    """
    This is meant to return the data arrays for daily, monhtly and annual (WY or cal year) for the selected parameter.
    These are the data arrays used to make the dataframes in the notebook.
    More development work may occur in the future to increase data handling efficiency in this notebook.
    """

    with xr.load_dataarray(out_dir / f"{output_var_sel}.nc") as da:
        # these machinations are to keep downstream things as they were before some refactoring
        da = da.to_dataset().rename_dims({"nhm_id": "nhru"})[da.name]
        var_units = da.units
        var_desc = da.desc
        var_daily = da.sel(time=slice(plot_start_date, plot_end_date))
        sum_var_monthly = var_daily.resample(time="m").sum()

        if water_years:
            sum_var_annual = var_daily.resample(time="A-SEP").sum()
        else:
            sum_var_annual = var_daily.resample(time="y").sum()
    del da

    return var_daily, sum_var_monthly, sum_var_annual, var_units, var_desc

def create_mean_var_dataarrays(
    out_dir,
    output_var_sel,
    plot_start_date,
    plot_end_date,
    water_years,
):
    """
    This is meant to return the data arrays for daily, monhtly and annual (WY or cal year) for the selected parameter.
    These are the data arrays used to make the dataframes in the notebook.
    More development work may occur in the future to increase data handling efficiency in this notebook.
    """

    with xr.load_dataarray(out_dir / f"{output_var_sel}.nc") as da:
        # these machinations are to keep downstream things as they were before some refactoring
        da = da.to_dataset().rename_dims({"nhm_id": "nhru"})[da.name]
        var_units = da.units
        var_desc = da.desc
        var_daily = da.sel(time=slice(plot_start_date, plot_end_date))
        mean_var_monthly = var_daily.resample(time="m").mean()

        if water_years:
            mean_var_annual = var_daily.resample(time="A-SEP").mean()
        else:
            mean_var_annual = var_daily.resample(time="y").mean()
    del da

    return var_daily, mean_var_monthly, mean_var_annual, var_units, var_desc


def create_sum_var_annual_gdf(
    output_var_sel,
    sum_var_annual,
    hru_gdf,
    year_list,
    
):

    df_output_var_annual = sum_var_annual.copy().to_dataframe(
        dim_order=["time", "nhru"]
    )
    df_output_var_annual.reset_index(inplace=True, drop=False)
    df_output_var_annual.set_index(
        "nhm_id", inplace=True, drop=True
    )  # resets the index to that new value and type

    df_output_var_annual["year"] = pd.DatetimeIndex(df_output_var_annual["time"]).year
    df_output_var_annual.year = df_output_var_annual.year.astype(str)
    df_output_var_annual.rename(
        columns={output_var_sel: "output_var"}, inplace=True
    )  # Rename the column to a general heading for later code

    df_output_var_annual.drop(columns=["time", "nhru"], inplace=True)

    table = pd.pivot_table(
        df_output_var_annual,
        values="output_var",
        index=["nhm_id"],
        columns=["year"],
    ).round(2)
    table.reset_index(inplace=True, drop=False)
    # output_filename = f"{shapefile_dir}/{output_var_sel}_annual_{var_units}.csv"  # Writes gpd GeoDataFrame our t as a shapefile for fun
    # table.to_csv(output_filename)

    """
    Merge in the geometry from the geodatabase with the dataframe.
    Keep the params in the gdf as well for future plotting calls--eventually will omit.
    """
    gdf_output_var_annual = hru_gdf.merge(table, on="nhm_id")
    gdf_output_var_annual.drop(
        columns=["hru_lat", "hru_lon", "hru_segment_nhm", "model_idx"], inplace=True
    )

    """
    Add the mean annual value for the model simulation period, year_list
    """
    for nhm_id in gdf_output_var_annual:
        gdf_output_var_annual["mean_annual"] = gdf_output_var_annual[year_list].mean(
            axis=1
        )

    """
    Determine the minimum and maximum values for annual recharge for the legend value bar.
    """
    value_min = gdf_output_var_annual[year_list].min().min()
    value_max = gdf_output_var_annual[year_list].max().max()

    if value_min == value_max:
        con.print(
            f"The min and max values are both {value_min}.",
            f"\nThe values of {value_min-.001} and {value_max +.001} will be used to set a values in the legend bar when mapping the annual data.",
        )
    else:
        con.print(
            f"The minimum and maximum annual values are: {value_min}, {value_max}.",
            "\nThis will be used to set a values in the legend bar when mapping the annual data.",
        )

    return gdf_output_var_annual, value_min, value_max


####################################################
def create_sum_var_annual_df(
    hru_gdf,
    poi_df,
    param_filename,
    plot_start_date,
    plot_end_date,
    sum_var_annual,
    output_var_sel,
):

    # Create a dataframe of ANNUAL recharge values for each HRU
    hru_area_df = hru_gdf[["hru_area", "nhm_id"]].set_index(
        "nhm_id", drop=True
    )  # Consider a dictionary from the par file and using .map() instead of merge

    sum_var_annual_df = sum_var_annual.to_dataframe(dim_order=["time", "nhru"])
    sum_var_annual_df.reset_index(inplace=True, drop=False)

    sum_var_annual_df = sum_var_annual_df.merge(
        hru_area_df, how="left", right_index=True, left_on="nhm_id"
    )

    # # # add the HRU area to the dataframe
    # for idx, row in output_var_annual_df.iterrows():
    #     output_var_annual_df.loc[idx, "hru_area"] = hru_gdf.loc[
    #         hru_gdf.nhm_id == row.nhm_id, "hru_area"
    #     ].item()

    # Add recharge volume, inch-acres, to the dataframe
    sum_var_annual_df["vol_inch_acres"] = (
        sum_var_annual_df[output_var_sel] * sum_var_annual_df["hru_area"]
    )
   
    # Drop unneeded columns
    sum_var_annual_df.drop(columns=["nhru"], inplace=True)
  

    return sum_var_annual_df