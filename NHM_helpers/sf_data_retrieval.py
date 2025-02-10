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


def owrd_scraper(station_nbr, start_date, end_date):
    # f string the args into the urldf
    url = f"https://apps.wrd.state.or.us/apps/sw/hydro_near_real_time/hydro_download.aspx?station_nbr={station_nbr}&start_date={start_date}&end_date={end_date}&dataset=MDF&format=html"

    # open and decode the url
    resource = request.urlopen(url)
    content = resource.read().decode(resource.headers.get_content_charset())

    # Ugly parsing between pre tags
    # initializing substrings
    sub1 = "<pre>"
    sub2 = "</pre>"

    # getting index of substrings
    idx1 = content.index(sub1)
    idx2 = content.index(sub2)

    res = ""
    # getting elements in between
    for idx in range(idx1 + len(sub1), idx2):
        res = res + content[idx]

    # make and return the pandas df

    # NOTE:
    # Read in the csv file taking care to set the data types exactly. This is important for stability and functionality.
    # This should be done everytime the databases are read into this and future notebooks!

    col_names = [
        "station_nbr",
        "record_date",
        "mean_daily_flow_cfs",
        #'published_status',
        #'estimated',
        #'revised',
        #'download_date',
    ]
    col_types = [
        np.str_,
        np.str_,
        float,
        # np.str_,
        # np.str_,
        # float,
        # np.str_,
    ]
    cols = dict(
        zip(col_names, col_types)
    )  # Creates a dictionary of column header and datatype called below.

    df = pd.read_csv(StringIO(res), sep="\t", header=0, dtype=cols)

    return df

def create_OR_sf_df(control, model_dir, gages_df):

    start_date = pd.to_datetime(str(control.start_time)).strftime("%m/%d/%Y")
    end_date = pd.to_datetime(str(control.end_time)).strftime("%m/%d/%Y")
    owrd_cache_file = model_dir / "notebook_output_files" / "nc_files" / "owrd_cache.nc" #(eventually comment out)

    if owrd_cache_file.exists():
        with xr.open_dataset(owrd_cache_file) as owrd_ds:
            owrd_df = owrd_ds.to_dataframe()
            print(
                "Cached copy of OWRD data exists. To re-download the data, remove the cache file."
            )
        del owrd_ds

    else:
        print("Retrieving OWRD daily streamflow observations.")
        lst = []
        
        for ii in gages_df.index:
            lst.append(owrd_scraper(ii, start_date, end_date))

        if lst:
            owrd_df = pd.concat(
                lst
            )  # Converts the list of df's to a single df  maybe move this to the owrd scraper function
    
            # Reformat owrd_df headers and data types
            # Rename column headers
            field_map = {
                "station_nbr": "poi_id",
                "record_date": "time",
                "mean_daily_flow_cfs": "discharge",
                "station_name": "poi_name",
            }
            owrd_df.rename(columns=field_map, inplace=True)
    
            # Change the datatype for 'poi_id' and 'time'
            dtype_map = {"poi_id": str, "time": "datetime64[ns]"}
            owrd_df = owrd_df.astype(dtype_map)
    
            # Drop the columns we don't need
            drop_cols = ["download_date", "estimated", "revised", "published_status"]
            owrd_df.drop(columns=drop_cols, inplace=True)
    
            # Add new field(s): 'agency_id' and set to 'OWRD'
            owrd_df["agency_id"] = "OWRD"  # Creates tags for all OWRD daily streamflow data
    
            # Set multi-index for df
            owrd_df.set_index(["poi_id", "time"], inplace=True)
    
            # Write df as netcdf fine (.nc)
            owrd_ds = xr.Dataset.from_dataframe(owrd_df)
    
            # Set attributes for the variables
            owrd_ds["discharge"].attrs = {"units": "ft3 s-1", "long_name": "discharge"}
            owrd_ds["poi_id"].attrs = {
                "role": "timeseries_id",
                "long_name": "Point-of-Interest ID",
                "_Encoding": "ascii",
            }
            owrd_ds["agency_id"].attrs = {"_Encoding": "ascii"}
    
            # Set encoding (see 'String Encoding' section at https://crusaderky-xarray.readthedocs.io/en/latest/io.html)
            owrd_ds["poi_id"].encoding.update(
                {"dtype": "S15", "char_dim_name": "poiid_nchars"}
            )
    
            owrd_ds["time"].encoding.update(
                {
                    "_FillValue": None,
                    "standard_name": "time",
                    "calendar": "standard",
                    "units": "days since 1940-01-01 00:00:00",
                }
            )
    
            owrd_ds["agency_id"].encoding.update(
                {"dtype": "S5", "char_dim_name": "agency_nchars"}
            )
    
            # Add fill values to the data variables
            var_encoding = dict(_FillValue=netCDF4.default_fillvals.get("f4"))
    
            for cvar in owrd_ds.data_vars:
                if cvar not in ["agency_id"]:
                    owrd_ds[cvar].encoding.update(var_encoding)
    
            # add global attribute metadata
            owrd_ds.attrs = {
                "Description": "Streamflow data for PRMS",
                "FeatureType": "timeSeries",
            }
    
            # Write the dataset to a netcdf file
            print(
                f"OWRD daily streamflow observations retrieved for {len(owrd_df.index)}, writing data to {owrd_cache_file}."
            )
            owrd_ds.to_netcdf(owrd_cache_file)
        else:
            owrd_df = pd.DataFrame()
            
    return owrd_df


def ecy_scrape(station, ecy_years, ecy_start_date, ecy_end_date):
    ecy_df_list = []
    for ecy_year in ecy_years:
        url = f"https://apps.ecology.wa.gov/ContinuousFlowAndWQ/StationData/Prod/{station}/{station}_{ecy_year}_DSG_DV.txt"
        try:
            # The string that is to be searched
            key = "DATE"

            # Opening the file and storing its data into the variable lines
            with urlopen(url) as file:
                lines = file.readlines()

            # Going over each line of the file
            dateline = []
            for number, line in enumerate(lines, 1):

                # Condition true if the key exists in the line
                # If true then display the line number
                if key in str(line):
                    dateline.append(number)
                    # print(f'{key} is at line {number}')
            # df = pd.read_csv(url, skiprows=11, sep = '\s{3,}', on_bad_lines='skip', engine = 'python')  # looks for at least three spaces as separator
            df = pd.read_fwf(
                url, skiprows=dateline[0]
            )  # seems to handle formatting for No Data and blanks together, above option is thrown off by blanks
            # df['Day'] = pd.to_numeric(df['Day'], errors='coerce') # day col to numeric
            # df = df[df['Day'].notna()].astype({'Day': int}) #
            # df = df.drop('Day.1', axis=1)
            if len(df.columns) == 3:
                df.columns = ["time", "discharge", "Quality"]
            elif len(df.columns) == 4:
                df.columns = ["time", "utc", "discharge", "Quality"]
                df.drop("utc", axis=1, inplace=True)
            try:
                df.drop(
                    "Quality", axis=1, inplace=True
                )  # drop quality for now, might use to filter later
            except KeyError:
                print(f"no Quality for {station} {ecy_year}")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"])
            df["poi_id"] = station
            df["discharge"] = pd.to_numeric(df["discharge"], errors="coerce")
            # specify data types
            dtype_map = {"poi_id": str, "time": "datetime64[ns]"}
            df = df.astype(dtype_map)

            df.set_index(["poi_id", "time"], inplace=True)
            # next two lines are new if this breaks...
            idx = pd.IndexSlice
            df = df.loc[
                idx[:, ecy_start_date:ecy_end_date], :
            ]  # filters to the date range
            df["agency_id"] = "ECY"

            ecy_df_list.append(df)
            print(f"good year {ecy_year}")
            print(url)
        except HTTPError:
            pass
        except ValueError as ex:
            print(ex)
            print(ecy_year)
    if len(df) != 0:
        temp_df = pd.concat(ecy_df_list)
        # ecy_df["discharge_cfs"] = pd.to_numeric(ecy_df["discharge_cfs"], errors = 'coerce')
        # maybe inster the rest of the df formatting here:

        return temp_df
    else:
        print(f"No data for station {station} for data range {ecy_years}.")
        return None

def create_ecy_sf_df(control, model_dir, gages_df):
    """Check the gages_df for ECY gages."""
    ecy_gages = []
    gage_list = gages_df.index.to_list()
    for i in gage_list:
        # if len(i) == 6 and i.matches("^[A-Z]{1}\\d{3}")
        if len(i) == 6 and i[0:2].isdigit() and i[2].isalpha() and i[4:6].isdigit():
            ecy_gages.append(i)
        else:
            pass

    if ecy_gages:
        ecy_df = pd.DataFrame()
        ecy_df_list = []
        ecy_cache_file = (
            model_dir / "notebook_output_files" / "nc_files" / "ecy_cache.nc"
        )# This too will go away eventually and so will the if loop below

        if ecy_cache_file.exists():
            with xr.open_dataset(ecy_cache_file) as ecy_ds:
                ecy_df = ecy_ds.to_dataframe()
            print(
                "Cached copy of ECY data exists. To re-download the data, remove the cache file."
            )
            del ecy_ds
        else:
            # Get start and end dates for ecy_scraper:
            ecy_start_date = pd.to_datetime(str(control.start_time)).strftime(
                "%Y-%m-%d"
            )
            ecy_end_date = pd.to_datetime(str(control.end_time)).strftime("%Y-%m-%d")

            # Get WY range in years (add 1 year to date range because ecy is water year, add another year because range is not inclusive)
            ecy_years = range(
                pd.to_datetime(str(control.start_time)).year,
                pd.to_datetime(str(control.end_time)).year + 2,
            )

            # 2) Go get the data
            for ecy_gage_id in ecy_gages:
                try:
                    ecy_df_list.append(
                        ecy_scrape(ecy_gage_id, ecy_years, ecy_start_date, ecy_end_date)
                    )

                except UnboundLocalError:
                    print(f"No data for {ecy_gage_id}")
                    pass

            ecy_df = pd.concat(
                ecy_df_list
            )  # Converts the list of ecy gage df's to a single df

            # set the multiIndex
            # ecy_df.set_index(['poi_id', 'time'], inplace=True)

            ecy_df = ecy_df[
                ~ecy_df.index.duplicated(keep="first")
            ]  # overlap in ecy records for 10-1, drop duplicates for xarray

            # Add new fields
            ecy_df["agency_id"] = (
                "ECY"  # Creates tags for all ECY daily streamflow data
            )

            # Write ecy_df as netcdf (.nc) file
            ecy_ds = xr.Dataset.from_dataframe(ecy_df)

            # Set attributes for the variables
            ecy_ds["discharge"].attrs = {"units": "ft3 s-1", "long_name": "discharge"}
            ecy_ds["poi_id"].attrs = {
                "role": "timeseries_id",
                "long_name": "Point-of-Interest ID",
                "_Encoding": "ascii",
            }
            ecy_ds["agency_id"].attrs = {"_Encoding": "ascii"}

            # Set encoding
            # See 'String Encoding' section at https://crusaderky-xarray.readthedocs.io/en/latest/io.html
            ecy_ds["poi_id"].encoding.update(
                {"dtype": "S15", "char_dim_name": "poiid_nchars"}
            )

            ecy_ds["time"].encoding.update(
                {
                    "_FillValue": None,
                    "standard_name": "time",
                    "calendar": "standard",
                    "units": "days since 1940-01-01 00:00:00",
                }
            )

            ecy_ds["agency_id"].encoding.update(
                {"dtype": "S5", "char_dim_name": "agency_nchars"}
            )

            # Add fill values to the data variables
            var_encoding = dict(_FillValue=netCDF4.default_fillvals.get("f4"))

            for cvar in ecy_ds.data_vars:
                if cvar not in ["agency_id"]:
                    ecy_ds[cvar].encoding.update(var_encoding)

            # add global attribute metadata
            ecy_ds.attrs = {
                "Description": "Streamflow data for PRMS",
                "FeatureType": "timeSeries",
            }

            # Write the dataset to a netcdf file
            ecy_ds.to_netcdf(ecy_cache_file)
    else:
        ecy_df = pd.DataFrame()
        pass
    return ecy_df

def create_nwis_sf_df(control, model_dir, gages_df):
    nwis_cache_file = (model_dir / "notebook_output_files" / "nc_files" / "nwis_cache.nc")
    
    if nwis_cache_file.exists():
        with xr.open_dataset(nwis_cache_file) as NWIS_ds:
            NWIS_df = NWIS_ds.to_dataframe()
            print(
                "Cached copy of NWIS data exists. To re-download the data, remove the cache file."
            )
            del NWIS_ds
    else:
        output_netcdf_filename = model_dir / "notebook_output_files" / "nc_files" / "sf_efc.nc"
        """
        This function returns a dataframe of mean daily streamflow data from NWIS using gages listed in the gages_df, 
        for the period of record defined in the NHM model control file control.default.bandit.
        Note: all gages in the gages_df that are not found in NWIS will be ignored.
        """
        
        nwis_start = pd.to_datetime(str(control.start_time)).strftime("%Y-%m-%d")
        nwis_end = pd.to_datetime(str(control.end_time)).strftime("%Y-%m-%d")
        NWIS_tmp = []
    
        with Progress() as progress:
            task = progress.add_task("[red]Downloading...", total=len(gages_df))
            err_list = []
            for ii in gages_df.index:
                try:
                    NWISgage_data = nwis.get_record(
                        sites=(str(ii)), service="dv", start=nwis_start, end=nwis_end
                    )
                    NWIS_tmp.append(NWISgage_data)
                except ValueError:
                    err_list.append(ii)
                    # con.print(f"Gage id {ii} not found in NWIS.")
                    pass
                progress.update(task, advance=1)
    
        NWIS_df = pd.concat(NWIS_tmp)
        con.print(
            f"No data for these {len(err_list)} gages, {err_list} were found in NWIS."
        )
        # we only need site_no and discharge (00060_Mean)
        NWIS_df = NWIS_df[["site_no", "00060_Mean"]].copy()
        NWIS_df["agency_id"] = "USGS"
    
        NWIS_df = NWIS_df.tz_localize(None)
        NWIS_df.reset_index(inplace=True)
    
        # rename cols to match other df
        NWIS_df.rename(
            columns={
                "datetime": "time",
                "00060_Mean": "discharge",
                "site_no": "poi_id",
            },
            inplace=True,
        )
    
        NWIS_df.set_index(["poi_id", "time"], inplace=True)

        #### Write the .nc file
        # Reformat data types
        # Change the datatype for 'poi_id' and 'time'
        # dtype_map = {"poi_id": str, "time": "datetime64[ns]"}
        # NWIS_df = NWIS_df.astype(dtype_map)
        
        # Write df as netcdf fine (.nc)
        NWIS_ds = xr.Dataset.from_dataframe(NWIS_df)
        
        # Set attributes for the variables
        NWIS_ds["discharge"].attrs = {"units": "ft3 s-1", "long_name": "discharge"}
        NWIS_ds["poi_id"].attrs = {
            "role": "timeseries_id",
            "long_name": "Point-of-Interest ID",
            "_Encoding": "ascii",
        }
        NWIS_ds["agency_id"].attrs = {"_Encoding": "ascii"}
        
        # Set encoding (see 'String Encoding' section at https://crusaderky-xarray.readthedocs.io/en/latest/io.html)
        NWIS_ds["poi_id"].encoding.update(
            {"dtype": "S15", "char_dim_name": "poiid_nchars"}
        )
        
        NWIS_ds["time"].encoding.update(
            {
                "_FillValue": None,
                "standard_name": "time",
                "calendar": "standard",
                "units": "days since 1940-01-01 00:00:00",
            }
        )
        
        NWIS_ds["agency_id"].encoding.update(
            {"dtype": "S5", "char_dim_name": "agency_nchars"}
        )
        
        # Add fill values to the data variables
        var_encoding = dict(_FillValue=netCDF4.default_fillvals.get("f4"))
        
        for cvar in NWIS_ds.data_vars:
            if cvar not in ["agency_id"]:
                NWIS_ds[cvar].encoding.update(var_encoding)
        
        # add global attribute metadata
        NWIS_ds.attrs = {
            "Description": "Streamflow data for PRMS",
            "FeatureType": "timeSeries",
        }
        
        # Write the dataset to a netcdf file
        print(
            f"NWIS daily streamflow observations retrieved, writing data to {nwis_cache_file}."
        )
        NWIS_ds.to_netcdf(nwis_cache_file)
        
    return NWIS_df