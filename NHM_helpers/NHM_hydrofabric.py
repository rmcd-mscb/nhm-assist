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

# from datetime import datetime

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


def create_hru_gdf(NHM_dir,
    model_dir,
    GIS_format,
    param_filename,
    nhru_params,
    nhru_nmonths_params,
):
    """
    Creates hru gdf for selected hru parameters from the parameter file.
    Selected in notebook 0a.

    This reads three layers (nhru, amd nsegments) into GeoPandas as DataFrames (_df) and if geometry is included (_gdb).
    Note:</b> Layer npoigages includes the poi gages that were included in the model and are limited. 
    Since pois will be added to the model paramter file, we provide another method of for retrieving poi metadata, such as 
    latitude (lat) and longitude (lon), for pois listed in the parameter file that uses NWIS and a supplimental gage ref
    table for gages that do not occur in NWIS. Locations may NOT be located exactly on the NHM segment. The POIs' assigned
    segment is displayed in the popup window when the gage icon is clicked.
    """

    # List of bynhru parameters to retrieve for the Notebook interactive maps.
    hru_params = [
        "hru_lat",  # the latitude if the hru centroid
        "hru_lon",  # the longitude if the hru centroid
        "hru_area",
        "hru_segment_nhm",  # The nhm_id of the segment recieving flow from the HRU
    ]
    cal_hru_params = nhru_params + nhru_nmonths_params
    gdb_hru_params = hru_params + nhru_params + nhru_nmonths_params

    """
    Projections are ascribed geometry from the HRUs geodatabase (GIS). 
    The NHM uses the NAD 1983 USGS Contiguous USA Albers projection EPSG# 102039. 
    The geometry units of this projection are not useful for many notebook packages. 
    The geodatabases are reprojected to World Geodetic System 1984.

    Options:
        crs = 3857, WGS 84 / Pseudo-Mercator - Spherical Mercator, Google Maps, OpenStreetMap, Bing, ArcGIS, ESRI.
        *crs = 4326, WGS 84 - WGS84 - World Geodetic System 1984, used in GPS
    """
    crs = 4326

    """
    Loading some pyPRMS helpers for parameter metadata: units, descriptions, etc.
    """
    prms_meta = MetaData(version=5, verbose=False).metadata  # loads metadata functions for pyPRMS
    pdb = ParameterFile(param_filename, metadata=prms_meta, verbose=False)  # loads parmaeterfile functions for pyPRMS
    

    if GIS_format == ".gpkg":
        hru_gdb = gpd.read_file(
            f"{model_dir}/GIS/model_layers.gpkg", layer="nhru"
        )  # Reads HRU file to Geopandas.

    if GIS_format == ".shp":
        hru_gdb = gpd.read_file(
            f"{model_dir}/GIS/model_nhru.shp"
        )  # Reads HRU file to Geopandas.
        hru_gdb = hru_gdb.set_index("nhm_id", drop=False).fillna(
            0
        )  # Set an index for HRU geodatabase.
        hru_gdb.index.name = "index"  # Index column must be renamed of the hru

    hru_gdb = hru_gdb.to_crs(crs)  # reprojects to the defined crs projection

    # Create a dataframe for parameter values
    first = True
    for vv in gdb_hru_params:
        if (
            first
        ):  # this creates the first iteration for the following iterations to concantonate to
            df = pdb.get_dataframe(vv)
            first = False
        else:
            df = pd.concat([df, pdb.get_dataframe(vv)], axis=1)  # , ignore_index=True)

    df.reset_index(inplace=True)
    df["model_idx"] = (
        df.index + 1
    )  #'model_idx' created here is the order of the parameters in the parameter file.
    # df

    # Join the HRU params values to the HRU geodatabase using Merge
    hru_gdb = pd.merge(df, hru_gdb, on="nhm_id")

    # Create a Goepandas GeoDataFrame for the HRU geodatabase
    hru_gdf = gpd.GeoDataFrame(hru_gdb, geometry="geometry")


    """
    NHM Calibration Levels for HRUs: (those hrus calibrated in byHW and byHWobs parts)
    
    HW basins were descritized using a drainage area maxiumum and minimum; HW HRUs, segments, outlet segment, and drainage area 
    available. 
    
    Gages used in byHWobs calibration, Part 3, for selected headwaters are also provided here.  
    
    FILES AND TABLES IN THIS SECTION ARE CONUS COVERAGE and will be subsetted later.
    """

    #### READ table (.csv) of HRU calibration level file
    hru_cal_levels_df = pd.read_csv(f"{NHM_dir}/nhm_v1_1_HRU_cal_levels.csv").fillna(0)
    hru_cal_levels_df["hw_id"] = hru_cal_levels_df.hw_id.astype("int64")
    print(hru_cal_levels_df["nhm_id"])

    hru_cal_levels_df = pd.merge(
    hru_cal_levels_df, hru_gdf, right_on="nhm_id", left_on="nhm_id"
    )
    hru_cal_levels_gdf = gpd.GeoDataFrame(
        hru_cal_levels_df, geometry="geometry"
    )  # Creates a Geopandas GeoDataFrame
    hru_cal_levels_gdf["nhm_id"] = hru_cal_levels_gdf["nhm_id"].astype(str)
    hru_cal_levels_gdf["hw_id"] = hru_cal_levels_gdf["hw_id"].astype(str)

    hru_gdf = hru_cal_levels_gdf.copy()
    
    print(
        "The number of HRUs in the byHRU calibration is",
        hru_cal_levels_gdf[hru_cal_levels_gdf["level"] > 0]["level"].count(),
    )
    print(
        "The number of HRUs in the byHW calibration is",
        hru_cal_levels_gdf[hru_cal_levels_gdf["level"] > 1]["level"].count(),
    )
    print(
        "The number of HRUs in the byHWobs calibration is",
        hru_cal_levels_gdf[hru_cal_levels_gdf["level"] > 2]["level"].count(),
    )

    return hru_gdf


def create_segment_gdf(
    model_dir,
    GIS_format,
    param_filename,
):
    """
    Creates segment gdf for selected segment parameters from the parameter file.
    Selected in notebook 0a.
    """

    # List of parameters values to retrieve for the segments.
    seg_params = ["tosegment_nhm", "tosegment", "seg_length", "obsin_segment"]

    """
    Projections are ascribed geometry from the HRUs geodatabase (GIS). 
    The NHM uses the NAD 1983 USGS Contiguous USA Albers projection EPSG# 102039. 
    The geometry units of this projection are not useful for many notebook packages. 
    The geodatabases are reprojected to World Geodetic System 1984.

    Options:
        crs = 3857, WGS 84 / Pseudo-Mercator - Spherical Mercator, Google Maps, OpenStreetMap, Bing, ArcGIS, ESRI.
        *crs = 4326, WGS 84 - WGS84 - World Geodetic System 1984, used in GPS
    """
    crs = 4326

    """
    Loading some pyPRMS helpers for parameter metadata: units, descriptions, etc.
    """
    prms_meta = MetaData(version=5, verbose=False).metadata  # loads metadata functions for pyPRMS
    pdb = ParameterFile(param_filename, metadata=prms_meta, verbose=False)  # loads parmaeterfile functions for pyPRMS
    

    if GIS_format == ".gpkg":
        seg_gdb = gpd.read_file(
            f"{model_dir}/GIS/model_layers.gpkg", layer="nsegment"
        ).fillna(
            0
        )  # Reads segemnt file to Geopandas.

    if GIS_format == ".shp":
        seg_gdb = gpd.read_file(f"{model_dir}/GIS/model_nsegment.shp").fillna(0)
        seg_gdb = seg_gdb.set_index(
            "nhm_seg", drop=False
        )  # Set an index for segment geodatabase(GIS)
        seg_gdb.index.name = "index"  # Index column must be renamed of the hru

    seg_gdb = seg_gdb.to_crs(crs)  # reprojects to the defined crs projection

    # Create a dataframe for parameter values
    first = True
    for vv in seg_params:
        if first:
            df = pdb.get_dataframe(vv)
            first = False
        else:
            df = pd.concat([df, pdb.get_dataframe(vv)], axis=1)  # , ignore_index=True)

    df.reset_index(inplace=True)
    df["model_idx"] = df.index + 1
    df.index.name = "index"  # Index column must be renamed

    # Join the HRU params values to the HRU geodatabase using Merge
    seg_gdb = pd.merge(df, seg_gdb, on="nhm_seg")

    # Create a Goepandas GeoDataFrame for the HRU geodatabase
    seg_gdf = gpd.GeoDataFrame(seg_gdb, geometry="geometry")

    return seg_gdf

from NHM_helpers.NHM_Assist_utilities import fetch_nwis_gage_info
def create_poi_df(
    model_dir,
    param_filename,
    control_file_name,
    hru_gdf,
    nwis_gages_aoi,
    gages_file,
):

    """
    Loading some pyPRMS helpers for parameter metadata: units, descriptions, etc.
    """
    prms_meta = MetaData(version=5, verbose=False).metadata  # loads metadata functions for pyPRMS
    pdb = ParameterFile(param_filename, metadata=prms_meta, verbose=False)  # loads parmaeterfile functions for pyPRMS
    
    """
    Create a dataframe of all POI-related parameters from the parameter file.
    """

    poi = pdb["poi_gage_id"].as_dataframe
    poi = poi.merge(
        pdb["poi_gage_segment"].as_dataframe, left_index=True, right_index=True
    )
    poi = poi.merge(pdb["poi_type"].as_dataframe, left_index=True, right_index=True)
    poi = poi.merge(
        pdb["nhm_seg"].as_dataframe, left_on="poi_gage_segment", right_index=True
    )

    control = pws.Control.load_prms(
        pl.Path(model_dir / control_file_name), warn_unused_options=False
    )  # loads the control file for pywatershed functions
    
    st_date = control.start_time
    en_date = control.end_time

    """
    Projections are ascribed geometry from the HRUs geodatabase (GIS). 
    The NHM uses the NAD 1983 USGS Contiguous USA Albers projection EPSG# 102039. 
    The geometry units of this projection are not useful for many notebook packages. 
    The geodatabases are reprojected to World Geodetic System 1984.

    Options:
        crs = 3857, WGS 84 / Pseudo-Mercator - Spherical Mercator, Google Maps, OpenStreetMap, Bing, ArcGIS, ESRI.
        *crs = 4326, WGS 84 - WGS84 - World Geodetic System 1984, used in GPS
    """
    crs = 4326

    """
    Make a list if the HUC2 region(s) the subbasin intersects for NWIS queries.
    """
    huc2_gdf = gpd.read_file("./data_dependencies/HUC2/HUC2.shp").to_crs(crs)
    model_domain_regions = list((huc2_gdf.clip(hru_gdf).loc[:]["huc2"]).values)

    """
    Create a dataframe for poi_gages from the parameter file with NWIS gage information data.
    """
    poi = poi.merge(
        nwis_gages_aoi, left_on="poi_gage_id", right_on="poi_id", how="left"
    )
    poi_df = pd.DataFrame(poi)  # Creates a Pandas DataFrame

    """
    Updates the poi_df with user altered metadata in the gages.csv file, if present
    """

    if gages_file.exists():
        for idx, row in poi_df.iterrows():
            if pd.isnull(row["poi_id"]):
                new_poi_id = row["poi_gage_id"]
                new_lat = gages_df.loc[
                    gages_df.index == row["poi_gage_id"], "latitude"
                ].values[0]
                new_lon = gages_df.loc[
                    gages_df.index == row["poi_gage_id"], "longitude"
                ].values[0]
                new_poi_agency = gages_df.loc[
                    gages_df.index == row["poi_gage_id"], "poi_agency"
                ].values[0]
                new_poi_name = gages_df.loc[
                    gages_df.index == row["poi_gage_id"], "poi_name"
                ].values[0]

                poi_df.loc[idx, "latitude"] = new_lat
                poi_df.loc[idx, "longitude"] = new_lon
                poi_df.loc[idx, "poi_id"] = new_poi_id
                poi_df.loc[idx, "poi_agency"] = new_poi_agency
                poi_df.loc[idx, "poi_name"] = new_poi_name

    else:
        pass

    
    return poi_df

def create_default_gages_file(
    model_dir,
    nwis_gages_aoi,
    poi_df,
):
    """
    Create default_gages.csv for your model extraction.
    NHM-Assist notebooks will display gages using the default gages file (default_gages.csv), if a modified gages file (gages.csv) is lacking.
    By default, this file will be composed of:

        1) the gages listed in the parameter file (poi_gages), and
        2) all streamflow gages from NWIS in the model domain that have at least 90 days of streamflow obervations.

    Note: all metadata in the default gages file is from NWIS if the gage is found NWIS.
    Note: Time-series data for streamflow observations will be collected using this gage list and the time range in the control file.
    Note: Initially, all gages listed in the parameter file exist in NWIS.
    """

    """
    First, select only gages from the gages file NOT in NWIS (to preserve NWIS metadata values in default_gages.csv)
    """
    non_NWIS_gages_from_poi_df = poi_df.loc[poi_df["poi_agency"] != "USGS"]
    non_NWIS_gages_from_poi_df.drop(
        columns=["poi_id", "nhm_seg", "poi_gage_segment", "poi_type"], inplace=True
    )
    non_NWIS_gages_from_poi_df.rename(columns={"poi_gage_id": "poi_id"}, inplace=True)

    """
    Projections are ascribed geometry from the HRUs geodatabase (GIS). 
    The NHM uses the NAD 1983 USGS Contiguous USA Albers projection EPSG# 102039. 
    The geometry units of this projection are not useful for many notebook packages. 
    The geodatabases are reprojected to World Geodetic System 1984.

    Options:
        crs = 3857, WGS 84 / Pseudo-Mercator - Spherical Mercator, Google Maps, OpenStreetMap, Bing, ArcGIS, ESRI.
        *crs = 4326, WGS 84 - WGS84 - World Geodetic System 1984, used in GPS
    """
    crs = 4326

    sta_file_col_order = [
        "poi_id",
        "poi_agency",
        "poi_name",
        "latitude",
        "longitude",
        "drainage_area",
        "drainage_area_contrib",
        #'nhm_seg', 'poi_gage_segment', 'poi_type'
    ]
    if pd.isnull(poi_df["poi_agency"]).values.any():
        temp = pd.concat(
            [nwis_gages_aoi, non_NWIS_gages_from_poi_df], ignore_index=True
        )
        temp2 = temp[sta_file_col_order]

    else:
        temp = nwis_gages_aoi.copy()
        temp2 = temp[sta_file_col_order]

    default_gages_file = model_dir / "default_gages.csv"
    temp2.to_csv(default_gages_file, index=False)

    return default_gages_file

def read_gages_file(
    model_dir,
    poi_df,
    nwis_gages_file,
    gages_file,
):

    """
    Read modified gages file.
    If there are gages in the parameter file that are not in NWIS (USGS gages), then latitude, longitude, and poi_name must be provided from another source,
    and appended to the "default_gages.csv" file. Once editing is complete, that file can be renamed "gages.csv"and will be used as the gages file. 
    If NO gages.csv is made, the default_gages.csv will be used.
    """
    default_gages_file = model_dir / "default_gages.csv"

    # Read in station file columns needed (You may need to tailor this to the particular file.
    col_names = [
        "poi_id",
        "poi_agency",
        "poi_name",
        "latitude",
        "longitude",
        "drainage_area",
        "drainage_area_contrib",
    ]
    col_types = [np.str_, np.str_, np.str_, float, float, float, float]
    cols = dict(
        zip(col_names, col_types)
    )  # Creates a dictionary of column header and datatype called below.

    if gages_file.exists():

        gages_df = pd.read_csv(gages_file)

        # Make poi_id the index
        gages_df.set_index("poi_id", inplace=True)
        exotic_gages = gages_df.loc[gages_df["poi_agency"] != "USGS"]
        gages_agencies_txt = ", ".join(
            f"{item}" for item in list(set(gages_df.poi_agency))
        )

        exotic_pois = poi_df.loc[poi_df["poi_agency"] != "USGS"]
        pois_agencies_txt = ", ".join(
            f"{item}" for item in list(set(poi_df.poi_agency))
        )

        con.print(
            f"[bold]Create hydrofabric files:\n",
            f"\nNHM-Assist notebooks will display gages using the modified gages file (gages.csv) that includes {len(gages_df)} gages managed by {gages_agencies_txt}.",
            f"Only {len(poi_df.index)} gages managed by {pois_agencies_txt} are found in the parameter file.",
        )

        """
        Checks the gages_df for missing meta data.
        """
        columns = ["latitude", "longitude", "poi_name", "poi_agency"]
        for item in columns:
            if pd.isnull(gages_df[item]).values.any():
                subset = gages_df.loc[pd.isnull(gages_df[item])]
                con.print(
                    f"The gages.csv is missing {item} data for {len(subset)} gages. Add missing data to the file and rename gages.csv."
                )
            else:
                pass
    else:
        gages_df = pd.read_csv(default_gages_file, dtype=cols)

        # Make poi_id the index
        gages_df.set_index("poi_id", inplace=True)
        exotic_gages = gages_df.loc[gages_df["poi_agency"] != "USGS"]
        gages_agencies_txt = ", ".join(
            f"{item}" for item in list(set(gages_df.poi_agency))
        )

        exotic_pois = poi_df.loc[poi_df["poi_agency"] != "USGS"]
        pois_agencies_txt = ", ".join(
            f"{item}" for item in list(set(poi_df.poi_agency))
        )

        con.print(
            f"[bold]Create hydrofabric files:\n",
            f"\nNHM-Assist notebooks will display gages using the default gages file (default_gages.csv) that includes {len(gages_df)} gages managed by {gages_agencies_txt}.",
            f"Only {len(poi_df.index)} gages managed by {pois_agencies_txt} are found in the parameter file.",
        )

        """
        Checks the gages_df for missing meta data.
        """
        columns = ["latitude", "longitude", "poi_name", "poi_agency"]
        for item in columns:
            if pd.isnull(gages_df[item]).values.any():
                subset = gages_df.loc[pd.isnull(gages_df[item])]
                con.print(
                    f"The default_gages.csv is missing {item} data for {len(subset)} gages. Add missing data to the file and rename gages.csv."
                )
            else:
                pass
    return gages_df

