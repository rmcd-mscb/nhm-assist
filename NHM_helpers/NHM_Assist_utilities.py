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

# List of bynhru parameters to retrieve for the Notebook interactive maps.
hru_params = [
    "hru_lat",  # the latitude if the hru centroid
    "hru_lon",  # the longitude if the hru centroid
    "hru_area",
    "hru_segment_nhm",  # The nhm_id of the segment recieving flow from the HRU
]

def bynhru_parameter_list(param_filename):
    pardat = pws.parameters.PrmsParameters.load(param_filename)
    bynhru_params = []
    for par in list(pardat.parameters.keys()):
        kk = list(pws.meta.parameters[par]["dims"])
        if kk == ["nhru"]:
            bynhru_params.append(par)
        else:
            pass
    return bynhru_params


def bynmonth_bynhru_parameter_list(param_filename):
    pardat = pws.parameters.PrmsParameters.load(param_filename)
    bynmonth_bynhru_params = []
    for par in list(pardat.parameters.keys()):
        kk = list(pws.meta.parameters[par]["dims"])
        if kk == ["nmonth", "nhru"]:
            bynmonth_bynhru_params.append(par)
        else:
            pass
    return bynmonth_bynhru_params


def bynsegment_parameter_list(param_filename):
    pardat = pws.parameters.PrmsParameters.load(param_filename)
    bynsegment_params = []
    for par in list(pardat.parameters.keys()):
        kk = list(pws.meta.parameters[par]["dims"])
        if kk == ["nsegment"]:
            bynsegment_params.append(par)
        else:
            pass
    return bynsegment_params


# Reads/Creates NWIS stations file if not already created
def fetch_nwis_gage_info(
    nwis_gage_nobs_min,
    model_domain_regions,
    st_date,
    en_date,
    hru_gdf,
    nwis_gages_file,
    crs,
):
    if nwis_gages_file.exists():
        col_names = [
            "poi_agency",
            "poi_id",
            "poi_name",
            "latitude",
            "longitude",
            "drainage_area",
            "drainage_area_contrib",
        ]
        col_types = [
            np.str_,
            np.str_,
            np.str_,
            float,
            float,
            float,
            float,
        ]
        cols = dict(
            zip(col_names, col_types)
        )  # Creates a dictionary of column header and datatype called below.

        nwis_gage_info_aoi = pd.read_csv(
            nwis_gages_file,
            dtype=cols,
            usecols=[
                "poi_agency",
                "poi_id",
                "poi_name",
                "latitude",
                "longitude",
                "drainage_area",
                "drainage_area_contrib",
            ],
        )
    else:
        siteINFO_huc = nwis.get_info(huc=model_domain_regions, siteType="ST")
        nwis_gage_info_gdf = siteINFO_huc[0].set_index("site_no").to_crs(crs)
        nwis_gage_info_aoi = nwis_gage_info_gdf.clip(hru_gdf)

        # Make a list of gages in the model domain that have discharge measurements > numer of specifed days
        siteINFO_huc = nwis.get_info(
            huc=model_domain_regions,
            startDt=st_date,
            endDt=en_date,
            seriesCatalogOutput=True,
            parameterCd="00060",
        )
        nwis_gage_info_gdf = siteINFO_huc[0].set_index("site_no").to_crs(crs)
        nwis_gage_nobs_aoi = nwis_gage_info_gdf.clip(hru_gdf)
        nwis_gage_nobs_aoi = nwis_gage_nobs_aoi.loc[
            nwis_gage_nobs_aoi.count_nu > nwis_gage_nobs_min
        ]
        nwis_gage_nobs_aoi_list = list(set(nwis_gage_nobs_aoi.index.to_list()))

        nwis_gage_info_aoi = nwis_gage_info_aoi.loc[
            nwis_gage_info_aoi.index.isin(nwis_gage_nobs_aoi_list)
        ]

        nwis_gage_info_aoi.reset_index(inplace=True)
        field_map = {
            "agency_cd": "poi_agency",
            "site_no": "poi_id",
            "station_nm": "poi_name",
            "dec_lat_va": "latitude",
            "dec_long_va": "longitude",
            "drain_area_va": "drainage_area",
            "contrib_drain_area_va": "drainage_area_contrib",
        }
        include_cols = list(field_map.keys())
        # include_cols = [
        #     "agency_cd",
        #     "site_no",
        #     "station_nm",
        #     "dec_lat_va",
        #     "dec_long_va",
        #     "drain_area_va",
        #     "contrib_drain_area_va",
        # ]
        nwis_gage_info_aoi = nwis_gage_info_aoi.loc[:, include_cols]

        nwis_gage_info_aoi.rename(columns=field_map, inplace=True)
        nwis_gage_info_aoi.set_index("poi_id", inplace=True)
        nwis_gage_info_aoi = nwis_gage_info_aoi.sort_index()
        nwis_gage_info_aoi.reset_index(inplace=True)

        # write out the file for later
        nwis_gage_info_aoi.to_csv(nwis_gages_file, index=False)  # , sep='\t')
    return nwis_gage_info_aoi