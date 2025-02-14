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


import jupyter_black

jupyter_black.load()


# Set approximate latitude, longitude and zoom level for subbasin is calculated for starting point of folium.map plot window.
pfile_lat = hru_gdf["hru_lat"].mean()
pfile_lon = hru_gdf["hru_lon"].mean()
zoom = 7  # Can be ad

# Set up a custom tile for base map

# This can be tricky with syntax but if you go to this link you will find resources that have options beyond the few defualt options in folium leaflet.
# http://leaflet-extras.github.io/leaflet-providers/preview/
# These tiles will also work in the minimap, but can get glitchy if the same tile var is used in the minimap and the main map child object.

USGStopo_layer = folium.TileLayer(
    tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
    attr="USGS_topo",
    zoom_start=zoom,
    name="USGSTopo",
)
USGSHydroCached_layer = folium.TileLayer(
    tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}",
    attr="USGSHydroCached",
    zoom_start=zoom,
    name="USGSHydroCached",
)

# Set up inset map
# This requires folium plugins. (from folium import plugins)
minimap = plugins.MiniMap(
    tile_layer="OpenStreetMap",
    # attr = 'USGS_topo',
    position="topleft",
    # zoom_level_offset=- 4,
    height=200,
    width=200,
    collapsed_height=25,
    collapsed_width=25,
    zoom_level_fixed=5,
    toggle_display=True,
    # collapsed = True
)

# Style functions
style_function_hru_map = lambda x: {
    "opacity": 1,
    "fillColor": "#00000000",  #'goldenrod',
    "color": "tan",
    "weight": 1.5,
}
highlight_function_hru_map = lambda x: {
    "opacity": 0.5,
    "color": "gray",
    "fillColor": "gray",
    "weight": 3,
}
style_function_seg_map = lambda x: {"opacity": 1, "color": "#217de7", "weight": 2}
highlight_function_seg_map = lambda x: {"opacity": 1, "color": "white", "weight": 4}
transparent = lambda x: {
    "fillColor": "#00000000",
    "color": "#00000000",
    "weight": 4,
}

########################################################################################