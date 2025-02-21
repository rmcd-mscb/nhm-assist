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

crs = 4326

admin_basin_style = lambda x: {'fillColor': '#00000000',
                               #'fill_opacity' : .8,
                               'color': 'black',
                               'weight': 2,
    }

transparent = lambda x: {
        "fillColor": "#00000000",
        "color": "#00000000",
        "weight": 4,
    }
    
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

popup_hru = folium.GeoJsonPopup(
        fields=["nhm_id", "hru_segment_nhm"],
        aliases=["hru", " flows to segment"],
        labels=True,
        localize=False,
        style=(
            "font-size: 16px;"
        ),  # Note that this tooltip style sets the style for all tool_tips.
        # background-color: #F0EFEF;border: 2px solid black;font-family: arial; padding: 10px; background-color: #F0EFEF;
    )

tooltip_hru = folium.GeoJsonTooltip(
        fields=["nhm_id", "hru_segment_nhm"],
        aliases=["hru", " flows to segment"],
        labels=True,
        # style=("background-color: #F0EFEF;border: 2px solid black;font-family: arial; font-size: 16px; padding: 10px;"),# Note that this tooltip style sets the style for all tool_tips.
    )

style_function_seg_map = lambda x: {
        "opacity": 1,
        "color": "#217de7",
        "weight": 2,
    }

highlight_function_seg_map = lambda x: {
        "opacity": 0.5,
        "color": "black",
        "weight": 3,
    }

popup_seg = folium.GeoJsonPopup(
        fields=["nhm_seg", "tosegment_nhm"],
        aliases=["segment", "flows to segment"],
        labels=True,
        localize=False,
    )
    
tooltip_seg = folium.GeoJsonTooltip(
        fields=["nhm_seg", "tosegment_nhm"],
        aliases=["segment", "flows to segment"],
        labels=True,
    )


def folium_map_elements(hru_gdf, poi_df, poi_id_sel):
    """
    Set approximate latitude, longitude and zoom level for subbasin is calculated for starting point of folium.map plot window.
    """
    if poi_id_sel:
        poi_lookup = poi_id_sel
        pfile_lat = poi_df.loc[poi_df.poi_id == poi_lookup, "latitude"].values[0]
        pfile_lon = poi_df.loc[poi_df.poi_id == poi_lookup, "longitude"].values[0]
        zoom = 12
        cluster_zoom = 8
    else:
        pfile_lat = hru_gdf["hru_lat"].mean()
        pfile_lon = hru_gdf["hru_lon"].mean()
        zoom = 8
        cluster_zoom = 8

    return pfile_lat, pfile_lon, zoom, cluster_zoom


def folium_map_tiles():
    """
    Set up a custom tile for base map
    This can be tricky with syntax but if you go to this link you will find resources that have options beyond the few defualt options in
    folium leaflet, http://leaflet-extras.github.io/leaflet-providers/preview/
    These tiles will also work in the minimap, but can get glitchy if the same tile var is used in the minimap and the main map child object.
    """

    USGSHydroCached_layer = folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}",
        attr="USGSHydroCached",
        #zoom_start=zoom,
        name="USGSHydroCached",
    )

    USGStopo_layer = folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attr="USGS_topo",
        #zoom_start=zoom,
        name="USGS Topography",
        show=False,
    )
    
    Esri_WorldImagery = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
        attr= 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        name="Esri_imagery",
        show=False,
        )

    OpenTopoMap = folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
        name='OpenTopoMap',
        show=False,
        )
    
    return USGSHydroCached_layer, USGStopo_layer, Esri_WorldImagery, OpenTopoMap


def create_minimap():

    """
    Set up inset map. This requires folium plugins. (from folium import plugins)
    """

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
    return minimap

def create_hru_map(hru_gdf):

    hru_map = folium.GeoJson(
        hru_gdf,
        style_function=style_function_hru_map,
        highlight_function=highlight_function_hru_map,
        name="NHM HRUs",
        # tooltip=tooltip_hru,
        popup=popup_hru,
    )
    return hru_map


def create_hru_label(hru_gdf, cluster_zoom):

    marker_cluster_label_hru = MarkerCluster(
        name="All HRU labels",
        overlay=True,
        control=True,
        show=False,  # False will not draw the child upon opening the map, but have it to draw in the Layer control.
        icon_create_function=None,
        disableClusteringAtZoom=cluster_zoom,
        # z_index_offset=4005,
    )

    for idx, row in hru_gdf.iterrows():
        text = f'{row["nhm_id"]}'
        label_lat = row["hru_lat"]
        label_lon = row["hru_lon"]
        marker_label = folium.map.Marker(
            [label_lat, label_lon],
            # z_index_offset=4008,
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html='<div style="font-family: verdona; font-size: 10pt; font-weight: bold; color: black; text-shadow: 1px 1px 2px white;">%s</div>'
                % text,
            ),
        ).add_to(marker_cluster_label_hru)
    return marker_cluster_label_hru

def create_segment_map_show(seg_gdf):
    seg_map_show = folium.GeoJson(
        seg_gdf,
        style_function=style_function_seg_map,
        highlight_function=highlight_function_seg_map,  # lambda feature: {"fillcolor": "white", "color": "white"},
        name="NHM Segments",
    )
    return seg_map_show                

def create_seg_map_hide(seg_gdf):
    seg_map_hide = folium.GeoJson(
        seg_gdf,
        style_function=transparent,
        highlight_function=highlight_function_seg_map,  # lambda feature: {"fillcolor": "white", "color": "white"},
        name="NHM Segments",
        # tooltip=tooltip_seg,
        popup=popup_seg,
    )
    return seg_map_hide

def create_poi_marker_cluster(
    poi_df,
    cluster_zoom: pd.DataFrame,
) -> tuple[folium.plugins.MarkerCluster, folium.plugins.MarkerCluster]:

    # add POI marker cluster child items for the map
    poi_marker_cluster = MarkerCluster(
        name="Model poi",
        overlay=True,
        control=True,
        icon_create_function=None,
        disableClusteringAtZoom=cluster_zoom,
    )
    poi_marker_cluster_label = MarkerCluster(
        name="Model poi label",
        overlay=True,
        control=True,
        show=False,  # False will not draw the child upon opening the map, but have it to draw in the Layer control.
        icon_create_function=None,
        disableClusteringAtZoom=cluster_zoom,
    )
    ##add POI markers and labels using row df.interowss loop
    for idx, row in poi_df.iterrows():
        text = f'{row["poi_id"]}'
        label_lat = row["latitude"]  # -0.01
        label_lon = row["longitude"]

        marker_label = folium.map.Marker(
            [label_lat, label_lon],
            icon=DivIcon(
                icon_size=(10, 10),  # (150,36),
                icon_anchor=(0, 0),
                html='<div style="font-size: 12pt; font-weight: bold">%s</div>' % text,
            ),
        ).add_to(poi_marker_cluster_label)

        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            name=row["poi_id"],
            popup=folium.Popup(
                f'<font size="3px">{row["poi_id"]} ({row["poi_agency"]})<br>{row["poi_name"]}<br> on <b>segment </b>{row["nhm_seg"]}</font>',
                max_width=280,
                max_height=2000,
            ),
            radius=5,
            weight=2,
            color="black",
            fill=True,
            fill_color="Black",
            fill_opacity=1.0,
        ).add_to(poi_marker_cluster)

    return poi_marker_cluster, poi_marker_cluster_label

def create_non_poi_marker_cluster(
    poi_df,
    nwis_gages_aoi,
    gages_df,
    cluster_zoom: pd.DataFrame,
) -> tuple[folium.plugins.MarkerCluster, folium.plugins.MarkerCluster]:

    # add non-poi gages marker cluster child items for the map
    non_poi_marker_cluster = MarkerCluster(
        name="Non-poi",
        overlay=True,
        control=True,
        icon_create_function=None,
        disableClusteringAtZoom=cluster_zoom,
    )
    non_poi_marker_cluster_label = MarkerCluster(
        name="Non-poi label",
        overlay=True,
        control=True,
        show=False,  # False will not draw the child upon opening the map, but have it to draw in the Layer control.
        icon_create_function=None,
        disableClusteringAtZoom=cluster_zoom,
    )

    ##add Non-poi gage markers and labels using row df.interowss loop
    gages_list = gages_df.index.to_list()
    additional_gages = list(set(gages_list) - set(poi_df.poi_id))

    for idx, row in nwis_gages_aoi.iterrows():
        if row["poi_id"] in additional_gages:

            text = f'{row["poi_id"]}'
            label_lat = row["latitude"]  # -0.01
            label_lon = row["longitude"]

            marker_label = folium.map.Marker(
                [label_lat, label_lon],
                icon=DivIcon(
                    icon_size=(10, 10),  # (150,36),
                    icon_anchor=(0, 0),
                    html='<div style="font-size: 12pt; font-weight: bold">%s</div>'
                    % text,
                ),
            ).add_to(non_poi_marker_cluster_label)

            marker = folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                name=row["poi_id"],
                popup=folium.Popup(
                    f'<font size="3px">{row["poi_id"]} ({row["poi_agency"]})<br>{row["poi_name"]}<br></font>',
                    max_width=280,
                    max_height=2000,
                ),
                radius=5,
                weight=2,
                color="gray",
                fill=True,
                fill_color="Gray",
                fill_opacity=1.0,
            ).add_to(non_poi_marker_cluster)
        else:
            pass

    return non_poi_marker_cluster, non_poi_marker_cluster_label

def nhru_par_map(
    param_filename,
    hru_gdf,
    par_sel,
    mo_sel,
    mo_name,
    nhru_params,
):

    cp_style_function = lambda feature: {
    "fillColor": linear(par_sel_color_dict[feature["id"]]),
    "color": "tan",
    "weight": 1,
    # "dashArray": "5, 5",
    "fillOpacity": 0.3,
}
    
    prms_meta = MetaData(version=5, verbose=False).metadata  # loads metadata functions for pyPRMS
    pdb = ParameterFile(param_filename, metadata=prms_meta, verbose=False)  # loads parmaeterfile functions for pyPRMS

    
    if par_sel in nhru_params:

        mo_sel = None  # set to none, this is not a monthly param
        hru_gdf_copy = hru_gdf.copy()
        hru_gdf_copy["nhm_id"] = hru_gdf_copy["nhm_id"].astype(str)
        hru_gdf_copy["hru_segment_nhm"] = hru_gdf_copy["hru_segment_nhm"].astype(str)

        hru_gdf_copy.set_index("nhm_id", inplace=True, drop=False)

        par_subset_df = hru_gdf.loc[:, ["nhm_id", par_sel]]
        par_subset_df["nhm_id"] = par_subset_df["nhm_id"].astype(str)
        par_subset_df.rename(columns={f"{par_sel}": "par_value"}, inplace=True)
        par_subset_df["par_value"] = np.round(par_subset_df["par_value"], 4)
        par_subset_df.set_index("nhm_id", inplace=True, drop=False)

        value_min = np.round(par_subset_df["par_value"].min(), 8)
        value_max = np.round(par_subset_df["par_value"].max(), 8)

        par_sel_color_dict = pd.Series(
            par_subset_df.par_value.values, index=par_subset_df.nhm_id
        ).to_dict()

        # Making par_bins
        sdv = par_subset_df["par_value"].std()
        mean = par_subset_df["par_value"].mean()

        par_bins = [
            value_min,
            np.round(value_min + (0.25 * (mean - value_min)), 5),
            np.round(value_min + (0.50 * (mean - value_min)), 5),
            np.round(value_min + (0.75 * (mean - value_min)), 5),
            np.round(mean, 3),
            np.round(value_max - (0.75 * (value_max - mean)), 5),
            np.round(value_max - (0.50 * (value_max - mean)), 5),
            np.round(value_max - (0.25 * (value_max - mean)), 5),
            value_max,
        ]

        #################################################
        if value_min != value_max:
            fig, ax = plt.subplots(figsize=(18, 0.5))
            fig.subplots_adjust(bottom=0.5)

            cmap = mplib.colors.ListedColormap(
                [
                    "#8B0000",
                    "#AC4800",
                    "#CD9100",
                    "#EEDA00",
                    "#DADA13",
                    "#91913B",
                    "#484863",
                    "#00008B",
                ]
            )
            cmap.set_over("0.25")
            cmap.set_under("0.75")

            bounds = par_bins
            norm = mplib.colors.BoundaryNorm(bounds, cmap.N)
            cb2 = mplib.colorbar.ColorbarBase(
                ax,
                cmap=cmap,
                norm=norm,
                boundaries=[0] + bounds + [13],
                extend=None,
                ticks=bounds,
                spacing="uniform",
                orientation="horizontal",
                alpha=0.45,
            )
            cb2.set_label(
                f'Discrete {par_sel} intervals, {pdb.get(par_sel).meta["units"]}'
            )

            fig.set_facecolor("lightgray")
            # fig.show()

        #######################################################

        if value_min == value_max:
            linear = cm.StepColormap(
                colors=[
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                ],
                index=par_bins,
                vmin=0.00,
                vmax=0.05,
                caption="Total Standard deviation at the point[mm]",
                # tick_labels= ('0.01', '0.02', '0.03', '0.04')
            )
        else:
            linear = cm.StepColormap(
                colors=[
                    "#8B0000",
                    "#AC4800",
                    "#CD9100",
                    "#EEDA00",
                    "#DADA13",
                    "#91913B",
                    "#484863",
                    "#00008B",
                ],
                index=par_bins,
                vmin=0.00,
                vmax=0.05,
                caption="Total Standard deviation at the point[mm]",
                # tick_labels= ('0.01', '0.02', '0.03', '0.04')
            )
        popup_hru = folium.GeoJsonPopup(
            fields=["nhm_id", "hru_segment_nhm", par_sel],
            aliases=["hru", " flows to segment", f"{par_sel}"],
            labels=True,
            localize=True,
            style=(
                "font-size: 16px;"
            ),  # Note that this tooltip style sets the style for all tool_tips.
            # background-color: #F0EFEF;border: 2px solid black;font-family: arial; padding: 10px; background-color: #F0EFEF;
        )
        hru_map = folium.GeoJson(
            hru_gdf_copy,
            style_function=cp_style_function,  # style_function_hru_map,
            highlight_function=highlight_function_hru_map,
            name="NHM HRUs",
            popup=popup_hru,
            z_index_offset=40002,
        )

        # tooltip_hru=folium.GeoJsonTooltip(fields= ["nhm_id","hru_segment_nhm",par_sel],
        #                               aliases=["HRU"," flows to segment", f"{par_sel}"],
        #                               localize=False,
        #                               style=("background-color: #F0EFEF;border: 2px solid black;font-family: arial; font-size: 16px; padding: 10px;"),
        #                                  )

    else:
        # mo_sel = '5'
        par_mo_sel = f"{par_sel}_{mo_sel}"
        value_min = hru_gdf[par_mo_sel].min()
        value_max = hru_gdf[par_mo_sel].max()

        hru_gdf_copy = hru_gdf.copy()
        hru_gdf_copy["nhm_id"] = hru_gdf_copy["nhm_id"].astype(str)
        hru_gdf_copy["hru_segment_nhm"] = hru_gdf_copy["hru_segment_nhm"].astype(str)

        hru_gdf_copy.set_index("nhm_id", inplace=True, drop=False)

        par_subset_df = hru_gdf.loc[:, ["nhm_id", par_mo_sel]]
        par_subset_df["nhm_id"] = par_subset_df["nhm_id"].astype(str)
        par_subset_df.rename(columns={f"{par_mo_sel}": "par_value"}, inplace=True)

        value_min = np.round(par_subset_df["par_value"].min(), 8)
        value_max = np.round(par_subset_df["par_value"].max(), 8)

        par_sel_color_dict = pd.Series(
            par_subset_df.par_value.values, index=par_subset_df.nhm_id
        ).to_dict()

        # Making par_bins
        sdv = par_subset_df["par_value"].std()
        mean = par_subset_df["par_value"].mean()

        par_bins = [
            value_min,
            np.round(value_min + (0.25 * (mean - value_min)), 5),
            np.round(value_min + (0.50 * (mean - value_min)), 5),
            np.round(value_min + (0.75 * (mean - value_min)), 5),
            np.round(mean, 3),
            np.round(value_max - (0.75 * (value_max - mean)), 5),
            np.round(value_max - (0.50 * (value_max - mean)), 5),
            np.round(value_max - (0.25 * (value_max - mean)), 5),
            value_max,
        ]

        #################################################
        if value_min != value_max:
            fig, ax = plt.subplots(figsize=(18, 0.5))
            fig.subplots_adjust(bottom=0.5)

            cmap = mplib.colors.ListedColormap(
                [
                    "#8B0000",
                    "#AC4800",
                    "#CD9100",
                    "#EEDA00",
                    "#DADA13",
                    "#91913B",
                    "#484863",
                    "#00008B",
                ]
            )
            cmap.set_over("0.25")
            cmap.set_under("0.75")

            bounds = par_bins
            norm = mplib.colors.BoundaryNorm(bounds, cmap.N)
            cb2 = mplib.colorbar.ColorbarBase(
                ax,
                cmap=cmap,
                norm=norm,
                boundaries=[0] + bounds + [13],
                extend=None,
                ticks=bounds,
                spacing="uniform",
                orientation="horizontal",
                alpha=0.45,
            )
            cb2.set_label(
                f'Discrete {par_sel} intervals, {pdb.get(par_sel).meta["units"]}'
            )  # {pdb.get(par_sel).units}

            fig.set_facecolor("lightgray")
            # fig.show()

        #######################################################
        if value_min == value_max:
            linear = cm.StepColormap(
                colors=[
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                    "#000000",
                ],
                index=par_bins,
                vmin=0.00,
                vmax=0.05,
                caption="Total Standard deviation at the point[mm]",
                # tick_labels= ('0.01', '0.02', '0.03', '0.04')
            )
        else:
            linear = cm.StepColormap(
                colors=[
                    "#8B0000",
                    "#AC4800",
                    "#CD9100",
                    "#EEDA00",
                    "#DADA13",
                    "#91913B",
                    "#484863",
                    "#00008B",
                ],
                index=par_bins,
                vmin=0.00,
                vmax=0.05,
                caption="Total Standard deviation at the point[mm]",
                # tick_labels= ('0.01', '0.02', '0.03', '0.04')
            )
        popup_hru = folium.GeoJsonPopup(
            fields=["nhm_id", "hru_segment_nhm", par_mo_sel],
            aliases=["hru", " flows to segment", f"{par_sel} for {mo_name}"],
            labels=True,
            localize=False,
            style=(
                "font-size: 16px;"
            ),  # Note that this tooltip style sets the style for all tool_tips.
            # background-color: #F0EFEF;border: 2px solid black;font-family: arial; padding: 10px; background-color: #F0EFEF;
        )
        hru_map = folium.GeoJson(
            hru_gdf_copy,
            style_function=cp_style_function,  # style_function_hru_map,
            highlight_function=highlight_function_hru_map,
            name="NHM HRUs",
            popup=popup_hru,
            # z_index_offset=40002,
        )
        # tooltip_hru=folium.GeoJsonTooltip(fields= ["nhm_id","hru_segment_nhm",par_mo_sel],
        #                               aliases=["HRU"," flows to segment", f"{par_sel} for {mo_name}"],
        #                               labels=True,
        #                               localize=False,
        #                               style=("background-color: #F0EFEF;border: 2px solid black;font-family: arial; font-size: 16px; padding: 10px;"),
        #                                  )

        # linear = cm.LinearColormap(["red", "yellow", "blue"], vmin= hru_gdf[par_mo_sel].min(), vmax=hru_gdf[par_mo_sel].max())

        # hru_map = folium.GeoJson(hru_gdf_copy,
        #                      style_function = cp_style_function,#style_function_hru_map,
        #                      highlight_function = highlight_function_hru_map,
        #                      name = "NHM HRUs",
        #                      z_index_offset = 40002).add_to(m3)
        # tooltip_hru=folium.GeoJsonTooltip(fields= ["nhm_id","hru_segment_nhm",par_mo_sel],
        #                               aliases=["HRU"," flows to segment", "par_value"],
        #                               labels=True)

    return fig, hru_map, value_min, value_max

def create_poi_paramplot_marker_cluster(
    poi_df,
    hru_gdf,
    # param_filename,
    # nhru_params,
    # nhru_nmonths_params,
    Folium_maps_dir,
    cluster_zoom,
    par_sel,
):

    marker_cluster = MarkerCluster(
        name="All the POIs",
        overlay=True,
        control=True,
        icon_create_function=None,
        disableClusteringAtZoom=cluster_zoom,
        # z_index_offset=5000,
    )
    marker_cluster_label_poi = MarkerCluster(
        name="All the POI labels",
        overlay=True,
        control=True,
        show=False,  # False will not draw the child upon opening the map, but have it to draw in the Layer control.
        icon_create_function=None,
        disableClusteringAtZoom=cluster_zoom,
        # z_index_offset=4004,
    )

    for idx, row in poi_df.iterrows():
        poi_id = row["poi_id"]
        # Read ploty plot of each poi
        with open(Folium_maps_dir / f"{par_sel}_{poi_id}.txt", "r") as f:
            div_txt = f.read()

        # Create html code to insert the plotly plot to the folium pop up
        html = (
            """
        <html>
        <head>
             <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
             <!-- Output from the Python script above: -->"""
            + div_txt
            + """</body>
        </html>"""
        )

        # Add the Plots to the popup
        iframe = folium.IFrame(html=html, width=845, height=475)
        # popup = folium.Popup(iframe, max_width=3250,parse_html=True)

        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            name=row["poi_id"],
            popup=folium.Popup(iframe, max_width=550, max_height=350, parse_html=True),
            radius=4,
            weight=2,
            color="black",
            fill=True,
            fill_color="Black",
            fill_opacity=1.0,
            draggable=True,
            # z_index_offset=4006,
        ).add_to(marker_cluster)

        # marker_cluster.add_child(marker)
        text = f'{row["poi_id"]}'
        label_lat = row["latitude"] - 0.01
        label_lon = row["longitude"]

        marker_label = folium.map.Marker(
            [label_lat, label_lon],
            z_index_offset=4007,
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html='<div style="font-size: 12pt; font-weight: bold">%s</div>' % text,
            ),
        ).add_to(marker_cluster_label_poi)

    return marker_cluster, marker_cluster_label_poi

def create_annual_output_var_map(
    gdf_output_var_annual,
    output_var_sel,
    sel_year,
):
    
    cp_style_function = lambda feature: {
        "fillColor": linear(var_sel_color_dict[feature["id"]]),
        "color": "tan",
        "weight": 1,
        # "dashArray": "5, 5",
        "fillOpacity": 0.3,
    }

    hru_gdf_copy = gdf_output_var_annual.copy().reset_index(drop=True).to_crs(crs)
    hru_gdf_copy["nhm_id"] = hru_gdf_copy["nhm_id"].astype(str)
    hru_gdf_copy.set_index("nhm_id", inplace=True, drop=False)

    var_subset_df = gdf_output_var_annual.loc[:, ["nhm_id", str(sel_year)]]
    var_subset_df["nhm_id"] = var_subset_df["nhm_id"].astype(str)
    var_subset_df.rename(columns={f"{sel_year}": "var_value"}, inplace=True)
    var_subset_df["var_value"] = np.round(var_subset_df["var_value"], 4)
    var_subset_df.set_index("nhm_id", inplace=True, drop=False)

    value_min = np.round(var_subset_df["var_value"].min(), 8)
    value_max = np.round(var_subset_df["var_value"].max(), 8)

    if value_min == value_max:
        value_min = value_min - 0.001
        value_max = value_min + 0.001
        color_bar = False
    else:
        color_bar = True

    var_sel_color_dict = pd.Series(
        var_subset_df.var_value.values, index=var_subset_df.nhm_id
    ).to_dict()

    # Making par_bins
    sdv = var_subset_df["var_value"].std()
    mean = var_subset_df["var_value"].mean()

    var_bins = [
        value_min,
        np.round(value_min + (0.25 * (mean - value_min)), 5),
        np.round(value_min + (0.50 * (mean - value_min)), 5),
        np.round(value_min + (0.75 * (mean - value_min)), 5),
        np.round(mean, 3),
        np.round(value_max - (0.75 * (value_max - mean)), 5),
        np.round(value_max - (0.50 * (value_max - mean)), 5),
        np.round(value_max - (0.25 * (value_max - mean)), 5),
        value_max,
    ]

    #################################################

    if not color_bar:
        scale_bar_txt = (
            f"All {output_var_sel} values are {value_min}. No value scale bar rendered."
        )
        fig, ax = plt.subplots(figsize=(18, 0.5))
    else:
        scale_bar_txt = ""

        fig, ax = plt.subplots(figsize=(18, 0.5))
        fig.subplots_adjust(bottom=0.5)

        cmap = mplib.colors.ListedColormap(
            [
                "#8B0000",
                "#AC4800",
                "#CD9100",
                "#EEDA00",
                "#DADA13",
                "#91913B",
                "#484863",
                "#00008B",
            ]
        )
        cmap.set_over("0.25")
        cmap.set_under("0.75")

        bounds = var_bins
        norm = mplib.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mplib.colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            boundaries=[0] + bounds + [13],
            extend=None,
            ticks=bounds,
            spacing="uniform",
            orientation="horizontal",
            alpha=0.45,
        )
        cb2.set_label(
            f"Discrete {output_var_sel} intervals"
        )  # , {pdb.get(output_var_sel).units}')

        fig.set_facecolor("lightgray")
        fig.show()

    #######################################################

    linear = cm.StepColormap(
        colors=[
            "#8B0000",
            "#AC4800",
            "#CD9100",
            "#EEDA00",
            "#DADA13",
            "#91913B",
            "#484863",
            "#00008B",
        ],
        index=var_bins,
        vmin=0.00,
        vmax=0.05,
        caption="Total Standard deviation at the point[mm]",
        # tick_labels= ('0.01', '0.02', '0.03', '0.04')
    )
    popup_hru = folium.GeoJsonPopup(
        fields=["nhm_id", str(sel_year)],
        aliases=["HRU", f"{output_var_sel} for {sel_year}"],
        labels=True,
        localize=True,
        style=(
            "font-size: 16px;"
        ),  # Note that this tooltip style sets the style for all tool_tips.
        # background-color: #F0EFEF;border: 2px solid black;font-family: arial; padding: 10px; background-color: #F0EFEF;
    )

    hru_map = folium.GeoJson(
        hru_gdf_copy,
        style_function=cp_style_function,  # style_function_hru_map,
        highlight_function=highlight_function_hru_map,
        name="NHM HRUs",
        popup=popup_hru,
        # z_index_offset=40002,
    )

    # tooltip_hru=folium.GeoJsonPopup(fields= ["nhm_id",str(sel_year)],
    #                                   aliases=["HRU", "var value"],
    #                                   labels=True)

    # tooltip_hru=folium.GeoJsonTooltip(fields= ["nhm_id",str(sel_year)],
    #                                   aliases=["HRU", f"{output_var_sel} for {sel_year}"],
    #                                   labels=True,
    #                                   localize=True,
    #                                   style=("background-color: #F0EFEF;border: 2px solid black;font-family: arial; font-size: 16px; padding: 10px;"),
    #                                      )

    # hru_map.add_child(tooltip_hru)

    return fig, hru_map, value_min, value_max, scale_bar_txt