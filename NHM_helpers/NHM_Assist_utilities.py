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

from NHM_helpers.NHM_helpers import hrus_by_poi

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
def fetch_nwis_gage_info(model_dir,
                         control_file_name,
                         nwis_gage_nobs_min,
                         hru_gdf,
):

    """
    This function creates a database of infomration for all gages in the model domain that
    are in NWIS, from 01-01-1949 to the end date listed in the control file. 
    """
    
    
    nwis_gages_file = model_dir / "NWISgages.csv"
    control = pws.Control.load_prms(pl.Path(model_dir / control_file_name, warn_unused_options=False))
    
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

    # Make a list if the HUC2 region(s) the subbasin intersects for NWIS queries
    huc2_gdf = gpd.read_file("./data_dependencies/HUC2/HUC2.shp").to_crs(crs)
    model_domain_regions = list((huc2_gdf.clip(hru_gdf).loc[:]["huc2"]).values)
    
        
    """
    Start date changed because gages were found in the par file that predate 1979 and tossing nan's into poi_df later.
    """
    
    st_date = '1949-01-01'#pd.to_datetime(str(control.start_time)).strftime("%Y-%m-%d")
    en_date = pd.to_datetime(str(control.end_time)).strftime("%Y-%m-%d")
    
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
        #siteINFO_huc = nwis.get_info(huc=model_domain_regions, siteType="ST")
        siteINFO_huc = gpd.GeoDataFrame()
        for i in model_domain_regions:
            zz = nwis.get_info(huc=i, siteType="ST", agencyCd="USGS",)[0]
            siteINFO_huc = pd.concat([siteINFO_huc, zz])
            
        nwis_gage_info_gdf = siteINFO_huc.set_index("site_no").to_crs(crs)
        nwis_gage_info_aoi = nwis_gage_info_gdf.clip(hru_gdf)

        # Make a list of gages in the model domain that have discharge measurements > numer of specifed days
        siteINFO_huc = gpd.GeoDataFrame()
        for i in model_domain_regions:
            zz = nwis.get_info(
                huc=i,
                startDt=st_date,
                endDt=en_date,
                seriesCatalogOutput=True,
                parameterCd="00060",
            )[0]
            siteINFO_huc = pd.concat([siteINFO_huc, zz])
            
        # siteINFO_huc = nwis.get_info(
        #     huc=model_domain_regions,
        #     startDt=st_date,
        #     endDt=en_date,
        #     seriesCatalogOutput=True,
        #     parameterCd="00060",
        # )
        nwis_gage_info_gdf = siteINFO_huc.set_index("site_no").to_crs(crs)
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
        #nwis_gage_info_aoi.to_csv(nwis_gages_file, index=False)  # , sep='\t')
    return nwis_gage_info_aoi

def make_plots_par_vals(
    poi_df,
    hru_gdf,
    param_filename,
    nhru_params,
    nhru_nmonths_params,
    Folium_maps_dir,
):
    """
    Builds plots, takes  >20 minutes to build all param plots for all pois
    """
    cal_hru_params = nhru_params + nhru_nmonths_params

    """
    First, group HRUs to the downstream gagepoi that they contribute flow.
    """
    poi_list = poi_df["poi_id"].values.tolist()

    """
    Make a dictionary of pois and the list of HRUs in the contributing area for each poi.
    """
    prms_meta = MetaData().metadata  # loads metadata functions for pyPRMS
    pdb = ParameterFile(param_filename, metadata=prms_meta, verbose=False)  # loads parmaeterfile functions for pyPRMS
    
    hru_poi_dict = hrus_by_poi(pdb, poi_list)  # Helper function from pyPRMS

    """
    Sort the dictionary: this is important for the reverse dictionary (next step) to accurately give a poi_group 
    to hrus that contribute to a downstream-gage.
    """
    sorted_items = sorted(
        hru_poi_dict.items(), key=lambda item: -len(item[1])
    )  # the - reverses the sorting order
    hru_poi_dict = dict(sorted_items[:])

    reversed_hru_poi_dict = {
        val: key for key in hru_poi_dict for val in hru_poi_dict[key]
    }

    # assigns poi_group value to all hrus #Keep for later application
    hru_gdf["poi_group"] = hru_gdf["nhm_id"].astype(int).map(reversed_hru_poi_dict)

    """
    Builds plots, takes  20 minutes to build all param plots for all pois
    """

    for idx, par in enumerate(cal_hru_params):
        try:
            pdb.get(par).dimensions["nmonths"].size

        except:
            #print(f"Checking for {par} dimensioned by nhru.")

            for idx, poi_id in enumerate(poi_list):
                par_plot_file = Folium_maps_dir / f"{par}_{poi_id}.txt"
                if par_plot_file.exists():
                    pass
                    # print(
                    #     f"{par}_{poi_id}.txt exists. To recreate the plot, remove the file from Folium_maps_dir"
                    # )
                    # print(par_plot_file)
                else:

                    ##%%time = par
                    # Preporcessing: pulling only the selected param values for the HRUs related to the selected POI to plot.
                    output_var_sel_plot_df = hru_gdf[
                        hru_gdf["nhm_id"].astype(int).isin(hru_poi_dict[poi_id])
                    ]
                    output_var_sel_plot_df = output_var_sel_plot_df.sort_values(
                        ["hru_area"], ascending=True
                    )
                    output_var_sel_plot_df.hru_area = (
                        output_var_sel_plot_df.hru_area.round()
                    )

                    x_axis_var = "hru_area"  # we broke this out separately to quickly generate new plots based on a different variable for the x-axis
                    fig = px.scatter(
                        output_var_sel_plot_df,
                        x=x_axis_var,
                        y=par,
                        # markers = True,
                        custom_data="nhm_id",
                        color="poi_group",
                        labels={"poi_group": "Downstream POI"},
                    )

                    fig.update_layout(
                        title=dict(
                            text=f"{par} for HRUs in {poi_id} catchment",
                            font=dict(size=18),
                            automargin=True,
                            yref="paper",
                        ),
                        width=500,
                        height=300,
                        showlegend=True,
                        # legend=dict(orientation="h",yanchor="bottom",y=1.02, xanchor="right", x=1),# this will put the legend above the plot
                        # margin=dict(l=20, r=20, t=20, b=20),#hesitate to customize until final plots are determined
                        font=dict(
                            family="Arial", size=10, color="#7f7f7f"
                        ),  # font color
                        paper_bgcolor="linen",
                        plot_bgcolor="white",
                    )

                    fig.update_yaxes(title_text=f'{par}, {pdb.get(par).meta["units"]}')
                    fig.update_xaxes(
                        title_text=f'{x_axis_var}, {pdb.get(x_axis_var).meta["units"]}'
                    )

                    fig.update_xaxes(
                        ticks="inside", tickwidth=2, tickcolor="black", ticklen=10
                    )
                    fig.update_yaxes(
                        ticks="inside", tickwidth=2, tickcolor="black", ticklen=10
                    )

                    # tick_step = np.round(((output_var_sel_plot_df.hru_area.max())*0.1)+500, -3)
                    # fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 0.0, dtick = tick_step))

                    fig.update_xaxes(
                        showline=True,
                        linewidth=2,
                        linecolor="black",
                        showgrid=False,
                        # gridcolor='lightgrey',
                    )
                    fig.update_yaxes(
                        showline=True,
                        linewidth=2,
                        linecolor="black",
                        showgrid=False,
                        # gridcolor='lightgrey',
                    )

                    # fig.update_xaxes(type='category')
                    fig.update_xaxes(autorange=True)

                    fig.update_traces(
                        hovertemplate="<br>".join(
                            [
                                "parameter value: %{y}",
                                "nhu area: %{x}",
                                "hru: %{customdata[0]}",
                            ]
                        )
                    )

                    fig.update_layout(hovermode="closest")
                    fig.update_layout(
                        hoverlabel=dict(
                            bgcolor="linen", font_size=13, font_family="Rockwell"
                        )
                    )

                    # Creating the html code for the plotly plot
                    text_div = plotly.offline.plot(
                        fig, include_plotlyjs=False, output_type="div"
                    )

                    # Saving the plot as txt file with the html code
                    # idx = 1
                    with open(Folium_maps_dir / f"{par}_{poi_id}.txt", "w") as f:
                        f.write(text_div)

                    # fig.show()

        else:
            #print(f"Checking for {par} dimensioned by nhru and nmonths")

            for idx, poi_id in enumerate(poi_list):

                par_plot_file = Folium_maps_dir / f"{par}_{poi_id}.txt"
                if par_plot_file.exists():
                    pass
                    # print(
                    #     f"{par}_{poi_id} exists. To recreate the plot, remove the file from"
                    # )
                    # print(par_plot_file)
                else:
                    # Reshapes the monthly data for plotting: assigns a "month" number
                    first = True
                    for vv in range(1, 13):
                        if first:
                            zz = f"{par}_{str(vv)}"
                            df = hru_gdf[["nhm_id", zz]]
                            df["month"] = vv
                            df[par] = df[zz]
                            df.drop(columns=zz, inplace=True)
                            first = False
                        else:
                            zz = f"{par}_{str(vv)}"
                            df2 = hru_gdf[["nhm_id", zz]]
                            df2["month"] = vv
                            df2[par] = df2[zz]
                            df2.drop(columns=zz, inplace=True)

                            df = pd.concat([df, df2], ignore_index=True)

                    nhru_params_nmonths_sel_df = df.copy()
                    ############################################################################################
                    nhru_params_nmonths_sel_plot_df = nhru_params_nmonths_sel_df[
                        nhru_params_nmonths_sel_df["nhm_id"].isin(hru_poi_dict[poi_id])
                    ]
                    # nhru_params_nmonths_sel_plot_df = nhru_params_nmonths_sel_df[nhru_params_nmonths_sel_df['poi_group'] == poi_id]

                    fig = px.line(
                        nhru_params_nmonths_sel_plot_df,
                        x="month",
                        y=par,
                        markers=True,
                        custom_data=nhru_params_nmonths_sel_plot_df[["nhm_id"]],
                        color="nhm_id",
                        labels={"nhm_id": "HRU"},
                    )

                    fig.update_layout(
                        title_text=f"{par} for HRUs in {poi_id} catchment",
                        width=500,
                        height=300,
                        showlegend=True,
                        # legend=dict(orientation="h",yanchor="bottom",y=1.02, xanchor="right", x=1),
                        font=dict(
                            family="Arial", size=10, color="#7f7f7f"
                        ),  # font color
                        paper_bgcolor="linen",
                        plot_bgcolor="white",
                    )

                    fig.update_yaxes(title_text=f"{par}, units")
                    fig.update_xaxes(title_text="Months")

                    fig.update_xaxes(
                        ticks="inside", tickwidth=2, tickcolor="black", ticklen=10
                    )
                    fig.update_yaxes(
                        ticks="inside", tickwidth=2, tickcolor="black", ticklen=10
                    )

                    fig.update_xaxes(
                        showline=True,
                        linewidth=2,
                        linecolor="black",
                        gridcolor="lightgrey",
                    )
                    fig.update_yaxes(
                        showline=True,
                        linewidth=2,
                        linecolor="black",
                        gridcolor="lightgrey",
                    )

                    fig.update_xaxes(autorange=True)

                    fig.update_traces(
                        hovertemplate="<br>".join(
                            [
                                "{parameter: %{y}",
                                "month: %{x}",
                                "hru: %{customdata[0]}",
                            ]
                        )
                    )

                    fig.update_layout(hovermode="closest")
                    fig.update_layout(
                        hoverlabel=dict(
                            bgcolor="linen", font_size=13, font_family="Rockwell"
                        )
                    )

                    # Creating the html code for the plotly plot
                    text_div = plotly.offline.plot(
                        fig, include_plotlyjs=False, output_type="div"
                    )

                    # Saving the plot as txt file with the html code
                    with open(Folium_maps_dir / f"{par}_{poi_id}.txt", "w") as f:
                        f.write(text_div)

def make_HW_cal_level_files(hru_gdf):
    """ 
    NHM Calibration Levels for HRUs: (those hrus calibrated in byHW and byHWobs parts)
    HW basins were descritized using a drainage area maxiumum and minimum; HW HRUs, segments, outlet segment, and drainage area are available.
    Gages used in byHWobs calibration, Part 3, for selected headwaters are also provided here.
    FILES AND TABLES IN THIS SECTION ARE CONUS COVERAGE and will be subsetted later.
    """
    crs = 4326
    byHW_basins_gdf = hru_gdf.loc[hru_gdf["byHW"] == 1]
    HW_basins_gdf = byHW_basins_gdf.dissolve(by="hw_id").to_crs(crs)
    HW_basins_gdf.reset_index(inplace=True, drop=False)
    HW_basins = HW_basins_gdf.boundary

    return HW_basins_gdf, HW_basins                       