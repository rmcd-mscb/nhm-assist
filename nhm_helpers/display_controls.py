from ipywidgets import widgets
from IPython.display import display, clear_output
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
from nhm_helpers.output_plots import create_streamflow_plot
from nhm_helpers.map_template import make_streamflow_map
from nhm_helpers.nhm_output_visualization import retrieve_hru_output_info

import pathlib as pl
import os
root_dir = pl.Path(os.getcwd().rsplit("nhm-assist", 1)[0] + "nhm-assist")


def _get_valid_poi() -> str:
    """
    Return a valid POI identifier: the combobox value if valid,
    otherwise the first available POI from poi_df.
    """
    ids = poi_df.poi_id.values
    return v2.value if v2.value in ids else ids[0]


def generate_map() -> None:
    """
    Generate and display the Folium map for the selected variable, year, and POI.
    """
    poi_id = _get_valid_poi()
    fmap = make_var_map(
        root_dir,
        out_dir,
        v.value,
        plot_start_date,
        plot_end_date,
        water_years,
        hru_gdf,
        poi_df,
        poi_id,
        seg_gdf,
        html_maps_dir,
        year_list,
        yr.value,
        Folium_maps_dir,
        HW_basins,
        subdomain,
    )
    display(fmap)


def generate_summary() -> None:
    """
    Generate and display the summary time-series plot of HRU contributions
    for the selected variable and POI.
    """
    poi_id = _get_valid_poi()
    fig1 = make_plot_var_for_hrus_in_poi_basin(
        out_dir,
        param_filename,
        water_years,
        hru_gdf,
        poi_df,
        v.value,
        poi_id,
        plot_start_date,
        plot_end_date,
        plot_colors,
        subdomain,
        html_plots_dir,
    )
    display(fig1)


def generate_flux() -> None:
    """
    Generate and display the flux rates time-series plot for the selected
    variable and POI.
    """
    poi_id = _get_valid_poi()
    fig2 = oopla(
        out_dir,
        param_filename,
        water_years,
        hru_gdf,
        poi_df,
        output_var_list,
        v.value,
        poi_id,
        plot_start_date,
        plot_end_date,
        plot_colors,
        var_colors_dict,
        leg_only_dict,
        subdomain,
        html_plots_dir,
    )
    display(fig2)


def on_generate_clicked(b: widgets.Button) -> None:
    """
    When the Generate button is clicked, clear all outputs and
    create only the selected plots.
    """
    clear_output(wait=True)
    display(
        VBox([v, yr, v2, plot_checks, btn_generate, out_map, out_summary, out_flux])
    )

    # Map
    if cb_map.value:
        with out_map:
            clear_output(wait=True)
            generate_map()

    # Summary TS
    if cb_summary.value:
        with out_summary:
            clear_output(wait=True)
            generate_summary()

    # Flux TS
    if cb_flux.value:
        with out_flux:
            clear_output(wait=True)
            generate_flux()

def on_map_clicked(b: widgets.Button) -> None:
    """
    When clicked, clear previous map, default to first POI if none entered,
    then generate and display the streamflow map.
    """
    with map_out:
        clear_output(wait=True)
        poi_id_sel = gage_txt.value.strip() or poi_df.poi_id.tolist()[0]
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
        if isinstance(map_file, str):
            display(IFrame(src=map_file, width="100%", height="500px"))
        else:
            display(map_file)

def on_plot_clicked(b: widgets.Button) -> None:
    """
    When clicked, clear previous plot, default to first POI if none entered,
    then generate and display the streamflow plot.
    """
    with plot_out:
        clear_output(wait=True)
        poi_id_sel = gage_txt.value.strip() or poi_df.poi_id.tolist()[0]
        fplot = create_streamflow_plot(
            poi_id_sel,
            plot_start_date,
            plot_end_date,
            water_years,
            html_plots_dir,
            output_netcdf_filename,
            out_dir,
            subdomain,
        )
        display(fplot)
