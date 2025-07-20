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



# ─── 2. Helper functions ────────────────────────────────────────────────
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


# ─── 3. Button callback ─────────────────────────────────────────────────
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


