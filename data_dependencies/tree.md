# Example directory tree

The following tree displays an example folder structure produced from the Willamette River subdomain model:

```
C:.
│   
└───nhm-assist
    │   0_workspace_setup.ipynb
    │   1_create_streamflow_observations.ipynb
    │   2_model_hydrofabric_visualization.ipynb
    │   3_model_parameter_visualization.ipynb
    │   4_run_model_using_pywatershed.ipynb
    │   5_hru_output_visualization.ipynb
    │   6_streamflow_output_visualization.ipynb
    │   code.json
    │   DISCLAIMER.md
    │   environment.yaml
    │   LICENSE.md
    │   pull_domain.py
    │   README.md
    │        
    ├───data_dependencies
    │   ├───HUC2
    │   │       HUC2.cpg
    │   │       HUC2.dbf
    │   │       HUC2.prj
    │   │       HUC2.shp
    │   │       HUC2.shx
    │   │       readme.txt
    │   │       
    │   └───NHM_v1_1
    │           nhm_v1_1_byhwobs_cal_gages.csv
    │           nhm_v1_1_HRU_cal_levels.csv
    │           readme.txt
    │           
    ├───domain_data
    │   └───willamette_river
    │       │   cbh.nc
    │       │   control.default.bandit
    │       │   default_gages.csv
    │       │   myparam.param
    │       │   NWISgages.csv
    │       │   prcp.nc
    │       │   sf_data
    │       │   tmax.nc
    │       │   tmin.nc
    │       │   
    │       ├───GIS
    │       │       model_layers.gpkg
    │       │       
    │       ├───notebook_output_files
    │       │   ├───Folium_maps
    │       │   │       adjmix_rain_14144800.txt
    │       │   │       etc.
    │       │   │       
    │       │   ├───html_maps
    │       │   │       gwflow_coef_map.html
    │       │   │       recharge_mean_annual_map.html
    │       │   │       streamflow_map.html
    │       │   │       
    │       │   ├───html_plots
    │       │   │       recharge_for_14152000_plot.html
    │       │   │       streamflow_eval_for_14152000_plot.html
    │       │   │       water_budget_fluxes_for_14152000_plot.html
    │       │   │       
    │       │   └───nc_files
    │       │           nwis_cache.nc
    │       │           owrd_cache.nc
    │       │           sf_efc.nc
    │       │           
    │       └───output
    │               gwres_flow.nc
    │               gwres_flow_vol.nc
    │               gwres_sink.nc
    │               gwres_stor.nc
    │               gwres_stor_change.nc
    │               hru_actet.nc
    │               hru_streamflow_out.nc
    │               net_ppt.nc
    │               net_rain.nc
    │               net_snow.nc
    │               recharge.nc
    │               seg_outflow.nc
    │               snowmelt.nc
    │               sroff.nc
    │               sroff_vol.nc
    │               ssres_flow.nc
    │               ssres_flow_vol.nc
    │               ssres_stor.nc
    │               unused_potet.nc
    │               
    └───nhm_helpers
        │   efc.py
        │   map_template.py
        │   nhm_assist_utilities.py
        │   nhm_helpers.py
        │   nhm_hydrofabric.py
        │   nhm_output_visualization.py
        │   output_plots.py
        └───sf_data_retrieval.py              
```
