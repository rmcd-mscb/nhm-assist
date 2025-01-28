### Helper Functions
# Reads/Creates NWIS stations file if not already created
def fetch_nwis_gage_info(
    nwis_gage_nobs_min,
    model_domain_regions,
    st_date,
    en_date,
    hru_gdf,
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
        include_cols = [
            "agency_cd",
            "site_no",
            "station_nm",
            "dec_lat_va",
            "dec_long_va",
            "drain_area_va",
            "contrib_drain_area_va",
        ]
        nwis_gage_info_aoi = nwis_gage_info_aoi.loc[:, include_cols]

        field_map = {
            "agency_cd": "poi_agency",
            "site_no": "poi_id",
            "station_nm": "poi_name",
            "dec_lat_va": "latitude",
            "dec_long_va": "longitude",
            "drain_area_va": "drainage_area",
            "contrib_drain_area_va": "drainage_area_contrib",
        }

        nwis_gage_info_aoi.rename(columns=field_map, inplace=True)
        nwis_gage_info_aoi.set_index("poi_id", inplace=True)
        nwis_gage_info_aoi = nwis_gage_info_aoi.sort_index()
        nwis_gage_info_aoi.reset_index(inplace=True)

        # write out the file for later
        nwis_gage_info_aoi.to_csv(nwis_gages_file, index=False)  # , sep='\t')
    return nwis_gage_info_aoi




def subset_stream_network(dag_ds, uscutoff_seg, dsmost_seg):  # (from Bandit)
    """Extract subset of stream network

    :param dag_ds: Directed, acyclic graph of downstream stream network
    :param uscutoff_seg: List of upstream cutoff segments
    :param dsmost_seg: List of outlet segments to start extraction from

    :returns: Stream network of extracted segments
    """

    # taken from Bandit bandit_helpers.py

    # Create the upstream graph
    dag_us = dag_ds.reverse()

    # Trim the u/s graph to remove segments above the u/s cutoff segments
    try:
        for xx in uscutoff_seg:
            try:
                dag_us.remove_nodes_from(nx.dfs_predecessors(dag_us, xx))

                # Also remove the cutoff segment itself
                dag_us.remove_node(xx)
            except KeyError:
                print(f"WARNING: nhm_segment {xx} does not exist in stream network")
    except TypeError:
        print(
            "\nSelected cutoffs should at least be an empty list instead of NoneType."
        )

    # =======================================
    # Given a d/s segment (dsmost_seg) create a subset of u/s segments

    # Get all unique segments u/s of the starting segment
    uniq_seg_us: Set[int] = set()
    if dsmost_seg:
        for xx in dsmost_seg:
            try:
                pred = nx.dfs_predecessors(dag_us, xx)
                uniq_seg_us = uniq_seg_us.union(
                    set(pred.keys()).union(set(pred.values()))
                )
            except KeyError:
                print(f"KeyError: Segment {xx} does not exist in stream network")

        # Get a subgraph in the dag_ds graph and return the edges
        dag_ds_subset = dag_ds.subgraph(uniq_seg_us).copy()

        node_outlets = [ee[0] for ee in dag_ds_subset.edges()]
        true_outlets = set(dsmost_seg).difference(set(node_outlets))

        # Add the downstream segments that exit the subgraph
        for xx in true_outlets:
            nhm_outlet = list(dag_ds.neighbors(xx))[0]
            dag_ds_subset.add_node(
                nhm_outlet, style="filled", fontcolor="white", fillcolor="grey"
            )
            dag_ds_subset.add_edge(xx, nhm_outlet)
            dag_ds_subset.nodes[xx]["style"] = "filled"
            dag_ds_subset.nodes[xx]["fontcolor"] = "white"
            dag_ds_subset.nodes[xx]["fillcolor"] = "blue"
    else:
        # No outlets specified so pull the full model
        dag_ds_subset = dag_ds

    return dag_ds_subset


def hrus_by_seg(pdb, segs):  # (custom code)
    # segs: global segment IDs

    if isinstance(segs, int):
        segs = [segs]
    elif isinstance(segs, KeysView):
        segs = list(segs)

    seg_hrus = {}
    seg_to_hru = pdb.seg_to_hru

    # Generate stream network for the model
    dag_streamnet = pdb.stream_network()

    for cseg in segs:
        # Lookup segment for the current POI
        dsmost_seg = [cseg]

        # Get subset of stream network for given POI
        dag_ds_subset = subset_stream_network(dag_streamnet, set(), dsmost_seg)

        # Create list of segments in the subset
        toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges))

        # Build list of HRUs that contribute to the POI
        final_hru_list = []

        for xx in toseg_idx:
            try:
                for yy in seg_to_hru[xx]:
                    final_hru_list.append(yy)
            except KeyError:
                # print(f'Segment {xx} has no HRUs connected to it') # comment this out and add pass to not print the KeyError
                pass
        final_hru_list.sort()

        seg_hrus[cseg] = final_hru_list

    return seg_hrus


def hrus_by_poi(pdb, poi):  # (custom code)
    if isinstance(poi, str):
        poi = [poi]
    elif isinstance(poi, KeysView):
        poi = list(poi)

    poi_hrus = {}
    nhm_seg = pdb.get("nhm_seg").data
    pois_dict = pdb.poi_to_seg
    seg_to_hru = pdb.seg_to_hru

    # Generate stream network for the model
    dag_streamnet = pdb.stream_network()

    for cpoi in poi:
        # Lookup global segment id for the current POI
        dsmost_seg = [nhm_seg[pois_dict[cpoi] - 1]]

        # Get subset of stream network for given POI
        dag_ds_subset = subset_stream_network(dag_streamnet, set(), dsmost_seg)

        # Create list of segments in the subset
        toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges))

        # Build list of HRUs that contribute to the POI
        final_hru_list = []

        for xx in toseg_idx:
            try:
                for yy in seg_to_hru[xx]:
                    final_hru_list.append(yy)
            except KeyError:
                # Not all segments have HRUs connected to them
                # print(f'{cpoi}: Segment {xx} has no HRUs connected to it')
                pass
        final_hru_list.sort()
        poi_hrus[cpoi] = final_hru_list

    return poi_hrus