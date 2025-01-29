#!/usr/bin/env python3

import numpy as np


def efc(df, flow_col):
    """Run all the steps for extreme flood classification and return
    a Pandas dataframe

    Expects a dataframe with streamflow values (additional columns are allowed).
    The `flow_col` argument specifies the name of the streamflow column"""
    df2 = df.copy()

    # Compute recurrence intervals
    df2['ri'] = compute_recurrence_interval(df2.loc[:, flow_col])

    # Compute high/low
    df2['high_low'] = compute_high_low(df2.loc[:, flow_col])

    # Compute EFC
    df2['efc'] = compute_efc(df2, flow_col=flow_col)

    return df2


def get_first_valid(ds):
    """Return index to the first valid non-zero streamflow value

    Expects a Pandas timeseries of streamflow values"""
    # Get the index to the first valid (zero or greater) streamflow value
    # Expects a Pandas time series
    first_valid = None

    for dd in range(0, len(ds.index) - 1):
        if (ds[dd] >= 0.0) and (ds[dd + 1] >= 0.0):
            first_valid = dd
            break
    return first_valid


def compute_efc(df, flow_col):
    """Compute the extreme flood classifications for a streamflow timeseries

    This routine expects a Pandas dataframe that has recurrence interval column
    named `ri` and a high_low classification column named `high_low`. Name of
    the streamflow column is specified with the `flow_col` argument."""
    # ~~~~~~~~
    # Part 3 - Assign EFC classifications
    # ~~~~~~~~
    # EFC classifications
    # 1 = Large floods
    # 2 = Small floods
    # 3 = High flow pulses
    # 4 = Low flows
    # 5 = Extreme low flows

    numdays = len(df.index)

    # Initialize the efc array
    efc_arr = np.zeros(numdays, dtype='int')
    efc_arr[:] = -1

    ts_q = df.loc[:, flow_col]

    # 10th percentile of the streamflow obs
    p10_q = ts_q[ts_q > 0.0].quantile(q=0.1, interpolation='nearest').item()

    for dd in range(0, numdays):
        if ts_q[dd] >= 0.0:
            if df['high_low'][dd] > 1:
                # High flows
                if df['ri'][dd] > 10.0:
                    efc_arr[dd] = 1  # Large flood
                elif df['ri'][dd] > 2.0:
                    efc_arr[dd] = 2  # Small flood
                else:
                    efc_arr[dd] = 3  # Default to high flow pulse
            elif df['high_low'][dd] == 1:
                # Low flow events
                if ts_q[dd] < p10_q:
                    # Extreme low flow event
                    efc_arr[dd] = 5
                else:
                    efc_arr[dd] = 4  # Default to low flow
    return efc_arr


def compute_high_low(df):
    """Classify streamflow in high flows (1, 2, 3) and low flows (4, 5)

    Expects a Pandas timeseries of streamflow.
    Returns an ndarray of high/low flow classification"""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Part 2 - Classify the high and low flows
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # High/Low flow classifications
    # 1 = low flow
    # 2 = Ascending limb
    # 3 = Descending limb

    numdays = len(df.index)
    ts_q = df

    # Setup array for high/low flow classifications
    high_low = np.zeros(numdays, dtype='int')
    high_low[:] = -1

    # Median and 75th percentile of the streamflow obs
    median_q = ts_q[ts_q > 0.0].quantile(q=0.5, interpolation='nearest').item()
    p75_q = ts_q[ts_q > 0.0].quantile(q=0.75, interpolation='nearest').item()

    has_gap = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Now classify the remaining daily flows
    # for cday in range(first_valid+2, numdays):
    for cday in range(1, numdays):
        prior_day = cday - 1

        if ts_q[cday] >= 0.0:
            if has_gap:
                # Initialize the first two valid days to a low flow
                high_low[cday] = 1
                high_low[cday+1] = 1

                # Compute the 25% greater and 25% lesser flows
                qq125 = ts_q[cday] * 1.25  # 25% greater
                qq75 = ts_q[cday] * .75   # 25% less

                # Check if the flow in day 1 greater than the median or if the flow on day 2 is
                # 25% greater than on day 1
                if (ts_q[cday] > median_q) or (ts_q[cday+1] >= qq125):
                    # Classify as ascending
                    high_low[cday] = 2
                    high_low[cday+1] = 2

                # If day 2 flow drops by more than 25% compared to day 1 then it is descending
                if ts_q[cday+1] < qq75:
                    high_low[cday] = 3
                    high_low[cday+1] = 3

                has_gap = False
                continue

            # Compute the 25% greater, 25% lesser and 10% lesser prior day flows
            qq125 = ts_q[prior_day] * 1.25  # 25% greater
            # qq75 = ts_q[prior_day] * .75   # 25% less
            qq90 = ts_q[prior_day] * .9    # 10% less

            if ts_q[cday] < median_q:
                # Classify as a low flow day
                high_low[cday] = 1
                continue

            if high_low[prior_day] == 1:
                # The prior day was a low flow day, check if today is still a
                # low flow or an ascending limb
                if ts_q[cday] > p75_q or (ts_q[cday] > median_q and ts_q[cday] > qq125):
                    # Ascending flow if Q > 75th percentile
                    high_low[cday] = 2
                else:
                    high_low[cday] = 1
            elif high_low[prior_day] == 2:
                # The prior day is an ascending limb (2) so continue ascending
                # until daily flow decreases by more than 10% at which time
                # descending (3) limb is initiated.
                if ts_q[cday] > qq90:
                    high_low[cday] = 2  # Ascending
                else:
                    high_low[cday] = 3  # Descending
            elif high_low[prior_day] == 3:
                # If the prior day is descending then ascending is restarted if
                # current day flow increases by more than 25%.
                if ts_q[cday] > qq125:
                    high_low[cday] = 2  # Ascending
                else:
                    high_low[cday] = 3  # Descending

                # If the rate of decrease drops below 10% per day
                # then event is ended unless the flow is still greater
                # than the 75th percentile.
                if ts_q[cday] > p75_q:
                    high_low[cday] = 3  # Descending
                else:
                    if ts_q[cday] < qq90:
                        high_low[cday] = 1  # Low flow
                    else:
                        high_low[cday] = 3  # Descending
        else:
            # We have a missing value
            has_gap = True

    return high_low


def compute_recurrence_interval(df):
    """Compute the recurrence intervals for streamflow

    Expects a Pandas timeseries of streamflow values.
    Returns an ndarray of recurrence intervals"""
    df_arr = df.values

    ri = np.zeros(len(df.index))
    ri[:] = np.nan

    # Get array of indices that would result in a sorted array
    sorted_ind = np.argsort(df_arr)

    numobs = df_arr[df_arr > 0.0, ].shape[0]  # Number of observations > 0.
    nyr = float(numobs) / 365.0  # Number of years of non-zero observations

    nz_cnt = 0  # non-zero value counter
    for si in sorted_ind:
        if df_arr[si] > 0.0:
            nz_cnt += 1
            rank = numobs - nz_cnt + 1
            ri[si] = (nyr + 1.0) / float(rank)
    return ri