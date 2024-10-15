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
    ri[:] = np.NAN

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


# def main():
#     write_dat = True  # Whether to output ri_dat, efc_dat, and highlow_dat
#     write_efc = True  # Whether to output the efc_subdivide file
#     write_highlow = True  # Whether to output the HIGH6.LOW7_subdivide file
#
#     debug_day = -1  # Index to print debug info for; -1 to disable
#
#     gage_id = '07249985'
#
#     streamflow_file = 'streamflow.data.07249985'
#
#     sf = prms.streamflow(streamflow_file)
#     sf_data = sf.data.iloc[:, 0]
#
#     numdays = sf_data.shape[0]  # total number of days of data (including Nan)
#
#     sf_nonans = sf_data.dropna(axis=0, how='any', inplace=False)
#
#     if debug_day != -1:
#         print(f'Number of days: {numdays}')
#
#     # ===========================================================================
#     # Compute and write the recurrence interval for runoff
#     # ===========================================================================
#     ri_runoff = get_recurrence_interval(sf_data)
#
#     if write_dat:
#         thedays = np.arange(1, numdays + 1)
#
#         ri_stack = np.column_stack((thedays, ri_runoff))
#         myhdr = 'rundays RI'
#         myfmt = '%5d ' + '%8.4f '  # Setup format for writing the data
#
#         if debug_day != -1:
#             print('Writing ri_dat')
#         np.savetxt('ri_dat.%s' % gage_id, ri_stack, delimiter=" ", header=myhdr, fmt=myfmt)
#     # ---------------------------------------------------------------------------
#
#     # ===========================================================================
#     # Classify High and Low flows
#     # ===========================================================================
#
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Part 1 - setup for the classification
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Requires:
#     #    median
#     #    10th percentile
#     #    75th percentile
#     #    number of days of record
#
#     # For median and 75th percentile we only deal with values > 0.0
#     q_data = sf_nonans[sf_nonans > 0.]
#
#     median_q = q_data.median(axis=0)
#     p75_q = np.percentile(q_data, 75)
#     p10_q = np.percentile(q_data, 10)
#
#     if debug_day != -1:
#         print(f'Median: {median_q}')
#         print(f'  75th: {p75_q}')
#         print(f'  10th: {p10_q}')
#
#     # Get the full data including nans as an ndarray
#     q_data = sf_data.values
#
#     # Get index to first value >= 0 for each station
#     first_valid = get_first_valid(q_data)
#
#     if first_valid is None:
#         print('No starting index greater than or equal to zero was found')
#         exit()
#
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Part 2 - Classify the high and low flows
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#     # Setup array for high/low flow classifications
#     high_low = np.zeros(q_data.shape, dtype='int')
#     high_low[:] = -1
#
#     # High/Low flow classifications
#     # 1 = low flow
#     # 2 = Ascending limb
#     # 3 = Descending limb
#
#     if debug_day != -1:
#         print('Classify high/low flows')
#
#     if q_data[first_valid] < 0:
#         # This should *never* happen
#         print(f'ERROR: First valid flow value is less than zero for day {first_valid}')
#         exit()
#
#     if q_data[first_valid + 1] < 0:
#         # This should *never* happen
#         print(f'ERROR: Second valid flow value is less than zero for day {first_valid}')
#         exit()
#
#     # Initialize the high/low class to a low flow
#     high_low[first_valid] = 1
#     high_low[first_valid + 1] = 1
#
#     # Compute the 25% greater and 25% lesser flows
#     qq125 = q_data[first_valid] * 1.25  # 25% greater
#     qq75 = q_data[first_valid] * .75  # 25% less
#
#     # Check if the flow in day 1 greater than the median or if the flow on day 2 is
#     # 25% greater than on day 1
#     if (q_data[first_valid] > median_q) or (q_data[first_valid + 1] >= qq125):
#         # Classify as ascending
#         high_low[first_valid] = 2
#         high_low[first_valid + 1] = 2
#
#     # If day 2 flow drops by more than 25% compared to day 1 then it is descending
#     if q_data[first_valid + 1] < qq75:
#         high_low[first_valid] = 3
#         high_low[first_valid + 1] = 3
#
#     # Now classify flows for the remaining days for this station
#     for dd in range(first_valid + 2, numdays):
#         if q_data[dd] >= 0.0:
#             # Compute the 25% greater, 25% lesser and 10% lesser prior day flows
#             qq125 = q_data[dd - 1] * 1.25  # 25% greater
#             qq75 = q_data[dd - 1] * .75  # 25% less
#             qq90 = q_data[dd - 1] * .9  # 10% less
#
#             if dd == debug_day:
#                 print('dd:%d obs:%0.1f median:%0.1f qq125:%0.1f qq75:%0.1f qq90:%0.1f' % (dd, q_data[dd], median_q, qq125, qq75, qq90))
#
#             if q_data[dd] < median_q:
#                 # Classify as a low flow day
#                 high_low[dd] = 1
#
#                 if dd == debug_day:
#                     print('q_data < median_q: HL=1')
#             elif high_low[dd - 1] == 1:
#                 # If the prior day was a low flow day, check if today is still a
#                 # low flow or an ascending limb
#                 if q_data[dd] > p75_q or (q_data[dd] > median_q and q_data[dd] > qq125):
#                     # Ascending flow if Q > 75th percentile or daily increase is > 25%
#                     high_low[dd] = 2
#                     if dd == debug_day:
#                         print('HL1: q_data > p75_q or (q_data > median_q and q_data > qq125): HL=2')
#                 else:
#                     high_low[dd] = 1
#                     if dd == debug_day:
#                         print('HL1: HL=1')
#             elif high_low[dd - 1] == 2:
#                 # If the prior day is an ascending limb (2) then continue ascending
#                 # until daily flow decreases by more than 10% at which time
#                 # descending (3) limb is initiated.
#                 if q_data[dd] > qq90:
#                     high_low[dd] = 2  # Ascending
#                     if dd == debug_day:
#                         print('HL2: q_data > qq90: HL=2')
#                 else:
#                     high_low[dd] = 3  # Descending
#                     if dd == debug_day:
#                         print('HL2: q_data <= qq90: HL=3')
#             elif high_low[dd - 1] == 3:
#                 # If the prior day is descending then ascending is restarted if
#                 # current day flow increases by more than 25%.
#                 if dd == debug_day:
#                     print('HL3: prior:%0.1f curr:%0.1f' % (q_data[dd - 1], q_data[dd]))
#
#                 if q_data[dd] > qq125:
#                     high_low[dd] = 2  # Ascending
#                     if dd == debug_day:
#                         print('HL3: q_data > qq125: HL=2')
#                 else:
#                     high_low[dd] = 3  # Descending
#                     if dd == debug_day:
#                         print('HL3: q_data <= qq125: HL=3')
#
#                 # If the rate of decrease drops below 10% per day
#                 # then event is ended unless the flow is still greater
#                 # than the 75th percentile.
#                 if q_data[dd] > p75_q:
#                     high_low[dd] = 3  # Descending
#                     if dd == debug_day:
#                         print('HL3: q_data > p75_q: HL=3')
#                 else:
#                     if q_data[dd] < qq90:
#                         high_low[dd] = 1  # Low flow
#                         if dd == debug_day:
#                             print('HL3: q_data <= p75_q and q_data < qq90: HL=1')
#                     else:
#                         high_low[dd] = 3  # Descending
#                         if dd == debug_day:
#                             print('HL3: q_data <= p75_q and q_data >= qq90: HL=3')
#         if dd == debug_day:
#             print(f'final high_low: {high_low[dd]}')
#
#     # ~~~~~~~~
#     # Part 3 - Assign EFC classifications
#     # ~~~~~~~~
#     if debug_day != -1:
#         print('-' * 40)
#         print('Extreme Flood Classification (EFC)')
#     # EFC classifications
#     # 1 = Large floods
#     # 2 = Small floods
#     # 3 = High flow pulses
#     # 4 = Low flows
#     # 5 = Extreme low flows
#
#     # Initialize the efc array
#     efc = np.zeros(q_data.shape, dtype='int')
#     efc[:] = -1
#
#     for dd in range(0, numdays):
#         if q_data[dd] >= 0.0:
#             if dd == debug_day:
#                 print(f'RI: {ri_runoff[dd]}')
#
#             if high_low[dd] > 1:
#                 # High flow events
#                 if ri_runoff[dd] > 10.0:
#                     efc[dd] = 1  # Large flood
#
#                     if dd == debug_day:
#                         print('EFC: \tri_runoff > 10; efc=1')
#                 elif ri_runoff[dd] > 2.0:
#                     efc[dd] = 2  # Small flood
#
#                     if dd == debug_day:
#                         print('EFC: \tri_runoff > 2; efc=2')
#                 else:
#                     # Default to a high flow pulse
#                     efc[dd] = 3
#
#                     if dd == debug_day:
#                         print('EFC: high_low > 1; efc=3')
#
#                 # efc[dd] = 3  # Default to high flow pulse
#                 #
#                 # if dd == debug_day:
#                 #     print('EFC: high_low > 1; efc=3')
#                 #
#                 # if ri_runoff[dd] > 2.0:
#                 #     efc[dd] = 2  # Small flood
#                 #
#                 #     if dd == debug_day:
#                 #         print('EFC: \tri_runoff > 2; efc=2')
#                 #
#                 # if ri_runoff[dd] > 10.0:
#                 #     efc[dd] = 1  # Large flood
#                 #
#                 #     if dd == debug_day:
#                 #         print('EFC: \tri_runoff > 10; efc=1')
#             elif high_low[dd] == 1:
#                 # Low flow events
#                 if q_data[dd] < p10_q:
#                     # Extreme low flow event
#                     efc[dd] = 5
#
#                     if dd == debug_day:
#                         print('EFC: \tq_data < p10_q; efc=5')
#                 else:
#                     efc[dd] = 4  # Default to low flow
#
#                     if dd == debug_day:
#                         print('EFC: high_low == 1; efc=4')
#
#                 # efc[dd] = 4  # Default to low flow
#                 #
#                 # if dd == debug_day:
#                 #     print('EFC: high_low == 1; efc=4')
#
#     if debug_day != -1:
#         print('-' * 40)
#
#     # Get the year, month, day information
#     # This is used for writing out thefiles
#     year = sf_data.index.year
#     wyear = sf_data.index.year
#     month = sf_data.index.month
#     day = sf_data.index.day
#
#     for ii, yy in enumerate(year):
#         if month[ii] > 9:
#             wyear[ii] += 1
#
#     timedata = np.column_stack((year, month, day))
#     thedays = np.arange(1, numdays + 1)
#
#     if write_dat:
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Write the efc data out
#         efc_stack = np.column_stack((wyear, timedata, thedays, efc))
#         myhdr = 'wyear year month day rundays efc'
#         myfmt = '%4d %4d %2d %2d %6d ' + '%2d '  # Setup format for writing the data
#
#         if debug_day != -1:
#             print('Writing efc_dat')
#         np.savetxt(f'efc_dat.{gage_id}', efc_stack, delimiter=' ', header=myhdr, fmt=myfmt)
#
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # Write the high_low data out
#         highlow_stack = np.column_stack((wyear, timedata, thedays, high_low))
#         myhdr = 'wyear year month day rundays highlow'
#         myfmt = '%4d %4d %2d %2d %6d ' + '%2d '  # Setup format for writing the data
#
#         if debug_day != -1:
#             print('Writing highlow_dat')
#
#         np.savetxt(f'highlow_dat.{gage_id}', highlow_stack, delimiter=' ', header=myhdr, fmt=myfmt)
#
#     # ---------------------------------------------------------------------------
#     # ---------------------------------------------------------------------------
#
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Write the efc_subdivide.* files
#     if debug_day != -1:
#         print('Writing efc_subdivide')
#
#     efc_stack = np.column_stack((timedata, efc[:]))
#     myhdr = 'year month day efc'
#     myfmt = '%4d %2d %2d ' + '%2d'
#
#     np.savetxt(f'efc_subdivide.{gage_id}', efc_stack, delimiter=' ', header=myhdr, fmt=myfmt)
#
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # Write the HIGH6.LOW7_subdivide.* files
#     if debug_day != -1:
#         print('Writing HIGH6.LOW7_subdivide')
#
#     # Create altered efc array for output
#     h6_l7 = np.copy(efc)
#     np.place(h6_l7, h6_l7 == 1, [6])
#     np.place(h6_l7, h6_l7 == 2, [6])
#     np.place(h6_l7, h6_l7 == 3, [6])
#     np.place(h6_l7, h6_l7 == 4, [7])
#     np.place(h6_l7, h6_l7 == 5, [7])
#     np.place(h6_l7, h6_l7 < 0, [0])
#
#     # Convert the nan values for streamflow to -999.0
#     q_data = np.where(np.isnan(q_data), -999.0, q_data)
#
#     h6_l7_stack = np.column_stack((timedata, h6_l7[:], q_data[:], efc[:]))
#     myhdr = 'year month day highlow Q efc'
#     myfmt = '%4d %2d %2d ' + '%2d %0.3f %2d'
#
#     np.savetxt(f'HIGH6.LOW7_subdivide.{gage_id}', h6_l7_stack, delimiter=' ', header=myhdr, fmt=myfmt)
#
#
# if __name__ == '__main__':
#     main()
