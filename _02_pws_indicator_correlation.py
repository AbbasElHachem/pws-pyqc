# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    Filter pws stations based on Indicator Correlation
Purpose: Find validity of pws Station for interpolation purposes

Created on: 2020-05-16

For every pws precipitation station, select the period of 2015-2019,
find the nearest prim_netw station, intersect both stations, same time period
construct the EDF for the pws and the prim_netw stations respectively 
based on a given percentage threshold, find the corresponding rainfall value,
and transform all values above ppt_thr to 1 and below to 0, 2 Boolean series
calculate the pearson correlation between the two boolean stations data
for the corresponding prim_netw station find it's prim_netw neighboring station
repeat same procedure now between the prim_netw-prim_netw station and find the pearson
indicator correlation between the transformed boolean data sets.

If the value of the indicator correlation between the Netamo-prim_netw pair is 
greater or equal to the value betweeb the prim_netw-prim_netw pait, keep pws station

Repeat this procedure for all pws station, or different
quantile threshold (a probabilistic threshold) and for
different neighbors and temporal resolution.

Save the result to a df

Parameters
----------

Input Files
    prim_netw station data
    prim_netw coordinates data
    pws precipitation station data
    pws station coordinates data
    Distance matrix between prim_netw and pws stations
    
Returns
-------

    
Df_correlations: df containing for every Netamo station, location (lon, lat),
    seperating distance, the pearson correlations of original data and
    for boolean transfomed data compared with the nearest prim_netw station.
    
Plot everything in the dataframe using a different script
Especially the change of correlation with distance
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

# generic Libs
import os
from pathlib import Path
import sys
import time
import timeit

import psutil
from scipy.spatial import cKDTree
from scipy.stats import pearsonr as pears

from _00_functions import (# resample_intersect_2_dfs,
    select_convective_season,
    select_df_within_period,
    get_cdf_part_abv_thr,
    # shift_dataframe_summer_time
)
from _01_2_read_hdf5 import HDF5
import multiprocessing as mp
import numpy as np
import pandas as pd

# other Libs
# own functions
#==============================================================================
# HOURLY DATA
#==============================================================================
# path_to_ppt_pws_data_hdf5 = (
#     r"C:\Users\hachem\Downloads\pws_filtered_2018_2019_5min_to_1hour.h5")
# assert os.path.exists(path_to_ppt_pws_data_hdf5), 'wrong pws Ppt file'
# path_to_ppt_pws_data_hdf5 = (
#     r"P:\2020_DFG_pws\03_data\01_pws\pws_Germany_5min_to_1hour.h5")
# _to_1hour_

# assert os.path.exists(path_to_ppt_pws_data_hdf5), 'wrong pws Ppt file'

path_to_ppt_pws_data_hdf5 = (
    r"X:\staff\elhachem\Data\pws_data\rain_RLP_2020_1hour\pws_RLP_2020__60min_2020.h5"
)
assert os.path.exists(path_to_ppt_pws_data_hdf5), 'wrong pws Ppt file'

path_to_ppt_prim_netw_data_hdf5 = (
    # r"P:\2020_DFG_pws\03_data\03_prim_netw\prim_netw_5min.h5")
    r"X:\staff\elhachem\ClimXtreme\03_data\00_prim_netw\prim_netw_comb_60min_data_agg_60min_2020.h5"
    # r"X:\staff\elhachem\ClimXtreme\03_data\00_prim_netw\prim_netw_comb_60min_1990_2020.h5"
)
assert os.path.exists(path_to_ppt_prim_netw_data_hdf5), 'wrong prim_netw Csv Ppt file'

# =============================================================================

# min distance threshold used for selecting neighbours
min_dist_thr_ppt = 100 * 1e4  # in m, for ex: 30km or 50km

# threshold for max ppt value per hour
max_ppt_thr = 100.  # ppt above this value are not considered

# only highest x% of the values are selected
lower_percentile_val_lst = [99.]  # 97., 98., 99., 99.5

# temporal frequencies on which the filtering should be done

aggregation_frequencies = ['60min']
_year = '2020'

# [0, 1, 2, 3, 4]  # refers to prim_netw neighbot (0=first)
neighbors_to_chose_lst = [0]  # , 1, 2, 3]  # 4, 5, 6, 7

# all pwss have more than 2 month data, this an extra check
min_req_ppt_vals = 0  # 2 * 24 * 30

# this is used to keep only data where month is not in this list
# not_convective_season = [10, 11, 12, 1, 2, 3, 4]  # oct till april
not_convective_season = [11, 12]  # this means all months are used

# date format of dataframes
date_fmt = '%Y-%m-%d %H:%M:%S'

# select data only within this period
start_date = '%s-01-01 00:00:00' % _year
end_date = '%s-08-30 23:00:00' % _year

# calculate no of workers
n_workers = 7  # 6  # mp.cpu_count() - 2

shift_pws_df = False
shift_by = -1

out_save_dir_orig = (
    r'X:\staff\elhachem\2020_05_20_pws_CML'
    r'\indicator_correlation_%s_%s_RLP2020' % (aggregation_frequencies[0],
                                               str(lower_percentile_val_lst[0]).replace('.', '_')))

if not os.path.exists(out_save_dir_orig):
    os.mkdir(out_save_dir_orig)
#==============================================================================
#
#==============================================================================


def process_manager(args):

    (# path_to_pws_coords_utm32,
        # path_to_prim_netw_coords_utm32,
        path_pws_ppt_df_hdf5,
        path_to_prim_netw_data_hdf5,
        tem_freq,
        neighbor_to_chose,
        val_thr_percent,
        min_req_ppt_vals) = args

    # get all station names for prim_netw
    HDF5_prim_netw = HDF5(infile=path_to_prim_netw_data_hdf5)
    all_prim_netw_stns_ids = HDF5_prim_netw.get_all_names()

    # get all station names for pws
    HDF5_pws = HDF5(infile=path_pws_ppt_df_hdf5)
    all_pws_ids = HDF5_pws.get_all_names()

    pws_coords = HDF5_pws.get_coordinates(all_pws_ids)

    in_pws_df_coords_utm32 = pd.DataFrame(
        index=all_pws_ids,
        data=pws_coords['easting'], columns=['X'])
    y_pws_coords = pws_coords['northing']
    in_pws_df_coords_utm32.loc[:, 'Y'] = y_pws_coords

#     in_pws_df_coords_utm32 = pd.read_csv(
#         path_to_pws_coords_utm32, sep=';',
#         index_col=0, engine='c')

    prim_netw_coords = HDF5_prim_netw.get_coordinates(all_prim_netw_stns_ids)

    in_prim_netw_df_coords_utm32 = pd.DataFrame(
        index=all_prim_netw_stns_ids,
        data=prim_netw_coords['easting'], columns=['X'])
    y_prim_netw_coords = prim_netw_coords['northing']
    in_prim_netw_df_coords_utm32.loc[:, 'Y'] = y_prim_netw_coords

#     in_prim_netw_df_coords_utm32 = pd.read_csv(
#         path_to_prim_netw_coords_utm32, sep=';',
#         index_col=0, engine='c')

    # create a tree from prim_netw coordinates

    prim_netw_coords_xy = [(x, y) for x, y in zip(
        in_prim_netw_df_coords_utm32.loc[:, 'X'].values,
        in_prim_netw_df_coords_utm32.loc[:, 'Y'].values)]

    # create a tree from coordinates
    prim_netw_points_tree = cKDTree(prim_netw_coords_xy)

    prim_netw_stns_ids = in_prim_netw_df_coords_utm32.index

    # df_results_correlations = pd.DataFrame(index=all_prim_netw_stns_ids

    print('Using Workers: ', n_workers)
    # devide stations on workers
    all_pws_stns_ids_worker = np.array_split(all_pws_ids, n_workers)
    args_worker = []

    for stns_list in all_pws_stns_ids_worker:
        df_results_correlations = pd.DataFrame(index=stns_list)
    # args_workers = list(repeat(args, n_worker))

        args_worker.append((path_to_prim_netw_data_hdf5,
                            in_prim_netw_df_coords_utm32,
                            path_pws_ppt_df_hdf5,
                            in_pws_df_coords_utm32,
                            stns_list,
                            prim_netw_points_tree,
                            prim_netw_stns_ids,
                            df_results_correlations,
                            tem_freq,
                            neighbor_to_chose,
                            val_thr_percent,
                            min_req_ppt_vals))

    # l = mp.Lock()
    # , initializer=init, initargs=(l,))
    my_pool = mp.Pool(processes=n_workers)
    # TODO: Check number of accounts

    results = my_pool.map(
        compare_pws_prim_netw_indicator_correlations, args_worker)

    # my_pool.terminate()

    my_pool.close()
    my_pool.join()

    results_df = pd.concat(results)

    if shift_pws_df and shift_by > 0:
        save_acc = 'sp%d' % abs(shift_by)
    elif shift_pws_df and shift_by < 0:
        save_acc = 'sm%d' % abs(shift_by)
    else:
        save_acc = ''

    results_df.to_csv(
        os.path.join(out_save_dir_orig,
                     'df_indic_corr_flt_sep_dist_%dkm_'
                     'freq_%s_upper_%s_per'
                     '_neighbor_%d_%s_%s.csv'  # filtered_95
                     % (min_dist_thr_ppt / 1e4, tem_freq,
                        str(val_thr_percent).replace('.', '_'),
                         neighbor_to_chose, save_acc, _year)),
        sep=';')

    return

#==============================================================================
#
#==============================================================================

# =============================================================================
# Main Function
# =============================================================================


def compare_pws_prim_netw_indicator_correlations(args):
    '''
     Find then for the pws station the neighboring prim_netw station
     intersect both stations, for the given probabilistic percentage
     threshold find the corresponding ppt_thr from the CDF of each station
     seperatly, make all values boolean (> 1, < 0) and calculate the pearson
     rank correlation between the two stations

     Add the result to a new dataframe and return it

    '''
    # print('\n######\n getting all station names, reading dfs \n#######\n')
    (path_to_prim_netw_data_hdf5,
     in_prim_netw_df_coords_utm32,
     path_pws_ppt_df_hdf5,
     in_pws_df_coords_utm32,
     all_pws_ids,
     prim_netw_points_tree,
     prim_netw_stns_ids,
     df_results_correlations,
     tem_freq,
     neighbor_to_chose,
     val_thr_percent,
     min_req_ppt_vals) = args

    # get all pws and prim_netw data
    HDF5_pws = HDF5(infile=path_pws_ppt_df_hdf5)

    HDF5_prim_netw = HDF5(infile=path_to_prim_netw_data_hdf5)

    alls_stns_len = len(all_pws_ids)  # to count number of stations

    # iterating through pws ppt stations
    for ppt_stn_id in all_pws_ids:

        print('\n**\n pws stations is %d/%d**\n'
              % (alls_stns_len, len(all_pws_ids)))

        alls_stns_len -= 1  # reduce number of remaining stations
        # ppt_stn_id = '70:ee:50:27:72:44'
        try:
            # read first pws station
            try:
                pws_ppt_stn1_orig = HDF5_pws.get_pandas_dataframe(
                    ppt_stn_id)

            except Exception as msg:
                print('error reading pws', msg)

            pws_ppt_stn1_orig = pws_ppt_stn1_orig[
                pws_ppt_stn1_orig < max_ppt_thr]

            # select df with period
            pws_ppt_stn1_orig = select_df_within_period(
                pws_ppt_stn1_orig,
                start=start_date,
                end=end_date)

            pws_ppt_season = select_convective_season(
                pws_ppt_stn1_orig, not_convective_season)

            # drop all index with nan values
            pws_ppt_season.dropna(axis=0, inplace=True)

            if pws_ppt_season.size > min_req_ppt_vals:

                # find distance to all prim_netw stations, sort them, select minimum
                (xpws, ynetamto) = (
                    in_pws_df_coords_utm32.loc[ppt_stn_id, 'X'],
                    in_pws_df_coords_utm32.loc[ppt_stn_id, 'Y'])

                # This finds the index of neighbours

                distances, indices = prim_netw_points_tree.query(
                    np.array([xpws, ynetamto]),
                    k=2)

                stn_2_prim_netw = prim_netw_stns_ids[indices[neighbor_to_chose]]

                min_dist_ppt_prim_netw = np.round(distances[neighbor_to_chose], 2)

                if min_dist_ppt_prim_netw <= min_dist_thr_ppt:

                    # check if prim_netw station is near, select and read prim_netw stn

                    try:
                        df_prim_netw_orig = HDF5_prim_netw.get_pandas_dataframe(stn_2_prim_netw)
                    except Exception as msg:
                        print('error reading prim_netw', msg)

#                     df_prim_netw_orig = select_df_within_period(df_prim_netw_orig,
#                                                           start=start_date,
#                                                           end=end_date)

                    df_prim_netw_orig.dropna(axis=0, inplace=True)

                    # select only data within same range
                    df_prim_netw_orig = select_df_within_period(
                        df_prim_netw_orig,
                        pws_ppt_season.index[0],
                        pws_ppt_season.index[-1])

                    if df_prim_netw_orig.size < min_req_ppt_vals:
                        stn_2_prim_netw = prim_netw_stns_ids[indices[neighbor_to_chose + 2]]
                        df_prim_netw_orig = HDF5_prim_netw.get_pandas_dataframe(
                            stn_2_prim_netw)
                        df_prim_netw_orig.dropna(axis=0, inplace=True)
                        df_prim_netw_orig = select_df_within_period(
                            df_prim_netw_orig,
                            pws_ppt_season.index[0],
                            pws_ppt_season.index[-1])
                    # =================================================
                    # Check neighboring prim_netw stations
                    # ==================================================
                    # for the prim_netw station, neighboring the pws
                    # get id, coordinates and distances of prim_netw
                    # neighbor
                    (xprim_netw, yprim_netw) = (
                        in_prim_netw_df_coords_utm32.loc[stn_2_prim_netw, 'X'],
                        in_prim_netw_df_coords_utm32.loc[stn_2_prim_netw, 'Y'])

                    distances_prim_netw, indices_prim_netw = prim_netw_points_tree.query(
                        np.array([xprim_netw, yprim_netw]),
                        k=5)
                    # +1 to get neighbor not same stn
                    stn_near_prim_netw = prim_netw_stns_ids[
                        indices_prim_netw[neighbor_to_chose + 1]]

                    min_dist_prim_netw_prim_netw = np.round(
                        distances_prim_netw[neighbor_to_chose + 1], 2)

                    try:
                        # read the neighboring prim_netw station

                        try:
                            df_prim_netw_ngbr = HDF5_prim_netw.get_pandas_dataframe(
                                stn_near_prim_netw)
                        except Exception as msg:
                            print('error reading prim_netw', msg)

                        df_prim_netw_ngbr.dropna(axis=0, inplace=True)
                        # select only data within same range
                        df_prim_netw_ngbr = select_df_within_period(
                            df_prim_netw_ngbr,
                            pws_ppt_season.index[0],
                            pws_ppt_season.index[-1])
                    except Exception:
                        raise Exception

                    if df_prim_netw_ngbr.size < min_req_ppt_vals:
                        stn_near_prim_netw = prim_netw_stns_ids[indices_prim_netw[neighbor_to_chose + 2]]
                        df_prim_netw_ngbr = HDF5_prim_netw.get_pandas_dataframe(stn_2_prim_netw)
                        df_prim_netw_ngbr.dropna(axis=0, inplace=True)
                        df_prim_netw_ngbr = select_df_within_period(
                            df_prim_netw_ngbr,
                            pws_ppt_season.index[0],
                            pws_ppt_season.index[-1])

#                         print('Second prim_netw Stn Id is', stn_near,
#                               'distance is', distance_near)
                        # calculate Indicator correlation between
                        # prim_netw-prim_netw
                    if min_dist_prim_netw_prim_netw < min_dist_thr_ppt:

                        cmn_idx = pws_ppt_season.index.intersection(
                            df_prim_netw_ngbr.index).intersection(
                                df_prim_netw_orig.index)

#                             print('\n done resampling data')
                        # same as before, now between prim_netw-prim_netw

                        if cmn_idx.size < min_req_ppt_vals:
                            stn_near_prim_netw = prim_netw_stns_ids[indices_prim_netw[neighbor_to_chose + 3]]
                            df_prim_netw_ngbr = HDF5_prim_netw.get_pandas_dataframe(
                                stn_2_prim_netw)
                            df_prim_netw_ngbr.dropna(axis=0, inplace=True)
                            df_prim_netw_ngbr = select_df_within_period(
                                df_prim_netw_ngbr,
                                pws_ppt_season.index[0],
                                pws_ppt_season.index[-1])

                            cmn_idx = pws_ppt_season.index.intersection(
                                df_prim_netw_ngbr.index).intersection(
                                    df_prim_netw_orig.index)

                        if cmn_idx.size > min_req_ppt_vals:

                            df_prim_netw_cmn_season = df_prim_netw_orig.loc[
                                cmn_idx, :]

                            df_pws_cmn_season = pws_ppt_season.loc[
                                cmn_idx, :]

                            df_prim_netw_ngbr_season = df_prim_netw_ngbr.loc[
                                cmn_idx, :]

                            assert (
                                df_prim_netw_cmn_season.isna().sum().values[0] == 0)
                            assert (
                                df_pws_cmn_season.isna().sum().values[0] == 0)
                            assert (
                                df_prim_netw_ngbr_season.isna().sum().values[0] == 0)

                            #======================================
                            # select only upper tail of values of both dataframes
                            #======================================
                            val_thr_float = val_thr_percent / 100
                            # this will calculate the EDF of pws
                            # station
                            pws_cdf_x, pws_cdf_y = get_cdf_part_abv_thr(
                                df_pws_cmn_season.values.ravel(), -0.1)
                            # find ppt value corresponding to quantile
                            # threshold
                            pws_ppt_thr_per = pws_cdf_x[np.where(
                                pws_cdf_y >= val_thr_float)][0]

                            # this will calculate the EDF of prim_netw
                            # station
                            prim_netw_cdf_x, prim_netw_cdf_y = get_cdf_part_abv_thr(
                                df_prim_netw_cmn_season.values.ravel(), -0.1)

                            # find ppt value corresponding to quantile
                            # threshold
                            prim_netw_ppt_thr_per = prim_netw_cdf_x[np.where(
                                prim_netw_cdf_y >= val_thr_float)][0]

        #                         print('\n****transform values to booleans*****\n')
                            # if Xi > Ppt_thr then 1 else 0
                            df_pws_cmn_Bool = (
                                df_pws_cmn_season > pws_ppt_thr_per
                            ).astype(int)

                            df_prim_netw_cmn_Bool = (
                                df_prim_netw_cmn_season > prim_netw_ppt_thr_per
                            ).astype(int)

                            # calculate pearson correlations of booleans 1,
                            # 0

                            bool_pears_corr = np.round(
                                pears(df_prim_netw_cmn_Bool.values.ravel(),
                                      df_pws_cmn_Bool.values.ravel())[0], 2)

                            #======================================
                            # select only upper tail both dataframes
                            #=====================================

                            prim_netw2_cdf_x, prim_netw2_cdf_y = get_cdf_part_abv_thr(
                                df_prim_netw_ngbr_season.values, -0.1)

                            # get prim_netw2 ppt thr from cdf
                            prim_netw2_ppt_thr_per = prim_netw2_cdf_x[np.where(
                                prim_netw2_cdf_y >= val_thr_float)][0]

                            df_prim_netw2_cmn_Bool = (
                                df_prim_netw_ngbr_season > prim_netw2_ppt_thr_per
                            ).astype(int)

                            # calculate pearson correlations of booleans
                            # 1, 0

                            bool_pears_corr_prim_netw = np.round(
                                pears(df_prim_netw_cmn_Bool.values.ravel(),
                                      df_prim_netw2_cmn_Bool.values.ravel())[0], 2)
                            # check if df_prim_netw2_cmn_Bool correlation between
                            # pws and prim_netw is higher than between
                            # prim_netw and prim_netw neighbours, if yes, keep
                            # pws

                            # pd.concat([in_df_april_mid_oct, in_df_mid_oct_mars])

                            if True:  # bool_pears_corr >= bool_pears_corr_prim_netw:
                                #
                                print('+++keeping pws+++')

                                #==================================
                                # append the result to df_correlations
                                #
                                #==================================
#                                     df_results_correlations.loc[
#                                         ppt_stn_id,
#                                         'lon'] = lon_stn_pws
#                                     df_results_correlations.loc[
#                                         ppt_stn_id,
#                                         'lat'] = lat_stn_pws
                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'Distance to neighbor'
                                ] = min_dist_ppt_prim_netw

                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'prim_netw neighbor ID'] = stn_2_prim_netw

                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'prim_netw-prim_netw neighbor ID'] = stn_near_prim_netw
                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'Distance prim_netw-prim_netw neighbor'
                                ] = min_dist_prim_netw_prim_netw
                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'pws_%s_Per_ppt_thr'
                                    % val_thr_percent] = pws_ppt_thr_per

                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'prim_netw_%s_Per_ppt_thr'
                                    % val_thr_percent] = prim_netw_ppt_thr_per

                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'Bool_Pearson_Correlation_pws_prim_netw'
                                ] = bool_pears_corr
                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'Bool_Pearson_Correlation_prim_netw_prim_netw'
                                ] = bool_pears_corr_prim_netw
                            else:
                                pass
                                print('---Removing pws---')

                        else:
                            print('not enough data')
    #                         print('\n********\n ADDED DATA TO DF RESULTS')
                    else:
                        pass
                        # print('After intersecting dataframes not enough data')
                else:
                    pass
                    # print('prim_netw Station is near but not enough data')
            else:
                pass
                # print('\n********\n prim_netw station is not near')

        except Exception as msg:
            print('error while finding neighbours ', msg)

            # import pdb
            # pdb.set_trace()

            continue
            # raise Exception
            # continue
    # assert alls_stns_len == 0, 'not all stations were considered'

    df_results_correlations.dropna(how='all', inplace=True)

    return df_results_correlations

#==============================================================================
# CALL FUNCTION HERE
#==============================================================================


if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    for lower_percentile_val in lower_percentile_val_lst:
        print('\n********\n Lower_percentile_val', lower_percentile_val)

        for temp_freq in aggregation_frequencies:
            print('\n********\n Time aggregation is', temp_freq)

            for neighbor_to_chose in neighbors_to_chose_lst:
                print('\n********\n prim_netw Neighbor is', neighbor_to_chose)

                path_to_df_correlations = ''
                if (not os.path.exists(path_to_df_correlations)):

                    print('\n Data frames do not exist, creating them\n')

                    args = (# path_to_pws_coords_utm32,
                        # path_to_prim_netw_coords_utm32,
                        path_to_ppt_pws_data_hdf5,
                        path_to_ppt_prim_netw_data_hdf5,
                        temp_freq,
                        neighbor_to_chose,
                        lower_percentile_val,
                        min_req_ppt_vals)

                    process_manager(args)
                else:
                    print('\n Data frames exist, not creating them\n')
#                     df_results_correlations = pd.read_csv(path_to_df_correlations,
# sep=';', index_col=0)

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))

# # !/usr/bin/env python.
# # -*- coding: utf-8 -*-
#
# """
# Name:    Calculate and plot statistical differences between neighbours
# Purpose: Find validity of pws Station compared to primary station
#
#
# For every pws precipitation station select the
# convective season period (Mai till Ocotber), find the nearest
# primary station, intersect both stations, based on a percentage threshold,
# select for every station seperatly the corresponding rainfall value based
# on the CDF, using the threshold, make all values above a 1 and below a 0,
# making everything boolean, calculate the spearman rank correlation between
# the two stations and save the result in a df, do it considering different
# neighbors and percentage threhsold ( a probabilistic threshold),
# this allows capturing the change of rank correlation with distance and thr
#
# Do it on using all data for a station
#
# Parameters
# ----------
#
# Input Files
#     primary station data
#     pws precipitation station data
#     pws station coordinates data
#
#     Optional:
#         # Distance matrix between primary and pws stations
#         # Shapefile of BW area
#
# Returns
# -------
#
#
# Df_correlations: df containing for every Netamo station,
#     the statistical difference in terms of Pearson and Spearman
#     Correlations for original data and boolean transfomed data
#     compared with the nearest primary station.
#
# Optional:
#     # Plot and save everything on a map using shapefile boundaries
# """
#
# __author__ = "Abbas El Hachem"
# __copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
# __email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
#
# # =============================================================================
# # generic Libs
# import os
# import time
# import timeit
#
# import psutil
# from scipy.spatial import cKDTree
# from scipy.stats import pearsonr as pears
# from scipy.stats import spearmanr as spr
#
# from _00_functions import (
#     select_convective_season,
#     select_df_within_period,
#     get_cdf_part_abv_thr)
#
# from _01_read_hdf5 import HDF5
# import multiprocessing as mp
# import numpy as np
# import pandas as pd
#
# #==============================================================================
# # HOURLY DATA
# #==============================================================================
#
# path_to_ppt_pws_data_hdf5 = (
#     r"X:\staff\elhachem\Data\pws_data"
#     r"\rain_RLP_2020_1hour\pws_RLP_2020__60min_2020.h5"
# )
# assert os.path.exists(path_to_ppt_pws_data_hdf5), 'wrong pws Ppt file'
#
# path_to_ppt_primary_data_hdf5 = (
#     r"X:\staff\elhachem\ClimXtreme\03_data\00_primary"
#     r"\primary_comb_60min_data_agg_60min_2020.h5")
# assert os.path.exists(path_to_ppt_primary_data_hdf5), 'wrong primary Csv Ppt file'
# #==============================================================================
# #
# #==============================================================================
# # min distance threshold used for selecting neighbours
# min_dist_thr_ppt = 100 * 1e4  # 5000  # m
#
# # threshold for max ppt value per hour
# max_ppt_thr = 200.  # ppt above this value are not considered
#
# # only highest x% of the values are selected
# lower_percentile_val_lst = [99.]  # [80, 85, 90, 95, 99]
#
# aggregation_frequencies = ['60min']
#
# # [0, 1, 2, 3, 4]  # refers to primary neighbot (0=first)
# neighbors_to_chose_lst = [0]
#
# # minimum hourly values that should be available per station
# min_req_ppt_vals = 0  # 30 * 24 * 1  # 2
# # this is used to keep only data where month is not in this list
# not_convective_season = [11, 12]  # oct till april
#
# date_fmt = '%Y-%m-%d %H:%M:%S'
#
# _year = '2019'
#
# # select data only within this period
# start_date = '%s-01-01 00:00:00' % _year
# end_date = '%s-10-30 23:00:00' % _year
#
# # calculate no of workers
# n_workers = 5  # psutil.cpu_count() - 2
#
# out_save_dir_orig = (
#     r'X:\staff\elhachem\2020_05_20_pws_CML'
#     r'\indicator_correlation_%s_%s_RLP2020' % (aggregation_frequencies[0],
#                                                str(lower_percentile_val_lst[0]).replace('.', '_')))
#
# # define out save directory (for example)
# out_save_dir_orig = (r'C´P:\results\indicator_correlation')
# if not os.path.exists(out_save_dir_orig):
#     os.mkdir(out_save_dir_orig)
# #==============================================================================
# #
# #==============================================================================
#
#
# def process_manager(args):
#
#     (path_pws_ppt_df_hdf5,  # path to hdf5 file containing pws data and coords
#         path_to_primary_data_hdf5,  # path to hdf5 file containing primary data and coords
#         tem_freq,  # temporal frequency of input data
#         neighbor_to_chose,  # calculate correlation between pws and which primary neighbor
#         val_thr_percent,  # percentile level for indicator correlation
#         min_req_ppt_vals  # minimum requested values between pws and primary station
#         ) = args
#
#     # get all station names for primary network
#     #===========================================================================
#     HDF5_primary = HDF5(infile=path_to_primary_data_hdf5)
#     all_primary_stns_ids = HDF5_primary.get_all_names()
#
#     # get corrdinates of primary network stations
#     primary_coords = HDF5_primary.get_coordinates(all_primary_stns_ids)
#
#     in_primary_df_coords_utm32 = pd.DataFrame(
#         index=all_primary_stns_ids,
#         data=primary_coords['easting'], columns=['X'])
#     y_primary_coords = primary_coords['northing']
#     in_primary_df_coords_utm32.loc[:, 'Y'] = y_primary_coords
#     primary_stns_ids = in_primary_df_coords_utm32.index
#
#     # get all station names for pws network
#     #===========================================================================
#     HDF5_pws = HDF5(infile=path_pws_ppt_df_hdf5)
#     all_pws_ids = HDF5_pws.get_all_names()
#
#     # get coordinates of pws data
#     pws_coords = HDF5_pws.get_coordinates(all_pws_ids)
#
#     # create a dataframe from pws coordinates
#     in_pws_df_coords_utm32 = pd.DataFrame(
#         index=all_pws_ids,
#         data=pws_coords['easting'], columns=['X'])
#     y_pws_coords = pws_coords['northing']
#     in_pws_df_coords_utm32.loc[:, 'Y'] = y_pws_coords
#
#     # create a tree from primary coordinates
#     #===========================================================================
#     primary_coords_xy = [(x, y) for x, y in zip(
#         in_primary_df_coords_utm32.loc[:, 'X'].values,
#         in_primary_df_coords_utm32.loc[:, 'Y'].values)]
#     # create a tree from coordinates
#     primary_points_tree = cKDTree(primary_coords_xy)
#
#     # initialize the data for each worker
#     #===========================================================================
#
#     print('Using %d Workers' % n_workers)
#
#     all_pws_stns_ids_worker = np.array_split(all_pws_ids, n_workers)
#     args_worker = []
#
#     for stns_list in all_pws_stns_ids_worker:
#         # dataframe to hold results fer every pws station list
#         df_results_correlations = pd.DataFrame(index=stns_list)
#
#         args_worker.append((path_to_primary_data_hdf5,
#                             path_pws_ppt_df_hdf5,
#                             in_pws_df_coords_utm32,
#                             stns_list,
#                             primary_points_tree,
#                             primary_stns_ids,
#                             df_results_correlations,
#                             neighbor_to_chose,
#                             val_thr_percent,
#                             min_req_ppt_vals))
#
#     # multiprocess the calculation and get results from all workers
#     #===========================================================================
#     my_pool = mp.Pool(processes=n_workers)
#
#     results = my_pool.map(indicator_correlation_pws_primary, args_worker)
#
#
#     my_pool.close()
#     my_pool.join()
#
#     # bring dfs together
#     results_df = pd.concat(results)
#
#     # save this results
#     results_df.to_csv(
#         os.path.join(out_save_dir_orig,
#                      'df_indic_corr_raw_data.csv'),
#         sep=';')
#
#     return
#
# #==============================================================================
# #
# #==============================================================================
#
# # @profile
#
#
# def indicator_correlation_pws_primary(args):
#     #         path_pws_ppt_df_hdf5,  # path to df of all pws ppt stations
#     #         path_to_primary_data_hdf5,  # path to primary ppt hdf5 data
#     #         path_to_pws_coords_utm32,  # path to pws coords stns utm32
#     #         path_to_primary_coords_utm32,  # path_to_primary coords snts utm32
#     #         neighbor_to_chose,  # which primary station neighbor to chose
#     #         min_dist_thr_ppt,  # distance threshold when selecting primary neigbours
#     #         temp_freq_resample,  # temp freq to resample dfs
#     #         val_thr_percent,  # value in percentage, select all values above it
#     #         min_req_ppt_vals  # threshold, minimum ppt values per station
#     # ):
#     '''
#      Find then for the pws station the neighboring primary station
#      intersect both stations, for the given probabilistic percentage
#      threshold find the corresponding ppt_thr from the CDF of each station
#      seperatly, make all values boolean (> 1, < 0) and calculate the spearman
#      rank correlation between the two stations
#
#      Add the result to a new dataframe and return it
#
#     # TODO: add documentation
#     '''
#     print('\n######\n getting all station names, reading dfs \n#######\n')
#
#     (path_to_primary_data_hdf5,
#      path_pws_ppt_df_hdf5,
#      in_pws_df_coords_utm32,
#      all_pws_ids,
#      primary_points_tree,
#      primary_stns_ids,
#      df_results_correlations,
#      neighbor_to_chose,
#      val_thr_percent,
#      min_req_ppt_vals) = args
#
#     # read hdf5
#     HDF5_pws = HDF5(infile=path_pws_ppt_df_hdf5)
#
#     HDF5_primary = HDF5(infile=path_to_primary_data_hdf5)
#
#     alls_stns_len = len(all_pws_ids)  # good_stns
#
#     for ppt_stn_id in all_pws_ids:
#
#         print('\n********\n Total number of pws stations is\n********\n',
#               alls_stns_len)
#         alls_stns_len -= 1
#
#         # iterating through pws ppt stations
#
#         print('\n********\n First Ppt Stn Id is', ppt_stn_id)
#         # data = HDF52.get_pandas_dataframe('P03668')
#         # orig stn name, for locating coordinates, appending to df_results
#         # ppt_stn_id_name_orig = ppt_stn_id.replace('_', ':')
#         try:
#             # read first pws station
#
#             try:
#                 pws_ppt_stn1_orig = HDF5_pws.get_pandas_dataframe(
#                     ppt_stn_id)
#
#             except Exception as msg:
#                 print('error reading primary', msg)
#
#             pws_ppt_stn1_orig = pws_ppt_stn1_orig[
#                 pws_ppt_stn1_orig < max_ppt_thr]
#
#             # select df with period
#             pws_ppt_stn1_orig = select_df_within_period(
#                 pws_ppt_stn1_orig,
#                 start=start_date,
#                 end=end_date)
#
#             # select only convective season
#             df_pws_cmn_season = select_convective_season(
#                 df=pws_ppt_stn1_orig,
#                 month_lst=not_convective_season)
#             # drop all index with nan values
#             df_pws_cmn_season.dropna(axis=0, inplace=True)
#
#             if df_pws_cmn_season.size > min_req_ppt_vals:
#
#                 # find distance to all primary stations, sort them, select minimum
#                 (xpws, ynetamto) = (
#                     in_pws_df_coords_utm32.loc[ppt_stn_id, 'X'],
#                     in_pws_df_coords_utm32.loc[ppt_stn_id, 'Y'])
#
#                 # This finds the index of all points within
#                 # radius
#     #             idxs_neighbours = primary_points_tree.query_ball_point(
#
#                 distances, indices = primary_points_tree.query(
#                     np.array([xpws, ynetamto]),
#                     k=5)
#
#                 stn_near = primary_stns_ids[indices[neighbor_to_chose]]
#
#                 min_dist_ppt_primary = np.round(
#                     distances[neighbor_to_chose], 2)
#
#                 if min_dist_ppt_primary <= min_dist_thr_ppt:
#                     # check if primary station is near, select and read primary stn
#
#                     try:
#                         df_primary_orig = HDF5_primary.get_pandas_dataframe(stn_near)
#                     except Exception as msg:
#                         print('error reading primary', msg)
#
#                     df_primary_orig.dropna(axis=0, inplace=True)
#
#                     cmn_idx = df_pws_cmn_season.index.intersection(
#                         df_primary_orig.index)
#
#                     if (cmn_idx.size > min_req_ppt_vals):
#                         # print('\n# Stations have more than 2month data
#                         # #\n')
#
#                         # select only convective season
#                         df_pws_cmn_season = df_pws_cmn_season.loc[
#                             cmn_idx, :]
#
#                         # select convective seasn
#                         df_primary_cmn_season = df_primary_orig.loc[cmn_idx, :]
#
#                         #==============================================
#                         # look for agreements, correlation between all values
#                         #==============================================
#
#                         # calculate pearson and spearman between original
#                         # values
#                         orig_pears_corr = np.round(
#                             pears(df_primary_cmn_season.values.ravel(),
#                                   df_pws_cmn_season.values.ravel())[0], 2)
#
#                         orig_spr_corr = np.round(
#                             spr(df_primary_cmn_season.values,
#                                 df_pws_cmn_season.values)[0], 2)
#
#                         #==============================================
#                         # select only upper tail of values of both dataframes
#                         #==============================================
#                         val_thr_float = val_thr_percent / 100
#
#                         pws_cdf_x, pws_cdf_y = get_cdf_part_abv_thr(
#                             df_pws_cmn_season.values.ravel(), -0.1)
#                         # get pws ppt thr from cdf
#                         pws_ppt_thr_per = pws_cdf_x[np.where(
#                             pws_cdf_y >= val_thr_float)][0]
#
#                         primary_cdf_x, primary_cdf_y = get_cdf_part_abv_thr(
#                             df_primary_cmn_season.values.ravel(), -0.1)
#
#                         # get primary ppt thr from cdf
#                         primary_ppt_thr_per = primary_cdf_x[np.where(
#                             primary_cdf_y >= val_thr_float)][0]
#
#                         # print('\n****transform values to booleans*****\n')
#
#                         df_pws_cmn_Bool = (
#                             df_pws_cmn_season > pws_ppt_thr_per
#                         ).astype(int)
#                         df_primary_cmn_Bool = (
#                             df_primary_cmn_season > primary_ppt_thr_per
#                         ).astype(int)
#
#                         # calculate spearman correlations of booleans
#                         # 1, 0
#
#                         bool_pears_corr = np.round(
#                             pears(df_primary_cmn_Bool.values.ravel(),
#                                   df_pws_cmn_Bool.values.ravel())[0], 2)
#                         print('bool_pears_corr', bool_pears_corr)
#
#                         df_results_correlations.loc[
#                             ppt_stn_id,
#                             'Distance to neighbor'] = min_dist_ppt_primary
#
#                         df_results_correlations.loc[
#                             ppt_stn_id,
#                             'primary neighbor ID'] = stn_near
#
#                         df_results_correlations.loc[
#                             ppt_stn_id,
#                             'pws_%s_Per_ppt_thr'
#                             % val_thr_percent] = pws_ppt_thr_per
#
#                         df_results_correlations.loc[
#                             ppt_stn_id,
#                             'primary_%s_Per_ppt_thr'
#                             % val_thr_percent] = primary_ppt_thr_per
#
#                         df_results_correlations.loc[
#                             ppt_stn_id,
#                             'Orig_Pearson_Correlation'] = orig_pears_corr
#
#                         df_results_correlations.loc[
#                             ppt_stn_id,
#                             'Orig_Spearman_Correlation'] = orig_spr_corr
#
#                         df_results_correlations.loc[
#                             ppt_stn_id,
#                             'Bool_Pearson_Correlation'] = bool_pears_corr
#
#                         print('\n********\n ADDED DATA TO DF RESULTS')
#
#                     else:
#                         print('primary Station is near but not enough data')
#                 else:
#                     print('\n********\n primary station is not near')
#
#         except Exception as msg:
#             print('error while finding neighbours ', msg)
#             continue
#
#     df_results_correlations.dropna(how='all', inplace=True)
#
#     return df_results_correlations
#
# #==============================================================================
# #
# #==============================================================================
#
#
# if __name__ == '__main__':
#
#     print('**** Started on %s ****\n' % time.asctime())
#     START = timeit.default_timer()  # to get the runtime of the program
#
#     for lower_percentile_val in lower_percentile_val_lst:
#         print('\n********\n Lower_percentile_val', lower_percentile_val)
#
#         for temp_freq in aggregation_frequencies:
#             print('\n********\n Time aggregation is', temp_freq)
#
#             for neighbor_to_chose in neighbors_to_chose_lst:
#                 print('\n********\n primary Neighbor is', neighbor_to_chose)
#
#                 # call this function to get the df, one containing
#                 # df_correlations comparing correlations
#                 path_to_df_correlations = ''
#
#                 if (not os.path.exists(path_to_df_correlations)):
#
#                     print('\n Data frames do not exist, creating them\n')
#
#                     args = (# path_to_pws_coords_utm32,
#                         # path_to_primary_coords_utm32,
#                         path_to_ppt_pws_data_hdf5,
#                         path_to_ppt_primary_data_hdf5,
#                         temp_freq,
#                         neighbor_to_chose,
#                         lower_percentile_val,
#                         min_req_ppt_vals)
#
#                     process_manager(args)
#                 else:
#                     print('\n Data frames exist, not creating them\n')
#                     # df_results_correlations = pd.read_csv(path_to_df_correlations,
#                     # sep=';', index_col=0)
#
#     STOP = timeit.default_timer()  # Ending time
#     print(('\n****Done with everything on %s.\nTotal run time was'
#            ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
