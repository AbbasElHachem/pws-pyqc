# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
# Description
#=========
Name:    Filter pws stations based on Indicator Correlation
Purpose: Find validity of pws Station for interpolation purposes

If the value of the indicator correlation between the pws-prim_netw pair is 
greater or equal to the value between the prim_netw-prim_netw pair,
keep pws station, else remove it

Repeat this procedure for all pws station, or different quantile threshold
and for different neighbors and temporal resolution.

Parameters
#=========
Input Files
    hdf5_file for the prim_netw station data and coordinates
    hdf5_file for pws precipitation station data and coordinates
    
Returns
#=========
Df_correlations: df containing for every pws station:
    ID1 neighbor in prim_netw
    Seperating distance pws-prim_netw, 
    
    The pws rainfall value corresponding for the percentile threshold
    The primary network rainfall value corresponding for the percentile threshold
    The spearman correlation between the boolean transformed prim_netw and
         prim_netw neighboring stations ID1 and ID2 for same time period pws-ID1
    The spearman correlation between the boolean transformed pws and
         primary network data ID1 for the same time period
    
Plot everything in the dataframe using a different script
Especially the change of correlation with distance

Reference
#=========

 Bárdossy, A., Seidel, J., and El Hachem, A.:
 The use of personal weather station observation for improving precipitation
 estimation and interpolation,
 
 Hydrol. Earth Syst. Sci. Discuss.,
 https://doi.org/10.5194/hess-2020-42

"""

__author__ = "Abbas El Hachem"
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
__version__ = 0.1
__last_update__ = '15.04.2020'

# ===========================================================

# generic Libs
import os

import time
import timeit

# other Libs
import numpy as np
import pandas as pd
import multiprocessing as mp

from scipy.spatial import cKDTree
from scipy.stats import pearsonr as prs

# own functions from script _00_functions
from _00_functions import (
    select_df_within_period,
    get_cdf_part_abv_thr)
# import class to read HDF5 data and coordinates
from _01_2_read_hdf5 import HDF5

#============================================================
# Path to pws and primary network data
#============================================================

path_to_ppt_pws_data_hdf5 = (
    r"X:\staff\elhachem\Data\pws_data\rain_RLP_2020_1hour"
    r"\pws_RLP_2020__60min_2020.h5")
assert os.path.exists(path_to_ppt_pws_data_hdf5), 'wrong pws file'

path_to_ppt_prim_netw_data_hdf5 = (
    r"X:\staff\elhachem\ClimXtreme\03_data\00_prim_netw"
    r"\prim_netw_comb_60min_data_agg_60min_2020.h5")
assert os.path.exists(path_to_ppt_prim_netw_data_hdf5), 'wrong prim_netw file'

# ===========================================================
# some parameters for the filtering
# ===========================================================
# min distance used for selecting neighbors
min_dist_thr_ppt = 100 * 1e4  # in m, for ex: 30km or 50km

# threshold for max pcp value per hour
# pcp above this value are not considered
max_ppt_thr = 100.

# only highest x% of the values are selected
# if value > threshold, then 1 else 0
lower_percentile_val_lst = [99.0]

# temporal frequencies on which the filtering should be done
aggregation_frequencies = ['60min']

# define for which year to do the filtering
_year = '2020'

# refers to prim_netw neighbor (0=first)
neighbors_to_chose_lst = [0]  # , 1, 2, 3]

# all pwss have more than 2 month data, this an extra check
min_req_ppt_vals = 0  # 2 * 24 * 30


# date format of dataframes
date_fmt = '%Y-%m-%d %H:%M:%S'

# select data only within this period
start_date = '%s-01-01 00:00:00' % _year
end_date = '%s-10-30 23:00:00' % _year

# nbr of workers for multiproccesing
n_workers = 7

# def out save directory
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

    prim_netw_coords = HDF5_prim_netw.get_coordinates(
        all_prim_netw_stns_ids)

    in_prim_netw_df_coords_utm32 = pd.DataFrame(
        index=all_prim_netw_stns_ids,
        data=prim_netw_coords['easting'], columns=['X'])
    y_prim_netw_coords = prim_netw_coords['northing']
    in_prim_netw_df_coords_utm32.loc[:, 'Y'] = y_prim_netw_coords
    prim_netw_stns_ids = in_prim_netw_df_coords_utm32.index
    # create a tree from prim_netw coordinates

    prim_netw_coords_xy = [(x, y) for x, y in zip(
        in_prim_netw_df_coords_utm32.loc[:, 'X'].values,
        in_prim_netw_df_coords_utm32.loc[:, 'Y'].values)]

    # create a tree from coordinates
    prim_netw_points_tree = cKDTree(prim_netw_coords_xy)

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


    results_df.to_csv(
        os.path.join(out_save_dir_orig,
                     'indic_corr_filter.csv'),
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
            pws_ppt_season = select_df_within_period(
                pws_ppt_stn1_orig,
                start=start_date,
                end=end_date)

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
                        df_prim_netw_orig = HDF5_prim_netw.get_pandas_dataframe(
                            stn_2_prim_netw)
                    except Exception as msg:
                        print('error reading prim_netw', msg)


                    df_prim_netw_orig.dropna(axis=0, inplace=True)

                    # select only data within same range
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

                    # calculate Indicator correlation between
                    # prim_netw-prim_netw
                    if min_dist_prim_netw_prim_netw < min_dist_thr_ppt:

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

                            # calculate pearson correlations of booleans 1, 0

                            bool_prs_corr = np.round(
                                prs(df_prim_netw_cmn_Bool.values.ravel(),
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

                            bool_prs_corr_prim_netw = np.round(
                                prs(df_prim_netw_cmn_Bool.values.ravel(),
                                      df_prim_netw2_cmn_Bool.values.ravel())[0], 2)

                            # check if df_prim_netw2_cmn_Bool correlation between
                            # pws and prim_netw is higher than between
                            # prim_netw and prim_netw neighbours, if yes, keep
                            # pws


                            if bool_prs_corr >= bool_prs_corr_prim_netw:

                                print('+++keeping pws+++')

                                #==================================
                                # append the result to df_correlations
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
                                    'prim_netw-prim_netw neighbor ID'
                                    ] = stn_near_prim_netw

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
                                ] = bool_prs_corr
                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'Bool_Pearson_Correlation_prim_netw_prim_netw'
                                ] = bool_prs_corr_prim_netw
                            else:
                                pass
                                print('---Removing pws---')

                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'Bool_Pearson_Correlation_pws_prim_netw'
                                ] = bool_prs_corr
                                df_results_correlations.loc[
                                    ppt_stn_id,
                                    'Bool_Pearson_Correlation_prim_netw_prim_netw'
                                ] = bool_prs_corr_prim_netw

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


            continue

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
                        neighbor_to_chose,
                        lower_percentile_val,
                        min_req_ppt_vals)

                    process_manager(args)
                else:
                    print('\n Data frames exist, not creating them\n')

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))

