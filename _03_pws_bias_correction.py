# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    Correct PWS using Primary Network Data
Purpose: Prepare PWS data for Interpolation

Created on: 2020-05-16

Correct each PWS observation using the neighboring primary network data
Using a unified exponential variogram interpolated each PWS observation
using the Neighboring primary Network data for the same time period,
recorrect the PWS Quantiles using the primary Network long observations
 
Period of focus: 2018-2019 Summer period

1. When transforming the Netatmo precipitation to Netatmo percentiles,
   we previously used a value of p0/2 for all values <0.1mm.
   Should I do the same here?
--> Yes that is the present way.

2. Should I correct only the Netamto precipitation or the precipitation 
   and the quantile values?
--> only precpitation
3. When using a unified variogram (exponential),
   what Sill and Range should I use, before the scaling factor?
--> Use a variogram without nugget and a range of 30 to 40 km first
4. When correcting a Netatmo station, what is the range or the maximum
   number of DWD stations to use?
Do not use more than 16


Parameters
----------

Input Files
    DWD station data
    DWD coordinates data
    Netatmo precipitation station data
    Netatmo station coordinates data
    
Returns
-------

    Corrected PWS data, save results in HDF5 format
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

# generic Libs
import os
# import timeit
import time
# import swifter
import gc

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# other Libs

# import sys

import pyximport
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
# from multiprocessing import cpu_count  # , Pool

# from scipy import spatial
from scipy.spatial import cKDTree
from pathlib import Path

# from statsmodels.distributions.empirical_distribution import ECDF
# own libraries, functions

# from spinterps import (OrdinaryKriging)
from pykrige.ok import OrdinaryKriging as OKpy
# from pandarallel import pandarallel

from _00_functions import (select_convective_season,
                           select_df_within_period,
                           build_edf_fr_vals,
                           calculate_probab_ppt_below_thr,
                           find_nearest,
                           resampleDf
                           )
# calculate_probab_ppt_below_thr)

from _01_2_read_hdf5 import HDF5

pyximport.install()
# cores = cpu_count() - 1  # Number of CPU cores on system
# pandarallel.initialize(nb_workers=2)
# set settings
gc.set_threshold(0, 0, 0)
# =============================================================================

start_date = '2018-01-01 00:00:00'
end_date = '2019-12-31 00:00:00'

# minimum hourly values that should be available per Netatmo station
min_req_ppt_vals = 30 * 24 * 2

# this is used to keep only data where month is not in this list
not_convective_season = [11, 12, 1, 2, 3]  # oct till april

# how many DWD stations to use
nbr_dwd_neighbours_to_use = 16

ppt_min_thr_0_vals = 0.1  # below it all values get p0/2
min_qt_to_correct = 0.75  # correct all qunatiles above it

vg_sill_b4_scale = 0.07
vg_range = 4e4
vg_model_str = 'spherical'
# vg_model_to_scale = '0.07 Sph(40000)'

n_workers = 1  # int(cores / 1)
# n_sub_proccess = cores - n_workers
# =============================================================================

# main_dir = Path(r"/home/IWS/hachem/Netatmo_CML")
main_dir = Path(r"X:\staff\elhachem\2020_05_20_Netatmo_CML")
os.chdir(main_dir)
# TODO: PATH
path_to_ppt_netatmo_data_hdf5 = (
    #    r"X:\exchange\ElHachem\Netatmo_correct_18_19\data_yearly\netatmo_stn_filtered_2018_2019_yearly.h5")
    r"P:\2020_DFG_Netatmo\03_data\01_netatmo\netatmo_Germany_5min_to_1hour_filter_00.h5")
# assert os.path.exists(path_to_ppt_netatmo_data_hdf5), 'wrong NETATMO Ppt file'

# path_to_ppt_netatmo_data_hdf5 = (
#     r"C:\Users\hachem\Downloads\netatmo_filtered_2018_2019_5min_to_1hour.h5")

path_to_ppt_dwd_data_hdf5 = (
    # r"P:\2020_DFG_Netatmo\03_data\03_dwd\DWD_5min_to_1hour.h5")
    r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD\dwd_comb_60min_SS1819.h5")
assert os.path.exists(path_to_ppt_dwd_data_hdf5), 'wrong DWD Csv Ppt file'

# NETATMO FIRST FILTER

path_to_netatmo_gd_stns_2018 = (main_dir / r'indicator_correlation_60min_99_0' /
                                (r'Netatmo_60min_Good_99_2018.csv'))
# Netatmo_60min_Good_99_2018
path_to_netatmo_gd_stns_2019 = (main_dir / r'indicator_correlation_60min_99_0' /
                                (r'Netatmo_60min_Good_99_2019.csv'))

# NETATMO FIRST FILTER
# path_to_netatmo_gd_stns = (
#     main_dir / "indicator_correlation/Netatmo_Good_99.csv")
# for output folder
title_ = r'corrected_PWS_DE_yearly'

out_save_dir = main_dir / title_

if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)

#==============================================================================
# MAIN FUNCTION
#==============================================================================


def convert_ppt_df_to_edf(df, stationname, ppt_min_thr_0_vals):
    df_col = df[~np.isnan(df)]
    p0 = calculate_probab_ppt_below_thr(df_col.values,
                                        ppt_min_thr_0_vals)

    try:
        x0, y0 = build_edf_fr_vals(df_col.values)
        y0[np.where(x0 <= ppt_min_thr_0_vals)] = p0 / 2
        y0[np.where(y0 == 1)] = 0.9999999999

    except Exception as msg:
        print('Error with EDF', msg)

    df_ppt_edf = pd.DataFrame(data=df_col.values,
                              index=df_col.index,
                              columns=[stationname])
    df_ppt_edf.sort_values(by=stationname, inplace=True)

    df_ppt_edf.loc[:, 'edf'] = y0
    df_ppt_edf.sort_index(inplace=True)
    df_ppt_edf.drop([stationname], axis=1, inplace=True)
    return df_ppt_edf
#==============================================================================
#
#==============================================================================


def process_manager(args):

    (

        path_to_netatmo_gd_stns_2018,
        path_to_netatmo_gd_stns_2019,

        path_to_neatmo_ppt_hdf5,
        # path_to_neatmo_edf_hdf5,
        path_to_dwd_ppt_hdf5) = args

    # get all station names for dwd
    HDF5_DWD = HDF5(infile=path_to_dwd_ppt_hdf5)
    all_dwd_stns_ids = HDF5_DWD.get_all_names()

    # get all station names for netatmo
    HDF5_Netatmo = HDF5(infile=path_to_neatmo_ppt_hdf5)
    all_netatmo_ids = HDF5_Netatmo.get_all_names()

    netatmo_coords = HDF5_Netatmo.get_coordinates(all_netatmo_ids)

    in_netatmo_df_coords_utm32 = pd.DataFrame(
        index=all_netatmo_ids,
        data=netatmo_coords['easting'], columns=['X'])
    y_netatmo_coords = netatmo_coords['northing']
    in_netatmo_df_coords_utm32.loc[:, 'Y'] = y_netatmo_coords

    #     in_netatmo_df_coords_utm32 = pd.read_csv(
    #         path_to_netatmo_coords_utm32, sep=';',
    #         index_col=0, engine='c')

    dwd_coords = HDF5_DWD.get_coordinates(all_dwd_stns_ids)

    in_dwd_df_coords_utm32 = pd.DataFrame(
        index=all_dwd_stns_ids,
        data=dwd_coords['easting'], columns=['X'])
    y_dwd_coords = dwd_coords['northing']
    in_dwd_df_coords_utm32.loc[:, 'Y'] = y_dwd_coords

    # Netatmo first filter
#     df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
#                              index_col=1,
#                              sep=';',
#                              encoding='utf-8')

    df_gd_stns_2018 = pd.read_csv(path_to_netatmo_gd_stns_2018,
                                  index_col=1,
                                  sep=';',
                                  encoding='utf-8')

    df_gd_stns_2019 = pd.read_csv(path_to_netatmo_gd_stns_2019,
                                  index_col=1,
                                  sep=';',
                                  encoding='utf-8')

    #=========================================================================
    # Netatmo data
    # HDF5_netatmo_edf = HDF5(infile=path_to_neatmo_edf_hdf5)
    # all_netatmo_ids = HDF5_netatmo_edf.get_all_names()^

    # all_netatmo_ids_with_good_data = df_gd_stns.index.to_list()

    all_netatmo_ids_with_good_data_2018 = df_gd_stns_2018.index.to_list()
    all_netatmo_ids_with_good_data_2019 = df_gd_stns_2019.index.to_list()

    # combine all good stns
    all_netatmo_ids_with_good_data = list(
        set(all_netatmo_ids_with_good_data_2018) |
        set(all_netatmo_ids_with_good_data_2019))
    # get indices of those stations

    r"""
        print('getting indices')
    stns_array = np.array(all_netatmo_ids)
    # good for 2018
    mask2018 = np.isin(
        all_netatmo_ids, all_netatmo_ids_with_good_data_2019) * 1
    mask2018_idx = mask2018.nonzero()[0]
    
    stns_name = stns_array[mask2018.nonzero()[0]]
    mask2018_idx_sr = pd.DataFrame(index=stns_name,
                                   data=mask2018_idx, columns=['Indices'])
    mask2018_idx_sr.to_csv('Netatmo_Good_2019_Indices.csv', sep=';')

    mask2018_idx_sr = pd.DataFrame(
                                   index=mask2018_idx)

    mask2018_idx_sr.to_csv('Netatmo_Good_2019_Indices_only.csv', sep=';')

        in_netatmo_df_coords_wgs84 = pd.DataFrame(
            index=all_netatmo_ids,
            data=netatmo_coords['lon'], columns=['lon'])
        lat_netatmo_coords = netatmo_coords['lat']
        z_netatmo_coords = netatmo_coords['z']
        in_netatmo_df_coords_wgs84.loc[:, 'lat'] = lat_netatmo_coords
        in_netatmo_df_coords_wgs84.loc[:, 'elevation'] = z_netatmo_coords
        
        in_netatmo_df_coords_wgs84.loc[all_netatmo_ids_with_good_data, :]
    
        df_metadata = pd.DataFrame(index=all_netatmo_ids_with_good_data,
                                   columns=['lon', 'lat', 'easting', 'northing',  'elevation'])
    
        df_metadata.loc[:, 'lon'] = in_netatmo_df_coords_wgs84.loc[:, 'lon']
        df_metadata.loc[:, 'lat'] = in_netatmo_df_coords_wgs84.loc[:, 'lat']
        df_metadata.loc[:,
                        'elevation'] = in_netatmo_df_coords_wgs84.loc[:, 'elevation']
        df_metadata.loc[:, 'easting'] = in_netatmo_df_coords_utm32.loc[:, 'X']
        df_metadata.loc[:, 'northing'] = in_netatmo_df_coords_utm32.loc[:, 'Y']
    
        df_metadata.to_csv(
            r'X:\exchange\ElHachem\Netatmo_correct_18_19\data_yearly\netatmo_stn_coords.csv',
            sep=';')
    """
    # DWD data
#     HDF5_dwd_ppt = HDF5(infile=path_to_dwd_ppt_hdf5)
#     # HDF5_dwd_edf = HDF5(infile=path_to_dwd_edf_hdf5)
#     all_dwd_ids = HDF5_dwd_ppt.get_all_names()
    # all_dwd_ids_with_data = [_id for _id in all_dwd_ids if len(_id) > 0]

    #=========================================================================
    # COORDS TREE DWD
    #=========================================================================
    # create a tree from DWD coordinates

    dwd_coords_xy = [(x, y) for x, y in zip(
        in_dwd_df_coords_utm32.loc[:, 'X'].values,
        in_dwd_df_coords_utm32.loc[:, 'Y'].values)]

    # create a tree from coordinates
    dwd_points_tree = cKDTree(dwd_coords_xy)

    dwd_stns_ids = in_dwd_df_coords_utm32.index

    # if debug mode use only one worker
    # if sys.gettrace():
    #    n_workers = 1

    print('Using %d Workers' % n_workers)

    all_netatmo_stns_ids_worker = np.array_split(
        all_netatmo_ids_with_good_data, n_workers)
    # args_worker = []

    procs = []
    for netatmo_ids_with_good_data in all_netatmo_stns_ids_worker:

        #         args_worker.append((in_netatmo_df_coords_utm32,
        #                             in_dwd_df_coords_utm32,
        #                             dwd_points_tree,
        #                             dwd_stns_ids,
        #                             # all_dwd_stns_ids,
        #                             netatmo_ids_with_good_data,
        #                             # path_to_neatmo_edf_hdf5,
        #                             path_to_neatmo_ppt_hdf5,
        #                             path_to_dwd_ppt_hdf5))

        procs.append(mp.Process(
            target=correct_pws, args=[(in_netatmo_df_coords_utm32,
                                       in_dwd_df_coords_utm32,
                                       dwd_points_tree,
                                       dwd_stns_ids,
                                       # ['70:ee:50:2a:e5:b2'],
                                       netatmo_ids_with_good_data,
                                       all_netatmo_ids_with_good_data_2018,
                                       all_netatmo_ids_with_good_data_2019,
                                       path_to_neatmo_ppt_hdf5,
                                       path_to_dwd_ppt_hdf5)]))
#         for _ in range(5):
#             while gc.collect() != 0:
#                 gc.collect()
#             del gc.garbage[:]
        # print('gc_collect', gc.collect())
    # l = mp.Lock()
    # , initializer=init, initargs=(l,))
    print(len(procs))
    [proc.start() for proc in procs]
    # my_pool = mp.Pool(processes=n_workers)

    # my_pool.map(correct_pws, args_worker)

    # my_pool.terminate()

    # my_pool.close()
    # my_pool.join()

    return

#==============================================================================
#
#==============================================================================


def correct_pws(args):

    (netatmo_in_coords_df,
     dwd_in_coords_df,
     dwd_points_tree,
     dwd_stns_ids,
     # all_dwd_ids_with_data,
     netatmo_ids,
     all_netatmo_ids_with_good_data_2018,
     all_netatmo_ids_with_good_data_2019,
     # path_to_neatmo_edf_hdf5,
     path_to_neatmo_ppt_hdf5,
     path_to_dwd_ppt_hdf5) = args
#     (path_to_netatmo_coords,
#      path_to_dwd_coords,
#      path_to_netatmo_gd_stns,
#      path_to_neatmo_ppt_hdf5,
#      path_to_neatmo_edf_hdf5,
#      path_to_dwd_ppt_hdf5,
#      all_netatmo_stns_ids_worker) = args
    HDF5_netatmo_ppt = HDF5(infile=path_to_neatmo_ppt_hdf5)
    # HDF5_netatmo_edf = HDF5(infile=path_to_neatmo_edf_hdf5)
    HDF5_dwd_ppt = HDF5(infile=path_to_dwd_ppt_hdf5)

#     for _ in range(5):
#         while gc.collect() != 0:
#             gc.collect()
#         del gc.garbage[:]
    # print('gc_collect', gc.collect())

    def get_dwd_ngbr_netatmo_stn(netatmo_stn):
        xnetatmo = netatmo_in_coords_df.loc[netatmo_stn, 'X']
        ynetatmo = netatmo_in_coords_df.loc[netatmo_stn, 'Y']

        # find neighboring DWD stations
        # find distance to all dwd stations, sort them, select minimum
        _, indices = dwd_points_tree.query(
            np.array([xnetatmo, ynetatmo]),
            k=nbr_dwd_neighbours_to_use + 1)

        dwd_stns_near = dwd_stns_ids[indices[:nbr_dwd_neighbours_to_use]]
        # dist_netatmo_dwd = distances[:nbr_dwd_neighbours_to_use]
#         dwd_stns_near = [stn for stn in dwd_stns_near_all
#                          if stn in all_dwd_ids_with_data]
        return xnetatmo, ynetatmo, dwd_stns_near

    def find_dwd_ppt_netatmo_edf(df_col, edf_netatmo):

        df_col = df_col[~np.isnan(df_col)]
        if df_col.size > 0:

            x0_dwd, y0_dwd = build_edf_fr_vals(df_col.values)
            y0_dwd[y0_dwd == 1] = 0.99999999
            # find nearest DWD ppt to Netatmo percentile
            nearst_dwd_edf = find_nearest(array=y0_dwd,
                                          value=edf_netatmo)
            ppt_idx = np.where(y0_dwd == nearst_dwd_edf)

            ppt_for_edf = x0_dwd[ppt_idx][0]
            if ppt_for_edf >= 0:
                return ppt_for_edf

    def scale_vg_based_on_dwd_ppt(ppt_dwd_vals, vg_sill_b4_scale):
        # sacle variogram based on dwd ppt
        # vg_sill = float(vg_model_to_scale.split(" ")[0])
        dwd_vals_var = np.var(ppt_dwd_vals)
        vg_scaling_ratio = dwd_vals_var / vg_sill_b4_scale

        if vg_scaling_ratio == 0:
            vg_scaling_ratio = vg_sill_b4_scale
        # rescale variogram
#         vgs_model_dwd_ppt = str(
#             np.round(vg_scaling_ratio, 4)
#         ) + ' ' + vg_model_to_scale.split(" ")[1]
#         vgs_model_dwd_ppt
        return vg_scaling_ratio

    def correct_pws_inner_loop(netatmo_edf):

        if netatmo_edf == 1:
            netatmo_edf = 0.999999999
        # print(netatmo_edf)

        dwd_ppt_netatmo_edf = dwd_ppt_neigbrs.apply(
            find_dwd_ppt_netatmo_edf, axis=0,
            args=netatmo_edf, raw=False)

        dwd_ppt_netatmo_edf.dropna(how='all', inplace=True)
        dwd_stns = dwd_ppt_netatmo_edf.reset_index()['level_0'].values
        dwd_xcoords = np.array(
            dwd_in_coords_df.loc[dwd_stns, 'X'])
        dwd_ycoords = np.array(
            dwd_in_coords_df.loc[dwd_stns, 'Y'])

        # gc.collect()
        # sacle variogram based on dwd ppt
        vg_scaling_ratio = scale_vg_based_on_dwd_ppt(
            dwd_ppt_netatmo_edf.values, vg_sill_b4_scale)

        # start kriging Netatmo location
        OK_dwd_netatmo_crt = OKpy(
            dwd_xcoords, dwd_ycoords, dwd_ppt_netatmo_edf.values,
            variogram_model=vg_model_str,
            variogram_parameters={
                'sill': vg_scaling_ratio,
                'range': vg_range,
                'nugget': 0})

        # sigma = _
        try:
            zvalues, _ = OK_dwd_netatmo_crt.execute(
                'points', np.array([xnetatmo]), np.array([ynetatmo]))
        except Exception:
            print('ror')
            pass
#         del(dwd_ppt_netatmo_edf, vg_scaling_ratio, OK_dwd_netatmo_crt)
#
#         while gc.collect() != 0:
        gc.collect()
        del gc.garbage[:]
        # print('gc_collect', gc.collect())
        # print(np.round(zvalues[0], 3))
        return np.round(zvalues[0], 3)

    def plot_obsv_vs_corrected(netatmo_stn, df_obsv, df_correct):

        plt.ioff()
        max_ppt = max(df_obsv.values.max(), df_correct.values.max())
        plt.scatter(df_obsv.values, df_correct.values, c='r')
        plt.xlabel('Original [mm/h] -| |- Sum: %0.2f mm' %
                   df_obsv.values.sum())
        plt.ylabel('Corrected [mm/h] -| |- Sum: %0.2f mm' %
                   df_correct.values.sum())
        plt.plot([0, max_ppt], [0, max_ppt], c='k', alpha=0.25)
        plt.title('%s \n %s-%s'
                  % (netatmo_stn, str(df_obsv.index[0]), str(df_obsv.index[-1])))
        plt.grid(alpha=0.25)
        plt.savefig(os.path.join(out_save_dir,
                                 'netatmo_stn_%s.png' % netatmo_stn))
        plt.close()

    def plot_max_daily_sums(netatmo_stn, df_obsv, df_correct):
        netatmo_stn_str = netatmo_stn.replace(':', '_')
        netatmo_orig_daily = resampleDf(df_obsv, 'D')
        netatmo_corr_daily = resampleDf(df_correct, 'D')

        max_10d_orig = netatmo_orig_daily.sort_values(
            by=[netatmo_stn])[-10:]
        max_10d_corr = netatmo_corr_daily.sort_values(
            by=[netatmo_stn])[-10:]

        cmn_days = max_10d_corr.index.intersection(
            max_10d_orig.index).sort_values()

        df_orig_daily_maxs = netatmo_orig_daily.loc[cmn_days, :]
        df_corr_daily_maxs = netatmo_corr_daily.loc[cmn_days, :]

        plt.ioff()
        max_ppt = max(df_orig_daily_maxs.values.max(),
                      df_corr_daily_maxs.values.max())
        plt.scatter(df_orig_daily_maxs.values,
                    df_corr_daily_maxs.values, c='b')
        plt.xlabel('Original [mm/d] -| |- Sum: %0.2f mm' %
                   df_orig_daily_maxs.values.sum())
        plt.ylabel('Corrected [mm/d] -| |- Sum: %0.2f mm' %
                   df_corr_daily_maxs.values.sum())
        plt.plot([0, max_ppt], [0, max_ppt], c='k', alpha=0.25)
        plt.title('Maximum daily sums for %s' % netatmo_stn)
        plt.grid(alpha=0.25)
        plt.savefig(os.path.join(
            out_save_dir,
            'max_10_days_netatmo_stn_%s.png' % netatmo_stn_str))
        plt.close()

    for ix, netatmo_stn in enumerate(netatmo_ids):
        start = time.time()
        # netatmo_stn = '70:ee:50:27:21:ea'
        netatmo_stn_str = netatmo_stn.replace(':', '_')
        print('Correcting ', netatmo_stn, ': ', ix, '/', len(netatmo_ids))

        if netatmo_stn in all_netatmo_ids_with_good_data_2018:
            start_date = '2018-04-01 00:00:00'
            end_date = '2018-10-31 23:00:00'

        if netatmo_stn in all_netatmo_ids_with_good_data_2019:
            start_date = '2019-04-01 00:00:00'
            end_date = '2019-10-31 23:00:00'

        try:
            netatmo_ppt_df = HDF5_netatmo_ppt.get_pandas_dataframe(netatmo_stn)

            netatmo_ppt_df_1819 = select_df_within_period(netatmo_ppt_df,
                                                          start=start_date,
                                                          end=end_date)

            # select only convective season
            netatmo_ppt_df_summer = select_convective_season(
                df=netatmo_ppt_df_1819,
                month_lst=not_convective_season).dropna(how='all')

            # netatmo_ppt_df_summer.loc['2018-05-29']

            # netatmo_ppt_df_summer[netatmo_ppt_df_summer>0].dropna()
            if netatmo_ppt_df_summer.size > min_req_ppt_vals:

                netatmo_edf_df = convert_ppt_df_to_edf(
                    df=netatmo_ppt_df_summer,
                    stationname=netatmo_stn,
                    ppt_min_thr_0_vals=ppt_min_thr_0_vals)

                netatmo_edf_df.dropna(how='all')
                # get all dwd ppt for corresponding netatmo edf
                # get netatmo coords and find dwd neighbors
                xnetatmo, ynetatmo, dwd_stns_near = get_dwd_ngbr_netatmo_stn(
                    netatmo_stn)
                empty_data = np.zeros(shape=(len(netatmo_edf_df.index), 1))
                empty_data[empty_data == 0] = np.nan

                netatmo_ppt_corrected_filled = pd.DataFrame(
                    index=netatmo_edf_df.index, data=empty_data,
                    columns=[netatmo_stn])

                # get DWD ppt data for this time period
                dwd_ppt_neigbrs = HDF5_dwd_ppt.get_pandas_dataframe_bet_dates(
                    dwd_stns_near, start_date=netatmo_edf_df.index[0],
                    end_date=netatmo_edf_df.index[-1])

                # dwd_ppt_neigbrs = dwd_ppt_neigbrs.loc[netatmo_edf_df.index, :]

                netatmo_edf_df_zeros = netatmo_edf_df[netatmo_edf_df.values <=
                                                      min_qt_to_correct]
                netatmo_edf_df_not_zeros = netatmo_edf_df[netatmo_edf_df.values >
                                                          min_qt_to_correct]

    #             print('DF shape: ', netatmo_edf_df_not_zeros.size)
    #
    # #             train = parallelize_dataframe(netatmo_edf_df,
    # #                  correct_pws_inner_loop, n_cores=4)
                netatmo_ppt_corrected_not_zeros = netatmo_edf_df_not_zeros.apply(
                    correct_pws_inner_loop, axis=1, raw=True)

                netatmo_ppt_corrected_filled.loc[
                    netatmo_edf_df_zeros.index,
                    netatmo_stn] = 0.0

                netatmo_ppt_corrected_filled.loc[
                    netatmo_ppt_corrected_not_zeros.index,
                    netatmo_stn] = netatmo_ppt_corrected_not_zeros.values.ravel()

                # netatmo_ppt_df_summer.loc['2019-06-03']
                # netatmo_ppt_corrected_filled.loc['2019-06-03']

                # resample to daily to find maximum values

                end = time.time()

                plot_max_daily_sums(
                    netatmo_stn, netatmo_ppt_df_summer,
                    netatmo_ppt_corrected_filled)

                plot_obsv_vs_corrected(
                    netatmo_stn_str, netatmo_ppt_df_summer,
                    netatmo_ppt_corrected_filled)

                netatmo_ppt_corrected_filled.to_csv(
                    os.path.join(out_save_dir,
                                 'netatmo_stn_%s.csv' % netatmo_stn_str),
                    sep=';', header=[netatmo_stn],
                    float_format='%0.3f')
                print('Needed time (s)', round(end - start, 2))

            else:
                print('+-+* +*+* not enough data')
                raise Exception
                break
        except Exception:
            raise Exception
            pass
#             del (netatmo_edf_df, netatmo_ppt_corrected_filled,
#                  dwd_ppt_neigbrs,
#                  xnetatmo, ynetatmo, netatmo_ppt_df)
#
# #             for _ in range(10):
#             while gc.collect() != 0:
#                 gc.collect()
#             del gc.garbage[:]
#             #print('gc_collect', gc.collect())
#             # break
#         # break
#         pass


# START HERE
if __name__ == '__main__':

    #     args = (path_to_netatmo_coords,
    #             path_to_dwd_coords,
    #             path_to_netatmo_gd_stns,
    #             path_to_neatmo_ppt_hdf5,
    #             path_to_neatmo_edf_hdf5,
    #             path_to_dwd_ppt_hdf5,
    #             all_netatmo_ids_with_good_data)
    #
    args = (# path_to_netatmo_coords,
        # path_to_dwd_coords,
        path_to_netatmo_gd_stns_2018,
        path_to_netatmo_gd_stns_2019,
        path_to_ppt_netatmo_data_hdf5,
        # path_to_ppt_netatmo_data_hdf5,
        path_to_ppt_dwd_data_hdf5)
    process_manager(args)
    # correct_pws(args)

    pass
