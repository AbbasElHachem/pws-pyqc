# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    Correct PWS using Primary Network Data
Purpose: Prepare PWS data for Interpolation

Correct each PWS observation using the neighboring primary network data
Using a unified exponential variogram interpolated each PWS observation
using the Neighboring primary Network data for the same time period,
recorrect the PWS Quantiles using the primary Network long observations
 

Parameters
----------

Input Files
    prim_netw station data
    prim_netw coordinates data
    pws precipitation station data
    pws station coordinates data
    
Returns
-------

    Corrected PWS data, save results in df
    
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

# =============================================================================

import os
import time
import gc

# os.environ[str('MKL_NUM_THREADS')] = str(1)
# os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
# os.environ[str('OMP_NUM_THREADS')] = str(1)

# other Libs

# import pyximport
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.spatial import cKDTree
from pathlib import Path

from spinterps import OrdinaryKriging as OKpy

# own Libs

from _00_functions import (
    select_convective_season,
                           select_df_within_period,
                           build_edf_fr_vals,
                           calculate_probab_ppt_below_thr,
                           find_nearest,
                           resampleDf
                           )

from _01_read_hdf5 import HDF5

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pyximport.install()
# cores = cpu_count() - 1  # Number of CPU cores on system
# pandarallel.initialize(nb_workers=2)
# set settings
gc.set_threshold(0, 0, 0)
# =============================================================================

start_date = '2018-01-01 00:00:00'
end_date = '2019-12-31 00:00:00'

# minimum hourly values that should be available per pws station
min_req_ppt_vals = 30 * 24 * 2

# this is used to keep only data where month is not in this list
not_convective_season = [11, 12, 1, 2, 3]  # oct till april

# how many prim_netw stations to use
nbr_prim_netw_neighbours_to_use = 16

ppt_min_thr_0_vals = 0.1  # below it all values get p0/2
min_qt_to_correct = 0.75  # correct all qunatiles above it

vg_sill_b4_scale = 0.07
vg_range = 4e4
vg_model_str = 'spherical'
vg_model_to_scale = '0.07 Sph(40000)'

n_workers = 1  # int(cores / 1)
# n_sub_proccess = cores - n_workers
# =============================================================================

# main_dir = Path(r"/home/IWS/hachem/pws_CML")
main_dir = Path(r"X:\staff\elhachem\2020_05_20_Netatmo_CML")
os.chdir(main_dir)
# TODO: PATH
path_to_ppt_pws_data_hdf5 = (
    #    r"X:\exchange\ElHachem\pws_correct_18_19\data_yearly\pws_stn_filtered_2018_2019_yearly.h5")
    r"P:\2020_DFG_Netatmo\03_data\01_netatmo\netatmo_Germany_5min_to_1hour_filter_00.h5")
# assert os.path.exists(path_to_ppt_pws_data_hdf5), 'wrong pws Ppt file'

# path_to_ppt_pws_data_hdf5 = (
#     r"C:\Users\hachem\Downloads\pws_filtered_2018_2019_5min_to_1hour.h5")

path_to_ppt_prim_netw_data_hdf5 = (
    # r"P:\2020_DFG_pws\03_data\03_prim_netw\prim_netw_5min_to_1hour.h5")
    r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD\DWD_60min_SS1819_new.h5")
assert os.path.exists(path_to_ppt_prim_netw_data_hdf5), 'wrong prim_netw Csv Ppt file'

# pws FIRST FILTER

path_to_pws_gd_stns_2018 = (main_dir / r'indicator_correlation_60min_99_0' /
                                (r'Netatmo_60min_Good_99_2018.csv'))
# pws_60min_Good_99_2018
path_to_pws_gd_stns_2019 = (main_dir / r'indicator_correlation_60min_99_0' /
                                (r'Netatmo_60min_Good_99_2019.csv'))

# pws FIRST FILTER
# path_to_pws_gd_stns = (
#     main_dir / "indicator_correlation/pws_Good_99.csv")
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

    (path_to_pws_gd_stns_2018,
        path_to_pws_gd_stns_2019,
        path_to_neatmo_ppt_hdf5,
        path_to_prim_netw_ppt_hdf5) = args

    # get all station names for prim_netw
    HDF5_prim_netw = HDF5(infile=path_to_prim_netw_ppt_hdf5)
    all_prim_netw_stns_ids = HDF5_prim_netw.get_all_names()

    # get all station names for pws
    HDF5_pws = HDF5(infile=path_to_neatmo_ppt_hdf5)
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

    # pws first filter
#     df_gd_stns = pd.read_csv(path_to_pws_gd_stns,
#                              index_col=1,
#                              sep=';',
#                              encoding='utf-8')

    df_gd_stns_2018 = pd.read_csv(path_to_pws_gd_stns_2018,
                                  index_col=1,
                                  sep=';',
                                  encoding='utf-8')

    df_gd_stns_2019 = pd.read_csv(path_to_pws_gd_stns_2019,
                                  index_col=1,
                                  sep=';',
                                  encoding='utf-8')

    #=========================================================

    all_pws_ids_with_good_data_2018 = df_gd_stns_2018.index.to_list()
    all_pws_ids_with_good_data_2019 = df_gd_stns_2019.index.to_list()

    # combine all good stns
    all_pws_ids_with_good_data = list(
        set(all_pws_ids_with_good_data_2018) |
        set(all_pws_ids_with_good_data_2019))
    # get indices of those stations

    #=========================================================================
    # COORDS TREE prim_netw
    #=========================================================================
    # create a tree from prim_netw coordinates

    prim_netw_coords_xy = [(x, y) for x, y in zip(
        in_prim_netw_df_coords_utm32.loc[:, 'X'].values,
        in_prim_netw_df_coords_utm32.loc[:, 'Y'].values)]

    # create a tree from coordinates
    prim_netw_points_tree = cKDTree(prim_netw_coords_xy)

    prim_netw_stns_ids = in_prim_netw_df_coords_utm32.index

    # if debug mode use only one worker
    # if sys.gettrace():
    #    n_workers = 1

    print('Using %d Workers' % n_workers)

    all_pws_stns_ids_worker = np.array_split(
        all_pws_ids_with_good_data, n_workers)
    # args_worker = []

    procs = []
    for pws_ids_with_good_data in all_pws_stns_ids_worker:


        procs.append(mp.Process(
            target=correct_pws, args=[(in_pws_df_coords_utm32,
                                       in_prim_netw_df_coords_utm32,
                                       prim_netw_points_tree,
                                       prim_netw_stns_ids,
                                       # ['70:ee:50:2a:e5:b2'],
                                       pws_ids_with_good_data,
                                       all_pws_ids_with_good_data_2018,
                                       all_pws_ids_with_good_data_2019,
                                       path_to_neatmo_ppt_hdf5,
                                       path_to_prim_netw_ppt_hdf5)]))

    print(len(procs))
    [proc.start() for proc in procs]

    return

#==============================================================================
#
#==============================================================================


def correct_pws(args):

    (pws_in_coords_df,
     prim_netw_in_coords_df,
     prim_netw_points_tree,
     prim_netw_stns_ids,
     # all_prim_netw_ids_with_data,
     pws_ids,
     all_pws_ids_with_good_data_2018,
     all_pws_ids_with_good_data_2019,
     path_to_neatmo_ppt_hdf5,
     path_to_prim_netw_ppt_hdf5) = args

    HDF5_pws_ppt = HDF5(infile=path_to_neatmo_ppt_hdf5)
    HDF5_prim_netw_ppt = HDF5(infile=path_to_prim_netw_ppt_hdf5)

    def get_prim_netw_ngbr_pws_stn(pws_stn):
        xpws = pws_in_coords_df.loc[pws_stn, 'X']
        ypws = pws_in_coords_df.loc[pws_stn, 'Y']

        # find neighboring prim_netw stations
        # find distance to all prim_netw stations,
        # sort them, select minimum
        _, indices = prim_netw_points_tree.query(
            np.array([xpws, ypws]),
            k=nbr_prim_netw_neighbours_to_use + 1)

        prim_netw_stns_near = prim_netw_stns_ids[
            indices[:nbr_prim_netw_neighbours_to_use]]
        return xpws, ypws, prim_netw_stns_near

    def find_prim_netw_ppt_pws_edf(df_col, edf_pws):

        df_col = df_col[~np.isnan(df_col)]
        if df_col.size > 0:

            x0_prim_netw, y0_prim_netw = build_edf_fr_vals(df_col.values)
            y0_prim_netw[y0_prim_netw == 1] = 0.99999999
            # find nearest prim_netw ppt to pws percentile
            nearst_prim_netw_edf = find_nearest(array=y0_prim_netw,
                                          value=edf_pws)
            ppt_idx = np.where(y0_prim_netw == nearst_prim_netw_edf)

            ppt_for_edf = x0_prim_netw[ppt_idx][0]
            if ppt_for_edf >= 0:
                return ppt_for_edf

    def scale_vg_based_on_prim_netw_ppt(ppt_dwd_vals, vg_sill_b4_scale):
        # sacle variogram based on dwd ppt
#         vg_sill = float(vg_model_to_scale.split(" ")[0])
        dwd_vals_var = np.var(ppt_dwd_vals)
        vg_scaling_ratio = dwd_vals_var / vg_sill_b4_scale

        if vg_scaling_ratio == 0:
            vg_scaling_ratio = vg_sill_b4_scale

        # rescale variogram
        vgs_model_dwd_ppt = str(
            np.round(vg_scaling_ratio, 4)
        ) + ' ' + vg_model_to_scale.split(" ")[1]
#         vgs_model_dwd_ppt
        return vgs_model_dwd_ppt  # vg_scaling_ratio

    def correct_pws_inner_loop(pws_edf):

        if pws_edf == 1:
            pws_edf = 0.999999999
        # print(pws_edf)

        prim_netw_ppt_pws_edf = prim_netw_ppt_neigbrs.apply(
            find_prim_netw_ppt_pws_edf, axis=0,
            args=pws_edf, raw=False)

        prim_netw_ppt_pws_edf.dropna(how='all', inplace=True)
        prim_netw_stns = prim_netw_ppt_pws_edf.reset_index()['level_0'].values
        prim_netw_xcoords = np.array(
            prim_netw_in_coords_df.loc[prim_netw_stns, 'X'])
        prim_netw_ycoords = np.array(
            prim_netw_in_coords_df.loc[prim_netw_stns, 'Y'])

        # gc.collect()
        # sacle variogram based on prim_netw ppt
        vgs_model_dwd_ppt = scale_vg_based_on_prim_netw_ppt(
            prim_netw_ppt_pws_edf.values, vg_sill_b4_scale)

        # start kriging pws location

        OK_prim_netw_pws_crt = OKpy(xi=prim_netw_xcoords,
                                            yi=prim_netw_ycoords,
                                            zi=prim_netw_ppt_pws_edf.values,
                                            xk=np.array([xpws]),
                                            yk=np.array([ypws]),
                                            model=vgs_model_dwd_ppt)
        # sigma = _
        try:
            OK_prim_netw_pws_crt.krige()
            zvalues = OK_prim_netw_pws_crt.zk.copy()
        except Exception:
            print('ror')
            pass

        gc.collect()
        del gc.garbage[:]

        return np.round(zvalues[0], 3)

    def plot_obsv_vs_corrected(pws_stn, df_obsv, df_correct):

        plt.ioff()
        max_ppt = max(df_obsv.values.max(), df_correct.values.max())
        plt.scatter(df_obsv.values, df_correct.values, c='r')
        plt.xlabel('Original [mm/h] -| |- Sum: %0.2f mm' %
                   df_obsv.values.sum())
        plt.ylabel('Corrected [mm/h] -| |- Sum: %0.2f mm' %
                   df_correct.values.sum())
        plt.plot([0, max_ppt], [0, max_ppt], c='k', alpha=0.25)
        plt.title('%s \n %s-%s'
                  % (pws_stn, str(df_obsv.index[0]), str(df_obsv.index[-1])))
        plt.grid(alpha=0.25)
        plt.savefig(os.path.join(out_save_dir,
                                 'pws_stn_%s.png' % pws_stn))
        plt.close()

    def plot_max_daily_sums(pws_stn, df_obsv, df_correct):
        pws_stn_str = pws_stn.replace(':', '_')
        pws_orig_daily = resampleDf(df_obsv, 'D')
        pws_corr_daily = resampleDf(df_correct, 'D')

        max_10d_orig = pws_orig_daily.sort_values(
            by=[pws_stn])[-10:]
        max_10d_corr = pws_corr_daily.sort_values(
            by=[pws_stn])[-10:]

        cmn_days = max_10d_corr.index.intersection(
            max_10d_orig.index).sort_values()

        df_orig_daily_maxs = pws_orig_daily.loc[cmn_days, :]
        df_corr_daily_maxs = pws_corr_daily.loc[cmn_days, :]

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
        plt.title('Maximum daily sums for %s' % pws_stn)
        plt.grid(alpha=0.25)
        plt.savefig(os.path.join(
            out_save_dir,
            'max_10_days_pws_stn_%s.png' % pws_stn_str))
        plt.close()

    for ix, pws_stn in enumerate(pws_ids):
        start = time.time()
        # pws_stn = '70:ee:50:27:21:ea'
        pws_stn_str = pws_stn.replace(':', '_')
        print('Correcting ', pws_stn, ': ', ix, '/', len(pws_ids))

        if pws_stn in all_pws_ids_with_good_data_2018:
            start_date = '2018-04-01 00:00:00'
            end_date = '2018-10-31 23:00:00'

        if pws_stn in all_pws_ids_with_good_data_2019:
            start_date = '2019-04-01 00:00:00'
            end_date = '2019-10-31 23:00:00'

        try:
            pws_ppt_df = HDF5_pws_ppt.get_pandas_dataframe(pws_stn)

            pws_ppt_df_1819 = select_df_within_period(pws_ppt_df,
                                                          start=start_date,
                                                          end=end_date)

            # select only convective season
            pws_ppt_df_summer = select_convective_season(
                df=pws_ppt_df_1819,
                month_lst=not_convective_season).dropna(how='all')

            # pws_ppt_df_summer.loc['2018-05-29']

            # pws_ppt_df_summer[pws_ppt_df_summer>0].dropna()
            if pws_ppt_df_summer.size > min_req_ppt_vals:

                pws_edf_df = convert_ppt_df_to_edf(
                    df=pws_ppt_df_summer,
                    stationname=pws_stn,
                    ppt_min_thr_0_vals=ppt_min_thr_0_vals)

                pws_edf_df.dropna(how='all')
                # get all prim_netw ppt for corresponding pws edf
                # get pws coords and find prim_netw neighbors
                xpws, ypws, prim_netw_stns_near = get_prim_netw_ngbr_pws_stn(
                    pws_stn)
                empty_data = np.zeros(shape=(len(pws_edf_df.index), 1))
                empty_data[empty_data == 0] = np.nan

                pws_ppt_corrected_filled = pd.DataFrame(
                    index=pws_edf_df.index, data=empty_data,
                    columns=[pws_stn])

                # get prim_netw ppt data for this time period
                prim_netw_ppt_neigbrs = (
                    HDF5_prim_netw_ppt.get_pandas_dataframe_bet_dates(
                    prim_netw_stns_near, start_date=pws_edf_df.index[0],
                    end_date=pws_edf_df.index[-1]))

                pws_edf_df_zeros = pws_edf_df[pws_edf_df.values <=
                                                      min_qt_to_correct]
                pws_edf_df_not_zeros = pws_edf_df[pws_edf_df.values >
                                                          min_qt_to_correct]

                pws_ppt_corrected_not_zeros = pws_edf_df_not_zeros.apply(
                    correct_pws_inner_loop, axis=1, raw=True)

                pws_ppt_corrected_filled.loc[
                    pws_edf_df_zeros.index,
                    pws_stn] = 0.0

                pws_ppt_corrected_filled.loc[
                    pws_ppt_corrected_not_zeros.index,
                    pws_stn] = pws_ppt_corrected_not_zeros.values.ravel()

                end = time.time()

                plot_max_daily_sums(
                    pws_stn, pws_ppt_df_summer,
                    pws_ppt_corrected_filled)

                plot_obsv_vs_corrected(
                    pws_stn_str, pws_ppt_df_summer,
                    pws_ppt_corrected_filled)

                pws_ppt_corrected_filled.to_csv(
                    os.path.join(out_save_dir,
                                 'pws_stn_%s.csv' % pws_stn_str),
                    sep=';', header=[pws_stn],
                    float_format='%0.3f')
                print('Needed time (s)', round(end - start, 2))

            else:
                print('+-+* +*+* not enough data')
                raise Exception
                break
        except Exception:
            raise Exception
            pass


#----------------------------------------------------------------------------
# # START HERE
#----------------------------------------------------------------------------
if __name__ == '__main__':

    args = (
        path_to_pws_gd_stns_2018,
        path_to_pws_gd_stns_2019,
        path_to_ppt_pws_data_hdf5,
        path_to_ppt_prim_netw_data_hdf5)
    process_manager(args)

