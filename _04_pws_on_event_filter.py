# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    Correct PWS using Primary Network Data
Purpose: Prepare PWS data for Interpolation

Created on: 2020-05-16


Parameters
----------

Input Files
    DWD station data
    DWD coordinates data
    Netatmo precipitation station data
    Netatmo station coordinates data
    
Returns
-------

    Corrected PWS data, save results in csv format
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

import os
import gc

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scipy import spatial
import pyximport
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

# from spinterps import (OrdinaryKriging)
from pykrige.ok import OrdinaryKriging as OKpy

from _01_2_read_hdf5 import HDF5

pyximport.install()

gc.set_threshold(0, 0, 0)
# =============================================================================

# this is used to keep only data where month is not in this list
not_convective_season = [11, 12, 1, 2, 3]  # oct till april

vg_sill_b4_scale = 0.07
vg_range = 4e4
vg_model_str = 'spherical'
# vg_model_to_scale = '0.07 Sph(40000)'

_year = '2019'

start_date = '%s-04-01 00:00:00' % _year
end_date = '%s-10-31 23:00:00' % _year
# =============================================================================

# main_dir = Path(r"/home/IWS/hachem/Netatmo_CML")
main_dir = Path(r"X:\staff\elhachem\2020_05_20_Netatmo_CML")
os.chdir(main_dir)

# "X:\exchange\ElHachem\Netatmo_correct_18_19
path_to_ppt_netatmo_data_hdf5 = (
    r"X:\exchange\ElHachem\Netatmo_correct_18_19"
    r"\data_%s_new\PWS_filtered_corrected_%s.h5"
    % (_year, _year))
assert os.path.exists(path_to_ppt_netatmo_data_hdf5), 'wrong NETATMO Ppt file'

path_to_ppt_dwd_data_hdf5 = (
    # r"P:\2020_DFG_Netatmo\03_data\03_dwd\DWD_5min_to_1hour.h5")
    r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD\dwd_comb_60min_SS1819.h5")
assert os.path.exists(path_to_ppt_dwd_data_hdf5), 'wrong DWD Csv Ppt file'

title_ = r'second_filter_PWS_%s_new' % _year

out_save_dir = main_dir / title_

if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)

n_workers = 7
#==============================================================================
#
#==============================================================================


def process_manager(args):

    (path_to_neatmo_ppt_hdf5,
        path_to_dwd_ppt_hdf5) = args

    HDF5_netatmo_ppt = HDF5(infile=path_to_neatmo_ppt_hdf5)
    all_netatmo_ids = HDF5_netatmo_ppt.get_all_names()

    HDF5_dwd_ppt = HDF5(infile=path_to_dwd_ppt_hdf5)
    all_dwd_stns_ids = HDF5_dwd_ppt.get_all_names()
    netatmo_coords = HDF5_netatmo_ppt.get_coordinates(all_netatmo_ids)

    netatmo_in_coords_df = pd.DataFrame(
        index=all_netatmo_ids,
        data=netatmo_coords['easting'], columns=['X'])
    y_netatmo_coords = netatmo_coords['northing']
    netatmo_in_coords_df.loc[:, 'Y'] = y_netatmo_coords
    # netatmo_in_coords_df.index.difference(all_netatmo_ids)
    dwd_coords = HDF5_dwd_ppt.get_coordinates(all_dwd_stns_ids)

    dwd_in_coords_df = pd.DataFrame(
        index=all_dwd_stns_ids,
        data=dwd_coords['easting'], columns=['X'])
    y_dwd_coords = dwd_coords['northing']
    dwd_in_coords_df.loc[:, 'Y'] = y_dwd_coords

    #=========================================================================
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    date_range_summer = pd.DatetimeIndex([date_ for date_ in date_range
                                          if date_.month not in not_convective_season])

    print('Using Workers: ', n_workers)
    # devide stations on workers
    all_timestamps_worker = np.array_split(date_range_summer, n_workers)
    args_worker = []

    for time_list in all_timestamps_worker:
        empty_data = np.zeros(shape=(len(time_list),
                                     len(all_netatmo_ids)))
        empty_data[empty_data == 0] = np.nan
        df_save_results = pd.DataFrame(index=time_list,
                                       columns=all_netatmo_ids, data=empty_data)
    # args_workers = list(repeat(args, n_worker))

        args_worker.append((path_to_dwd_ppt_hdf5,
                            dwd_in_coords_df,
                            path_to_neatmo_ppt_hdf5,
                            netatmo_in_coords_df,
                            time_list,
                            df_save_results))

    my_pool = mp.Pool(processes=n_workers)
    # TODO: Check number of accounts

    results = my_pool.map(
        on_evt_filter_pws, args_worker)

    # my_pool.terminate()

    my_pool.close()
    my_pool.join()

    results_df = pd.concat(results)

    results_df.to_csv(
        os.path.join(out_save_dir,
                     'netatmo_flagged_%s.csv' % (_year)),
        sep=';')

    return
#==============================================================================
# MAIN FUNCTION
#==============================================================================


def on_evt_filter_pws(args):

    (path_to_dwd_ppt_hdf5,
     dwd_in_coords_df,
     path_to_neatmo_ppt_hdf5,
     netatmo_in_coords_df,
     time_list,
     df_save_results) = args

    HDF5_netatmo_ppt = HDF5(infile=path_to_neatmo_ppt_hdf5)
    all_netatmo_ids = HDF5_netatmo_ppt.get_all_names()

    HDF5_dwd_ppt = HDF5(infile=path_to_dwd_ppt_hdf5)
    all_dwd_ids = HDF5_dwd_ppt.get_all_names()

    def scale_vg_based_on_dwd_ppt(ppt_dwd_vals, vg_sill_b4_scale):
        # sacle variogram based on dwd ppt
        # vg_sill = float(vg_model_to_scale.split(" ")[0])
        dwd_vals_var = np.var(ppt_dwd_vals)
        vg_scaling_ratio = dwd_vals_var / vg_sill_b4_scale

        if vg_scaling_ratio == 0:
            vg_scaling_ratio = vg_sill_b4_scale
        # rescale variogram

        return vg_scaling_ratio

    def plot_good_bad_stns(netatmo_in_coords_df,
                           ids_netatmo_stns_gd,
                           ids_netatmo_stns_bad,
                           zvalues,
                           dwd_ppt,
                           zvalues_bad,
                           dwdx, dwdy,
                           event_date):

        xstns_good = netatmo_in_coords_df.loc[
            ids_netatmo_stns_gd, 'X'].values.ravel()
        ystns_good = netatmo_in_coords_df.loc[
            ids_netatmo_stns_gd, 'Y'].values.ravel()

        xstns_bad = netatmo_in_coords_df.loc[
            ids_netatmo_stns_bad, 'X'].values.ravel()
        ystns_bad = netatmo_in_coords_df.loc[
            ids_netatmo_stns_bad, 'Y'].values.ravel()
        max_ppt = max(np.nanmax(zvalues), np.nanmax(dwd_ppt))

        interval_ppt = np.linspace(0.0, 0.99)
        colors_ppt = plt.get_cmap('Blues')(interval_ppt)
        cmap_ppt = LinearSegmentedColormap.from_list('name', colors_ppt)
        # cmap_ppt = plt.get_cmap('jet_r')
        cmap_ppt.set_over('navy')

        interval_ppt_bad = np.linspace(0.02, 0.95)
        colors_ppt_bad = plt.get_cmap('autumn')(interval_ppt_bad)
        cmap_ppt_bad = LinearSegmentedColormap.from_list(
            'name', colors_ppt_bad)

        plt.ioff()
        plt.figure(figsize=(12, 8), dpi=100)
        plt.scatter(dwdx, dwdy, c=dwd_ppt, cmap=cmap_ppt,
                    marker=',', s=10, alpha=0.75, vmin=0, vmax=max_ppt,
                    label='DWD %d' % dwdx.size)

        sc = plt.scatter(xstns_good, ystns_good, c=zvalues,
                         cmap=cmap_ppt,
                         marker='.', s=10, alpha=0.75, vmin=0, vmax=max_ppt,
                         label='PWS Gd %d' % xstns_good.size)

#         plt.tricontourf(xstns_good,
#                         ystns_good, zvalues, levels=14,
#                         cmap=plt.get_cmap('jet_r'))

        plt.scatter(xstns_bad, ystns_bad,
                    alpha=0.75, c=zvalues_bad, cmap=cmap_ppt_bad,
                    marker='x', s=10, vmin=0, vmax=max_ppt,
                    label='PWS Bd %d' % xstns_bad.size)
        # plt.show()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.legend(loc=0)
        cbar = plt.colorbar(sc, extend='max')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('[mm/hr]')
        plt.title('Event date %s ' % (event_date))
        plt.grid(alpha=0.25)
        plt.savefig(os.path.join(out_save_dir,
                                 'event_date_%s.png'
                                 % (str(event_date).replace('-', '_').replace(':', '_'))))
        plt.close()

    #==========================================================================
    #
    #==========================================================================

    for ix, date_to_correct in enumerate(time_list):

        print(ix, '/', len(time_list), '--', date_to_correct)
        # netatmo_data = pd.read_feather(path_netatamo_edf_fk, columns=netatmo_ids_str)
        netatmo_data_evt = HDF5_netatmo_ppt.get_pandas_dataframe_for_date(
            ids=all_netatmo_ids, event_date=date_to_correct).dropna(how='all', axis=1)

        if len(netatmo_data_evt.columns) > 0:
            netatmo_stns_evt = netatmo_data_evt.columns.to_list()

            # coords of stns to correct
            xstns_interp = netatmo_in_coords_df.loc[
                netatmo_stns_evt, 'X'].values.ravel()
            ystns_interp = netatmo_in_coords_df.loc[
                netatmo_stns_evt, 'Y'].values.ravel()

            # dwd data
            ppt_dwd_vals_evt = HDF5_dwd_ppt.get_pandas_dataframe_for_date(
                ids=all_dwd_ids, event_date=date_to_correct)
            dwd_stns_evt = ppt_dwd_vals_evt.columns.to_list()

            dwd_xcoords = dwd_in_coords_df.loc[
                dwd_stns_evt, 'X'].values.ravel()
            dwd_ycoords = dwd_in_coords_df.loc[
                dwd_stns_evt, 'Y'].values.ravel()

            vg_scaling_ratio = scale_vg_based_on_dwd_ppt(
                ppt_dwd_vals=ppt_dwd_vals_evt.values.ravel(),
                vg_sill_b4_scale=vg_sill_b4_scale)

            # start kriging Netatmo location
            OK_dwd_netatmo_crt = OKpy(
                dwd_xcoords, dwd_ycoords,
                ppt_dwd_vals_evt.values,
                variogram_model=vg_model_str,
                variogram_parameters={
                    'sill': vg_scaling_ratio,
                    'range': vg_range,
                    'nugget': 0})

            # sigma = _
            try:
                zvalues, est_var = OK_dwd_netatmo_crt.execute(
                    'points', np.array([xstns_interp]), np.array([ystns_interp]))
            except Exception as msg:
                print('ror', msg)

            zvalues = np.round(zvalues.data, 2)
            # if neg assign 0
            zvalues[zvalues < 0] = 0
            # calcualte standard deviation of estimated values
            std_est_vals = np.sqrt(est_var).data
            # calculate difference observed and estimated
            # values
            diff_obsv_interp = np.abs(netatmo_data_evt.values - zvalues)

            idx_good_stns = np.where(
                diff_obsv_interp <= 3 * std_est_vals)[1]
            idx_bad_stns = np.where(
                diff_obsv_interp > 3 * std_est_vals)[1]

            if len(idx_bad_stns) or len(idx_good_stns) > 0:

                # use additional filter
                try:
                    ids_netatmo_stns_gd = np.take(netatmo_stns_evt,
                                                  idx_good_stns).ravel()
                    ids_netatmo_stns_bad = np.take(netatmo_stns_evt,
                                                   idx_bad_stns).ravel()

                except Exception as msg:
                    print(msg)
                # ids of bad stns
                xstns_bad = netatmo_in_coords_df.loc[
                    ids_netatmo_stns_bad, 'X'].values.ravel()
                ystns_bad = netatmo_in_coords_df.loc[
                    ids_netatmo_stns_bad, 'Y'].values.ravel()
#                 # check if bad are truly bad
                xstns_good = netatmo_in_coords_df.loc[
                    ids_netatmo_stns_gd, 'X'].values.ravel()
                ystns_good = netatmo_in_coords_df.loc[
                    ids_netatmo_stns_gd, 'Y'].values.ravel()

                # coords of neighbors good
                neighbors_coords = np.array(
                    [(x, y) for x, y in zip(xstns_good, ystns_good)])

                # create a tree from coordinates
                points_tree = spatial.cKDTree(neighbors_coords)

                neighbors_coords_dwd = np.array(
                    [(x, y) for x, y in zip(dwd_xcoords, dwd_ycoords)])

                points_tree_dwd = spatial.cKDTree(neighbors_coords_dwd)

#                 plt.ioff()
#                 plt.scatter(xstns_good, ystns_good, c='b')
#                 plt.scatter(dwd_xcoords, dwd_ycoords, c='g')
#                 plt.scatter(xstns_bad, ystns_bad, c='r')
#                 plt.show()
                if len(idx_bad_stns) > 0 or len(idx_good_stns) > 0:
                    for stn_ix, stn_bad in zip(idx_bad_stns, ids_netatmo_stns_bad):
                        ppt_bad = netatmo_data_evt.loc[:, stn_bad].values
                        # print('ppt_bad', ppt_bad)
                        if ppt_bad >= 0.:

                            xstn_bd = netatmo_in_coords_df.loc[
                                stn_bad, 'X']
                            ystn_bd = netatmo_in_coords_df.loc[
                                stn_bad, 'Y']

                            idxs_neighbours = points_tree.query_ball_point(
                                np.array((xstn_bd, ystn_bd)), 1e4)

                            ids_neighbours = ids_netatmo_stns_gd[idxs_neighbours]
                            ids_neighbours_evt = np.in1d(
                                netatmo_stns_evt, ids_neighbours)

                            idxs_neighbours_dwd = points_tree_dwd.query_ball_point(
                                np.array((xstn_bd, ystn_bd)), 1e4)

                            ids_neighbours_dwd_evt = np.array(
                                dwd_stns_evt)[idxs_neighbours_dwd]

                            if len(ids_neighbours_evt) > 0:
                                ppt_netatmo_ngbrs = netatmo_data_evt.loc[:,
                                                                         ids_neighbours_evt]
                                ppt_netatmo_data = ppt_netatmo_ngbrs.values
                                xstn_ngbr = netatmo_in_coords_df.loc[
                                    ppt_netatmo_ngbrs.columns, 'X'].values.ravel()
                                ystn_ngbr = netatmo_in_coords_df.loc[
                                    ppt_netatmo_ngbrs.columns, 'Y'].values.ravel()

                                if ppt_netatmo_data.size == 0:
                                    ppt_netatmo_data = 1000
                            else:
                                ppt_netatmo_data = 1000

                            if len(ids_neighbours_dwd_evt) > 0:
                                ppt_dwd_ngbrs = ppt_dwd_vals_evt.loc[:,
                                                                     ids_neighbours_dwd_evt]
                                ppt_dwd_data = ppt_dwd_ngbrs.values
                                dwd_xstn_ngbr = dwd_in_coords_df.loc[
                                    ppt_dwd_ngbrs.columns, 'X'].values.ravel()
                                dwd_ystn_ngbr = dwd_in_coords_df.loc[
                                    ppt_dwd_ngbrs.columns, 'Y'].values.ravel()

#                                 plt.ioff()
#                                 plt.scatter(xstn_bd, ystn_bd, c='r')
#                                 plt.scatter(xstn_ngbr, ystn_ngbr, c='b')
#                                 plt.scatter(dwd_xstn_ngbr,
#                                             dwd_ystn_ngbr, c='g')
#                                 plt.show()
                                if ppt_dwd_ngbrs.size == 0:
                                    ppt_dwd_ngbrs = 1000
                            else:
                                ppt_dwd_ngbrs = 1000  # always wrong
                            try:
                                if (ppt_bad > np.nanmin(ppt_netatmo_data) or
                                        ppt_bad > np.nanmin(ppt_dwd_data)):
                                    # print('added bad to good\n')
                                    # print('ppt_bad', ppt_bad)
                                    ids_netatmo_stns_gd_final = np.append(
                                        ids_netatmo_stns_gd, stn_bad)
                                    idx_good_stns_final = np.append(
                                        idx_good_stns, stn_ix)

                                    ids_netatmo_stns_bad_final = np.setdiff1d(
                                        ids_netatmo_stns_bad, stn_bad)
                                    ids_netatmo_stns_bad_final.size
                                    idx_bad_stns_final = np.setdiff1d(
                                        idx_bad_stns, stn_ix)

                                    assert stn_bad in ids_netatmo_stns_gd_final
                                    assert stn_bad not in ids_netatmo_stns_bad_final
                                else:
                                    pass
                                    # print('not added bad stn')

                                idx_good_stns_final = np.sort(
                                    idx_good_stns_final)
                                idx_bad_stns_final = np.sort(
                                    idx_bad_stns_final)
                            except Exception as msg:
                                print('error', msg)
                                continue
                                # raise Exception

                else:
                    ids_netatmo_stns_gd_final = ids_netatmo_stns_gd
                    idx_good_stns_final = idx_good_stns
                    ids_netatmo_stns_bad_final = ids_netatmo_stns_bad
                    idx_bad_stns_final = idx_bad_stns

                try:
                    print('Number of Stations with bad index \n',
                          len(idx_bad_stns_final))
                    print('Number of Stations with good index \n',
                          len(idx_good_stns_final))

                    # save results gd+1, bad -1
                    df_save_results.loc[
                        date_to_correct, ids_netatmo_stns_gd_final] = 1
                    df_save_results.loc[
                        date_to_correct, ids_netatmo_stns_bad_final] = -1
                except Exception as msg3:
                    print(msg3)
                    continue

                try:
                    zvalues_good = netatmo_data_evt.loc[date_to_correct,
                                                        ids_netatmo_stns_gd_final].values.ravel()
                    zvalues_bad = netatmo_data_evt.loc[date_to_correct,
                                                       ids_netatmo_stns_bad_final].values.ravel()

                    # plot configuration
                    max_ppt = max(np.nanmax(zvalues_good),
                                  np.nanmax(ppt_dwd_vals_evt.values.ravel()))
                    if max_ppt >= 30:
                        print('plotting map')
                        plot_good_bad_stns(netatmo_in_coords_df,
                                           ids_netatmo_stns_gd_final,
                                           ids_netatmo_stns_bad_final,
                                           zvalues_good,
                                           ppt_dwd_vals_evt.values.ravel(),
                                           zvalues_bad,
                                           dwd_xcoords, dwd_ycoords,
                                           date_to_correct)
                        plt.close()
                except Exception as msg2:
                    print('error plotting ', msg2)

    df_save_results.dropna(how='all', inplace=True)

    return df_save_results
#     df_save_results.to_csv(
#         os.path.join(out_save_dir, 'netatmos_flagged_%s.csv' % _year),
#         sep=';')


# START HERE
if __name__ == '__main__':

    args = (
        path_to_ppt_netatmo_data_hdf5,
        path_to_ppt_dwd_data_hdf5)

    process_manager(args)

    pass
