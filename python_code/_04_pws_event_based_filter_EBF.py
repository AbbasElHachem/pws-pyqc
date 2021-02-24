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
from win32comext.shell.demos.servers.folder_view import IDS_5ORGREATER

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

# from spinterps import OrdinaryKriging as OKpy
from pykrige.ok import OrdinaryKriging as OKpy

from _01_read_hdf5 import HDF5

pyximport.install()

gc.set_threshold(0, 0, 0)
# =============================================================================

# this is used to keep only data where month is not in this list
not_convective_season = [11, 12, 1, 2, 3]  # oct till april

vg_sill_b4_scale = 0.07
vg_range = 4e4
vg_model_str = 'spherical'
vg_model_to_scale = '0.07 Sph(40000)'

_year = '2019'

start_date = '%s-04-01 00:00:00' % _year
end_date = '%s-10-31 23:00:00' % _year

n_workers = 7
# =============================================================================

# need to input the corrected data after the bias correction,
# this is just a test data
path_to_ppt_pws_data_hdf5 = (
    r"X:\staff\elhachem\GitHub\pws-pyqc\test_data\pws_test_data.h5")

assert os.path.exists(path_to_ppt_pws_data_hdf5), 'wrong pws file'

path_to_ppt_prim_netw_data_hdf5 = (
    r"X:\staff\elhachem\GitHub\pws-pyqc\test_data\primary_network_test_data.h5")
assert os.path.exists(path_to_ppt_prim_netw_data_hdf5), 'wrong prim_netw file'

path_to_filtered_pws = (
    r"X:\staff\elhachem\GitHub\pws-pyqc\test_results\remaining_pws.csv")

# def out save directory
out_save_dir = (
    r"X:\staff\elhachem\GitHub\pws-pyqc\test_results")
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)

#==============================================================================
#
#==============================================================================


def process_manager(args):

    (path_to_neatmo_ppt_hdf5,
        path_to_prim_netw_ppt_hdf5,
        path_to_filtered_pws) = args

    #=========================================================
    HDF5_pws_ppt = HDF5(infile=path_to_neatmo_ppt_hdf5)
    all_pws_ids = HDF5_pws_ppt.get_all_names()
    pws_coords = HDF5_pws_ppt.get_coordinates(all_pws_ids)
    pws_in_coords_df = pd.DataFrame(
        index=all_pws_ids,
        data=pws_coords['easting'], columns=['X'])
    y_pws_coords = pws_coords['northing']
    pws_in_coords_df.loc[:, 'Y'] = y_pws_coords
    pws_in_coords_df.dropna(how='all', inplace=True)
    assert pws_in_coords_df.isna().sum().sum() == 0
    #=========================================================
    HDF5_prim_netw_ppt = HDF5(infile=path_to_prim_netw_ppt_hdf5)
    all_prim_netw_stns_ids = HDF5_prim_netw_ppt.get_all_names()

    prim_netw_coords = HDF5_prim_netw_ppt.get_coordinates(
        all_prim_netw_stns_ids)
    prim_netw_in_coords_df = pd.DataFrame(
        index=all_prim_netw_stns_ids,
        data=prim_netw_coords['easting'], columns=['X'])
    y_prim_netw_coords = prim_netw_coords['northing']
    prim_netw_in_coords_df.loc[:, 'Y'] = y_prim_netw_coords
    prim_netw_in_coords_df.dropna(how='all', inplace=True)
    assert prim_netw_in_coords_df.isna().sum().sum() == 0
    #=========================================================
    # select on 'good' pws
    ids_pws_to_use = pd.read_csv(
        path_to_filtered_pws, index_col=0).index.to_list()
    #=========================================================
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    date_range_summer = pd.DatetimeIndex([date_ for date_ in date_range
                                          if date_.month not in not_convective_season])

    print('Using Workers: ', n_workers)
    # devide stations on workers
    all_timestamps_worker = np.array_split(date_range_summer, n_workers)
    args_worker = []

    for time_list in all_timestamps_worker:
        empty_data = np.zeros(shape=(len(time_list),
                                     len(all_pws_ids)))
        empty_data[empty_data == 0] = np.nan
        df_save_results = pd.DataFrame(index=time_list,
                                       columns=all_pws_ids, data=empty_data)
    # args_workers = list(repeat(args, n_worker))

        args_worker.append((path_to_prim_netw_ppt_hdf5,
                            prim_netw_in_coords_df,
                            path_to_neatmo_ppt_hdf5,
                            pws_in_coords_df,
                            ids_pws_to_use,
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
                     'pws_flagged_%s.csv' % (_year)),
        sep=';')

    return
#==============================================================================
# MAIN FUNCTION
#==============================================================================


def on_evt_filter_pws(args):

    (path_to_prim_netw_ppt_hdf5,
     prim_netw_in_coords_df,
     path_to_neatmo_ppt_hdf5,
     pws_in_coords_df,
     ids_pws_to_use,
     time_list,
     df_save_results) = args

    HDF5_pws_ppt = HDF5(infile=path_to_neatmo_ppt_hdf5)
#     all_pws_ids = HDF5_pws_ppt.get_all_names()
    all_pws_ids_to_use = ids_pws_to_use

    HDF5_prim_netw_ppt = HDF5(infile=path_to_prim_netw_ppt_hdf5)
    all_prim_netw_ids = HDF5_prim_netw_ppt.get_all_names()

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

    def plot_good_bad_stns(pws_in_coords_df,
                           ids_pws_stns_gd,
                           ids_pws_stns_bad,
                           zvalues,
                           prim_netw_ppt,
                           zvalues_bad,
                           prim_netwx, prim_netwy,
                           event_date):

        xstns_good = pws_in_coords_df.loc[
            ids_pws_stns_gd, 'X'].values.ravel()
        ystns_good = pws_in_coords_df.loc[
            ids_pws_stns_gd, 'Y'].values.ravel()

        xstns_bad = pws_in_coords_df.loc[
            ids_pws_stns_bad, 'X'].values.ravel()
        ystns_bad = pws_in_coords_df.loc[
            ids_pws_stns_bad, 'Y'].values.ravel()
        max_ppt = max(np.nanmax(zvalues), np.nanmax(prim_netw_ppt))

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
        plt.scatter(prim_netwx, prim_netwy, c=prim_netw_ppt, cmap=cmap_ppt,
                    marker=',', s=10, alpha=0.75, vmin=0, vmax=max_ppt,
                    label='prim_netw %d' % prim_netwx.size)

        sc = plt.scatter(xstns_good, ystns_good, c=zvalues,
                         cmap=cmap_ppt,
                         marker='.', s=10, alpha=0.75, vmin=0, vmax=max_ppt,
                         label='PWS Good %d' % xstns_good.size)

        plt.scatter(xstns_bad, ystns_bad,
                    alpha=0.75, c=zvalues_bad, cmap=cmap_ppt_bad,
                    marker='X', s=20, vmin=0, vmax=max_ppt,
                    label='PWS Bad %d' % xstns_bad.size)
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
        plt.savefig(os.path.join(
            out_save_dir,
            'event_date_%s.png'
            % (str(event_date
                   ).replace('-', '_').replace(':', '_'))))
        plt.close()

    #==========================================================================
    #
    #==========================================================================

    for ix, date_to_correct in enumerate(time_list):

        print(ix, '/', len(time_list), '--', date_to_correct)
        # pws_data = pd.read_feather(path_netatamo_edf_fk, columns=pws_ids_str)
        pws_data_evt = HDF5_pws_ppt.get_pandas_dataframe_for_date(
            ids=all_pws_ids_to_use,
            event_date=date_to_correct).dropna(how='all', axis=1)

        if len(pws_data_evt.columns) > 0:
            pws_stns_evt = pws_data_evt.columns.to_list()
            cmn_pws_event = pws_in_coords_df.index.intersection(
                pws_stns_evt)
            # coords of stns to correct
            xstns_interp = pws_in_coords_df.loc[
                cmn_pws_event, 'X'].values.ravel()
            ystns_interp = pws_in_coords_df.loc[
                cmn_pws_event, 'Y'].values.ravel()
            cmn_pws_data_evt = pws_data_evt.loc[:, cmn_pws_event]
            # prim_netw data
            ppt_prim_netw_vals_evt = (
                HDF5_prim_netw_ppt.get_pandas_dataframe_for_date(
                    ids=all_prim_netw_ids, event_date=date_to_correct))
            prim_netw_stns_evt = ppt_prim_netw_vals_evt.columns.to_list()

            cmn_prim_netw_stns_evt = prim_netw_in_coords_df.index.intersection(
                prim_netw_stns_evt)

            prim_netw_xcoords = prim_netw_in_coords_df.loc[
                cmn_prim_netw_stns_evt, 'X'].values.ravel()
            prim_netw_ycoords = prim_netw_in_coords_df.loc[
                cmn_prim_netw_stns_evt, 'Y'].values.ravel()
            # primary network values for event
            cmn_ppt_prim_netw_vals_evt = ppt_prim_netw_vals_evt.loc[
                :, cmn_prim_netw_stns_evt]
            # scale variogram
            vgs_model_dwd_ppt = scale_vg_based_on_prim_netw_ppt(
                ppt_prim_netw_vals_evt.values, vg_sill_b4_scale)

            # start kriging pws location
#             OK_prim_netw_pws_crt = OKpy(xi=prim_netw_xcoords,
#                                         yi=prim_netw_ycoords,
#                                         zi=cmn_ppt_prim_netw_vals_evt.values.ravel(),
#                                         xk=xstns_interp,
#                                         yk=ystns_interp,
#                                         model=vgs_model_dwd_ppt)
            # using PYkrige
            dwd_vals_var = np.var(cmn_ppt_prim_netw_vals_evt.values)
            vg_scaling_ratio = dwd_vals_var / vg_sill_b4_scale
#
            if vg_scaling_ratio == 0:
                vg_scaling_ratio = vg_sill_b4_scale
            OK_prim_netw_pws_crt = OKpy(
                prim_netw_xcoords, prim_netw_ycoords,
                cmn_ppt_prim_netw_vals_evt.values.ravel(),
                variogram_model=vg_model_str,
                variogram_parameters={
                    'sill': vg_scaling_ratio,
                    'range': vg_range,
                    'nugget': 0})
            try:
                #                 OK_prim_netw_pws_crt.krige()
                #                 zvalues = OK_prim_netw_pws_crt.zk.copy()
                #
                #                 # calcualte standard deviation of estimated values
                #                 std_est_vals = np.sqrt(OK_prim_netw_pws_crt.est_vars)

                zvalues, est_var = OK_prim_netw_pws_crt.execute(
                    'points', np.array([xstns_interp]),  np.array([ystns_interp]))
                std_est_vals = np.sqrt(est_var).data
            except Exception as msg:
                print('ror', msg)

            # if neg assign 0
            zvalues[zvalues < 0] = 0
            # calcualte standard deviation of estimated values

            # calculate difference observed and estimated
            # values
            diff_obsv_interp = np.abs(cmn_pws_data_evt.values - zvalues)

            idx_good_stns = np.where(
                diff_obsv_interp <= 3 * std_est_vals)[1]
            idx_bad_stns = np.where(
                diff_obsv_interp > 3 * std_est_vals)[1]

            if len(idx_bad_stns) > 0:

                # use additional filter
                try:
                    ids_pws_stns_gd = np.take(cmn_pws_event,
                                              idx_good_stns).ravel()
                    ids_pws_stns_bad = np.take(cmn_pws_event,
                                               idx_bad_stns).ravel()

                except Exception as msg:
                    print(msg)
                # ids of bad stns
                xstns_bad = pws_in_coords_df.loc[
                    ids_pws_stns_bad, 'X'].values.ravel()
                ystns_bad = pws_in_coords_df.loc[
                    ids_pws_stns_bad, 'Y'].values.ravel()
#                 # check if bad are truly bad
                xstns_good = pws_in_coords_df.loc[
                    ids_pws_stns_gd, 'X'].values.ravel()
                ystns_good = pws_in_coords_df.loc[
                    ids_pws_stns_gd, 'Y'].values.ravel()

                # coords of neighbors good
                neighbors_coords = np.array(
                    [(x, y) for x, y in zip(xstns_good, ystns_good)])

                # create a tree from coordinates
                points_tree = spatial.cKDTree(neighbors_coords)

                neighbors_coords_prim_netw = np.array(
                    [(x, y) for x, y in zip(prim_netw_xcoords, prim_netw_ycoords)])

                points_tree_prim_netw = spatial.cKDTree(
                    neighbors_coords_prim_netw)

#                 plt.ioff()
#                 plt.scatter(xstns_good, ystns_good, c='b')
#                 plt.scatter(prim_netw_xcoords, prim_netw_ycoords, c='g')
#                 plt.scatter(xstns_bad, ystns_bad, c='r')
#                 plt.show()
                if len(idx_bad_stns) > 0 > 0:
                    for stn_ix, stn_bad in zip(idx_bad_stns, ids_pws_stns_bad):
                        ppt_bad = cmn_pws_data_evt.loc[:, stn_bad].values
                        # print('ppt_bad', ppt_bad)
                        if ppt_bad >= 0.:

                            xstn_bd = pws_in_coords_df.loc[
                                stn_bad, 'X']
                            ystn_bd = pws_in_coords_df.loc[
                                stn_bad, 'Y']

                            idxs_neighbours = points_tree.query_ball_point(
                                np.array((xstn_bd, ystn_bd)), 1e4)

                            ids_neighbours = ids_pws_stns_gd[idxs_neighbours]
                            ids_neighbours_evt = np.in1d(
                                cmn_pws_event, ids_neighbours)

                            idxs_neighbours_prim_netw = points_tree_prim_netw.query_ball_point(
                                np.array((xstn_bd, ystn_bd)), 1e4)

                            ids_neighbours_prim_netw_evt = np.array(
                                cmn_prim_netw_stns_evt)[idxs_neighbours_prim_netw]

                            if len(ids_neighbours_evt) > 0:
                                ppt_pws_ngbrs = cmn_pws_data_evt.loc[:,
                                                                     ids_neighbours_evt]
                                ppt_pws_data = ppt_pws_ngbrs.values
                                #---------------------------------------------
                                # xstn_ngbr = pws_in_coords_df.loc[
                                #     ppt_pws_ngbrs.columns, 'X'].values.ravel()
                                # ystn_ngbr = pws_in_coords_df.loc[
                                #     ppt_pws_ngbrs.columns, 'Y'].values.ravel()
                                #---------------------------------------------

                                if ppt_pws_data.size == 0:
                                    ppt_pws_data = 1000
                            else:
                                ppt_pws_data = 1000

                            if len(ids_neighbours_prim_netw_evt) > 0:
                                ppt_prim_netw_ngbrs = cmn_ppt_prim_netw_vals_evt.loc[:,
                                                                                     ids_neighbours_prim_netw_evt]
                                ppt_prim_netw_data = ppt_prim_netw_ngbrs.values
                                #---------------------------------------------
                                # prim_netw_xstn_ngbr = prim_netw_in_coords_df.loc[
                                #     ppt_prim_netw_ngbrs.columns, 'X'].values.ravel()
                                # prim_netw_ystn_ngbr = prim_netw_in_coords_df.loc[
                                #     ppt_prim_netw_ngbrs.columns, 'Y'].values.ravel()
                                #---------------------------------------------

#                                 plt.ioff()
#                                 plt.scatter(xstn_bd, ystn_bd, c='r')
#                                 plt.scatter(xstn_ngbr, ystn_ngbr, c='b')
#                                 plt.scatter(prim_netw_xstn_ngbr,
#                                             prim_netw_ystn_ngbr, c='g')
#                                 plt.show()
                                if ppt_prim_netw_ngbrs.size == 0:
                                    ppt_prim_netw_ngbrs = 1000
                            else:
                                ppt_prim_netw_ngbrs = 1000  # always wrong
                            try:
                                if (ppt_bad > np.nanmin(ppt_pws_data) or
                                        ppt_bad > np.nanmin(ppt_prim_netw_data)):
                                    # print('added bad to good\n')
                                    # print('ppt_bad', ppt_bad)
                                    ids_pws_stns_gd_final = np.append(
                                        ids_pws_stns_gd, stn_bad)
                                    idx_good_stns_final = np.append(
                                        idx_good_stns, stn_ix)

                                    ids_pws_stns_bad_final = np.setdiff1d(
                                        ids_pws_stns_bad, stn_bad)
                                    ids_pws_stns_bad_final.size
                                    idx_bad_stns_final = np.setdiff1d(
                                        idx_bad_stns, stn_ix)

                                    assert stn_bad in ids_pws_stns_gd_final
                                    assert stn_bad not in ids_pws_stns_bad_final
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
                    ids_pws_stns_gd_final = ids_pws_stns_gd
                    idx_good_stns_final = idx_good_stns
                    ids_pws_stns_bad_final = ids_pws_stns_bad
                    idx_bad_stns_final = idx_bad_stns

                try:
                    print('Number of Stations with bad index \n',
                          len(idx_bad_stns_final))
                    print('Number of Stations with good index \n',
                          len(idx_good_stns_final))

                    # save results gd+1, bad -1
                    df_save_results.loc[
                        date_to_correct, ids_pws_stns_gd_final] = 1
                    df_save_results.loc[
                        date_to_correct, ids_pws_stns_bad_final] = -1
                except Exception as msg3:
                    print(msg3)
                    continue

                try:
                    zvalues_good = cmn_pws_data_evt.loc[date_to_correct,
                                                        ids_pws_stns_gd_final].values.ravel()
                    zvalues_bad = cmn_pws_data_evt.loc[date_to_correct,
                                                       ids_pws_stns_bad_final].values.ravel()

                    # plot configuration
                    max_ppt = max(np.nanmax(zvalues_good),
                                  np.nanmax(cmn_ppt_prim_netw_vals_evt.values.ravel()))
                    if max_ppt >= 0:
                        print('plotting map')
                        plot_good_bad_stns(pws_in_coords_df,
                                           ids_pws_stns_gd_final,
                                           ids_pws_stns_bad_final,
                                           zvalues_good,
                                           cmn_ppt_prim_netw_vals_evt.values.ravel(),
                                           zvalues_bad,
                                           prim_netw_xcoords, prim_netw_ycoords,
                                           date_to_correct)
                        plt.close()
                except Exception as msg2:
                    print('error plotting ', msg2)
#         break
    df_save_results.dropna(how='all', inplace=True)

    return df_save_results
#     df_save_results.to_csv(
#         os.path.join(out_save_dir, 'pwss_flagged_%s.csv' % _year),
#         sep=';')


# START HERE
if __name__ == '__main__':

    args = (
        path_to_ppt_pws_data_hdf5,
        path_to_ppt_prim_netw_data_hdf5,
        path_to_filtered_pws)

    process_manager(args)

    pass
