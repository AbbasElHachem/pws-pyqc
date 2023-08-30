#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:  filter and correct unreliable/reliable PWS

How to cite:
https://doi.org/10.5281/zenodo.4501920

Reference paper:

Bárdossy, A., Seidel, J., and El Hachem, A.: 
The use of personal weather station observations to improve precipitation estimation and interpolation,
Hydrol. Earth Syst. Sci., 25, 583–601, https://doi.org/10.5194/hess-25-583-2021, 2021.

"""

__author__ = "Abbas El Hachem", "Micha Eisele", "Jochen Seidel", "Andras Bardossy"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

#from pathlib import Path

import os
import pyproj
import tqdm
import fnmatch
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

from pyproj import Transformer

from pykrige import OrdinaryKriging as OKpy
from sklearn.metrics import mean_squared_error as rmse


#==============================================================================


def list_all_full_path(ext, file_dir):
    """
    Purpose: To return full path of files in all dirs of a given folder with a
    -------  given extension in ascending order.

    Keyword arguments:
    ------------------
        ext (string) = Extension of the files to list
            e.g. '.txt', '.tif'.
        file_dir (string) = Full path of the folder in which the files
            reside.
    """
    new_list = []
    patt = '*' + ext
    for root, _, files in os.walk(file_dir):
        for elm in files:
            if fnmatch.fnmatch(elm, patt):
                full_path = os.path.join(root, elm)
                new_list.append(full_path)
    return(sorted(new_list))


# In[3]:


def read_pcp_csv_file(path_to_file, sep_type, index_col):
    '''
    Read csv file and return df data
    '''
    
    in_df = pd.read_csv(path_to_file, sep=sep_type,
                        index_col=index_col,
                       encoding='latin-1')
    if index_col is not None:
        in_df.index = pd.to_datetime(in_df.index,
                                 format='%Y-%m-%d %H:%M:%S')
    return in_df



# In[4]:


def read_metadata_csv_file(path_to_file, sep_type, index_col):
    '''
    Read metadata csv file and return df coords also in utm
    '''
    
    df_coords = pd.read_csv(path_to_file, sep=sep_type,
                        index_col=index_col,
                       encoding='latin-1')
    
    lon_vals = df_coords.loc[:, 'lon'].values.ravel()
    lat_vals = df_coords.loc[:, 'lat'].values.ravel()
    
    # convert to utm32 for distance calculation
    x_vals, y_vals = LatLon_To_XY(lon_vals, lat_vals)

    stn_ids = [str(ii) for ii in range(len(lon_vals))]
    # make a df and combine all info
    df_coords_utm32 = pd.DataFrame(
            index=stn_ids,
            data=lon_vals, columns=['lon'])
    df_coords_utm32['lat'] = lat_vals
    
    df_coords_utm32['X'] = x_vals
    df_coords_utm32['Y'] = y_vals
    
    coords_xy = np.array([(x, y) for x, y in zip(
        df_coords_utm32.loc[:, 'X'].values,
        df_coords_utm32.loc[:, 'Y'].values)])

    # create a tree from coordinates
    # coords_points_tree = cKDTree(coords_xy)
    
    return df_coords_utm32, coords_xy


# In[5]:


def LatLon_To_XY(i_area, j_area):
    
    ''' convert coordinates from wgs84 to utm 32'''
    P = pyproj.Proj(proj='utm', zone=32,
                    ellps='WGS84',
                    preserve_units=True)

    x, y = P.transform(i_area, j_area)
    # G = pyproj.Geod(ellps='WGS84')

    #xy_area = np.array(
    #[(i, j)
    #for i, j in zip(x, y)])

    return x, y


def XY_to_latlon(i_area, j_area):
    ''' convert coordinates from utm32 to wgs84'''
    transformer = Transformer.from_crs(
        "EPSG:25832",
       "EPSG:4326",
        always_xy=True)
    x, y = transformer.transform(i_area, j_area)
    # G = pyproj.Geod(ellps='WGS84')

    xy_area = np.array(
        [(i, j)
         for i, j in zip(x, y)])

    return xy_area


# In[6]:

def calc_prs_spr_corr(var_A, var_B):
        prs_obsv_raw = st.pearsonr(var_A, var_B)[0]
        spr_obsv_raw = st.spearmanr(var_A, var_B)[0]
        rmse_obsv_raw = rmse(var_A,var_B, squared=True)
        return prs_obsv_raw, spr_obsv_raw, rmse_obsv_raw
    
if __name__ == '__main__':
    # read input data
    
    # settings
    
    
    main_dir = r"https://raw.githubusercontent.com/AbbasElHachem/pws-pyqc/main/OpenSense/Data/"
    # =============================================================================
    # AWS data from Netherlands
    path_primary_network = os.path.join(main_dir, r"AWS_stns_data.csv")
    path_primary_metadata = os.path.join(main_dir, r"AWS_stns_coords.csv")
      
    # 20 radar grid cell data from Amsterdam
    path_primary_network2 = os.path.join(
        main_dir, r"Radar_grid_cell_vals.csv")
    path_primary_metadata2 = os.path.join(
        main_dir, r"selected_radar_grid_lonlat.csv") 

    path_pws_data_raw = (
        r"X:\staff\elhachem\2022_02_01_OpenSense\results_Lotte\AmsterdamPWSdataset_hourly_201605010100_201806010000.csv")  
    
    # PWS hourly data

    path_pws_data = r"X:\staff\elhachem\2022_02_01_OpenSense\interpolation\pws_hourly_on_event_10outof12.csv"
    #os.path.join(main_dir, r"pws_bias_corr.csv")
    path_pws_metadata = os.path.join(main_dir, r"AMS_metadata.csv")
    
    # interpolation grid
    path_interp_grid_AMS = os.path.join(
        main_dir, r"interpolation_grid_AMS.csv") 
    
    date_range_all = pd.date_range(start='2016-05-01 01:00:00',
                                    end='2018-06-01 00:00:00', freq='H')
    # read primary network 1
    print('Reading first primary network data')
    in_primary_pcp = read_pcp_csv_file(path_to_file=path_primary_network,
                           sep_type=';',
                           index_col=0)

    df_prim_coords, prim_coords_xy = read_metadata_csv_file(
        path_primary_metadata,
                                                           sep_type=';', index_col=0,)
    df_prim_coords.index = in_primary_pcp.columns
    # read primary network 2
    print('Reading secondary primary network data')
    in_primary_pcp_2 = read_pcp_csv_file(
        path_to_file=path_primary_network2,
                           sep_type=';',
                           index_col=0)
    
    
    df_prim_coords_2, prim_coords_xy_2 = read_metadata_csv_file(
            path_primary_metadata2,
            sep_type=',', index_col=0,)
    df_prim_coords_2.index = in_primary_pcp_2.columns
    
    in_primary_pcp_2[in_primary_pcp_2 < 0] = np.nan
    in_primary_pcp_2.dropna(how='all', inplace=True, axis=1)
    
    # read pws data
    print('Reading PWS data')
    
    df_pws_pcp_hourly_raw = read_pcp_csv_file(
        path_to_file=path_pws_data_raw,  sep_type=',',  index_col=None)
    df_pws_pcp_hourly_raw.index = date_range_all
    df_pws_pcp_hourly_raw[df_pws_pcp_hourly_raw == -9] = np.nan
    df_pws_pcp_hourly_raw.dropna(how='all', axis=0, inplace=True)

    df_pws_pcp_hourly = read_pcp_csv_file(
        path_to_file=path_pws_data,
                           sep_type=';',
                           index_col=0)
    
    df_pws_pcp_hourly = df_pws_pcp_hourly.shift(1)
    df_pws_pcp_hourly[df_pws_pcp_hourly == -9] = np.nan
    df_pws_pcp_hourly.dropna(how='all', axis=0, inplace=True)
    
    pws_ids_list = [int(_id.split('s')[1]) 
                    for _id in df_pws_pcp_hourly.columns]
    
    
    df_pws_coords, pws_coords_xy = read_metadata_csv_file(
        path_to_file=path_pws_metadata,
                           sep_type=',',
                           index_col=0)
    df_pws_coords.loc[:,'idx'] = range(1, len(df_pws_coords.index)+1, 1)
    df_pws_coords.set_index('idx', inplace=True)
    
    df_pws_coords.index = df_pws_pcp_hourly_raw.columns
    # only filtered ids
    
    
    
    #df_pws_coords_indic_flt = df_pws_coords.loc[df_pws_coords.index.intersection(pws_ids_list),:]
    #df_pws_coords_indic_flt.index = df_pws_pcp_hourly.columns

    path_interp_holland = pd.read_csv(
    #r"https://raw.githubusercontent.com/overeem11/RAINLINK/master/InterpolationGrid.dat",
        path_interp_grid_AMS,
        index_col=0,
        sep=',')
    
    path_data_intense = pd.read_csv(
        r"X:\staff\elhachem\2022_02_01_OpenSense\data_Roberto"
        r"\all_Intense_output.csv",
        index_col=0,
        sep=',', parse_dates=True)
    
    path_data_intense = path_data_intense.shift(1)
    
    df_lotte = pd.read_csv(
        r"X:\staff\elhachem\2022_02_01_OpenSense\data_Lotte"
        r"\AmsterdamPWSdataset_hourly_afterPWSQC_201605010100_201806010000.csv",
        sep=',', index_col=None, parse_dates=True)
    
    df_lotte.index = date_range_all
    
    df_lotte = df_lotte.loc['2016-06-01 01:00:00':,:]
    # stn in amsterdam
    id_stn_prim_ams = '240'
    
    
    
    corrds_cross_val_stns = pd.read_csv(
        r"X:\staff\elhachem\2022_02_01_OpenSense\data_Netherland_PWS"
        r"\Radar_052018\05\selected_radar_grid_lonlat_crossval.csv",
        index_col=0, sep=',')
    
    xobsv, yobsv = LatLon_To_XY(corrds_cross_val_stns.x_lon.values.ravel(),
                            corrds_cross_val_stns.y_lat.values.ravel())
    
    df_obsc_cross_val = pd.read_csv(
    r"X:\staff\elhachem\2022_02_01_OpenSense\data_Netherland_PWS\Radar_052018\05\grid_cell_vals_cross_val.csv",
    index_col=0, parse_dates=True, sep=';')

    xstn = df_prim_coords.loc[id_stn_prim_ams,:].X
    ystn = df_prim_coords.loc[id_stn_prim_ams,:].Y
    
    temp_agg_list = ['1440min']
    
    data_locs = os.chdir(r"X:\staff\elhachem\2022_02_01_OpenSense\interpolation")
    all_csv_fles = glob.glob('*.csv')
    
    for temp_agg in temp_agg_list:
        
        pws_raw_file = [_f for _f in all_csv_fles if temp_agg in _f
                        and 'pws_raw' in _f][0]
                             
        pws_pwsqc_file = [_f for _f in all_csv_fles if temp_agg in _f
                        and 'pws_pwsqc' in _f][0]    
        
        pws_pwspyqc_file = [_f for _f in all_csv_fles if temp_agg in _f
                        and 'pws_pwspyqc' in _f][0]      
        
        pws_intense_file = [_f for _f in all_csv_fles if temp_agg in _f
                        and 'pws_intense' in _f][0]      
        
        pws_radar_file = [_f for _f in all_csv_fles if temp_agg in _f
                        and 'pws_radar' in _f][0]
                        
        obsv_vals_res_file = [_f for _f in all_csv_fles if temp_agg in _f
                        and 'obsv_res' in _f][0]
                          
        df_pws_raw = pd.read_csv(pws_raw_file, index_col=0, sep=',',
                                 parse_dates=True)
        
        df_pws_pwsqc = pd.read_csv(pws_pwsqc_file, index_col=0, sep=',',
                                 parse_dates=True)
        
        df_pws_pwspyqc= pd.read_csv(pws_pwspyqc_file, index_col=0, sep=',',
                                 parse_dates=True)
        
        df_pws_intense = pd.read_csv(pws_intense_file, index_col=0, sep=',',
                                 parse_dates=True)
        
        df_pws_radar = pd.read_csv(pws_radar_file, index_col=0, sep=',',
                                 parse_dates=True)
        
        df_obsv_res = pd.read_csv(obsv_vals_res_file, index_col=0, sep=',',
                                 parse_dates=True)
        
        df_obsv_res = df_obsv_res.shift(1)
        cmn_idx_final = df_pws_raw.dropna(
            axis=0, how='all').index.intersection(df_pws_pwspyqc.dropna(
            axis=0, how='all').index).intersection(df_pws_pwsqc.dropna(
            axis=0, how='all').index).intersection(df_pws_intense.dropna(
            axis=0, how='all').index).intersection(df_pws_radar.dropna(
            axis=0, how='all').index).intersection(df_obsv_res.index)
            
        df_pws_raw = df_pws_raw.loc[cmn_idx_final, :]
        df_pws_pwsqc = df_pws_pwsqc.loc[cmn_idx_final, :]
        df_pws_pwspyqc = df_pws_pwspyqc.loc[cmn_idx_final, :]
        df_pws_intense = df_pws_intense.loc[cmn_idx_final, :]
        df_pws_radar = df_pws_radar.loc[cmn_idx_final, :]
        df_obsv_res = df_obsv_res.loc[cmn_idx_final, :]
        pcp_thr_list = [0, 1, 5]
        
        dfs_list = [df_pws_raw, df_pws_pwsqc, df_pws_pwspyqc, df_pws_intense]
        df_cmn = pd.DataFrame(columns=['PWS-Raw','PWSQC', 'PWS-pyQC', 'GSDR-QC', 'Avg. Nbr. Val.', 'False zero', 'False positive'],
                              index=pcp_thr_list)
        
        for pcp_thr in pcp_thr_list:
            
            for il, _df in enumerate(dfs_list):
                avg_stns = []
                avg_vals = []
                obsv_pos_pws_zer = []
                obsv_zero_pws_pos = []
                col_name = df_cmn.columns[il]
                for _i in df_obsv_res.columns.astype(int):
                    if pcp_thr == 0:
                        cmn_vals = np.intersect1d(np.where(df_obsv_res.iloc[:,_i].values == pcp_thr)[0],
                               np.where(_df.iloc[:,_i].values == pcp_thr)[0])
                        cmn_vals_abv_below = np.intersect1d(np.where(df_obsv_res.iloc[:,_i].values > pcp_thr)[0],
                               np.where(_df.iloc[:,_i].values == pcp_thr)[0])
                        cmn_vals_below_abv = np.intersect1d(np.where(df_obsv_res.iloc[:,_i].values == pcp_thr)[0],
                               np.where(_df.iloc[:,_i].values > pcp_thr)[0])
                        vals_shape=np.where(df_obsv_res.iloc[:,_i].values == pcp_thr)[0].shape[0]
                        pcp_thr = 0
                        obsv_pos_pws_zer.append(100*cmn_vals_abv_below.shape[0]/np.where(df_obsv_res.iloc[:,_i].values > pcp_thr)[0].shape[0])
                        obsv_zero_pws_pos.append(100*cmn_vals_below_abv.shape[0]/np.where(df_obsv_res.iloc[:,_i].values == pcp_thr)[0].shape[0])
                    
                    else:
                        cmn_vals = np.intersect1d(np.where(df_obsv_res.iloc[:,_i].values >= pcp_thr)[0],
                               np.where(_df.iloc[:,_i].values >= pcp_thr)[0])
                        vals_shape =np.where(df_obsv_res.iloc[:,_i].values >= pcp_thr)[0].shape[0]
                    
                    avg_stns.append(100*cmn_vals.shape[0]/vals_shape)
                    avg_vals.append(vals_shape)
                    
                    
                avg_val_df = np.mean(avg_stns)
                avg_vals_df = np.mean(avg_vals)
                avg_vals_false_zero = np.mean(obsv_pos_pws_zer)
                avg_vals_false_pos = np.mean(obsv_zero_pws_pos)
                df_cmn.loc[pcp_thr, col_name] =  avg_val_df
                
                # df_cmn.loc[pcp_thr, 'False zero'] =      avg_vals_false_zero
                # df_cmn.loc[pcp_thr, 'False pos'] =      avg_vals_false_pos
                df_cmn.loc[pcp_thr, 'Avg. Nbr. Val.'] =    avg_vals_df
                df_cmn.dropna(how='all', axis=1)
                      
            np.where(df_obsv_res.iloc[:,_i] == 0)
            break
        pass
        
    #
    #     
    #
    #     obsv_vals = obsv_res.loc[cmn_idx_final, :].dropna()#.shift(1)       
    #     obsv_vals_shift = obsv_vals.shift(0).dropna()
    #
    #     cmn_idx_final = df_pws_raw.dropna(
    #         axis=0, how='all').index.intersection(df_pws_pwspyqc.dropna(
    #         axis=0, how='all').index).intersection(df_pws_pwsqc.dropna(
    #         axis=0, how='all').index).intersection(df_pws_intense.dropna(
    #         axis=0, how='all').index).intersection(df_pws_radar.dropna(
    #         axis=0, how='all').index).intersection(obsv_vals_shift.index)
    #
    #     prs_obsv_raw, spr_obsv_raw, rmse_obsv_raw = calc_prs_spr_corr(
    #         obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(),                    df_pws_raw.loc[cmn_idx_final, :].values.ravel())
    #
    #     prs_obsv_pwspyqc, spr_obsv_pwspyqc, rmse_obsv_pwspyqc = calc_prs_spr_corr(
    #         obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(),                    df_pws_pwspyqc.loc[cmn_idx_final, :].values.ravel())
    #
    #     prs_obsv_pwsqc, spr_obsv_pwsqc, rmse_obsv_pwsqc = calc_prs_spr_corr(
    #         obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(),                    df_pws_pwsqc.loc[cmn_idx_final, :].values.ravel())
    #
    #     prs_obsv_intense, spr_obsv_intense, rmse_obsv_intense = calc_prs_spr_corr(
    #         obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(),                    df_pws_intense.loc[cmn_idx_final, :].values.ravel())
    #
    #
    #     prs_obsv_radar, spr_obsv_radar, rmse_obsv_radar = calc_prs_spr_corr(
    #         obsv_vals.loc[cmn_idx_final, :].values.ravel(),  df_pws_radar.loc[cmn_idx_final, :].values.ravel())
    #
    #     df_tables_metric = pd.DataFrame(columns=['Raw-PWS', 'PWSQC', 'PWS-pyQC', 'GSDR-QC', 'Radar'],
    #                                     index=['prs', 'spr', 'rmse'])
    #     df_tables_metric.loc['prs', :] = [prs_obsv_raw, prs_obsv_pwsqc, prs_obsv_pwspyqc, prs_obsv_intense, prs_obsv_radar]
    #     df_tables_metric.loc['spr', :] = [spr_obsv_raw, spr_obsv_pwsqc, spr_obsv_pwspyqc, spr_obsv_intense, spr_obsv_radar]
    #     df_tables_metric.loc['rmse', :] = [rmse_obsv_raw, rmse_obsv_pwsqc, rmse_obsv_pwspyqc, rmse_obsv_intense, rmse_obsv_radar]
    #     df_tables_metric.to_csv(r'X:\staff\elhachem\2022_02_01_OpenSense\interpolation'
    #                 r'\%s_cross_val.csv' % temp_agg, sep=',')
    #     plt.ioff()
    #     plt.figure(figsize=(6, 6), dpi=300)
    #
    #
    #     #df_pws_raw
    #     plt.scatter(obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(),     df_pws_raw.loc[cmn_idx_final, :].values.ravel(),
    #                  c='orange', marker='2', label=r'$PWS-Raw-\rho=%0.2f$' % prs_obsv_raw, alpha=0.5)
    #
    #
    #     plt.scatter(obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(), df_pws_pwsqc.loc[cmn_idx_final, :].values.ravel(),
    #                  c='r', marker='o', label=r'$PWSQC-\rho=%0.2f$' % prs_obsv_pwsqc, alpha=0.5)
    #
    #     plt.scatter(obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(), df_pws_pwspyqc.loc[cmn_idx_final, :].values.ravel(),
    #                  alpha=0.5, c='b', marker=',',label=r'$PWS-pyQC-\rho=%0.2f$' % prs_obsv_pwspyqc)
    #
    #     plt.scatter(obsv_vals_shift.loc[cmn_idx_final, :].values.ravel(),  df_pws_intense.loc[cmn_idx_final, :].values.ravel(),
    #                    alpha=0.5,  c='lime',  marker='d', label=r'$GSDR-QC-\rho=%0.2f$' % prs_obsv_intense)
    #
    #     plt.scatter(obsv_vals.loc[cmn_idx_final, :].values.ravel(),  df_pws_radar.loc[cmn_idx_final, :].values.ravel(),
    #                    alpha=0.5, c='k',  marker='3', label=r'$Radar-\rho=%0.2f$' % prs_obsv_radar)
    #
    #     # plt.scatter(obsv_vals.loc[cmn_idx_final, :].values.ravel(),
    #                 # obsv_vals.loc[cmn_idx_final, :].values.ravel(), c='gray', alpha=0.25,marker='.')
    #     max_obsv =obsv_vals.values.max()
    #     plt.ylim([-1,max_obsv+2])
    #     plt.xlim([-1, max_obsv+2])
    #     plt.grid(alpha=0.5)
    #
    #     plt.xlabel('Observed precipitation -  [mm/%s]' % temp_agg)
    #     plt.ylabel('Interpoated precipitation - [mm/%s]' % temp_agg)
    #
    #     plt.legend(loc=0)
    #     plt.tight_layout()
    #     plt.savefig(r'X:\staff\elhachem\2022_02_01_OpenSense\interpolation'
    #                 r'\%s_pcp_b4_cross2.png' % temp_agg, bbox_inches='tight')
    #     plt.close()
    #
    #
    # #===========================================================================
    # # 
    # #===========================================================================
    #
    #     min_pcp_thr_list = [0]#, 1, 5, 10]
    #
    #     df_mean_corr = pd.DataFrame(index=min_pcp_thr_list,
    #                                 columns=['Raw-PWS', 'PWS-pyQC', 'PWSQC', 'GSDR-QC', 'Radar'])
    #     # In[410]:
    #     for min_pcp_thr in min_pcp_thr_list:
    #         print(min_pcp_thr)
    #         df_spr_corr = pd.DataFrame(index=obsv_vals.columns,
    #                                columns=['Raw-PWS', 'PWS-pyQC',
    #                                          'PWSQC', 'GSDR-QC', 'Radar'],
    #                                data=-9)
    #         df_pears_corr = pd.DataFrame(index=obsv_vals.columns,
    #                                     columns=['Raw-PWS', 'PWS-pyQC',
    #                                               'PWSQC', 'GSDR-QC', 'Radar'],
    #                                     data=-9)
    #         df_indic_corr = pd.DataFrame(index=obsv_vals.columns,
    #                                    columns=['Raw-PWS', 'PWS-pyQC',
    #                                              'PWSQC', 'GSDR-QC', 'Radar'],
    #                                    data=-9)
    #
    #
    #         df_rmse = pd.DataFrame(index=obsv_vals.columns,
    #                                     columns=['Raw-PWS', 'PWS-pyQC',
    #                                               'PWSQC', 'GSDR-QC', 'Radar'],
    #                                     data=-9)
    #
    #         for _loc in obsv_vals.columns:
    #             obsv_vals_loc = obsv_vals_shift.loc[:, _loc].dropna()
    #             obsv_val_loc_orig = obsv_vals.loc[:, _loc].dropna()
    #             obsv_vals_loc = obsv_vals_loc[obsv_vals_loc >= min_pcp_thr].dropna()
    #             interp_pws_raw = df_pws_raw.loc[:, _loc].dropna()
    #             interp_pws_pwspyqc = df_pws_pwspyqc.loc[:, _loc].dropna()
    #             interp_pws_pwsqc = df_pws_pwsqc.loc[:, _loc].dropna()
    #             interp_pws_intense = df_pws_intense.loc[:, _loc].dropna()
    #             interp_pws_radar = df_pws_radar.loc[:, _loc].dropna()
    #
    #             cmn_idx_all = obsv_vals_loc.index.intersection(
    #             interp_pws_raw.index).intersection(interp_pws_pwspyqc.index).intersection(
    #             interp_pws_pwsqc.index).intersection(interp_pws_intense.index).intersection(
    #                 interp_pws_radar.index)
    #
    #             prs_obsv_raw, spr_obsv_raw, rms_raw = calc_prs_spr_corr(
    #                 obsv_vals_loc.loc[cmn_idx_all].values.ravel(),
    #                     interp_pws_raw.loc[cmn_idx_all].values.ravel())
    #
    #             prs_obsv_pwspyqc, spr_obsv_pwspyqc, rms_pwspyqc = calc_prs_spr_corr(
    #                 obsv_vals_loc.loc[cmn_idx_all].values.ravel(),
    #                 interp_pws_pwspyqc.loc[cmn_idx_all].values.ravel())
    #
    #             prs_obsv_pwsqc, spr_obsv_pwsqc, rms_pwsqc = calc_prs_spr_corr(
    #                 obsv_vals_loc.loc[cmn_idx_all].values.ravel(),
    #                 interp_pws_pwsqc.loc[cmn_idx_all].values.ravel())
    #
    #             prs_obsv_intense, spr_obsv_intense, rms_intense = calc_prs_spr_corr(
    #                 obsv_vals_loc.loc[cmn_idx_all].values.ravel(),
    #                     interp_pws_intense.loc[cmn_idx_all].values.ravel())
    #
    #             prs_obsv_radar, spr_obsv_radar, rms_radar = calc_prs_spr_corr(
    #                 obsv_val_loc_orig.loc[cmn_idx_all].values.ravel(),
    #                 interp_pws_radar.loc[cmn_idx_all].values.ravel())
    #
    #
    #             df_pears_corr.loc[_loc,:] = [prs_obsv_raw, prs_obsv_pwspyqc, 
    #                                     prs_obsv_pwsqc, prs_obsv_intense, prs_obsv_radar]
    #
    #             df_spr_corr.loc[_loc,:] = [spr_obsv_raw, spr_obsv_pwspyqc,
    #                                         spr_obsv_pwsqc, spr_obsv_intense,
    #                                          spr_obsv_radar]
    #
    #             df_rmse.loc[_loc,:] = [rms_raw, rms_pwspyqc, 
    #                                     rms_pwsqc, rms_intense, rms_radar]
    #             #print(spr_obsv_raw, spr_obsv_pwspyqc, spr_obsv_pwsqc, spr_obsv_intense, spr_obsv_radar)
    #
    #             #print(df_spr_corr.head())
    #
    #         df_pears_corr.to_csv(r'X:\staff\elhachem\2022_02_01_OpenSense\interpolation'
    #                     r'\%s_cross_val_pears_corr_%d.csv' 
    #                     % (temp_agg, min_pcp_thr), sep=',')
    #         df_spr_corr.to_csv(r'X:\staff\elhachem\2022_02_01_OpenSense\interpolation'
    #                     r'\%s_cross_val_spr_corr_%d.csv' 
    #                     % (temp_agg,min_pcp_thr), sep=',')
    #         df_rmse.to_csv(r'X:\staff\elhachem\2022_02_01_OpenSense\interpolation'
    #                     r'\%s_cross_val_rmse_%d.csv' % (temp_agg, min_pcp_thr), sep=',')
    #
    #
    #
    #         df_mean_corr.loc[min_pcp_thr,:] = df_pears_corr.mean(axis=0)
    #         #===========================================================================
    #         # 
    #         #===========================================================================
    #
    #         # xlabels = ['Raw-PWS', 'PWSQC', 'PWS-pyQC', 'GSDR-QC', 'Radar']
    #         xlabels = ['Raw-PWS', 'PWSQC', 'PWS-pyQC', 'GSDR-QC']
    #         labels = [0, 1, 2, 3, 4]
    #         plt.ioff()
    #         # df_spr_corr
    #         plt.rcParams["figure.autolayout"] = True
    #         fig, (ax1) = plt.subplots(1, 1, figsize=(6,4), dpi=300)
    #
    #         ax1.set_xlabel('Datasets')
    #
    #
    #         # Create a legend for the first line.
    #
    #         data2 = [df_spr_corr.iloc[:,0],
    #                  df_spr_corr.iloc[:,2],
    #                 df_spr_corr.iloc[:,1],
    #                 df_spr_corr.iloc[:,3],
    #                 # df_spr_corr.iloc[:,4]
    #                 ]
    #
    #
    #         bp0=ax1.boxplot(data2,
    #            vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      #labels=['60 min'],
    #                         sym='o',
    #                 showfliers=True,
    #                                           notch=False,
    #                    zorder=1,
    #                        )
    #         for ii, box in enumerate(bp0['boxes']):
    #             if ii == 0:
    #                 box.set(color='orange', linewidth=1)
    #                 box.set(facecolor = 'yellow' )
    #             if ii == 1:
    #                 box.set(color='r', linewidth=1)
    #                 box.set(facecolor = 'darkred' )
    #             if ii == 2:
    #                 box.set(color='darkblue', linewidth=1)
    #                 box.set(facecolor = 'blue' )
    #             if ii == 3:
    #                 box.set(color='darkgreen', linewidth=1)
    #                 box.set(facecolor = 'lime' )
    #             if ii == 4:
    #                 box.set(color='k', linewidth=1)
    #                 box.set(facecolor = 'gray' )
    #
    #         #second_legend = plt.legend([bp0["boxes"][0]], [ 'RCP8.5 Raw'], loc='upper left')
    #
    #         ax1.set_xticks(ticks=range(1, len(xlabels)+1), labels=xlabels)
    #
    #         ax1.set_ylim([0., 1])
    #         ax1.grid(alpha=0.25)
    #         #ax1.legend(loc='lower right')
    #         ax1.set_ylabel('Spearman Correlation')
    #
    #
    #         plt.savefig(r'X:\staff\elhachem\2022_02_01_OpenSense'
    #                     r'\data_Netherland_PWS\split_sampling'               
    #                     r'\spr_corr_abv_%d_%s_nord.png' % (min_pcp_thr, temp_agg),
    #                     bbox_inches='tight',transparent=False)
    #
    #
    #         #===========================================================================
    #         # 
    #         #===========================================================================
    #
    #         labels = [0, 1, 2, 3, 4]
    #
    #         # df_spr_corr
    #         plt.ioff()
    #         plt.rcParams["figure.autolayout"] = True
    #         fig, (ax1) = plt.subplots(1, 1, figsize=(6,4), dpi=300)
    #
    #         ax1.set_xlabel('Datasets')
    #
    #
    #         # Create a legend for the first line.
    #
    #         data2 = [df_rmse.iloc[:,0],
    #                  df_rmse.iloc[:,2],
    #                 df_rmse.iloc[:,1],
    #                 df_rmse.iloc[:,3],
    #                 # df_rmse.iloc[:,4]
    #                 ]
    #
    #
    #         bp0=ax1.boxplot(data2,
    #            vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      #labels=['60 min'],
    #                         sym='o',
    #                 showfliers=True,
    #                                           notch=False,
    #                    zorder=1,
    #                        )
    #         for ii, box in enumerate(bp0['boxes']):
    #             if ii == 0:
    #                 box.set(color='orange', linewidth=1)
    #                 box.set(facecolor = 'yellow' )
    #             if ii == 1:
    #                 box.set(color='r', linewidth=1)
    #                 box.set(facecolor = 'darkred' )
    #             if ii == 2:
    #                 box.set(color='darkblue', linewidth=1)
    #                 box.set(facecolor = 'blue' )
    #             if ii == 3:
    #                 box.set(color='darkgreen', linewidth=1)
    #                 box.set(facecolor = 'lime' )
    #             if ii == 4:
    #                 box.set(color='k', linewidth=1)
    #                 box.set(facecolor = 'gray' )
    #
    #         #second_legend = plt.legend([bp0["boxes"][0]], [ 'RCP8.5 Raw'], loc='upper left')
    #
    #         ax1.set_xticks(ticks=range(1, len(xlabels)+1), labels=xlabels)
    #
    #         ax1.set_ylim([-5, 5])
    #         ax1.grid(alpha=0.25)
    #         #ax1.legend(loc='lower right')
    #         ax1.set_ylabel('RMSE')
    #
    #
    #         plt.savefig(r'X:\staff\elhachem\2022_02_01_OpenSense'
    #                     r'\data_Netherland_PWS\split_sampling'               
    #                     r'\rmse_abv_%d_%s_nord.png' % (min_pcp_thr, temp_agg),
    #                     bbox_inches='tight',transparent=False)
    #
    #
    #         #===================================================================
    #         # 
    #         #===================================================================
    #
    #         # df_spr_corr
    #         plt.rcParams["figure.autolayout"] = True
    #         fig, (ax1) = plt.subplots(1, 1, figsize=(6,4), dpi=300)
    #
    #         ax1.set_xlabel('Datasets')
    #
    #
    #         # Create a legend for the first line.
    #
    #         data2 = [df_pears_corr.iloc[:,0],
    #                  df_pears_corr.iloc[:,2],
    #                 df_pears_corr.iloc[:,1],
    #                 df_pears_corr.iloc[:,3],
    #                 # df_pears_corr.iloc[:,4]
    #                 ]
    #
    #
    #         bp0=ax1.boxplot(data2,
    #            vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      #labels=['60 min'],
    #                         sym='o',
    #                 showfliers=True,
    #                                           notch=False,
    #                    zorder=1,
    #                        )
    #         for ii, box in enumerate(bp0['boxes']):
    #             if ii == 0:
    #                 box.set(color='orange', linewidth=1)
    #                 box.set(facecolor = 'yellow' )
    #             if ii == 1:
    #                 box.set(color='r', linewidth=1)
    #                 box.set(facecolor = 'darkred' )
    #             if ii == 2:
    #                 box.set(color='darkblue', linewidth=1)
    #                 box.set(facecolor = 'blue' )
    #             if ii == 3:
    #                 box.set(color='darkgreen', linewidth=1)
    #                 box.set(facecolor = 'lime' )
    #             if ii == 4:
    #                 box.set(color='k', linewidth=1)
    #                 box.set(facecolor = 'gray' )
    #
    #         #second_legend = plt.legend([bp0["boxes"][0]], [ 'RCP8.5 Raw'], loc='upper left')
    #
    #         ax1.set_xticks(ticks=range(1, len(xlabels)+1), labels=xlabels)
    #
    #         ax1.set_ylim([0., 1])
    #         ax1.grid(alpha=0.25)
    #         #ax1.legend(loc='lower right')
    #         ax1.set_ylabel('Pearson Correlation')
    #
    #
    #         plt.savefig(r'X:\staff\elhachem\2022_02_01_OpenSense'
    #                     r'\data_Netherland_PWS\split_sampling'               
    #                     r'\prs_corr_abv_%d_%snord.png' % (min_pcp_thr, temp_agg),
    #                     bbox_inches='tight',transparent=False)
    #
    #     plt.ioff()
    #     plt.figure(figsize=(6, 4), dpi=300)
    #     for _min_thr in df_mean_corr.index:
    #
    #         plt.plot(range(len(df_mean_corr.columns)),                     df_mean_corr.loc[_min_thr,:].values,
    #                  label='Pcp thr =%d mm' % _min_thr)
    #
    #     plt.xticks(ticks=range(0, len(xlabels)), labels=xlabels)
    #     plt.grid(alpha=0.5)
    #
    #     plt.xlabel('Datasets')
    #
    #     plt.ylabel('Mean Pearson Correaltion')
    #
    #     plt.legend(loc=0)
    #     plt.tight_layout()
    #     plt.savefig(r'X:\staff\elhachem\2022_02_01_OpenSense\interpolation'
    #                 r'\%s_mean_corr.png' % temp_agg, bbox_inches='tight')
    #     plt.close()
    #
    #
