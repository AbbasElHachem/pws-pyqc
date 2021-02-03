# pws-pyqc
  Personal Weather Station (PWS) quality control python filter.
  
## How to cite:
Bárdossy, A., Seidel, J., and El Hachem, A.: The use of personal weather station observation for improving precipitation estimation and interpolation, Hydrol. Earth Syst. Sci. Discuss. [preprint], https://doi.org/10.5194/hess-2020-42, in review, 2020. 

-----------------------------------------------------------------------------------------------
### Main Procedure
------------------

**Flowchart from raw PWS data to filtered data for interpolation**

The three main codes corresponding to the IBF, Bias correction and EBF are available in the python_code folder

![flowchart_netatmo_paper](https://user-images.githubusercontent.com/22959071/106765543-3303fb00-6639-11eb-92d8-d0e06a6044f1.png)

-----------------------------------------------------------------------------------------------

diff
### Indicator based Filter IBF


**Corresponsing code**
_02_pws_indicator_correlation_IBF.py

****Required Input****
  1. Hdf5 data file containing the PWS station data, the corresponding timestamps and 
    their corresponding coordinates, in a metric coordinate system
  2. Hdf5 data file containing the primary network station data, the corresponding timestamps and
    their corresponding coordinates, in a metric coordinate system (same as PWS)
  
****Output****
  1. A dataframe containing mainly the correlation values between each PWS and the corresponding neighboring station data
    and the correlation between the neighboring primary network stations .
  2. The final result is obtained by keeping all the PWS where the correlation between PWS and 
    primary network is greater or equal to the correlation between the primary network stations.

-----------------------------------------------------------------------------------------------

### Bias correction

**Corresponsing code**
_02_pws_bias_correction_BC.py

****Required Input****
  1. Hdf5 data file containing the ***filtered*** PWS station data, the corresponding timestamps
    and their corresponding coordinates, in a metric coordinate system
  2. Hdf5 data file containing the primary network station data, the corresponding timestamps
    and their corresponding coordinates, in a metric coordinate system (same as PWS)
  
****Output****
  1. A dataframe for each PWS with the new data, a 'complete' new timeseries, used later on (for example in the interpolation)
 
-----------------------------------------------------------------------------------------------
### Event based filter (EBF)

**Corresponsing code**
_04_event_based_filter_EBF.py

****Required Input****
  1. Hdf5 data file containing the ***filtered and bias corrected*** PWS station data,
    the corresponding timestamps and their corresponding coordinates, in a metric coordinate system
  2. Hdf5 data file containing the primary network station data, the corresponding timestamps
    and their corresponding coordinates, in a metric coordinate system (same as PWS)
  
****Output****
  1. A dataframe containing for every event (or timestamp) the PWS that should be flagged and
    not used for the interpolation of the corresponding event or timestep
 

-----------------------------------------------------------------------------------------------
#### Examples
