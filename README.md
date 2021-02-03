# pypws

## How to cite:
Bárdossy, A., Seidel, J., and El Hachem, A.: The use of personal weather station observation for improving precipitation estimation and interpolation, Hydrol. Earth Syst. Sci. Discuss. [preprint], https://doi.org/10.5194/hess-2020-42, in review, 2020. 

### Process Pearsonal Weather Station (PWS) data

The number of personal weather stations
(PWSs) with data available through the internet is
increasing gradually in many parts of the world. The purpose
of this study is to investigate the applicability of these
data for the spatial interpolation of precipitation using a novel
approach based on indicator correlations and rank statistics.
Due to unknown errors and biases of the observations, rainfall
amounts from the PWS network are not considered directly.
Instead, it is assumed that the temporal order of the
ranking of these data is correct. The crucial step is to find the
stations which fulfil this condition. This is done in two steps
– first, by selecting the locations using the time series of indicators
of high precipitation amounts. Then, the remaining
stations are then checked for whether they fit into the spa15
tial pattern of the other stations. Thus, it is assumed that the
quantiles of the empirical distribution functions are accurate.
These quantiles are then transformed to precipitation
amounts by a quantile mapping using the distribution functions
which were interpolated from the information from the
German NationalWeather Service (DeutscherWetterdienst –
DWD) data only. The suggested procedure was tested for the
state of Baden-Württemberg in Germany. A detailed cross
validation of the interpolation was carried out for aggregated
precipitation amount of 1, 3, 6, 12 and 24 h. For each of these
temporal aggregations, nearly 200 intense events were evaluated,
and the improvement of the interpolation was quantified.
The results show that the filtering of observations from
PWSs is necessary as the interpolation error after the filtering
and data transformation decreases significantly. The biggest
improvement is achieved for the shortest temporal aggregations.


**Flowchart from raw PWS data to filtered data for interpolation**
![flowchart_netatmo_paper](https://user-images.githubusercontent.com/22959071/106765543-3303fb00-6639-11eb-92d8-d0e06a6044f1.png)

#### Indicator Based Filtering (IBF) ###
![indic_corr](https://user-images.githubusercontent.com/22959071/106766133-d6eda680-6639-11eb-8dab-9a6b000752f5.png)

#### Bias Correction ###


**EBF**
