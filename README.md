# pypws
Process Pearsonal Weather Station (PWS) data

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
