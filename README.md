# SeaStats

Simple package to compare and anylyse 2 time series. We use the following conventionn in this repo:
 * `sim`: modelled surge time series
 * `mod`: observed surge time series

## Metrics available:

These are following metrics available in this repository:

We need metrics to assess the quality of the model.
We define the most important ones, as stated on this [Stats wiki](https://cirpwiki.info/wiki/Statistics):
### A. Dimensional Statistics:
#### Mean Error (or Bias)
$$\langle x_c - x_m \rangle = \langle x_c \rangle - \langle x_m \rangle$$
#### RMSE (Root Mean Squared Error)
$$\sqrt{\langle(x_c - x_m)^2\rangle}$$
#### Mean-Absolute Error (MAE):
$$\langle |x_c - x_m| \rangle$$
### B. Dimentionless Statistics (best closer to 1)

#### Performance Scores (PS) or Nash-Sutcliffe Eff (NSE): $$1 - \frac{\langle (x_c - x_m)^2 \rangle}{\langle (x_m - x_R)^2 \rangle}$$
#### Correlation Coefficient (R):
$$\frac {\langle x_{m}x_{c}\rangle -\langle x_{m}\rangle \langle x_{c}\rangle }{{\sqrt {\langle x_{m}^{2}\rangle -\langle x_{m}\rangle ^{2}}}{\sqrt {\langle x_{c}^{2}\rangle -\langle x_{c}\rangle ^{2}}}}$$
#### Kling–Gupta Efficiency (KGE):
$$1 - \sqrt{(r-1)^2 + b^2 + (g-1)^2}$$
with :
 * `r` the correlation
 * `b` the modified bias term (see [ref](https://journals.ametsoc.org/view/journals/clim/34/16/JCLI-D-21-0067.1.xml)) $$\frac{\langle x_c \rangle - \langle x_m \rangle}{\sigma_m}$$
 * `g` the std dev term $$\frac{\sigma_c}{\sigma_m}$$

#### Lambda index ($\lambda$), values closer to 1 indicate better agreement:
$$\lambda = 1 - \frac{\sum{(x_c - x_m)^2}}{\sum{(x_m - \overline{x}_m)^2} + \sum{(x_c - \overline{x}_c)^2} + n(\overline{x}_m - \overline{x}_c)^2 + \kappa}$$
 * with `kappa` $$2 \cdot \left| \sum{((x_m - \overline{x}_m) \cdot (x_c - \overline{x}_c))} \right|$$

### Storm metrics:
The metrics are returned by the function
```python
storm_metrics(sim: pd.Series, obs: pd.Series, quantile: float, cluster_duration:int = 72)
```
which returns this dictionary:
```
"db_match" : val,
"R1_norm": val,
"R1": val,
"R3_norm": val,
"R3": val,
"error": val,
"error_metric": val
```
we defined the following metrics for the storms events:

* `R1`/`R3`/`error_metric`: we select the biggest observated storms, and then calculate error (so the absolute value of differenc between the model and the observed peaks)
  * `R1` is the error for the biggest storm
  * `R3` is the mean error for the 3 biggest storms
  * `error_metric` is the mean error for all the storms above the threshold.

* `R1_norm`/`R3_norm`/`error`: Same methodology, but values are in normalised (in %) by the observed peaks.

### case of NaNs
The `storm_metrics()` might return:
```
"db_match" : np.nan,
"R1_norm": np.nan,
"R1": np.nan,
"R3_norm": np.nan,
"R3": np.nan,
"error": np.nan,
"error_metric": np.nan
```

this happens when the function `storms/match_extremes.py` couldn't finc concomitent storms for the observed and modeled time series.

## Usage
