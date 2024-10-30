# SeaStats

`seastats` is a simple package to compare and analyse 2 time series. We use the following convention in this repo:
 * `sim`: modelled surge time series
 * `mod`: observed surge time series

The main functions are:
* `get_stats()`: to get [general comparison metrics](#general-metrics) between two time series
* `match_extremes()`: a PoT selection is done on the observed signal. Function returns the decreasing extreme event peak values for observed and modeled signals (and time lag between events). See details [below](#extreme-events)
* `storm_metrics()`: functions returns storm metrics. See details [below](#storm-metrics)

# General metrics

```python
stats = get_stats(sim: pd.Series, obs: pd.Series)
```
returns the following dictionary:

```
{
  'bias': 0.01,
  'rmse': 0.105,
  'rms': 0.106,
  'rms_95': 0.095,
  'sim_mean': 0.01,
  'obs_mean': -0.0,
  'sim_std': 0.162,
  'obs_std': 0.142,
  'nse': 0.862,
  'lamba': 0.899,
  'cr': 0.763,
  'cr_95': 0.489,
  'slope': 0.215,
  'intercept': 0.01,
  'slope_pp': 0.381,
  'intercept_pp': 0.012,
  'mad': 0.062,
  'madp': 0.207,
  'madc': 0.269,
  'kge': 0.71
}
```

with:
* `bias`: Bias
* `rmse`: Root Mean Square Error
* `rms`: Root Mean Square
* `rms_95`: Root Mean Square for data points above 95th percentile
* `sim_mean`: Mean of simulated values
* `obs_mean`: Mean of observed values
* `sim_std`: Standard deviation of simulated values
* `obs_std`: Standard deviation of observed values
* `nse`: Nash-Sutcliffe Efficiency
* `lamba`: Lambda index
* `cr`: Pearson Correlation coefficient
* `cr_95`: Pearson Correlation coefficient for data points above 95th percentile
* `slope`: Slope of Model/Obs correlation
* `intercept`: Intercept of Model/Obs correlation
* `slope_pp`: Slope of Model/Obs correlation of percentiles
* `intercept_pp`: Intercept of Model/Obs correlation of percentiles
* `mad`: Mean Absolute Deviation
* `madp`: Mean Absolute Deviation of percentiles
* `madc`: `mad + madp`
* `kge`: Kling–Gupta Efficiency

Most of the paremeters are detailed below:

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

# Extreme events

Example of implementation:
```python
from seastats.storms import match_extremes
extremes_df = match_extremes(sim, obs, 0.99, cluster = 72)
extremes_df
```
The modeled peaks are matched with the observed peaks. Function returns a pd.DataFrame of the decreasing observed storm peaks as follows:

| time observed       |   observed | time observed       |    model | time model          |       diff |     error |   error_norm |   tdiff |
|:--------------------|-----------:|:--------------------|---------:|:--------------------|-----------:|----------:|-------------:|--------:|
| 2022-01-29 19:30:00 |   0.803 | 2022-01-29 19:30:00 | 0.565 | 2022-01-29 17:00:00 | -0.237  | 0.237  |    0.296  |    -2.5 |
| 2022-02-20 20:30:00 |   0.639 | 2022-02-20 20:30:00 | 0.577 | 2022-02-20 20:00:00 | -0.062 | 0.062 |    0.0963 |    -0.5 |
...
| 2022-11-27 15:30:00 |   0.386  | 2022-11-27 15:30:00 | 0.400 | 2022-11-27 17:00:00 |  0.014 | 0.014 |    0.036 |     1.5 |

with:
 * `diff` the difference between modeled and observed peaks
 * `error` the absolute difference between modeled and observed peaks
 * `tdiff` the time difference between modeled and observed peaks

NB: the function uses [pyextremes](https://georgebv.github.io/pyextremes/quickstart/) in the background, with PoT method, using the `quantile` value of the observed signal as physical threshold and passes the `cluster_duration` argument.

# Storm metrics
The functions uses the above `match_extremes()` and returns:
 * `R1`: the error for the biggest storm
 * `R3`: the mean error for the 3 biggest storms
 * `error`: the mean error for all the storms above the threshold.
 * `R1_norm`/`R3_norm`/`error`: Same methodology, but values are in normalised (in %) relatively to the observed peaks.

Example of implementation:
```python
from seastats.storms import storm_metrics
storm_metrics(sim: pd.Series, obs: pd.Series, quantile: float, cluster_duration:int = 72)
```
returns this dictionary:
```python
{'R1': 0.237,
 'R1_norm': 0.296,
 'R3': 0.147,
 'R3_norm': 0.207,
 'error': 0.0938,
 'error_norm': 0.178}
```

### case of NaNs
The `storm_metrics()` might return:
```python
{'R1': np.nan,
 'R1_norm': np.nan,
 'R3': np.nan,
 'R3_norm': np.nan,
 'error': np.nan,
 'error_norm': np.nan}
```

this happens when the function `storms/match_extremes.py` couldn't finc concomitent storms for the observed and modeled time series.

## Usage
see [notebook](/notebooks/example_abed.ipynb) for details

get all metrics in a 3 liner:
```python
stats = get_stats(sim, obs)
metrics = storm_metrics(sim, obs, quantile=0.99, cluster=72)
pd.DataFrame(dict(stats, **metrics), index=['abed'])
```

|      |   bias |   rmse |   rms |   rms_95 |   sim_mean |   obs_mean |   sim_std |   obs_std |   nse |   lamba |    cr |   cr_95 |   slope |   intercept |   slope_pp |   intercept_pp |   mad |   madp |   madc |   kge |       R1 |   R1_norm |       R3 |   R3_norm |     error |   error_norm |
|:-----|-------:|-------:|------:|---------:|-----------:|-----------:|----------:|----------:|------:|--------:|------:|--------:|--------:|------------:|-----------:|---------------:|------:|-------:|-------:|------:|---------:|----------:|---------:|----------:|----------:|-------------:|
| abed | -0.007 |  0.086 | 0.086 |    0.088 |         -0 |      0.007 |     0.142 |     0.144 | 0.677 |   0.929 | 0.817 |   0.542 |   0.718 |      -0.005 |      1.401 |         -0.028 | 0.052 |  0.213 |  0.265 |  0.81 | 0.237364 |  0.295719 | 0.147163 |  0.207019 | 0.0938142 |     0.177533 |
