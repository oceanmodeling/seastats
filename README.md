[![Available on pypi](https://img.shields.io/pypi/v/seastats.svg)](https://pypi.python.org/pypi/seastats/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/seastats.svg)](https://anaconda.org/conda-forge/seastats)
[![CI](https://github.com/oceanmodeling/seastats/actions/workflows/run_tests.yml/badge.svg)](https://github.com/oceanmodeling/seastats/actions/workflows/run_tests.yml)

# SeaStats

`seastats` is a simple package to compare and analyse 2 time series. We use the following convention in this repo:
 * `sim`: modelled surge time series
 * `mod`: observed surge time series

The main function is:

```python
def get_stats(
    sim: Series,
    obs: Series,
    metrics: Sequence[str] = SUGGESTED_METRICS,
    quantile: float = 0,
    cluster: int = 24,
    round: int = -1
) -> dict[str, float]
```
Which calculates various statistical metrics between the simulated and observed time series data.

## Install

### PiPy
```
pip install seastats
```

### Conda / Mamba
```
mamba install -c conda-forge seastats
```

## Parameters:
 * **sim** (pd.Series). The simulated time series data.
 * **obs** (pd.Series). The observed time series data.
 * **metrics** (list[str]). (Optional) The list of statistical metrics to calculate. If metrics = ["all"], all items in `SUPPORTED_METRICS` will be calculated. Default is all items in `SUGGESTED_METRICS`.
 * **quantile** (float). (Optional) Quantile used to calculate the metrics. Default is `0` (no selection)
 * **cluster** (int). (Optional) Cluster duration for grouping storm events. Default is `72` hours.
 * **round** (int). (Optional) Apply rounding to the results to. Default is no rounding (value is `-1`)

Returns a dictionary containing the calculated metrics and their corresponding values. With 2 types of metrics:
* [The "general" metrics](#general-metrics): All the basic metrics needed for signal comparison (RMSE, RMS, Correlation etc..). See details below
  * `bias`: Bias
  * `rmse`: Root Mean Square Error
  * `rms`: Root Mean Square
  * `rms_95`: Root Mean Square for data points above 95th percentile
  * `sim_mean`: Mean of simulated values
  * `obs_mean`: Mean of observed values
  * `sim_std`: Standard deviation of simulated values
  * `obs_std`: Standard deviation of observed values
  * `mae`: Mean Absolute Error
  * `mse`: Mean Square Error
  * `nse`: Nash-Sutcliffe Efficiency
  * `lambda`: Lambda index
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
* [The storm metrics](#storm-metrics): a PoT selection is done on the observed signal (using the `match_extremes()` function). Function returns the decreasing extreme event peak values for observed and modeled signals (and time lag between events). See details below.
  * `R1`: Difference between observed and modelled for the biggest storm
  * `R1_norm`: Normalized R1 (R1 divided by observed value)
  * `R3`: Average difference between observed and modelled for the three biggest storms
  * `R3_norm`: Normalized R3 (R3 divided by observed value)
  * `error`: Average difference between observed and modelled for all storms
  * `error_norm`: Normalized error (error divided by observed value)

## General metrics
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

## Storm metrics
The functions uses the `match_extremes()` function (detailed below) and returns:
 * `R1`: the error for the biggest storm
 * `R3`: the mean error for the 3 biggest storms
 * `error`: the mean error for all the storms above the threshold.
 * `R1_norm`/`R3_norm`/`error`: Same methodology, but values are in normalised (in %) relatively to the observed peaks.


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
## Extreme events

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


this happens when the function `storms/match_extremes.py` couldn't find concomitent storms for the observed and modeled time series.

## Usage
see [notebook](/notebooks/example_abed.ipynb) for details

get all metrics in a 3 liner:
```python
from seastats import get_stats, GENERAL_METRICS_ALL, STORM_METRICS_ALL
general = get_stats(sim, obs, metrics = GENERAL_METRICS)
storm = get_stats(sim, obs, quantile = 0.99, metrics = STORM_METRICS) # we use a different quantile for PoT selection
pd.DataFrame(dict(general, **storm), index=['abed'])
```

|      |   bias |   rmse |   rms |   rms_95 |   sim_mean |   obs_mean |   sim_std |   obs_std |   nse |  lambda |    cr |   cr_95 |   slope |   intercept |   slope_pp |   intercept_pp |   mad |   madp |   madc |   kge |       R1 |   R1_norm |       R3 |   R3_norm |     error |   error_norm |
|:-----|-------:|-------:|------:|---------:|-----------:|-----------:|----------:|----------:|------:|--------:|------:|--------:|--------:|------------:|-----------:|---------------:|------:|-------:|-------:|------:|---------:|----------:|---------:|----------:|----------:|-------------:|
| abed | -0.007 |  0.086 | 0.086 |    0.088 |         -0 |      0.007 |     0.142 |     0.144 | 0.677 |   0.929 | 0.817 |   0.542 |   0.718 |      -0.005 |      1.401 |         -0.028 | 0.052 |  0.213 |  0.265 |  0.81 | 0.237364 |  0.295719 | 0.147163 |  0.207019 | 0.0938142 |     0.177533 |
