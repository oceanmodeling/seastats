from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from seastats.stats import align_ts
from seastats.stats import get_bias
from seastats.stats import get_corr
from seastats.stats import get_kge
from seastats.stats import get_lambda
from seastats.stats import get_mad
from seastats.stats import get_madc
from seastats.stats import get_madp
from seastats.stats import get_mae
from seastats.stats import get_mse
from seastats.stats import get_nse
from seastats.stats import get_percentiles
from seastats.stats import get_rms
from seastats.stats import get_rmse
from seastats.stats import get_slope_intercept
from seastats.stats import get_slope_intercept_pp
from seastats.storms import match_extremes

__version__ = importlib.metadata.version(__name__)
__all__ = [
    "align_ts",
    "GENERAL_METRICS",
    "GENERAL_METRICS_ALL",
    "get_bias",
    "get_corr",
    "get_kge",
    "get_lambda",
    "get_mad",
    "get_madc",
    "get_madp",
    "get_mae",
    "get_mse",
    "get_nse",
    "get_percentiles",
    "get_rms",
    "get_rmse",
    "get_slope_intercept",
    "get_slope_intercept_pp",
    "get_stats",
    "match_extremes",
    "STORM_METRICS",
    "STORM_METRICS_ALL",
    "SUGGESTED_METRICS",
    "SUPPORTED_METRICS",
]

logger = logging.getLogger(__name__)

GENERAL_METRICS_ALL = [
    "bias",
    "rmse",
    "mae",
    "mse",
    "rms",
    "sim_mean",
    "obs_mean",
    "sim_std",
    "obs_std",
    "nse",
    "lambda",
    "cr",
    "slope",
    "intercept",
    "slope_pp",
    "intercept_pp",
    "mad",
    "madp",
    "madc",
    "kge",
]
GENERAL_METRICS = ["bias", "rms", "rmse", "cr", "nse", "kge"]
STORM_METRICS = ["R1", "R3", "error"]
STORM_METRICS_ALL = ["R1", "R1_norm", "R3", "R3_norm", "error", "error_norm"]

SUGGESTED_METRICS = sorted(GENERAL_METRICS + STORM_METRICS)
SUPPORTED_METRICS = sorted(GENERAL_METRICS_ALL + STORM_METRICS_ALL)


def get_stats(  # noqa: C901
    sim: pd.Series[float],
    obs: pd.Series[float],
    metrics: Sequence[str] = SUGGESTED_METRICS,
    quantile: float = 0,
    cluster: int = 72,
    round: int = -1,  # noqa: A002
) -> dict[str, float]:
    """
    Calculates various statistical metrics between the simulated and observed time series data.

    :param pd.Series sim: The simulated time series data.
    :param pd.Series obs: The observed time series data.
    :param list[str] metrics: (Optional) The list of statistical metrics to calculate. If metrics = ["all"], all items in SUPPORTED_METRICS will be calculated. Default is all items in SUGGESTED_METRICS.
    :param float quantile: (Optional) Quantile used to calculate the metrics. Default is 0 (no selection)
    :param int cluster: (Optional) Cluster duration for grouping storm events. Default is 72 hours.
    :param int round: (Optional) Apply rounding to the results to. Default is no rounding (value is -1)

    :return dict stats: dictionary containing the calculated metrics.

    The dictionary contains the following keys and their corresponding values:

    - `bias`: The bias between the simulated and observed time series data.
    - `rmse`: The Root Mean Square Error between the simulated and observed time series data.
    - `mae`: The Mean Absolute Error the simulated and observed time series data.
    - `mse`: The Mean Square Error the simulated and observed time series data.
    - `rms`: The raw mean square error between the simulated and observed time series data.
    - `sim_mean`: The mean of the simulated time series data.
    - `obs_mean`: The mean of the observed time series data.
    - `sim_std`: The standard deviation of the simulated time series data.
    - `obs_std`: The standard deviation of the observed time series data.
    - `nse`: The Nash-Sutcliffe efficiency between the simulated and observed time series data.
    - `lambda`: The lambda statistic between the simulated and observed time series data.
    - `cr`: The correlation coefficient between the simulated and observed time series data.
    - `slope`: The slope of the linear regression between the simulated and observed time series data.
    - `intercept`: The intercept of the linear regression between the simulated and observed time series data.
    - `slope_pp`: The slope of the linear regression between the percentiles of the simulated and observed time series data.
    - `intercept_pp`: The intercept of the linear regression between the percentiles of the simulated and observed time series data.
    - `mad`: The median absolute deviation of the simulated time series data from its median.
    - `madp`: The median absolute deviation of the simulated time series data from its median, calculated using the percentiles of the observed time series data.
    - `madc`: The median absolute deviation of the simulated time series data from its median, calculated by adding `mad` to `madp`
    - `kge`: The Kling-Gupta efficiency between the simulated and observed time series data.
    - `R1`: Difference between observed and modelled for the biggest storm
    - `R1_norm`: Normalized R1 (R1 divided by observed value)
    - `R3`: Average difference between observed and modelled for the three biggest storms
    - `R3_norm`: Normalized R3 (R3 divided by observed value)
    - `error`: Average difference between observed and modelled for all storms
    - `error_norm`: Normalized error (error divided by observed value)
    """
    if not isinstance(metrics, list):
        raise ValueError("metrics must be a list")

    if metrics == ["all"]:
        metrics = SUPPORTED_METRICS

    if not np.any([m in SUPPORTED_METRICS for m in metrics]):
        raise ValueError("metrics must be a list of supported variables in SUPPORTED_METRICS or ['all']")

    # Storm metrics part with PoT Selection
    if np.any([m in STORM_METRICS_ALL for m in metrics]):
        extreme_df = match_extremes(sim, obs, quantile=quantile, cluster=cluster)
    else:
        extreme_df = pd.DataFrame()  # Just to make mypy happy

    if quantile:  # signal subsetting is only to be done on general metrics
        sim = sim[sim > sim.quantile(quantile)]
        obs = obs[obs > obs.quantile(quantile)]

    stats = {}
    for metric in metrics:
        match metric:
            case "bias":
                stats["bias"] = get_bias(sim, obs)
            case "rmse":
                stats["rmse"] = get_rmse(sim, obs)
            case "rms":
                stats["rms"] = get_rms(sim, obs)
            case "sim_mean":
                stats["sim_mean"] = sim.mean()
            case "obs_mean":
                stats["obs_mean"] = obs.mean()
            case "sim_std":
                stats["sim_std"] = sim.std()
            case "obs_std":
                stats["obs_std"] = obs.std()
            case "mae":
                stats["mae"] = get_mae(sim, obs)
            case "mse":
                stats["mse"] = get_mse(sim, obs)
            case "nse":
                stats["nse"] = get_nse(sim, obs)
            case "lambda":
                stats["lambda"] = get_lambda(sim, obs)
            case "cr":
                stats["cr"] = get_corr(sim, obs)
            case "slope":
                stats["slope"], _ = get_slope_intercept(sim, obs)
            case "intercept":
                _, stats["intercept"] = get_slope_intercept(sim, obs)
            case "slope_pp":
                stats["slope_pp"], _ = get_slope_intercept_pp(sim, obs)
            case "intercept_pp":
                _, stats["intercept_pp"] = get_slope_intercept_pp(sim, obs)
            case "mad":
                stats["mad"] = get_mad(sim, obs)
            case "madp":
                stats["madp"] = get_madp(sim, obs)
            case "madc":
                stats["madc"] = get_madc(sim, obs)
            case "kge":
                stats["kge"] = get_kge(sim, obs)
            case "R1":
                stats["R1"] = extreme_df["error"].iloc[0]
            case "R1_norm":
                stats["R1_norm"] = extreme_df["error_norm"].iloc[0]
            case "R3":
                stats["R3"] = extreme_df["error"].iloc[0:3].mean()
            case "R3_norm":
                stats["R3_norm"] = extreme_df["error_norm"].iloc[0:3].mean()
            case "error":
                stats["error"] = extreme_df["error"].mean()
            case "error_norm":
                stats["error_norm"] = extreme_df["error_norm"].mean()

    if round > 0:
        for metric in metrics:
            stats[metric] = np.round(stats[metric], round)

    return stats
