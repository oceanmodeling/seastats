from __future__ import annotations

import importlib.metadata
import logging
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from seastats.stats import align_ts
from seastats.stats import get_bias
from seastats.stats import get_cc
from seastats.stats import get_corr
from seastats.stats import get_kge
from seastats.stats import get_lambda
from seastats.stats import get_mad
from seastats.stats import get_madc
from seastats.stats import get_madp
from seastats.stats import get_mae
from seastats.stats import get_mb
from seastats.stats import get_mse
from seastats.stats import get_nse
from seastats.stats import get_percentiles
from seastats.stats import get_rms
from seastats.stats import get_rmsd
from seastats.stats import get_rmse
from seastats.stats import get_slope_intercept
from seastats.stats import get_slope_intercept_pp
from seastats.stats import get_urmsd
from seastats.stats import get_vd
from seastats.stats import get_vs
from seastats.storms import match_extremes

__version__ = importlib.metadata.version(__name__)
__all__ = [
    "align_ts",
    "GENERAL_METRICS",
    "GENERAL_METRICS_ALL",
    "get_bias",
    "get_cc",
    "get_corr",
    "get_kge",
    "get_lambda",
    "get_mad",
    "get_madc",
    "get_madp",
    "get_mae",
    "get_mb",
    "get_mse",
    "get_nse",
    "get_percentiles",
    "get_rms",
    "get_rmsd",
    "get_rmse",
    "get_slope_intercept",
    "get_slope_intercept_pp",
    "get_stats",
    "get_urmsd",
    "get_vd",
    "get_vs",
    "match_extremes",
    "STORM_METRICS",
    "STORM_METRICS_ALL",
    "SUGGESTED_METRICS",
    "SUPPORTED_METRICS",
]

logger = logging.getLogger(__name__)

# Mapping from deprecated metric names to new names
_DEPRECATED_METRIC_NAMES: dict[str, str] = {
    "bias": "mb",
    "rmse": "rmsd",
    "rms": "urmsd",
    "cr": "cc",
}

GENERAL_METRICS_ALL = [
    "mb",
    "rmsd",
    "mae",
    "mse",
    "urmsd",
    "sim_mean",
    "obs_mean",
    "sim_std",
    "obs_std",
    "nse",
    "lambda",
    "cc",
    "slope",
    "intercept",
    "slope_pp",
    "intercept_pp",
    "mad",
    "madp",
    "madc",
    "kge",
    "vs",
    "vd",
]
GENERAL_METRICS = ["mb", "urmsd", "rmsd", "cc", "nse", "kge", "vs", "vd"]
STORM_METRICS = ["R1", "R3", "error"]
STORM_METRICS_ALL = [
    "R1",
    "R1_abs",
    "R1_abs_norm",
    "R3",
    "R3_abs",
    "R3_abs_norm",
    "error",
    "abs_error",
    "abs_error_norm",
]

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

    - `mb`: Mean Bias between the simulated and observed time series data.
    - `rmsd`: Root Mean Square Difference between the simulated and observed time series data.
    - `mae`: The Mean Absolute Error the simulated and observed time series data.
    - `mse`: The Mean Square Error the simulated and observed time series data.
    - `urmsd`: Unbiased Root Mean Square Difference (centered RMSD) between the simulated and observed time series data.
    - `sim_mean`: The mean of the simulated time series data.
    - `obs_mean`: The mean of the observed time series data.
    - `sim_std`: The standard deviation of the simulated time series data.
    - `obs_std`: The standard deviation of the observed time series data.
    - `nse`: The Nash-Sutcliffe efficiency between the simulated and observed time series data.
    - `lambda`: The lambda statistic between the simulated and observed time series data.
    - `cc`: Pearson Correlation Coefficient between the simulated and observed time series data.
    - `slope`: The slope of the linear regression between the simulated and observed time series data.
    - `intercept`: The intercept of the linear regression between the simulated and observed time series data.
    - `slope_pp`: The slope of the linear regression between the percentiles of the simulated and observed time series data.
    - `intercept_pp`: The intercept of the linear regression between the percentiles of the simulated and observed time series data.
    - `mad`: The median absolute deviation of the simulated time series data from its median.
    - `madp`: The median absolute deviation of the simulated time series data from its median, calculated using the percentiles of the observed time series data.
    - `madc`: The median absolute deviation of the simulated time series data from its median, calculated by adding `mad` to `madp`
    - `kge`: The Kling-Gupta efficiency between the simulated and observed time series data.
    - `vs`: Variance Similarity (Koh et al., 2012), unitless, ranges from 0 to 1 (1 = equal variances).
    - `vd`: Variance Dissimilarity, 0 = equal variances, positive for var(M)>var(O), negative for var(M)<var(O), range from -1 (limit of 0 model var or inf obs var) to +1 (limit of inf model var or 0 obs var)".
    - `R1`: Difference between observed and modelled for the biggest storm
    - `R1_abs`: Absolute R1 (R1 divided by observed value)
    - `R1_abs_norm`: Absolute normalized R1 (R1_abs divided by observed max peak)
    - `R3`: Averaged difference between observed and modelled for the three biggest storms
    - `R3_abs`: Averaged absolute difference between observed and modelled for the three biggest storms
    - `R3_abs_norm`: Normalized absolute R3 (R3_abs divided by observed value)
    - `error`: Averaged difference between modelled values and observed detected storms
    - `abs_error`: Averaged absolute difference between modelled values and observed detected storms
    - `abs_error_norm`: Averaged normalised absolute difference between modelled values and observed detected storms

    .. deprecated::
        The metric names ``bias``, ``rmse``, ``rms``, and ``cr`` are deprecated.
        Use ``mb``, ``rmsd``, ``urmsd``, and ``cc`` instead.
    """
    if not isinstance(metrics, list):
        raise ValueError("metrics must be a list")

    if metrics == ["all"]:
        metrics = SUPPORTED_METRICS

    # Handle deprecated metric names
    resolved_metrics: list[str] = []
    for m in metrics:
        if m in _DEPRECATED_METRIC_NAMES:
            new_name = _DEPRECATED_METRIC_NAMES[m]
            warnings.warn(
                f"Metric name '{m}' is deprecated, use '{new_name}' instead",
                FutureWarning,
                stacklevel=2,
            )
            resolved_metrics.append(new_name)
        else:
            resolved_metrics.append(m)
    metrics = resolved_metrics

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
            case "mb":
                stats["mb"] = get_mb(sim, obs)
            case "rmsd":
                stats["rmsd"] = get_rmsd(sim, obs)
            case "urmsd":
                stats["urmsd"] = get_urmsd(sim, obs)
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
            case "cc":
                stats["cc"] = get_cc(sim, obs)
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
            case "vs":
                stats["vs"] = get_vs(sim, obs)
            case "vd":
                stats["vd"] = get_vd(sim, obs)
            case "R1":
                stats["R1"] = extreme_df["error"].iloc[0]
            case "R1_abs":
                stats["R1_abs"] = extreme_df["abs_error"].iloc[0]
            case "R1_abs_norm":
                stats["R1_abs_norm"] = extreme_df["abs_error_norm"].iloc[0]
            case "R3":
                stats["R3"] = extreme_df["error"].iloc[0:3].mean()
            case "R3_abs":
                stats["R3_abs"] = extreme_df["abs_error"].iloc[0:3].mean()
            case "R3_abs_norm":
                stats["R3_abs_norm"] = extreme_df["abs_error_norm"].iloc[0:3].mean()
            case "error":
                stats["error"] = extreme_df["error"].mean()
            case "abs_error":
                stats["abs_error"] = extreme_df["abs_error"].mean()
            case "abs_error_norm":
                stats["abs_error_norm"] = extreme_df["abs_error_norm"].mean()

    if round > 0:
        for metric in metrics:
            stats[metric] = np.round(stats[metric], round)

    return stats
