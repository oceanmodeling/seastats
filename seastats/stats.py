from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_bias(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    return float(sim.mean() - obs.mean())


def get_mse(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    return float(np.square(np.subtract(obs, sim)).mean())


def get_rmse(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    return float(np.sqrt(get_mse(sim, obs)))


def get_mae(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    return float(np.abs(np.subtract(obs, sim)).mean())


def get_mad(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    return float(np.abs(np.subtract(obs, sim)).std())


def get_madp(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    pc1, pc2 = get_percentiles(sim, obs)
    return get_mad(pc1, pc2)


def get_madc(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    madp = get_madp(sim, obs)
    return get_mad(sim, obs) + madp


def get_rms(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    crmsd = ((sim - sim.mean()) - (obs - obs.mean())) ** 2
    return float(np.sqrt(crmsd.mean()))


def get_corr(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    return float(sim.corr(obs))


def get_nse(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    nse = 1 - np.nansum(np.subtract(obs, sim) ** 2) / np.nansum((obs - float(np.nanmean(obs))) ** 2)
    return float(nse)


def get_lambda(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    Xmean = float(np.nanmean(obs))
    Ymean = float(np.nanmean(sim))
    nObs = len(obs)
    corr = get_corr(sim, obs)
    if corr >= 0:
        kappa = 0
    else:
        kappa = 2 * abs(np.nansum((obs - Xmean) * (sim - Ymean)))

    numerator = np.nansum((obs - sim) ** 2)
    denominator = (
        np.nansum((obs - Xmean) ** 2)
        + np.nansum((sim - Ymean) ** 2)
        + nObs * ((Xmean - Ymean) ** 2)
        + kappa
    )
    lambda_index = 1 - numerator / denominator
    return float(lambda_index)


def get_kge(sim: pd.Series[float], obs: pd.Series[float]) -> float:
    corr = get_corr(sim, obs)
    b = (sim.mean() - obs.mean()) / obs.std()
    g = sim.std() / obs.std()
    return float(1 - np.sqrt((corr - 1) ** 2 + b**2 + (g - 1) ** 2))


def truncate_seconds(ts: pd.Series[float]) -> pd.Series[float]:
    df = pd.DataFrame({"time": ts.index, "value": ts.values})
    df = df.assign(time=df.time.dt.floor("min"))
    if df.time.duplicated().any():
        # There are duplicates. Keep the first datapoint per minute.
        msg = "Duplicate timestamps have been detected after the truncation of seconds. Keeping the first datapoint per minute"
        logger.warning(msg)
        df = df.iloc[df.time.drop_duplicates().index].reset_index(drop=True)
    df.index = df.time  # type: ignore[assignment]
    df = df.drop("time", axis=1)
    ts = pd.Series(index=df.index, data=df.value)
    return ts


def align_ts(
    sim: pd.Series[float],
    obs: pd.Series[float],
    resample_to_model: bool = True,
) -> tuple[pd.Series[float], pd.Series[float]]:
    # requisite: obs is the observation time series
    obs = obs.dropna()
    obs = truncate_seconds(obs)
    if resample_to_model:
        freq = sim.index.to_series().diff().median()
        obs = obs.resample(pd.Timedelta(freq)).mean()
    sim_, obs_ = sim.align(obs, axis=0)
    nan_mask1 = pd.isna(sim_)
    nan_mask2 = pd.isna(obs_)
    nan_mask = np.logical_or(nan_mask1, nan_mask2)
    sim_ = sim_[~nan_mask]
    obs_ = obs_[~nan_mask]
    return sim_, obs_


def get_percentiles(
    sim: pd.Series[float],
    obs: pd.Series[float],
    higher_tail: bool = False,
) -> tuple[pd.Series[float], pd.Series[float]]:
    x = np.arange(0, 0.99, 0.01)
    if higher_tail:
        x = np.hstack([x, np.arange(0.99, 1, 0.001)])

    pc_sim = sim.quantile(x).to_numpy()
    pc_obs = obs.quantile(x).to_numpy()

    return pd.Series(pc_sim), pd.Series(pc_obs)


def get_slope_intercept(sim: pd.Series[float], obs: pd.Series[float]) -> tuple[float, float]:
    # Calculate means of x and y
    x_mean = float(np.mean(obs))
    y_mean = float(np.mean(sim))

    numerator = np.sum((obs - x_mean) * (sim - y_mean))
    denominator = np.sum((obs - x_mean) ** 2)

    # Calculate slope (A) and intercept (B) in A*X + B
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def get_slope_intercept_pp(sim: pd.Series[float], obs: pd.Series[float]) -> tuple[float, float]:
    pc1, pc2 = get_percentiles(sim, obs)
    slope, intercept = get_slope_intercept(pc1, pc2)
    return slope, intercept
