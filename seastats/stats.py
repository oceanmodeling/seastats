from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd


def get_bias(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    bias = sim.mean() - obs.mean()
    return np.round(bias, round)


def get_mse(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    mse = np.square(np.subtract(obs, sim)).mean()
    return np.round(mse, round)


def get_rmse(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    rmse = np.sqrt(get_mse(sim, obs, 10))
    return np.round(rmse, round)


def get_mae(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    mae = np.abs(np.subtract(obs, sim)).mean()
    return np.round(mae, round)


def get_mad(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    mae = np.abs(np.subtract(obs, sim)).std()
    return np.round(mae, round)


def get_madp(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    pc1, pc2 = get_percentiles(sim, obs)
    return get_mad(pc1, pc2, round)


def get_madc(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    madp = get_madp(sim, obs, round)
    return get_mad(sim, obs, round) + madp


def get_rms(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    crmsd = ((sim - sim.mean()) - (obs - obs.mean())) ** 2
    return np.round(np.sqrt(crmsd.mean()), round)


def get_corr(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    corr = sim.corr(obs)
    return np.round(corr, round)


def get_rms_quantile(
    sim: pd.Series, obs: pd.Series, quantile: float = 0.95, round: int = 3
) -> float:
    df1_95 = sim[sim > sim.quantile(quantile)]
    df2_95 = obs[obs > obs.quantile(quantile)]
    crmsd = ((df1_95 - df1_95.mean()) - (df2_95 - df2_95.mean())) ** 2
    return np.round(np.sqrt(crmsd.mean()), round)


def get_corr_quantile(
    sim: pd.Series, obs: pd.Series, quantile: float = 0.95, round: int = 3
) -> float:
    df1_95 = sim[sim > sim.quantile(quantile)]
    df2_95 = obs[obs > obs.quantile(quantile)]
    corr = df1_95.corr(df2_95)
    return np.round(corr, round)


def get_nse(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    nse = 1 - np.nansum(np.subtract(obs, sim) ** 2) / np.nansum(
        (obs - np.nanmean(obs)) ** 2
    )
    return np.round(nse, round)


def get_lambda(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    Xmean = np.nanmean(obs)
    Ymean = np.nanmean(sim)
    nObs = len(obs)
    corr = get_corr(sim, obs, 10)
    if corr >= 0:
        kappa = 0
    else:
        kappa = 2 * abs(np.nansum((obs - Xmean) * (sim - Ymean)))

    Nomin = np.nansum((obs - sim) ** 2)
    Denom = (
        np.nansum((obs - Xmean) ** 2)
        + np.nansum((sim - Ymean) ** 2)
        + nObs * ((Xmean - Ymean) ** 2)
        + kappa
    )
    lambda_index = 1 - Nomin / Denom
    return np.round(lambda_index, round)


def get_kge(sim: pd.Series, obs: pd.Series, round: int = 3) -> float:
    corr = get_corr(sim, obs, 10)
    b = (sim.mean() - obs.mean()) / obs.std()
    g = sim.std() / obs.std()
    kge = 1 - np.sqrt((corr - 1) ** 2 + b**2 + (g - 1) ** 2)
    return np.round(kge, round)


def truncate_seconds(ts: pd.Series) -> pd.Series:
    df = pd.DataFrame({"time": ts.index, "value": ts.values})
    df = df.assign(time=df.time.dt.floor("min"))
    if df.time.duplicated().any():
        # There are duplicates. Keep the first datapoint per minute.
        msg = "Duplicate timestamps have been detected after the truncation of seconds. Keeping the first datapoint per minute"
        print(msg)
        df = df.iloc[df.time.drop_duplicates().index].reset_index(drop=True)
    df.index = df.time
    df = df.drop("time", axis=1)
    ts = pd.Series(index=df.index, data=df.value)
    return ts


def align_ts(
    sim: pd.Series, obs: pd.Series, resample_to_model: bool = True
) -> Tuple[pd.Series, pd.Series]:
    # requisite: obs is the observation time series
    obs = obs.dropna()
    obs = truncate_seconds(obs)
    if resample_to_model:
        freq = sim.index.to_series().diff().median()
        obs = obs.resample(pd.Timedelta(freq)).mean()
    sim_, obs_ = sim.align(obs, axis=0)
    nan_mask1 = pd.isna(sim_)
    nan_mask2 = pd.isna(obs_)
    nan_mask = np.logical_or(nan_mask1.values, nan_mask2.values)
    sim_ = sim_[~nan_mask]
    obs_ = obs_[~nan_mask]
    return sim_, obs_


def get_percentiles(
    sim: pd.Series, obs: pd.Series, higher_tail: bool = False
) -> Tuple[pd.Series, pd.Series]:
    x = np.arange(0, 0.99, 0.01)
    if higher_tail:
        x = np.hstack([x, np.arange(0.99, 1, 0.001)])
    pc_sim = np.zeros(len(x))
    pc_obs = np.zeros(len(x))
    for it, thd in enumerate(x):
        pc_sim[it] = sim.quantile(thd)
        pc_obs[it] = obs.quantile(thd)
    return pd.Series(pc_sim), pd.Series(pc_obs)


def get_slope_intercept(
    sim: pd.Series, obs: pd.Series, round: int = 3
) -> Tuple[float, float]:
    # Calculate means of x and y
    x_mean = np.mean(obs)
    y_mean = np.mean(sim)

    # Calculate the terms needed for the numerator and denominator of the slope
    numerator = np.sum((obs - x_mean) * (sim - y_mean))
    denominator = np.sum((obs - x_mean) ** 2)

    # Calculate slope (beta1) and intercept (beta0)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return np.round(slope, round), np.round(intercept, round)


def get_slope_intercept_pp(
    sim: pd.Series, obs: pd.Series, round: int = 3
) -> Tuple[float, float]:
    pc1, pc2 = get_percentiles(sim, obs)
    slope, intercept = get_slope_intercept(pc1, pc2)
    return np.round(slope, round), np.round(intercept, round)


def get_stats(sim: pd.Series, obs: pd.Series) -> Dict[str, float]:
    slope, intercept = get_slope_intercept(sim, obs)
    slopepp, interceptpp = get_slope_intercept_pp(sim, obs)
    version_stat = {
        "bias": get_bias(sim, obs),
        "rmse": get_rmse(sim, obs),
        "rms": get_rms(sim, obs),
        "rms_95": get_rms_quantile(sim, obs),
        "sim_mean": np.round(sim.mean(), 3),
        "obs_mean": np.round(obs.mean(), 3),
        "sim_std": np.round(sim.std(), 3),
        "obs_std": np.round(obs.std(), 3),
        "nse": get_nse(sim, obs),
        "lamba": get_lambda(sim, obs),
        "cr": get_corr(sim, obs),
        "cr_95": get_corr_quantile(sim, obs),
        "slope": slope,
        "intercept": intercept,
        "slope_pp": slopepp,
        "intercept_pp": interceptpp,
        "mad": get_mad(sim, obs),
        "madp": get_madp(sim, obs),
        "madc": get_madc(sim, obs),
        "kge": get_kge(sim, obs),
    }
    return version_stat