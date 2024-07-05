import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Tuple, Dict


def get_bias(df1: pd.Series, df2: pd.Series, round:int = 3)->float: 
    bias = df1.mean() - df2.mean()
    return np.round(bias, round)


def get_mse(df1: pd.Series, df2: pd.Series, round: int = 3)->float: 
    mse = np.square(np.subtract(df2, df1)).mean()
    return np.round(mse, round)


def get_rmse(df1: pd.Series, df2: pd.Series, round:int = 3)->float:
    rmse = np.sqrt(get_mse(df1, df2, 10))
    return np.round(rmse, round)


def get_mae(df1: pd.Series, df2: pd.Series, round:int = 3)->float:
    mae = np.abs(np.subtract(df2, df1)).mean()
    return np.round(mae, round)

def get_mad(df1: pd.Series, df2: pd.Series, round:int = 3)->float:
    mae = np.abs(np.subtract(df2, df1)).std()
    return np.round(mae, round)


def get_madp(df1: pd.Series, df2: pd.Series, round:int = 3)->float:
    pc1, pc2 = get_percentiles(df1, df2)
    return get_mad(pc1, pc2, round)


def get_madc(df1: pd.Series, df2: pd.Series, round:int = 3)->float:
    madp = get_madp(df1, df2, round)
    return get_mad(df1, df2, round) + madp


def get_rms(df1: pd.Series, df2: pd.Series, round:int = 3)->float:
    crmsd = ((df1 - df1.mean()) - (df2 - df2.mean()))**2
    return  np.round(np.sqrt(crmsd.mean()), round)


def get_corr(df1: pd.Series, df2: pd.Series, round: int = 3)->float:
    corr = df1.corr(df2)
    return np.round(corr, round)


def get_rms_quantile(df1: pd.Series, df2: pd.Series, quantile: float = 0.95, round:int = 3)->float:
    df1_95 = df1[df1 > df1.quantile(quantile)]
    df2_95 = df2[df2 > df2.quantile(quantile)]
    crmsd = ((df1_95 - df1_95.mean()) - (df2_95 - df2_95.mean()))**2
    return  np.round(np.sqrt(crmsd.mean()), round)


def get_corr_quantile(df1: pd.Series, df2: pd.Series, quantile: float = 0.95, round: int = 3)->float:
    df1_95 = df1[df1 > df1.quantile(quantile)]
    df2_95 = df2[df2 > df2.quantile(quantile)]
    corr = df1_95.corr(df2_95)
    return np.round(corr, round)


def get_nse(df1: pd.Series, df2: pd.Series, round: int = 3)->float:
    nse = 1 - np.nansum(np.subtract(df2, df1) ** 2) / np.nansum((df2 - np.nanmean(df2)) ** 2)
    return np.round(nse, round)


def get_lambda(df1: pd.Series, df2: pd.Series, round: int = 3)->float:
    Xmean = np.nanmean(df2)
    Ymean = np.nanmean(df1)
    nObs = len(df2)
    corr = get_corr(df1, df2, 10)
    if corr >= 0:
        kappa = 0
    else:
        kappa = 2 * abs(np.nansum((df2 - Xmean) * (df1 - Ymean)))

    Nomin = np.nansum((df2 - df1) ** 2)
    Denom = (
        np.nansum((df2 - Xmean) ** 2)
        + np.nansum((df1 - Ymean) ** 2)
        + nObs * ((Xmean - Ymean) ** 2)
        + kappa
    )
    lambda_index = 1 - Nomin / Denom
    return np.round(lambda_index, round)
                          

def get_kge(df1: pd.Series, df2: pd.Series, round: int =3)->float:
    corr = get_corr(df1, df2, 10)
    b = (df1.mean() - df2.mean())/df2.std()
    g = df1.std()/df2.std()
    kge = 1 - np.sqrt((corr-1)**2 + b**2 + (g-1)**2)
    return np.round(kge, round)


def truncate_seconds(ts: pd.Series)->pd.Series:
    df = pd.DataFrame({"time": ts.index, "value": ts.values})
    df = df.assign(time=df.time.dt.floor("min"))
    if df.time.duplicated().any():
        # There are duplicates. Keep the first datapoint per minute.
        msg = "Duplicate timestamps have been detected after the truncation of seconds. Keeping the first datapoint per minute"
        print(msg)
        df = df.iloc[df.time.drop_duplicates().index].reset_index(drop=True)
    df.index = df.time
    df = df.drop("time", axis=1)
    ts  = pd.Series(index=df.index, data=df.value)
    return ts


def align_ts(df1: pd.Series, df2: pd.Series, resample_to_model: bool = True)->Tuple[pd.Series, pd.Series]:
    # requisite: df2 is the observation time series
    df2 = df2.dropna()
    df2 = truncate_seconds(df2)
    if resample_to_model:
        freq = df1.index.to_series().diff().median()
        df2 = df2.resample(pd.Timedelta(freq)).mean()
    sim, obs = df1.align(df2, axis = 0)
    nan_mask1 = pd.isna(sim)
    nan_mask2 = pd.isna(obs)
    nan_mask = np.logical_or(nan_mask1.values, nan_mask2.values)
    sim = sim[~nan_mask]
    obs = obs[~nan_mask]
    return sim, obs


def get_percentiles(df1: pd.Series, df2: pd.Series, higher_tail:bool = False) -> Tuple[pd.Series, pd.Series]:
    x = np.arange(0, 0.99, 0.01)
    if higher_tail:
        x = np.hstack([x, np.arange(0.99, 1, 0.001)])
    pc1 = np.zeros(len(x))
    pc2 = np.zeros(len(x))
    for it, thd in enumerate(x):
        pc1[it] = df1.quantile(thd)
        pc2[it] = df2.quantile(thd)
    return pd.Series(pc1), pd.Series(pc2)


def get_slope_intercept_pp(df1: pd.Series, df2: pd.Series, round: int = 3)->Tuple[float,float]:
    pc1, pc2 = get_percentiles(df1, df2)
    slope, intercept, _, _, _ =  linregress(pc1, pc2)
    return np.round(slope, round), np.round(intercept, round)


def get_stats(sim: pd.Series, obs: pd.Series)->Dict[str, float]:
    slope, intercept, _, _, _ =  linregress(sim, obs)
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
        "mad" : get_mad(sim, obs),
        "madp" : get_madp(sim, obs),
        "madc" : get_madc(sim, obs),
        "kge": get_kge(sim, obs)
    }
    return version_stat
