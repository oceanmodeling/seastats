from typing import Dict

import numpy as np
import pandas as pd
from pyextremes import get_extremes


def get_extremes_ts(
    sim: pd.Series, obs: pd.Series, quantile: float, cluster_duration: int = 72
) -> pd.DataFrame:
    # first ts
    threshold = sim.quantile(quantile)
    ext_ = get_extremes(sim, "POT", threshold=threshold, r=f"{cluster_duration}h")
    extremes1 = pd.DataFrame(
        {"modeled": ext_, "time_model": ext_.index}, index=ext_.index
    )
    # second ts
    threshold = obs.quantile(quantile)
    ext_ = get_extremes(obs, "POT", threshold=threshold, r=f"{cluster_duration}h")
    extremes2 = pd.DataFrame(
        {"observed": ext_, "time_obs": ext_.index}, index=ext_.index
    )
    extremes = pd.concat([extremes1, extremes2], axis=1)
    if extremes.empty:
        return pd.DataFrame()
    else:
        extremes = (
            extremes.groupby(pd.Grouper(freq=f"{cluster_duration//2}h"))
            .mean()
            .dropna(how="all")
        )
        return extremes


def match_extremes(
    sim: pd.Series, obs: pd.Series, quantile: float, cluster: int = 72
) -> pd.DataFrame:
    """
    Calculate metrics for comparing simulated and observed storm events.
    Parameters:
    - sim (pd.Series): Simulated storm series.
    - obs (pd.Series): Observed storm series.
    - quantile (float): Quantile value for defining extreme events.
    - cluster (int, optional): Cluster duration for grouping storm events. Default is 72 hours.

    Returns:
    - df: pd.DataFrame of the matched extremes between the observed and modeled pd.Series.
          with following columns:
           * `observed`: observed extreme event value
           * `time observed`: observed extreme event time
           * `model`: modeled extreme event value
           * `time model`: modeled extreme event time
           * `diff`: difference between model and observed
           * `error`: absolute difference between model and observed
           * `error_norm`: normalised difference between model and observed
           * `tdiff`: time difference between model and observed (in hours)

    !Important: The modeled values are matched on the observed events calculated by POT analysis.
                The user needs to be mindful about the order of the observed and modeled pd.Series.
    """
    # resample observation to 1H time series
    obs = obs.resample("1h").mean().shift(freq="30min")
    # get observed extremes
    ext = get_extremes(obs, "POT", threshold=obs.quantile(quantile), r=f"{cluster}h")
    ext_values_dict = {}
    ext_values_dict["observed"] = ext.values
    ext_values_dict["time observed"] = ext.index
    #
    max_in_window = []
    tmax_in_window = []
    # match simulated values with observed events
    for it, itime in enumerate(ext.index):
        snippet = sim[
            itime
            - pd.Timedelta(hours=cluster / 2) : itime
            + pd.Timedelta(hours=cluster / 2)
        ]
        try:
            tmax_in_window.append(snippet.index[snippet.argmax()])
            max_in_window.append(snippet.max())
        except Exception:
            tmax_in_window.append(itime)
            max_in_window.append(np.nan)
    ext_values_dict["model"] = max_in_window
    ext_values_dict["time model"] = tmax_in_window
    #
    df = pd.DataFrame(ext_values_dict)
    df.index = df["time observed"]
    df = df.dropna(subset="model")
    df = df.sort_values("observed", ascending=False)
    df["diff"] = df["model"] - df["observed"]
    df["error"] = abs(df["diff"])
    df["error_norm"] = abs(df["diff"] / df["observed"])
    df["tdiff"] = df["time model"] - df["time observed"]
    df["tdiff"] = df["tdiff"].apply(lambda x: x.total_seconds() / 3600)
    return df


def storm_metrics(
    sim: pd.Series, obs: pd.Series, quantile: float, cluster: int = 72
) -> Dict[str, float]:
    """
    Calculate metrics for comparing simulated and observed storm events
    Parameters:
    - sim (pd.Series): Simulated storm series
    - obs (pd.Series): Observed storm series
    - quantile (float): Quantile value for defining extreme events
    - cluster (int, optional): Cluster duration for grouping storm events. Default is 72 hours

    Returns:
    - Dict[str, float]: Dictionary containing calculated metrics:
        - R1: Difference between observed and modelled for the biggest storm
        - R1_norm: Normalized R1 (R1 divided by observed value)
        - R3: Average difference between observed and modelled for the three biggest storms
        - R3_norm: Normalized R3 (R3 divided by observed value)
        - error: Average difference between observed and modelled for all storms
        - error_norm: Normalized error (error divided by observed value)
    """
    df = match_extremes(sim, obs, quantile=quantile, cluster=cluster)

    # calculate R1, R3 and error
    R1 = df["error"].iloc[0]
    R1n = df["error_norm"].iloc[0]
    R3 = df["error"].iloc[0:3].mean()
    R3n = df["error_norm"].iloc[0:3].mean()
    ERROR = df["error"].mean()
    ERRORn = df["error_norm"].mean()
    # R3: Difference between observed and modelled for the biggest storm
    metrics = {
        "R1": R1,
        "R1_norm": R1n,
        "R3": R3,
        "R3_norm": R3n,
        "error": ERROR,
        "error_norm": ERRORn,
    }
    return metrics
