from __future__ import annotations

import typing as T

import numpy as np
import pandas as pd
from pyextremes import get_extremes


def match_extremes(
    sim: pd.Series[float],
    obs: pd.Series[float],
    quantile: float,
    cluster: int = 24,
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
    # get observed extremes
    ext = get_extremes(obs, "POT", threshold=obs.quantile(quantile), r=f"{cluster}h")
    ext_values_dict: dict[str, T.Any] = {}
    ext_values_dict["observed"] = ext.values
    ext_values_dict["time observed"] = ext.index.values
    #
    max_in_window = []
    tmax_in_window = []
    # match simulated values with observed events
    for it, itime in enumerate(ext.index):
        snippet = sim[itime - pd.Timedelta(hours=cluster / 2) : itime + pd.Timedelta(hours=cluster / 2)]
        try:
            tmax_in_window.append(snippet.index[int(snippet.argmax())])
            max_in_window.append(snippet.max())
        except Exception:
            tmax_in_window.append(itime)
            max_in_window.append(np.nan)
    ext_values_dict["model"] = max_in_window
    ext_values_dict["time model"] = tmax_in_window
    #
    df = pd.DataFrame(ext_values_dict)
    df = df.dropna(subset="model")
    df = df.sort_values("observed", ascending=False)
    df["diff"] = df["model"] - df["observed"]
    df["error"] = abs(df["diff"])
    df["error_norm"] = abs(df["diff"] / df["observed"])
    df["tdiff"] = df["time model"] - df["time observed"]
    df["tdiff"] = df["tdiff"].apply(lambda x: x.total_seconds() / 3600)
    df = df.set_index("time observed")
    return df
