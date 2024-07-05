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
        extremes = extremes.groupby(pd.Grouper(freq="2D")).mean().dropna(how="all")
        return extremes


def match_extremes(extremes: pd.DataFrame) -> pd.DataFrame:
    if extremes.empty:
        return pd.DataFrame()
    extremes_match = extremes.groupby(pd.Grouper(freq="2D")).mean().dropna()
    if len(extremes_match) == 0:
        return pd.DataFrame()
    else:
        extremes_match["difference"] = (
            extremes_match["observed"] - extremes_match["modeled"]
        )
        extremes_match["error"] = np.abs(
            extremes_match["difference"] / extremes_match["observed"]
        )
        extremes_match["error_m"] = extremes_match["error"] * extremes_match["observed"]
        return extremes_match


def storm_metrics(
    sim: pd.Series, obs: pd.Series, quantile: float, cluster_duration: int = 72
) -> Dict[str, float]:
    extremes = get_extremes_ts(sim, obs, quantile, cluster_duration)
    extremes_match = match_extremes(extremes)
    if extremes_match.empty:
        return {
            "db_match": np.nan,
            "R1_norm": np.nan,
            "R1": np.nan,
            "R3_norm": np.nan,
            "R3": np.nan,
            "error": np.nan,
            "error_metric": np.nan,
        }
    else:
        # R1: diff for the biggest storm in each dataset
        idx_max = extremes_match["observed"].idxmax()
        R1_norm = extremes_match["error"][idx_max]
        R1 = extremes_match["error_m"][idx_max]
        # R3: Difference between observed and modelled for the biggest storm
        idx_max = extremes_match["observed"].nlargest(3).index
        R3_norm = extremes_match["error"][idx_max].mean()
        R3 = extremes_match["error_m"][idx_max].mean()
        metrics = {
            "db_match": len(extremes_match) / len(extremes),
            "R1_norm": R1_norm,
            "R1": R1,
            "R3_norm": R3_norm,
            "R3": R3,
            "error": extremes_match["error"].mean(),
            "error_metric": extremes_match["error_m"].mean(),
        }
        return metrics
