import pandas as pd

from seastats.stats import get_stats
from seastats.storms import storm_metrics


def test_get_stats():
    sim = pd.read_parquet("tests/data/abed_sim.parquet")
    obs = pd.read_parquet("tests/data/abed_obs.parquet")
    # sim and obs need to be Series
    obs = obs[obs.columns[0]]
    sim = sim[sim.columns[0]]
    stats = get_stats(sim, obs)
    metric99 = storm_metrics(sim, obs, quantile=0.99, cluster_duration=72)
    metric95 = storm_metrics(sim, obs, quantile=0.95, cluster_duration=72)
    # Create a dictionary for the current version's statistics
    stats["R1"] = metric99["R1"]
    stats["R1_norm"] = metric99["R1_norm"]
    stats["R3"] = metric99["R3"]
    stats["R3_norm"] = metric99["R3_norm"]
    stats["error99"] = metric99["error"]
    stats["error99m"] = metric99["error_metric"]
    stats["error95"] = metric95["error"]
    stats["error95m"] = metric95["error_metric"]
