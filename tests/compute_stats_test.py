import pandas as pd
import pytest

from seastats import get_stats

SIM = pd.read_parquet("tests/data/abed_sim.parquet")
SIM = SIM[SIM.columns[0]]
OBS = pd.read_parquet("tests/data/abed_obs.parquet")
OBS = OBS[OBS.columns[0]]

RESULTS = {
    "bias": 0.007,
    "rmse": 0.086,
    "rms": 0.086,
    "nse": 0.908,
    "lamba": 0.929,
    "cr": 0.817,
    "slope": 0.204,
    "intercept": 0.007,
    "slope_pp": 0.336,
    "intercept_pp": 0.008,
    "mad": 0.052,
    "madp": 0.213,
    "madc": 0.265,
    "kge": 0.81,
    "R1": 0.237,
    "R3": 0.147,
    "error": 0.094,
}

# Define test cases
test_cases = [
    (
        "all_stats_and_extremes",
        {"quantile": 0, "metrics": ["rmse", "rms", "nse", "lamba", "cr", "slope"]},
    ),
    ("all_stats", {"quantile": 0.99, "metrics": ["R1", "R3", "error"]}),
]


@pytest.mark.parametrize("test_type, args", test_cases)
def test_metrics(test_type, args):
    stats = get_stats(SIM, OBS, **args)

    # Assert all metrics
    check_dict = {m: RESULTS[m] for m in args["metrics"]}
    for metric, value in check_dict.items():
        assert stats[metric] == pytest.approx(value, abs=1.0e-3)
