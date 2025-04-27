import pandas as pd
import pytest

import seastats

EXPECTED_000 = {
    "R1": 0.237,
    "R1_norm": 0.296,
    "R3": 0.233,
    "R3_norm": 0.366,
    "bias": 0.007,
    "cr": 0.817,
    "error": 0.233,
    "error_norm": 0.366,
    "intercept": 0.007,
    "intercept_pp": 0.008,
    "kge": 0.81,
    "lambda": 0.929,
    "mad": 0.052,
    "madc": 0.266,
    "madp": 0.213,
    "mae": 0.068,
    "mse": 0.007,
    "nse": 0.908,
    "obs_mean": -0.0,
    "obs_std": 0.142,
    "rms": 0.086,
    "rmse": 0.086,
    "sim_mean": 0.007,
    "sim_std": 0.144,
    "slope": 0.204,
    "slope_pp": 0.336,
}
EXPECTED_099 = {
    "R1": 0.237,
    "R1_norm": 0.296,
    "R3": 0.147,
    "R3_norm": 0.207,
    "bias": -0.028,
    "cr": 0.453,
    "error": 0.094,
    "error_norm": 0.178,
    "intercept": 0.433,
    "intercept_pp": 0.153,
    "kge": 0.25,
    "lambda": 0.852,
    "mad": 0.07,
    "madc": 0.1,
    "madp": 0.03,
    "mae": 0.077,
    "mse": 0.011,
    "nse": 0.822,
    "obs_mean": 0.482,
    "obs_std": 0.087,
    "rms": 0.093,
    "rmse": 0.104,
    "sim_mean": 0.454,
    "sim_std": 0.053,
    "slope": 0.043,
    "slope_pp": 0.626,
}


@pytest.fixture(scope="session")
def sim():
    sim = pd.read_parquet("tests/data/abed_sim.parquet")
    sim = sim[sim.columns[0]]
    return sim


@pytest.fixture(scope="session")
def obs():
    obs = pd.read_parquet("tests/data/abed_obs.parquet")
    obs = obs[obs.columns[0]]
    return obs


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(dict(quantile=0.00, expected=EXPECTED_000, metrics=seastats.GENERAL_METRICS), id="general"),
        pytest.param(dict(quantile=0.00, expected=EXPECTED_000, metrics=seastats.GENERAL_METRICS_ALL), id="general_all"),
        pytest.param(dict(quantile=0.99, expected=EXPECTED_099, metrics=seastats.STORM_METRICS), id="storm"),
        pytest.param(dict(quantile=0.99, expected=EXPECTED_099, metrics=seastats.STORM_METRICS_ALL), id="storm_all"),
        pytest.param(dict(quantile=0.99, expected=EXPECTED_099, metrics=seastats.SUGGESTED_METRICS), id="suggested"),
        pytest.param(dict(quantile=0.99, expected=EXPECTED_099, metrics=seastats.SUPPORTED_METRICS), id="supported"),
    ],
)  # fmt: skip
def test_get_stats(sim, obs, args):
    expected_results = args.pop("expected")
    stats = seastats.get_stats(sim, obs, **args)
    for metric in args["metrics"]:
        result = stats[metric]
        expected = expected_results[metric]
        assert result == pytest.approx(expected, abs=1e-3)


def test_get_stats_raise_for_non_supported_metrics(sim, obs):
    with pytest.raises(ValueError) as exc:
        seastats.get_stats(sim, obs, metrics=["gibberish"])
    assert "metrics must be a list of supported variables in SUPPORTED_METRICS or ['all']" in str(exc)


def test_get_stats_raise_if_metrics_is_not_a_list(sim, obs):
    with pytest.raises(ValueError) as exc:
        seastats.get_stats(sim, obs, metrics="all")
    assert "metrics must be a list" in str(exc)


def test_get_stats_ensure_that_all_returns_supported_metrics(sim, obs):
    all_results = seastats.get_stats(sim, obs, metrics=["all"])
    supported_results = seastats.get_stats(sim, obs, metrics=seastats.SUPPORTED_METRICS)
    assert all_results == supported_results
