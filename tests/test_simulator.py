"""Tests for the GBM market data simulator."""

import numpy as np
import pytest

from src.simulator.gbm import simulate_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_asset_params():
    return dict(
        S0=np.array([100.0]),
        mu=np.array([0.05]),
        sigma=np.array([0.2]),
        T=1.0,
        dt=1 / 252,
        n_simulations=1000,
        seed=42,
    )


@pytest.fixture
def multi_asset_params():
    corr = np.array([
        [1.0, 0.6],
        [0.6, 1.0],
    ])
    return dict(
        S0=np.array([100.0, 50.0]),
        mu=np.array([0.05, 0.08]),
        sigma=np.array([0.2, 0.3]),
        corr_matrix=corr,
        T=1.0,
        dt=1 / 252,
        n_simulations=1000,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_identical_output(self, single_asset_params):
        paths1 = simulate_paths(**single_asset_params)
        paths2 = simulate_paths(**single_asset_params)
        np.testing.assert_array_equal(paths1, paths2)

    def test_same_seed_multi_asset(self, multi_asset_params):
        paths1 = simulate_paths(**multi_asset_params)
        paths2 = simulate_paths(**multi_asset_params)
        np.testing.assert_array_equal(paths1, paths2)

    def test_different_seed_different_output(self, single_asset_params):
        paths1 = simulate_paths(**single_asset_params)
        single_asset_params["seed"] = 99
        paths2 = simulate_paths(**single_asset_params)
        assert not np.array_equal(paths1, paths2)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_single_asset_shape(self, single_asset_params):
        paths = simulate_paths(**single_asset_params)
        n_steps = int(single_asset_params["T"] / single_asset_params["dt"])
        assert paths.shape == (1000, 1, n_steps + 1)

    def test_multi_asset_shape(self, multi_asset_params):
        paths = simulate_paths(**multi_asset_params)
        n_steps = int(multi_asset_params["T"] / multi_asset_params["dt"])
        assert paths.shape == (1000, 2, n_steps + 1)

    def test_initial_prices(self, multi_asset_params):
        paths = simulate_paths(**multi_asset_params)
        np.testing.assert_array_almost_equal(paths[:, 0, 0], 100.0)
        np.testing.assert_array_almost_equal(paths[:, 1, 0], 50.0)


# ---------------------------------------------------------------------------
# Single asset (no correlation matrix)
# ---------------------------------------------------------------------------

class TestSingleAsset:
    def test_no_corr_matrix(self):
        paths = simulate_paths(
            S0=np.array([100.0]),
            mu=np.array([0.05]),
            sigma=np.array([0.2]),
            T=10 / 252,
            dt=1 / 252,
            n_simulations=500,
            seed=7,
        )
        assert paths.shape == (500, 1, 11)
        assert np.all(paths > 0)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

class TestStatisticalProperties:
    def test_mean_drift(self):
        """With large n, realized mean log-return should approximate mu."""
        mu_input = 0.05
        sigma_input = 0.2
        T = 1.0
        dt = 1 / 252
        n_sims = 50_000

        paths = simulate_paths(
            S0=np.array([100.0]),
            mu=np.array([mu_input]),
            sigma=np.array([sigma_input]),
            T=T,
            dt=dt,
            n_simulations=n_sims,
            seed=123,
        )

        # Total log-return over the full horizon
        log_returns = np.log(paths[:, 0, -1] / paths[:, 0, 0])
        realized_mean = np.mean(log_returns)

        # Under GBM: E[ln(S(T)/S(0))] = (mu - sigma^2/2) * T
        expected_mean = (mu_input - 0.5 * sigma_input**2) * T
        assert abs(realized_mean - expected_mean) < 0.02

    def test_volatility(self):
        """Realized volatility should be close to input sigma."""
        sigma_input = 0.3
        T = 1.0
        dt = 1 / 252
        n_sims = 50_000

        paths = simulate_paths(
            S0=np.array([100.0]),
            mu=np.array([0.0]),
            sigma=np.array([sigma_input]),
            T=T,
            dt=dt,
            n_simulations=n_sims,
            seed=456,
        )

        # Total log-return variance over T
        log_returns = np.log(paths[:, 0, -1] / paths[:, 0, 0])
        realized_vol = np.std(log_returns) / np.sqrt(T)
        assert abs(realized_vol - sigma_input) < 0.02

    def test_prices_positive(self, multi_asset_params):
        """GBM paths should always be positive (log-normal property)."""
        paths = simulate_paths(**multi_asset_params)
        assert np.all(paths > 0)


# ---------------------------------------------------------------------------
# Correlation structure
# ---------------------------------------------------------------------------

class TestCorrelationStructure:
    def test_realized_correlation(self):
        """Realized cross-asset correlation should approximate input."""
        target_corr = 0.7
        corr_matrix = np.array([
            [1.0, target_corr],
            [target_corr, 1.0],
        ])

        paths = simulate_paths(
            S0=np.array([100.0, 100.0]),
            mu=np.array([0.05, 0.05]),
            sigma=np.array([0.2, 0.2]),
            corr_matrix=corr_matrix,
            T=1.0,
            dt=1 / 252,
            n_simulations=50_000,
            seed=789,
        )

        # Compute per-step log-returns and flatten across sims & steps
        log_ret = np.diff(np.log(paths), axis=2)  # (n_sims, 2, n_steps)
        r0 = log_ret[:, 0, :].flatten()
        r1 = log_ret[:, 1, :].flatten()
        realized_corr = np.corrcoef(r0, r1)[0, 1]

        assert abs(realized_corr - target_corr) < 0.02


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_price(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_paths(
                S0=np.array([-100.0]),
                mu=np.array([0.05]),
                sigma=np.array([0.2]),
            )

    def test_zero_price(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_paths(
                S0=np.array([0.0]),
                mu=np.array([0.05]),
                sigma=np.array([0.2]),
            )

    def test_negative_volatility(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_paths(
                S0=np.array([100.0]),
                mu=np.array([0.05]),
                sigma=np.array([-0.2]),
            )

    def test_zero_volatility(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_paths(
                S0=np.array([100.0]),
                mu=np.array([0.05]),
                sigma=np.array([0.0]),
            )

    def test_zero_horizon(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_paths(
                S0=np.array([100.0]),
                mu=np.array([0.05]),
                sigma=np.array([0.2]),
                T=0.0,
            )

    def test_dt_exceeds_horizon(self):
        with pytest.raises(ValueError, match="must not exceed"):
            simulate_paths(
                S0=np.array([100.0]),
                mu=np.array([0.05]),
                sigma=np.array([0.2]),
                T=1 / 252,
                dt=2 / 252,
            )

    def test_zero_simulations(self):
        with pytest.raises(ValueError, match="at least 1"):
            simulate_paths(
                S0=np.array([100.0]),
                mu=np.array([0.05]),
                sigma=np.array([0.2]),
                n_simulations=0,
            )

    def test_mismatched_mu_length(self):
        with pytest.raises(ValueError, match="mu length"):
            simulate_paths(
                S0=np.array([100.0, 50.0]),
                mu=np.array([0.05]),
                sigma=np.array([0.2, 0.3]),
            )

    def test_mismatched_sigma_length(self):
        with pytest.raises(ValueError, match="sigma length"):
            simulate_paths(
                S0=np.array([100.0, 50.0]),
                mu=np.array([0.05, 0.08]),
                sigma=np.array([0.2]),
            )

    def test_non_square_corr_matrix(self):
        with pytest.raises(ValueError):
            simulate_paths(
                S0=np.array([100.0, 50.0]),
                mu=np.array([0.05, 0.08]),
                sigma=np.array([0.2, 0.3]),
                corr_matrix=np.array([[1.0, 0.5]]),
            )

    def test_asymmetric_corr_matrix(self):
        with pytest.raises(ValueError, match="symmetric"):
            simulate_paths(
                S0=np.array([100.0, 50.0]),
                mu=np.array([0.05, 0.08]),
                sigma=np.array([0.2, 0.3]),
                corr_matrix=np.array([[1.0, 0.5], [0.3, 1.0]]),
            )

    def test_non_pd_corr_matrix(self):
        with pytest.raises(ValueError, match="positive-definite"):
            simulate_paths(
                S0=np.array([100.0, 50.0]),
                mu=np.array([0.05, 0.08]),
                sigma=np.array([0.2, 0.3]),
                corr_matrix=np.array([[1.0, 1.5], [1.5, 1.0]]),
            )

    def test_empty_assets(self):
        with pytest.raises(ValueError, match="at least one"):
            simulate_paths(
                S0=np.array([]),
                mu=np.array([]),
                sigma=np.array([]),
            )
