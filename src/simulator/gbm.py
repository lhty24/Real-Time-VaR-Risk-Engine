"""
Geometric Brownian Motion (GBM) market data simulator.

Generates synthetic price paths using the Euler-Maruyama discretization:
    S(t+dt) = S(t) * exp[(mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z]

For correlated multi-asset simulation, standard normals are correlated
via Cholesky decomposition of the correlation matrix.
"""

import numpy as np
from numpy.typing import NDArray


def _validate_inputs(
    S0: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    corr_matrix: NDArray[np.float64] | None,
    T: float,
    dt: float,
    n_simulations: int,
) -> None:
    """Validate all inputs to simulate_paths."""
    n_assets = len(S0)

    if n_assets == 0:
        raise ValueError("S0 must contain at least one asset.")

    if len(mu) != n_assets:
        raise ValueError(
            f"mu length ({len(mu)}) must match number of assets ({n_assets})."
        )

    if len(sigma) != n_assets:
        raise ValueError(
            f"sigma length ({len(sigma)}) must match number of assets ({n_assets})."
        )

    if np.any(S0 <= 0):
        raise ValueError("All initial prices (S0) must be positive.")

    if np.any(sigma <= 0):
        raise ValueError("All volatilities (sigma) must be positive.")

    if T <= 0:
        raise ValueError("Time horizon (T) must be positive.")

    if dt <= 0:
        raise ValueError("Time step (dt) must be positive.")

    if dt > T:
        raise ValueError(
            f"Time step (dt={dt}) must not exceed time horizon (T={T})."
        )

    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    if corr_matrix is not None:
        if corr_matrix.ndim != 2:
            raise ValueError("Correlation matrix must be 2-dimensional.")

        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"Correlation matrix shape {corr_matrix.shape} must match "
                f"({n_assets}, {n_assets})."
            )

        if not np.allclose(corr_matrix, corr_matrix.T):
            raise ValueError("Correlation matrix must be symmetric.")

        if not np.allclose(np.diag(corr_matrix), 1.0):
            raise ValueError("Correlation matrix diagonal must be all ones.")

        try:
            np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix must be positive-definite.")


def simulate_paths(
    S0: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    corr_matrix: NDArray[np.float64] | None = None,
    T: float = 1.0,
    dt: float = 1.0,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Simulate correlated GBM price paths for multiple assets.

    Parameters
    ----------
    S0 : array of shape (n_assets,)
        Initial prices for each asset.
    mu : array of shape (n_assets,)
        Annualized drift (expected return) for each asset.
    sigma : array of shape (n_assets,)
        Annualized volatility for each asset.
    corr_matrix : array of shape (n_assets, n_assets), optional
        Correlation matrix. If None, assets are simulated independently.
    T : float
        Time horizon in years (e.g., 1/252 for 1 trading day).
    dt : float
        Time step in years (e.g., 1/252 for daily steps).
    n_simulations : int
        Number of Monte Carlo paths to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    paths : ndarray of shape (n_simulations, n_assets, n_steps + 1)
        Simulated price paths. paths[:, :, 0] == S0.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if corr_matrix is not None:
        corr_matrix = np.asarray(corr_matrix, dtype=np.float64)

    _validate_inputs(S0, mu, sigma, corr_matrix, T, dt, n_simulations)

    n_assets = len(S0)
    n_steps = int(T / dt)

    rng = np.random.default_rng(seed)

    # Draw iid standard normals: (n_simulations, n_steps, n_assets)
    Z = rng.standard_normal((n_simulations, n_steps, n_assets))

    # Correlate normals via Cholesky decomposition
    if corr_matrix is not None and n_assets > 1:
        L = np.linalg.cholesky(corr_matrix)
        Z = Z @ L.T  # (n_simulations, n_steps, n_assets)

    # GBM drift and diffusion terms
    drift = (mu - 0.5 * sigma**2) * dt  # (n_assets,)
    diffusion = sigma * np.sqrt(dt)  # (n_assets,)

    # Log-returns for each step: (n_simulations, n_steps, n_assets)
    log_returns = drift + diffusion * Z

    # Cumulative sum of log-returns, prepend zeros for initial prices
    cum_log_returns = np.cumsum(log_returns, axis=1)
    zeros = np.zeros((n_simulations, 1, n_assets))
    cum_log_returns = np.concatenate([zeros, cum_log_returns], axis=1)

    # Price paths: S(t) = S0 * exp(cumulative log-returns)
    paths = S0 * np.exp(cum_log_returns)

    # Transpose to (n_simulations, n_assets, n_steps + 1)
    paths = np.transpose(paths, (0, 2, 1))

    return paths
