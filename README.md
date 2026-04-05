# Real-Time VaR Risk Engine

A production-grade Value at Risk engine computing portfolio risk via Monte Carlo simulation, exposed through FastAPI. Implements three VaR methods (Monte Carlo, Parametric, Historical) plus Expected Shortfall (CVaR), designed with bank-grade auditability, reproducibility, and regulatory alignment.

## Features

- **Monte Carlo VaR** — Cholesky-correlated GBM simulation across multi-asset portfolios
- **Parametric VaR** — Closed-form variance-covariance method for fast validation
- **Historical VaR** — Distribution-free empirical simulation
- **Expected Shortfall (CVaR)** — Basel III/IV-aligned coherent risk measure
- **Backtesting** — Kupiec test, Christoffersen test, Basel traffic-light classification
- **Audit logging** — Immutable append-only logs for full reproducibility

## Tech Stack

- **Python** — Core language
- **FastAPI** — Async API layer
- **NumPy / SciPy** — Monte Carlo simulation, statistical tests, linear algebra
- **Pandas** — Time series processing
- **PostgreSQL** — Persistent storage (portfolios, results, audit logs)

## Architecture

```
Market Data Simulator → Data Processing → VaR Engine → FastAPI API
                                              ↓
                                     Audit & Logging → PostgreSQL
```

**Key principle:** The computation layer is stateless. All inputs and seeds are logged for exact reproducibility.

### Components

1. **Market Data Simulator** — GBM-based synthetic price generation with seed reproducibility
2. **Data Processing** — Log-returns, annualized statistics, correlation matrix estimation
3. **VaR Engine** — `simulate_paths()` → `calculate_pnl()` → `compute_var()` returning (VaR, ES) tuples
4. **API Layer** — `POST /portfolio`, `GET /var`, `GET /backtest`, `GET /health`
5. **Audit & Logging** — Immutable logs of all inputs, seeds, and model versions
6. **Storage** — PostgreSQL for portfolios, historical results, and audit logs
7. **Backtesting** — Statistical validation of model accuracy

## Development Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | GBM simulator, Monte Carlo/Parametric VaR, ES, FastAPI endpoints | In progress |
| **Phase 2** | PostgreSQL, audit logging, portfolio CRUD, backtesting, Historical VaR | Planned |
| **Phase 3** | Variance reduction, fat-tailed distributions, stress testing | Planned |
| **Phase 4** | Kafka, Redis, Streamlit dashboard, monitoring | Planned |

## Documentation

- **Design document:** [`docs/design-doc.md`](docs/design-doc.md) — Full mathematical model, architecture, and roadmap
