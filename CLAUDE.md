# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time Value at Risk (VaR) engine computing portfolio risk via Monte Carlo simulation, exposed through FastAPI. Implements three VaR methods (Monte Carlo, Parametric, Historical) plus Expected Shortfall (CVaR). Designed to mimic bank-grade risk infrastructure with auditability, reproducibility, and regulatory alignment.

## Tech Stack

- **Language:** Python
- **API:** FastAPI
- **Compute:** NumPy / SciPy (simulation, statistical tests, linear algebra), Pandas (time series)
- **Storage:** PostgreSQL (portfolios, results, audit logs)
- **Infra (Phase 3+):** Docker, Redis (caching), Kafka (streaming), Kubernetes
- **UI (Phase 4):** Streamlit

## Architecture

**Mathematical core:** GBM price dynamics → Cholesky-correlated Monte Carlo paths → P&L distribution → VaR/ES at configurable confidence and horizon.

Seven components:

1. **Market Data Simulator** — GBM-based synthetic price generation with seed reproducibility
2. **Data Processing** — log-returns, annualized stats, correlation matrix estimation
3. **VaR Engine** — `simulate_paths()` → `calculate_pnl()` → `compute_var()` returning (VaR, ES) tuples. Also: `parametric_var()` and `historical_var()` for cross-validation
4. **API Layer (FastAPI)** — `POST /portfolio`, `GET /var`, `GET /backtest`, `GET /health`
5. **Audit & Logging** — immutable append-only logs of all inputs, seeds, model versions
6. **Storage (PostgreSQL)** — portfolios, historical results, audit logs
7. **Backtesting Module** — Kupiec test, Christoffersen test, Basel traffic-light classification

Key principle: computation layer is **stateless**. All inputs + seeds logged for exact reproducibility.

## Development Phases

- **Phase 1 (MVP):** GBM simulator, Monte Carlo VaR + ES, Parametric VaR, FastAPI endpoints
- **Phase 2:** PostgreSQL, audit logging, portfolio CRUD, backtesting framework, Historical VaR
- **Phase 3:** Variance reduction, fat-tailed distributions, stress testing, sensitivity analysis
- **Phase 4:** Kafka, Redis, Streamlit dashboard, monitoring

## Reference

- Design document: `docs/design-doc.md`
- Task planning: `docs/plans/`
