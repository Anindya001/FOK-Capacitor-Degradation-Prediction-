"""Uncertainty quantification utilities for the FK model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from fractional_model import FKParams, fractional_capacitance, time_to_threshold
from fractional_estimation import FKFitResult


def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -700.0))
    return z / (1.0 + z)


def _xi_to_params(xi: Sequence[float]) -> FKParams:
    kappa, a, b, c0 = xi
    return FKParams(
        C0=math.exp(c0),
        k=math.exp(kappa),
        alpha=min(max(_logistic(a), 1e-6), 1.0 - 1e-6),
        f_inf=min(max(_logistic(b), 1e-8), 1.0 - 1e-6),
    )


def _stabilize_covariance(
    cov: np.ndarray,
    *,
    max_condition: float = 1e5,
    min_variance: float = 1e-10,
) -> np.ndarray:
    """Spectrally regularise ill-conditioned covariance matrices.

    The plateau parameter is often weakly identified on short monotone storage
    records. In those cases the raw inverse Hessian can contain a single
    variance direction that is many orders of magnitude larger than the others,
    producing spurious posterior draws and unrealistically wide predictive
    intervals. We keep the covariance symmetric, bound its condition number, and
    preserve the identifiable directions.
    """

    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.shape != (4, 4):
        raise ValueError("Covariance matrix must be 4x4 for FK parameters.")
    cov_arr = 0.5 * (cov_arr + cov_arr.T)
    eigvals, eigvecs = np.linalg.eigh(cov_arr)
    positive = eigvals[eigvals > 0.0]
    if positive.size == 0:
        raise ValueError("Covariance matrix does not contain any positive eigenvalues.")
    floor = max(float(np.min(positive)), float(min_variance))
    ceiling = floor * float(max_condition)
    clipped = np.clip(eigvals, floor, ceiling)
    stabilized = eigvecs @ np.diag(clipped) @ eigvecs.T
    return 0.5 * (stabilized + stabilized.T)


def laplace_draws(
    fit: FKFitResult,
    n_draws: int = 2000,
    *,
    random_state: int | None = None,
) -> tuple[list[FKParams], np.ndarray]:
    """Generate Laplace approximation draws for parameters and sigma."""
    rng = np.random.default_rng(random_state)
    cov = np.asarray(fit.covariance, dtype=float)
    if cov.shape != (4, 4):
        raise ValueError("Covariance matrix must be 4x4 for FK parameters.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix contains non-finite entries.")
    cov = _stabilize_covariance(cov)

    # Stabilise covariance prior to Cholesky.
    jitter = 1e-12
    for _ in range(5):
        try:
            L = np.linalg.cholesky(cov + np.eye(4) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:  # pragma: no cover
        raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

    draws = rng.standard_normal((n_draws, 4)) @ L.T + fit.transformed
    params_draws = [_xi_to_params(xi) for xi in draws]

    dof = max(1, fit.residuals.size - 4)
    chi2 = rng.chisquare(dof, size=n_draws)
    sigma_draws = fit.sigma * np.sqrt(dof / chi2)

    return params_draws, sigma_draws


@dataclass
class PredictiveResults:
    time: np.ndarray
    mean: np.ndarray
    mu_samples: np.ndarray
    total_samples: np.ndarray


def posterior_predictive(
    params_draws: Sequence[FKParams],
    sigma_draws: Sequence[float],
    t: Iterable[float],
    *,
    random_state: int | None = None,
) -> PredictiveResults:
    """Compute posterior predictive samples for a time grid."""
    t_arr = np.asarray(t, dtype=float)
    n_draws = len(params_draws)
    if len(sigma_draws) != n_draws:
        raise ValueError("params_draws and sigma_draws must have the same length.")

    if n_draws == 0:
        empty = np.empty((0, t_arr.size), dtype=float)
        return PredictiveResults(time=t_arr, mean=np.empty_like(t_arr), mu_samples=empty, total_samples=empty.copy())

    sigma_arr = np.asarray(sigma_draws, dtype=float).reshape(-1)
    if sigma_arr.size != n_draws:
        raise ValueError("sigma_draws must be a flat array matching params_draws length.")

    mu_samples = np.empty((n_draws, t_arr.size), dtype=float)

    try:
        from scipy.special import mittag_leffler as scipy_mittag_leffler  # type: ignore
    except ImportError:
        scipy_mittag_leffler = None  # type: ignore

    computed = False
    if scipy_mittag_leffler is not None:
        try:
            C0 = np.fromiter((theta.C0 for theta in params_draws), dtype=float, count=n_draws)
            k = np.fromiter((theta.k for theta in params_draws), dtype=float, count=n_draws)
            alpha = np.fromiter((theta.alpha for theta in params_draws), dtype=float, count=n_draws)
            f_inf = np.fromiter((theta.f_inf for theta in params_draws), dtype=float, count=n_draws)

            alpha_grid = alpha[:, None]
            t_grid = t_arr[None, :]
            z = -k[:, None] * np.power(t_grid, alpha_grid, where=np.isfinite(t_grid), out=np.zeros_like(t_grid))
            ml_values = scipy_mittag_leffler(z, alpha_grid, 1.0)
            mu_samples[:] = C0[:, None] * (f_inf[:, None] + (1.0 - f_inf[:, None]) * ml_values)
            computed = True
        except Exception:
            computed = False

    if not computed:
        for idx, theta in enumerate(params_draws):
            mu_samples[idx] = fractional_capacitance(t_arr, theta)

    rng = np.random.default_rng(random_state)
    log_mu = np.log(np.clip(mu_samples, 1e-15, np.inf))
    total_samples = np.exp(rng.normal(loc=log_mu, scale=sigma_arr[:, None]))

    mean_curve = mu_samples.mean(axis=0)
    return PredictiveResults(
        time=t_arr,
        mean=mean_curve,
        mu_samples=mu_samples,
        total_samples=total_samples,
    )


def failure_time_samples(
    params_draws: Sequence[FKParams],
    thresholds: Sequence[float],
) -> dict[float, np.ndarray]:
    """Compute failure-time samples for each threshold."""
    if not thresholds:
        return {}
    out: dict[float, list[float]] = {q: [] for q in thresholds}
    for theta in params_draws:
        for q in thresholds:
            try:
                out[q].append(time_to_threshold(theta, q))
            except Exception:
                out[q].append(np.nan)
    return {q: np.asarray(times, dtype=float) for q, times in out.items()}


def mcmc_draws(
    fit: FKFitResult,
    times: Iterable[float],
    values: Iterable[float],
    *,
    n_draws: int = 2000,
    burn_in: int = 500,
    step_scale: float = 0.5,
    random_state: Optional[int] = None,
) -> tuple[list[FKParams], np.ndarray, float]:
    """Simple Metropolis-Hastings sampler over FK parameters and log-sigma.

    Parameters
    ----------
    fit
        Result returned by :func:`fractional_estimation.fit_fractional_model`.
    times, values
        Training data used for fitting. Values must be strictly positive.
    n_draws
        Number of posterior samples to return *after* burn-in.
    burn_in
        Number of initial iterations to discard.
    step_scale
        Scaling applied to the proposal covariance. Increase to explore more
        broadly, decrease to improve acceptance.

    Returns
    -------
    params_draws, sigma_draws, acceptance_rate
        Tuple containing sampled FK parameters, sampled noise scales, and the
        overall acceptance rate for diagnostics.
    """
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative.")

    t_arr = np.asarray(times, dtype=float)
    y_arr = np.asarray(values, dtype=float)
    if t_arr.shape != y_arr.shape:
        raise ValueError("times and values must have identical shapes for MCMC.")
    if t_arr.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if np.any(y_arr <= 0):
        raise ValueError("values must be strictly positive for log-normal noise.")

    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    y_arr = y_arr[order]
    log_y = np.log(y_arr)
    n_obs = log_y.size

    base_cov = np.asarray(fit.covariance, dtype=float)
    if base_cov.shape != (4, 4):
        raise ValueError("Covariance matrix must be 4x4 for FK parameter draws.")
    prop_cov = np.eye(5, dtype=float)
    prop_cov[:4, :4] = base_cov
    prop_cov[4, 4] = max(fit.sigma**2, 1e-6)

    jitter = 1e-10
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(prop_cov + np.eye(5) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:  # pragma: no cover
        raise np.linalg.LinAlgError("Proposal covariance not positive definite.")

    rng = np.random.default_rng(random_state)

    def _log_posterior(state: np.ndarray) -> float:
        xi = state[:4]
        sigma_log = state[4]
        sigma = math.exp(sigma_log)
        if not np.isfinite(sigma) or sigma <= 0:
            return float("-inf")
        params = _xi_to_params(xi)
        try:
            mu = fractional_capacitance(t_arr, params)
        except Exception:
            return float("-inf")
        if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
            return float("-inf")
        log_mu = np.log(np.clip(mu, 1e-15, np.inf))
        resid = log_y - log_mu
        residual_sum = float(np.dot(resid, resid))
        log_lik = (
            -float(np.sum(log_y))
            - n_obs * sigma_log
            - 0.5 * n_obs * math.log(2 * math.pi)
            - 0.5 * residual_sum / (sigma**2)
        )
        log_prior = -0.5 * float(np.dot(xi, xi)) / (5.0**2) - 0.5 * (sigma_log / 2.0) ** 2
        return log_lik + log_prior

    state_dim = 5
    current = np.concatenate([fit.transformed.copy(), [fit.sigma_log]])
    current_lp = _log_posterior(current)
    if not np.isfinite(current_lp):
        raise RuntimeError("Initial log-posterior is not finite; cannot start MCMC.")

    total_iters = n_draws + burn_in
    chain = np.empty((n_draws, state_dim), dtype=float)
    accepted = 0

    for idx in range(total_iters):
        proposal = current + step_scale * (chol @ rng.standard_normal(state_dim))
        prop_lp = _log_posterior(proposal)
        if np.isfinite(prop_lp):
            log_accept = prop_lp - current_lp
            if math.log(rng.random()) < log_accept:
                current = proposal
                current_lp = prop_lp
                accepted += 1
        if idx >= burn_in:
            chain[idx - burn_in] = current

    acceptance_rate = accepted / max(total_iters, 1)
    params_draws = [_xi_to_params(row[:4]) for row in chain]
    sigma_draws = np.exp(chain[:, 4])
    return params_draws, sigma_draws, float(acceptance_rate)


__all__ = [
    "laplace_draws",
    "posterior_predictive",
    "PredictiveResults",
    "failure_time_samples",
    "mcmc_draws",
]
