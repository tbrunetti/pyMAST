"""
pymast/ebayes.py
================
Empirical Bayes variance shrinkage for the continuous (Gaussian) component
of the MAST hurdle model.

Reimplements the R MAST empirical Bayes functions from ``ebayes-helpers.R``
using only ``scipy`` and ``numpy`` — zero R dependencies.

The prior on gene-level variances is: ``sigma^2_g ~ Inverse-Gamma(a0, b0)``
where the hyperparameters ``(a0, b0)`` are estimated by marginal maximum
likelihood via L-BFGS-B optimization.

R reference: RGLab/MAST main branch, R/ebayes-helpers.R
             https://github.com/RGLab/MAST/blob/main/R/ebayes-helpers.R

Citation: Smyth GK (2004). Linear models and empirical Bayes methods for
assessing differential expression in microarray experiments.
Statistical Applications in Genetics and Molecular Biology 3(1):3.
DOI: 10.2202/1544-6115.1027
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma


# ---------------------------------------------------------------------------
# Data class for per-gene sufficient statistics
# ---------------------------------------------------------------------------

@dataclass
class GeneSufficientStats:
    """Sufficient statistics for the continuous component of one gene.

    Parameters
    ----------
    ss_g : np.ndarray
        Per-gene sum of squares of residuals (length n_genes).
    r_ng : np.ndarray
        Per-gene residual degrees of freedom (length n_genes).
    """
    ss_g: np.ndarray    # sum of squares per gene
    r_ng: np.ndarray    # residual df per gene


# ---------------------------------------------------------------------------
# get_ss_g_r_ng
# R: ebayes-helpers.R — getSSg_rNg(object)
# ---------------------------------------------------------------------------

def get_ss_g_r_ng(
    residuals: np.ndarray,
    df_resid: np.ndarray,
) -> GeneSufficientStats:
    """Extract per-gene sum of squares and residual degrees of freedom.

    R: ebayes-helpers.R ``getSSg_rNg(object)``
       ``SSg <- colSums(resid^2, na.rm=TRUE)``
       ``rNg <- colSums(!is.na(resid)) - p``

    Parameters
    ----------
    residuals : np.ndarray
        Residual matrix (cells × genes) from OLS fits of the continuous
        component (NaN for cells with zero expression).
    df_resid : np.ndarray
        Per-gene residual degrees of freedom vector (length n_genes).
        Typically ``n_expressing_cells_g - n_covariates``.

    Returns
    -------
    GeneSufficientStats
    """
    # R: ebayes-helpers.R L~3: SSg <- colSums(resid^2, na.rm=TRUE)
    ss_g = np.nansum(residuals ** 2, axis=0)       # sum of squared residuals per gene

    # R: ebayes-helpers.R L~4: rNg <- colSums(!is.na(resid)) - p
    r_ng = np.asarray(df_resid, dtype=np.float64)

    return GeneSufficientStats(ss_g=ss_g, r_ng=r_ng)


# ---------------------------------------------------------------------------
# get_marginal_hyperlikelihood
# R: ebayes-helpers.R — getMarginalHyperLikelihood(a0, b0, SSg, rNg)
# ---------------------------------------------------------------------------

def get_marginal_hyperlikelihood(
    a0: float,
    b0: float,
    ss_g: np.ndarray,
    r_ng: np.ndarray,
) -> float:
    """Compute the total marginal log-likelihood for hyperparameters (a0, b0).

    For each gene g, the marginal log-likelihood is:
    ``L_g = -lbeta(rNg/2, a0) - rNg/2 * log(b0)``
    ``       - log(1 + SSg/(2*b0)) * (rNg/2 + a0)``

    The total is the sum over all genes with positive rNg.

    R: ebayes-helpers.R ``getMarginalHyperLikelihood(a0, b0, SSg, rNg)``
       Lines ~6-18.

    Parameters
    ----------
    a0 : float
        Inverse-Gamma shape hyperparameter.
    b0 : float
        Inverse-Gamma scale hyperparameter.
    ss_g : np.ndarray
        Per-gene sum of squared residuals.
    r_ng : np.ndarray
        Per-gene residual degrees of freedom.

    Returns
    -------
    float
        Total marginal log-likelihood (sum over genes).
    """
    # R: ebayes-helpers.R L~6: valid <- rNg > 0
    valid = r_ng > 0
    if not valid.any():
        return 0.0

    rn = r_ng[valid]
    ss = ss_g[valid]

    # R: ebayes-helpers.R L~8-10:
    # Li <- -lbeta(rNg/2, a0) - rNg/2 * log(b0)
    #        - log(1 + SSg/(2*b0)) * (rNg/2 + a0)
    li = (
        -betaln(rn / 2.0, a0)
        - (rn / 2.0) * np.log(b0)
        - np.log(1.0 + ss / (2.0 * b0)) * (rn / 2.0 + a0)
    )
    return float(li.sum())


# ---------------------------------------------------------------------------
# Score functions (gradient of the marginal hyperlikelihood)
# R: ebayes-helpers.R — gradient inside getMarginalHyperLikelihood
# ---------------------------------------------------------------------------

def _marginal_hyperlikelihood_gradient(
    a0: float,
    b0: float,
    ss_g: np.ndarray,
    r_ng: np.ndarray,
) -> tuple[float, float]:
    """Compute analytic gradients of the marginal hyperlikelihood w.r.t. a0, b0.

    R: ebayes-helpers.R L~21-23:
       ``score.a0 <- digamma(rNg/2 + a0) - digamma(a0) - log(1 + SSg/(2*b0))``
       ``score.b0 <- (a0*SSg - rNg*b0) / (SSg*b0 + 2*b0^2)``

    Returns
    -------
    tuple[float, float]
        ``(grad_a0, grad_b0)`` — partial derivatives.
    """
    valid = r_ng > 0
    if not valid.any():
        return 0.0, 0.0

    rn = r_ng[valid]
    ss = ss_g[valid]

    # R: ebayes-helpers.R L~21
    score_a0 = digamma(rn / 2.0 + a0) - digamma(a0) - np.log(1.0 + ss / (2.0 * b0))
    # R: ebayes-helpers.R L~22
    score_b0 = (a0 * ss - rn * b0) / (ss * b0 + 2.0 * b0 ** 2)

    return float(score_a0.sum()), float(score_b0.sum())


# ---------------------------------------------------------------------------
# solve_mom — method of moments initialization
# R: ebayes-helpers.R — solveMoM(SSg, rNg)
# ---------------------------------------------------------------------------

def solve_mom(ss_g: np.ndarray, r_ng: np.ndarray) -> tuple[float, float]:
    """Estimate initial hyperparameters (a0, b0) via method of moments.

    Used to initialize the L-BFGS-B optimizer for the marginal hyperlikelihood.

    R: ebayes-helpers.R ``solveMoM(SSg, rNg)``
       Lines ~29-50.

    Parameters
    ----------
    ss_g : np.ndarray
        Per-gene sum of squares.
    r_ng : np.ndarray
        Per-gene residual df.

    Returns
    -------
    tuple[float, float]
        Initial ``(a0, b0)`` estimates.
    """
    # R: ebayes-helpers.R L~30: sigma2g <- SSg / rNg
    valid = r_ng > 0
    if valid.sum() < 2:
        return 1.0, 1.0

    sigma2g = ss_g[valid] / r_ng[valid]
    # Remove non-positive or non-finite
    sigma2g = sigma2g[np.isfinite(sigma2g) & (sigma2g > 0)]

    if len(sigma2g) < 2:
        return 1.0, 1.0

    # R: ebayes-helpers.R L~33-36: method of moments for Inv-Gamma params
    # E[sigma^2] = b0 / (a0 - 1);  Var[sigma^2] = b0^2 / ((a0-1)^2 * (a0-2))
    m = float(np.mean(sigma2g))
    v = float(np.var(sigma2g, ddof=1))

    if v <= 0 or not np.isfinite(v):
        return 1.0, max(m, 1e-4)

    # Inverse-Gamma MoM: a0 = m^2/v + 2; b0 = m*(m^2/v + 1)
    a0_init = max(m ** 2 / v + 2.0, 1.1)   # must be > 1 for finite mean
    b0_init = max(m * (m ** 2 / v + 1.0), 1e-4)

    return float(a0_init), float(b0_init)


# ---------------------------------------------------------------------------
# ebayes — main entry point
# R: ebayes-helpers.R — ebayes(object, ...)
# ---------------------------------------------------------------------------

@dataclass
class EBayesResult:
    """Result of empirical Bayes hyperparameter estimation.

    Attributes
    ----------
    a0 : float
        Estimated Inverse-Gamma shape hyperparameter.
    b0 : float
        Estimated Inverse-Gamma scale hyperparameter.
    v : float
        Prior variance = b0 / a0 (roughly the prior mean of sigma^2).
    df_prior : float
        Prior degrees of freedom = 2 * a0.
    sigma2_post : np.ndarray
        Posterior (shrunk) per-gene variance estimates.
    df_total : np.ndarray
        Total degrees of freedom per gene (prior df + residual df).
    converged : bool
        Whether L-BFGS-B optimizer converged.
    """
    a0: float
    b0: float
    v: float
    df_prior: float
    sigma2_post: np.ndarray
    df_total: np.ndarray
    converged: bool


def ebayes(
    stats: GeneSufficientStats,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> EBayesResult:
    """Estimate empirical Bayes hyperparameters and compute posterior variances.

    Fits the Inverse-Gamma prior on gene-level variances by maximising the
    marginal hyperlikelihood using L-BFGS-B (matching R's ``nlminb`` call).

    Then computes posterior (shrunk) variance estimates by combining the
    prior with the per-gene OLS variance:
    ``sigma2_post_g = (b0 + SS_g/2) / (a0 + rNg/2 - 1)``
    which is the posterior mode of the Inverse-Gamma posterior.

    R: ebayes-helpers.R ``ebayes(object, ...)`` — lines ~80-110.

    Parameters
    ----------
    stats : GeneSufficientStats
        Per-gene sufficient statistics from :func:`get_ss_g_r_ng`.
    max_iter : int
        Maximum iterations for L-BFGS-B. R uses ``nlminb`` with default 200.
    tol : float
        Gradient tolerance. Equivalent to R's ``rel.tol``.

    Returns
    -------
    EBayesResult
    """
    ss_g = stats.ss_g
    r_ng = stats.r_ng

    # Method of moments initialization
    a0_init, b0_init = solve_mom(ss_g, r_ng)

    def neg_ll(params: np.ndarray) -> float:
        """Negative marginal hyperlikelihood (minimization target)."""
        a0, b0 = params
        if a0 <= 0 or b0 <= 0:
            return np.inf
        return -get_marginal_hyperlikelihood(a0, b0, ss_g, r_ng)

    def neg_grad(params: np.ndarray) -> np.ndarray:
        """Gradient of negative marginal hyperlikelihood."""
        a0, b0 = params
        if a0 <= 0 or b0 <= 0:
            return np.array([0.0, 0.0])
        ga, gb = _marginal_hyperlikelihood_gradient(a0, b0, ss_g, r_ng)
        return np.array([-ga, -gb])

    # R: ebayes-helpers.R L~93: nlminb(start=c(a0, b0), objective=negLL, gradient=negGrad,
    #    lower=c(0, 0), control=list(iter.max=200))
    result = minimize(
        neg_ll,
        x0=np.array([a0_init, b0_init]),
        jac=neg_grad,
        method="L-BFGS-B",
        bounds=[(1e-4, None), (1e-4, None)],
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    a0 = float(result.x[0])
    b0 = float(result.x[1])

    # R: ebayes-helpers.R L~99-101:
    # v <- max(th[2]/th[1], 0)      # prior variance estimate
    # df.prior <- max(2*th[1], 0)   # prior degrees of freedom
    v = max(b0 / a0, 0.0)
    df_prior = max(2.0 * a0, 0.0)

    # Posterior (shrunk) variance per gene:
    # sigma2_post_g = (b0 + SS_g/2) / (a0 + rNg/2 - 1)
    # R: ebayes-helpers.R L~105-108
    df_total = df_prior + r_ng
    sigma2_post = np.where(
        r_ng > 0,
        (b0 + ss_g / 2.0) / (a0 + r_ng / 2.0),
        v,   # use prior mean for genes with no observations
    )

    return EBayesResult(
        a0=a0,
        b0=b0,
        v=v,
        df_prior=df_prior,
        sigma2_post=sigma2_post,
        df_total=df_total,
        converged=result.success,
    )
