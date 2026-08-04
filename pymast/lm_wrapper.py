"""
pymast/lm_wrapper.py
====================
Abstract base class and utilities for MAST linear model wrappers.

Provides the ``LMlike`` ABC that both ``BayesGLMLike`` (discrete/logistic)
and ``GLMLike`` (discrete/logistic) and the continuous OLS fitter inherit
from, plus helper functions for chi-squared statistics and Wald tests.

R reference: RGLab/MAST main branch, R/LmWrapper.R
             https://github.com/RGLab/MAST/blob/main/R/LmWrapper.R

Citation: McDavid A, Finak G, et al. (2013). Data exploration, quality control
and testing in single-cell qPCR-based gene expression experiments.
Bioinformatics 29(4):461–467. DOI: 10.1093/bioinformatics/btt087
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# LMlike — abstract base class
# R: LmWrapper.R — LMlike virtual class
# ---------------------------------------------------------------------------

class LMlike(ABC):
    """Abstract base class for MAST linear model wrappers.

    Defines the interface that all model wrappers (discrete/continuous) must
    implement so they can be used interchangeably in ``ZlmFitter``.

    R: LmWrapper.R ``LMlike`` virtual S4 class.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coef_names: list[str] | None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ):
        """Fit the model and return a result object with .coef, .vcov, .loglik."""
        ...

    @property
    @abstractmethod
    def family(self) -> str:
        """Return the model family: 'binomial' or 'gaussian'."""
        ...


# ---------------------------------------------------------------------------
# OLSResult — result from the continuous component OLS fit
# ---------------------------------------------------------------------------

@dataclass
class OLSResult:
    """Result of an OLS (linear regression) fit for the continuous component.

    Attributes
    ----------
    coef : np.ndarray
        OLS coefficients.
    coef_names : list[str]
        Coefficient names.
    vcov : np.ndarray
        Covariance matrix of coefficients.
    residuals : np.ndarray
        Residuals for expressing cells.
    sigma2 : float
        Residual variance estimate.
    df_resid : int
        Residual degrees of freedom.
    loglik : float
        Log-likelihood at fitted values.
    converged : bool
        Always True for OLS.
    n_obs : int
        Number of expressing cells used.
    """
    coef: np.ndarray
    coef_names: list[str]
    vcov: np.ndarray
    residuals: np.ndarray
    sigma2: float
    df_resid: int
    loglik: float
    converged: bool
    n_obs: int


# ---------------------------------------------------------------------------
# OLSLike — continuous component: standard OLS via numpy
# R: LmWrapper.R — LMlike subclass for the continuous component
# ---------------------------------------------------------------------------

class OLSLike:
    """Ordinary least squares for the continuous component.

    Fits ``X_gi | Z_gi=1 ~ X_i[pos] @ beta_C`` using ``numpy.linalg.lstsq``,
    matching the R ``lm()`` call for the continuous component of ZLM.

    R: LmWrapper.R — the continuous component uses base R ``lm()``; Python
       equivalent is ``numpy.linalg.lstsq``.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coef_names: list[str] | None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> OLSResult:
        """Fit OLS.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_expressing × n_params).
        y : np.ndarray
            Log-expression among expressing cells (n_expressing,).
        coef_names : list[str] or None
            Coefficient names.
        sample_weight : np.ndarray or None
            Not used (OLS is unweighted). Kept for interface compatibility.

        Returns
        -------
        OLSResult
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_obs, n_params = X.shape

        if coef_names is None:
            coef_names = [f"x{i}" for i in range(n_params)]

        # R: LmWrapper.R — object@fitC <- lm(y ~ X - 1)
        # numpy lstsq: minimize ||Xb - y||^2
        coef, residuals_sq, rank, _ = np.linalg.lstsq(X, y, rcond=None)

        y_hat = X @ coef
        resid = y - y_hat
        df_resid = max(n_obs - n_params, 0)
        sigma2 = float(np.sum(resid ** 2) / df_resid) if df_resid > 0 else 0.0

        # Covariance matrix of OLS: sigma2 * (X'X)^{-1}
        XtX = X.T @ X
        try:
            vcov = sigma2 * np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            vcov = sigma2 * np.linalg.pinv(XtX)

        # Log-likelihood: Gaussian log-likelihood
        # R: LmWrapper.R — logLik.lm uses: -n/2 * log(2*pi*sigma2) - SS/(2*sigma2)
        ss = float(np.sum(resid ** 2))
        if sigma2 > 0 and n_obs > 0:
            loglik = (
                -n_obs / 2.0 * np.log(2.0 * np.pi * sigma2)
                - ss / (2.0 * sigma2)
            )
        else:
            loglik = -np.inf

        return OLSResult(
            coef=coef,
            coef_names=list(coef_names),
            vcov=vcov,
            residuals=resid,
            sigma2=sigma2,
            df_resid=df_resid,
            loglik=float(loglik),
            converged=True,
            n_obs=n_obs,
        )


# ---------------------------------------------------------------------------
# make_chisq_table
# R: LmWrapper.R — makeChiSqTable(lambda_C, lambda_D, df_C, df_D)
# ---------------------------------------------------------------------------

@dataclass
class ChiSqTable:
    """Chi-squared test table for the combined hurdle LRT.

    Contains statistics for the continuous, discrete, and combined (hurdle)
    components.
    """
    lambda_C: float   # continuous LR statistic
    lambda_D: float   # discrete LR statistic
    lambda_hurdle: float   # combined hurdle = lambda_C + lambda_D
    df_C: int         # continuous df
    df_D: int         # discrete df
    df_hurdle: int    # combined df = df_C + df_D
    p_C: float        # continuous p-value
    p_D: float        # discrete p-value
    p_hurdle: float   # combined hurdle p-value


def make_chisq_table(
    lambda_C: float,
    lambda_D: float,
    df_C: int,
    df_D: int,
) -> ChiSqTable:
    """Build the chi-squared table for a hurdle LRT result.

    The combined hurdle statistic is the sum of the continuous and discrete
    LR statistics, tested against a chi-squared with combined df.

    R: LmWrapper.R ``makeChiSqTable()`` — lines ~105-127.
       ``lambda.hurdle <- lambda.C + lambda.D``
       ``df.hurdle <- df.C + df.D``
       ``p.hurdle <- 1 - pchisq(lambda.hurdle, df.hurdle)``

    Parameters
    ----------
    lambda_C : float
        LR statistic for the continuous (Gaussian) component.
    lambda_D : float
        LR statistic for the discrete (binomial) component.
    df_C : int
        Degrees of freedom for continuous component.
    df_D : int
        Degrees of freedom for discrete component.

    Returns
    -------
    ChiSqTable
    """
    # R: LmWrapper.R L~105: lambda.hurdle <- lambda.C + lambda.D
    lambda_hurdle = lambda_C + lambda_D
    df_hurdle = df_C + df_D

    # R: LmWrapper.R L~112: 1 - pchisq(lambda, df)  = sf(lambda, df)
    p_C = float(chi2.sf(lambda_C, df=df_C)) if df_C > 0 and np.isfinite(lambda_C) else np.nan
    p_D = float(chi2.sf(lambda_D, df=df_D)) if df_D > 0 and np.isfinite(lambda_D) else np.nan
    p_hurdle = (
        float(chi2.sf(lambda_hurdle, df=df_hurdle))
        if df_hurdle > 0 and np.isfinite(lambda_hurdle)
        else np.nan
    )

    return ChiSqTable(
        lambda_C=float(lambda_C),
        lambda_D=float(lambda_D),
        lambda_hurdle=float(lambda_hurdle),
        df_C=df_C,
        df_D=df_D,
        df_hurdle=df_hurdle,
        p_C=p_C,
        p_D=p_D,
        p_hurdle=p_hurdle,
    )


# ---------------------------------------------------------------------------
# wald_test
# R: LmWrapper.R — .waldTest(contrast, coef, vcov)
# ---------------------------------------------------------------------------

def wald_test(
    contrast: np.ndarray,
    coef: np.ndarray,
    vcov: np.ndarray,
) -> tuple[float, int, float]:
    """Compute the Wald test statistic for a linear contrast.

    Tests ``H0: L' @ beta = 0`` where ``L`` is the contrast matrix.

    Wald statistic: ``lambda = (L' @ beta)' @ (L' @ Sigma @ L)^{-1} @ (L' @ beta)``

    R: LmWrapper.R ``.waldTest(contrast, coef, vcov)`` — lines ~142-167.

    Parameters
    ----------
    contrast : np.ndarray
        Contrast matrix L of shape (n_params, k). Each column is a contrast.
    coef : np.ndarray
        Fitted coefficient vector (n_params,).
    vcov : np.ndarray
        Variance-covariance matrix (n_params × n_params).

    Returns
    -------
    tuple[float, int, float]
        ``(lambda_, df, p_value)``
    """
    contrast = np.asarray(contrast, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    vcov = np.asarray(vcov, dtype=np.float64)

    # contr = L' @ beta  (k,)
    # R: LmWrapper.R L~142: contrC <- t(contrast) %*% coef_C
    contr = contrast.T @ coef

    # contrCov = L' @ Sigma @ L  (k × k)
    # R: LmWrapper.R L~143: contrCovC <- t(contrast) %*% vcov_C %*% contrast
    contr_cov = contrast.T @ vcov @ contrast

    df = contrast.shape[1]   # number of contrasts

    try:
        # lambda = contr' @ inv(contrCov) @ contr
        # R: LmWrapper.R L~145: lambda_C <- t(contrC) %*% solve(contrCovC) %*% contrC
        lambda_ = float(contr @ np.linalg.solve(contr_cov, contr))
    except np.linalg.LinAlgError:
        lambda_ = float(contr @ np.linalg.lstsq(contr_cov, contr, rcond=None)[0])

    p_value = float(chi2.sf(lambda_, df=df)) if np.isfinite(lambda_) else np.nan
    return lambda_, df, p_value


# ---------------------------------------------------------------------------
# rotate_model_matrix
# R: LmWrapper.R — .rotateMM(mm, testInd)
# ---------------------------------------------------------------------------

def rotate_model_matrix(
    X: np.ndarray,
    test_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate/permute a design matrix to put test columns last.

    Used internally to construct the reduced model by removing test columns,
    then the full model matrix.  Equivalent to R's ``.rotateMM()`` which
    re-orders columns so tested coefficients are last.

    R: LmWrapper.R ``.rotateMM(mm, testInd)`` — lines ~70-90.

    Parameters
    ----------
    X : np.ndarray
        Full design matrix (n_obs × n_params).
    test_indices : list[int]
        Column indices corresponding to the tested coefficients.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X_null, X_full)`` where ``X_null`` is the design matrix with test
        columns removed and ``X_full`` is the original full design matrix.
    """
    all_indices = list(range(X.shape[1]))
    null_indices = [i for i in all_indices if i not in test_indices]
    X_null = X[:, null_indices]
    return X_null, X
