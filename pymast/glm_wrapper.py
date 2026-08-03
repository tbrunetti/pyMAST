"""
pymast/glm_wrapper.py
=====================
Standard (non-Bayesian) logistic regression wrapper for the discrete
component of the MAST hurdle model.

Wraps ``statsmodels`` GLM (binomial family) to provide the same interface
as :class:`pymast.bayes_glm.BayesGLMLike`, so the two can be used
interchangeably in ``ZlmFitter``.

R reference: RGLab/MAST main branch, R/lmWrapper-glm.R
             https://github.com/RGLab/MAST/blob/main/R/lmWrapper-glm.R

Citation: Nelder JA & Wedderburn RWM (1972). Generalized Linear Models.
J Royal Statistical Society A 135(3):370–384. DOI: 10.2307/2344614
Statsmodels: Seabold & Perktold (2010). Statsmodels: Econometric and
Statistical Modeling with Python. Proc 9th Python in Science Conf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# GLMResult — mirrors BayesGLMResult interface
# ---------------------------------------------------------------------------

@dataclass
class GLMResult:
    """Result of a standard logistic regression fit (GLM binomial).

    Attributes
    ----------
    coef : np.ndarray
        Fitted coefficients.
    coef_names : list[str]
        Coefficient names.
    vcov : np.ndarray
        Variance-covariance matrix.
    loglik : float
        Log-likelihood at fitted coefficients.
    converged : bool
        Whether the GLM optimizer converged.
    n_iter : int
        Number of IRLS iterations used.
    n_obs : int
        Number of observations.
    """
    coef: np.ndarray
    coef_names: list[str]
    vcov: np.ndarray
    loglik: float
    converged: bool
    n_iter: int
    n_obs: int


# ---------------------------------------------------------------------------
# GLMLike — standard statsmodels logit wrapper
# R: lmWrapper-glm.R — GLMlike S4 class
# ---------------------------------------------------------------------------

class GLMLike:
    """Standard logistic regression via statsmodels (GLM binomial).

    Provides the same ``fit()`` interface as :class:`pymast.bayes_glm.BayesGLMLike`
    so either can be used as the discrete-component fitter in ``ZlmFitter``.

    R: lmWrapper-glm.R ``GLMlike`` — wraps ``glm(family=binomial)`` from base R.

    Parameters
    ----------
    max_iter : int
        Maximum IRLS iterations. Passed to statsmodels. Default: 100.
    tol : float
        Convergence tolerance. Default: 1e-8.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coef_names: list[str] | None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> GLMResult:
        """Fit standard logistic regression.

        R: lmWrapper-glm.R ``GLMlike@.fit()`` — ``glm(y ~ X - 1, family=binomial)``
           (design matrix already includes intercept if desired).

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_obs × n_params).
        y : np.ndarray
            Binary response (0 / 1).
        coef_names : list[str] or None
            Names for the coefficients.
        sample_weight : np.ndarray or None
            Per-observation weights. None means uniform weight 1.

        Returns
        -------
        GLMResult
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_obs, n_params = X.shape

        if coef_names is None:
            coef_names = [f"x{i}" for i in range(n_params)]

        try:
            # R: lmWrapper-glm.R — glm(family=binomial, x=TRUE, y=TRUE)
            model = sm.GLM(
                y,
                X,
                family=sm.families.Binomial(),
                freq_weights=sample_weight,
            )
            result = model.fit(
                maxiter=self.max_iter,
                tol=self.tol,
                disp=False,
            )

            coef = np.asarray(result.params, dtype=np.float64)
            vcov = np.asarray(result.cov_params(), dtype=np.float64)
            loglik = float(result.llf)
            converged = result.converged
            n_iter = getattr(result, "fit_history", {}).get("iteration", self.max_iter)
            if not isinstance(n_iter, int):
                n_iter = self.max_iter

        except Exception:
            # Fallback: return zero coefficients on fitting failure
            coef = np.zeros(n_params, dtype=np.float64)
            vcov = np.zeros((n_params, n_params), dtype=np.float64)
            loglik = -np.inf
            converged = False
            n_iter = 0

        return GLMResult(
            coef=coef,
            coef_names=list(coef_names),
            vcov=vcov,
            loglik=loglik,
            converged=converged,
            n_iter=n_iter,
            n_obs=n_obs,
        )
