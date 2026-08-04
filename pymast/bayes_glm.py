"""
pymast/bayes_glm.py
===================
Bayesian logistic regression with Cauchy prior — used for the discrete
(detection) component of the MAST hurdle model.

Reimplements R's ``arm::bayesglm()`` Cauchy-prior penalized logistic
regression via Iteratively Reweighted Least Squares (IRLS) with a Cauchy
prior augmentation on each coefficient.

The algorithm is equivalent to the ``BayesGLMlike`` S4 class in R MAST,
which wraps ``arm::bayesglm``.

R reference: RGLab/MAST main branch, R/lmWrapper-bayesglm.R
             https://github.com/RGLab/MAST/blob/main/R/lmWrapper-bayesglm.R

Citation: Gelman A, Jakulin A, Pittau MG, Su Y-S (2008). A weakly informative
default prior distribution for logistic and other regression models.
Annals of Applied Statistics 2(4):1360–1383.
DOI: 10.1214/08-AOAS161
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit  # sigmoid / logistic function

# ---------------------------------------------------------------------------
# BayesGLMResult — result container
# ---------------------------------------------------------------------------

@dataclass
class BayesGLMResult:
    """Result of a Bayesian logistic regression fit.

    Attributes
    ----------
    coef : np.ndarray
        Fitted coefficient vector (length n_params).
    coef_names : list[str]
        Names of coefficients (e.g. from patsy design).
    vcov : np.ndarray
        Variance-covariance matrix of coefficients (n_params × n_params).
    loglik : float
        Log-likelihood at the fitted coefficients.
    converged : bool
        Whether IRLS converged within ``max_iter`` iterations.
    n_iter : int
        Number of IRLS iterations run.
    n_obs : int
        Number of observations used in fitting.
    """
    coef: np.ndarray
    coef_names: list[str]
    vcov: np.ndarray
    loglik: float
    converged: bool
    n_iter: int
    n_obs: int


# ---------------------------------------------------------------------------
# BayesGLMLike — Cauchy-prior logistic regression via IRLS
# R: lmWrapper-bayesglm.R — BayesGLMlike S4 class
# ---------------------------------------------------------------------------

class BayesGLMLike:
    """Bayesian logistic regression (Cauchy prior) via IRLS.

    Implements the ``BayesGLMlike`` class from R MAST which wraps
    ``arm::bayesglm(family=binomial)``.  This class performs penalized
    logistic regression where each coefficient is penalized by an independent
    Cauchy prior (Gelman et al. 2008 default: scale = 2.5 for all predictors).

    The IRLS algorithm augments the working response and weights with a
    Cauchy prior contribution on each coefficient at each iteration.

    Parameters
    ----------
    prior_scale : float
        Scale of the Cauchy prior on each coefficient. R/arm default: 2.5.
    prior_scale_intercept : float
        Scale of the Cauchy prior on the intercept. R/arm default: 10.0.
    max_iter : int
        Maximum IRLS iterations. R/arm default: 100.
    tol : float
        Convergence tolerance on coefficient change. R/arm default: 1e-8.
    """

    def __init__(
        self,
        prior_scale: float = 2.5,
        prior_scale_intercept: float = 10.0,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        self.prior_scale = prior_scale
        self.prior_scale_intercept = prior_scale_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coef_names: list[str] | None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> BayesGLMResult:
        """Fit Bayesian logistic regression with Cauchy prior.

        Implements the Gelman et al. (2008) IRLS-with-prior algorithm:
        at each iteration, the prior is incorporated as a pseudo-observation
        appended to the working response vector with prior-determined weight.

        R: lmWrapper-bayesglm.R ``BayesGLMlike@.fit()``
           arms the standard IRLS (glm binomial family) with Cauchy prior
           augmentation on each coefficient.

        R: arm/bayesglm.fit() — iterates IRLS with:
           ``prior.weights[k] = 1 / (coef[k]^2 + scale[k]^2)``  (Cauchy kernel)

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_obs × n_params). Typically from patsy.
        y : np.ndarray
            Binary response (0 / 1) for each observation.
        coef_names : list[str] or None
            Names for each coefficient column.
        sample_weight : np.ndarray or None
            Per-observation weights. None means uniform weight 1.

        Returns
        -------
        BayesGLMResult
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_obs, n_params = X.shape

        if sample_weight is None:
            W_obs = np.ones(n_obs, dtype=np.float64)
        else:
            W_obs = np.asarray(sample_weight, dtype=np.float64)

        if coef_names is None:
            coef_names = [f"x{i}" for i in range(n_params)]

        # Build prior scales per coefficient
        # R: arm/bayesglm.fit() — scales[1] = prior.scale.for.intercept (default 10)
        #                          scales[-1] = prior.scale (default 2.5)
        scales = np.full(n_params, self.prior_scale, dtype=np.float64)
        # Assume intercept is first column (all-ones) — standard design
        if n_params > 0:
            scales[0] = self.prior_scale_intercept

        # Initialize coefficients at 0
        # R: arm/bayesglm.fit() — coef.old <- rep(0, ncol(X))
        beta = np.zeros(n_params, dtype=np.float64)

        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            # ---- Standard logistic IRLS update ----
            # Predicted probabilities: mu = sigma(X @ beta)
            # R: arm/bayesglm.fit() — mu <- binomial()$linkinv(eta)
            eta = X @ beta
            mu = expit(eta)                              # sigmoid
            mu = np.clip(mu, 1e-8, 1 - 1e-8)           # numerical stability

            # IRLS weights: W = W_obs * mu * (1 - mu)
            # R: arm/bayesglm.fit() — W <- diag(weights * mu * (1-mu))
            W = W_obs * mu * (1.0 - mu)

            # Working response: z = eta + (y - mu) / (mu * (1 - mu))
            # R: arm/bayesglm.fit() — z <- eta + (y - mu) / (mu * (1-mu))
            z = eta + (y - mu) / np.where(W > 0, mu * (1.0 - mu), 1.0)

            # ---- Cauchy prior augmentation ----
            # R: arm/bayesglm.fit() L~170:
            #    prior.weight[k] <- 1 / (coef[k]^2 + scale[k]^2)
            # Append n_params pseudo-observations — one per coefficient
            prior_weights = 1.0 / (beta ** 2 + scales ** 2)  # (n_params,)

            # Augmented design: [X; I_p * sqrt(prior_weights)]
            # Augmented response: [z; 0] (prior centers at 0)
            # Augmented weights: [W; prior_weights]
            X_aug = np.vstack([X, np.diag(np.sqrt(prior_weights))])
            z_aug = np.concatenate([z, np.zeros(n_params)])
            W_aug = np.concatenate([W, prior_weights])

            # WLS solve: beta_new = (X_aug.T W_aug X_aug)^-1 X_aug.T W_aug z_aug
            # R: arm/bayesglm.fit() — solve(t(X.aug) %*% diag(W.aug) %*% X.aug,
            #                              t(X.aug) %*% diag(W.aug) %*% z.aug)
            XtW = (X_aug * W_aug[:, np.newaxis]).T          # (n_params, n_aug)
            XtWX = XtW @ X_aug                              # (n_params, n_params)
            XtWz = XtW @ z_aug                              # (n_params,)

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                # Singular — use least-squares fallback
                beta_new, _, _, _ = np.linalg.lstsq(XtWX, XtWz, rcond=None)

            # Convergence check
            # R: arm/bayesglm.fit() — max(abs(coef.new - coef.old)) < tol
            delta = np.max(np.abs(beta_new - beta))
            beta = beta_new

            if delta < self.tol:
                converged = True
                break

        # ---- Compute variance-covariance matrix ----
        # vcov = (X.T W X)^-1  (using original obs, not augmented)
        # R: arm/bayesglm.fit() — vcov <- solve(t(X) %*% diag(W) %*% X)
        eta = X @ beta
        mu = expit(np.clip(eta, -500, 500))
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        W_final = W_obs * mu * (1.0 - mu)
        XtWX_obs = (X * W_final[:, np.newaxis]).T @ X
        try:
            vcov = np.linalg.inv(XtWX_obs)
        except np.linalg.LinAlgError:
            vcov = np.linalg.pinv(XtWX_obs)

        # ---- Log-likelihood ----
        # loglik = sum(y * log(mu) + (1-y) * log(1-mu))
        loglik = float(
            np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu))
        )

        return BayesGLMResult(
            coef=beta,
            coef_names=list(coef_names),
            vcov=vcov,
            loglik=loglik,
            converged=converged,
            n_iter=n_iter,
            n_obs=n_obs,
        )
