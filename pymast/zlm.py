"""
pymast/zlm.py
=============
Main ZLM (Zero-inflated Linear Model) fitting engine.

The ``zlm()`` function is the primary entry point for fitting the MAST hurdle
model gene by gene across all genes in an AnnData object.  It dispatches to
either :class:`~pymast.bayes_glm.BayesGLMLike` (default, Cauchy prior) or
:class:`~pymast.glm_wrapper.GLMLike` for the discrete component, and to
:class:`~pymast.lm_wrapper.OLSLike` for the continuous component.

R reference: RGLab/MAST main branch, R/zeroinf.R
             https://github.com/RGLab/MAST/blob/main/R/zeroinf.R

Citation: Finak G et al. (2015). MAST: a flexible statistical framework.
Genome Biology 16:278. DOI: 10.1186/s13059-015-0844-5
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import patsy
from anndata import AnnData
from joblib import Parallel, delayed
from tqdm import tqdm

from .anndata_utils import (
    filter_to_highly_variable,
    get_expression_matrix,
    validate_anndata,
)
from .bayes_glm import BayesGLMLike
from .ebayes import GeneSufficientStats, ebayes, get_ss_g_r_ng
from .glm_wrapper import GLMLike
from .lm_wrapper import OLSLike
from .utils import compute_cdr
from .zlm_fit import ZlmFit


# ---------------------------------------------------------------------------
# _fit_one_gene — fit a single gene's hurdle model
# ---------------------------------------------------------------------------

def _fit_one_gene(
    g: int,
    y: np.ndarray,
    X_full: np.ndarray,
    X_null: np.ndarray,
    coef_names: list[str],
    null_coef_names: list[str],
    discrete_fitter,
    n_params: int,
) -> dict:
    """Fit discrete and continuous components for one gene.

    Internal worker function dispatched via joblib. Returns a dict of
    per-gene statistics collected into ``ZlmFit``.

    R: zeroinf.R ``.zlm()`` — per-gene fit loop.
    """
    n_params_null = X_null.shape[1]

    # ---- Detection indicator ----
    # R: zeroinf.R — Zg <- y > 0
    z = (y > 0).astype(float)

    # ---- Discrete component (binomial) ----
    # R: zeroinf.R — fitD <- glm/bayesglm(Zg ~ X, family=binomial)
    try:
        fit_D_full = discrete_fitter.fit(X_full, z, coef_names=coef_names)
        fit_D_null = discrete_fitter.fit(X_null, z, coef_names=null_coef_names)
        coef_D = fit_D_full.coef
        vcov_D = fit_D_full.vcov
        loglik_D = fit_D_full.loglik
        loglik_D_null = fit_D_null.loglik
        conv_D = fit_D_full.converged
    except Exception:
        coef_D = np.zeros(n_params)
        vcov_D = np.zeros((n_params, n_params))
        loglik_D = -np.inf
        loglik_D_null = -np.inf
        conv_D = False

    # ---- Continuous component (Gaussian) ----
    # R: zeroinf.R — fitC <- lm(y[Zg > 0] ~ X[Zg > 0,])
    pos_mask = z > 0
    n_expressing = int(pos_mask.sum())
    ols_fitter = OLSLike()

    coef_C = np.zeros(n_params)
    vcov_C = np.zeros((n_params, n_params))
    loglik_C = -np.inf
    loglik_C_null = -np.inf
    residuals = np.full(n_expressing, np.nan)
    sigma2 = np.nan
    df_resid_C = 0
    conv_C = False

    if n_expressing > n_params:
        X_pos = X_full[pos_mask]
        y_pos = y[pos_mask]
        X_null_pos = X_null[pos_mask]

        try:
            fit_C_full = ols_fitter.fit(X_pos, y_pos, coef_names=coef_names)
            fit_C_null = ols_fitter.fit(X_null_pos, y_pos, coef_names=null_coef_names)
            coef_C = fit_C_full.coef
            vcov_C = fit_C_full.vcov
            loglik_C = fit_C_full.loglik
            loglik_C_null = fit_C_null.loglik
            residuals = fit_C_full.residuals
            sigma2 = fit_C_full.sigma2
            df_resid_C = fit_C_full.df_resid
            conv_C = fit_C_full.converged
        except Exception:
            pass

    # df for LRT = n_params_full - n_params_null
    df_diff = n_params - n_params_null

    return {
        "coef_C": coef_C,
        "coef_D": coef_D,
        "vcov_C": vcov_C,
        "vcov_D": vcov_D,
        "loglik_C": loglik_C,
        "loglik_D": loglik_D,
        "loglik_C_null": loglik_C_null,
        "loglik_D_null": loglik_D_null,
        "df_C": df_diff,
        "df_D": df_diff,
        "conv_C": conv_C,
        "conv_D": conv_D,
        "n_expressing": n_expressing,
        "residuals": residuals,
        "sigma2": sigma2,
        "df_resid_C": df_resid_C,
    }


# ---------------------------------------------------------------------------
# ZlmFitter — dispatches method selection
# R: zeroinf.R — methodDict
# ---------------------------------------------------------------------------

class ZlmFitter:
    """Dispatches discrete component fitter based on method name.

    R: zeroinf.R ``methodDict`` — maps method string to glm/bayesglm class.

    Parameters
    ----------
    method : str
        ``"bayesglm"`` (default) or ``"glm"``.
    """

    METHOD_DICT: dict[str, type] = {
        "bayesglm": BayesGLMLike,
        "glm": GLMLike,
    }

    def __init__(self, method: Literal["bayesglm", "glm"] = "bayesglm") -> None:
        if method not in self.METHOD_DICT:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {list(self.METHOD_DICT.keys())}."
            )
        self.method = method
        self._fitter = self.METHOD_DICT[method]()

    @property
    def discrete_fitter(self):
        return self._fitter


# ---------------------------------------------------------------------------
# zlm — main entry point
# R: zeroinf.R — zlm(formula, sca, method, ...)
# ---------------------------------------------------------------------------

def zlm(
    adata: AnnData,
    formula: str = "~ group + cdr",
    *,
    contrast_column: str | None = None,
    layer: str | None = None,
    method: Literal["bayesglm", "glm"] = "bayesglm",
    use_highly_variable: bool = True,
    use_ebayes: bool = True,
    n_jobs: int = -1,
    verbose: bool = True,
    cdr_key: str = "cdr",
) -> ZlmFit:
    """Fit the MAST hurdle model gene-wise via ZLM.

    This is the core ``zlm()`` function, mirroring the R MAST interface.
    For each gene, fits:
    - **Discrete component**: Bayesian or standard logistic regression on the
      detection indicator ``Z_gi = 1(X_gi > 0)`` vs. null model without the
      test contrast.
    - **Continuous component**: OLS on log-expression among expressing cells
      vs. null model without test contrast.

    The null model omits the ``contrast_column`` term from the formula.
    If ``contrast_column`` is ``None``, the last term in the formula is used.

    R: zeroinf.R ``zlm(formula, sca, method, ...)`` — gene-wise loop with
       ``mapply(.zlm, ...)`` dispatching per gene.

    Parameters
    ----------
    adata : AnnData
        Input data. Must have CDR in ``adata.obs[cdr_key]`` (or it will be
        computed automatically).
    formula : str
        Patsy formula for the linear model. Must start with ``~``.
        E.g. ``"~ group + cdr"`` or ``"~ condition + batch + cdr"``.
    contrast_column : str or None
        The covariate to test (the column dropped to form the null model).
        If ``None``, inferred as the first non-intercept, non-CDR term.
    layer : str or None
        Expression layer (``None`` = ``adata.X``).
    method : str
        Discrete component fitter: ``"bayesglm"`` (default) or ``"glm"``.
    use_highly_variable : bool
        If ``True`` and ``adata.var["highly_variable"]`` exists, restrict
        fitting to HVGs.
    use_ebayes : bool
        If ``True``, apply empirical Bayes shrinkage to the continuous
        component variance estimates.
    n_jobs : int
        Parallelism. ``-1`` = all cores. ``1`` = single-threaded.
    verbose : bool
        Show progress bar.
    cdr_key : str
        Column name in ``adata.obs`` for CDR covariate.

    Returns
    -------
    ZlmFit
        Fitted ZlmFit object with per-gene coefficients, log-likelihoods,
        and convergence flags.
    """
    validate_anndata(adata)

    # Compute CDR if not present
    if cdr_key not in adata.obs.columns:
        compute_cdr(adata, layer=layer, key_added=cdr_key, inplace=True)

    # Optionally restrict to HVGs
    if use_highly_variable:
        adata_use = filter_to_highly_variable(adata)
    else:
        adata_use = adata

    # Expression matrix (cells × genes)
    X_expr = get_expression_matrix(adata_use, layer=layer)  # (n_obs, n_genes)
    gene_names = list(adata_use.var_names)
    n_genes = len(gene_names)

    # ---- Build design matrices via patsy ----
    # R: zeroinf.R — model.matrix(formula, data=colData(sca))
    obs_data = adata_use.obs.copy()

    try:
        # Full model
        _, X_full_df = patsy.dmatrices(formula + " - 1", data=obs_data, return_type="dataframe")
    except patsy.PatsyError as e:
        raise ValueError(
            f"Formula '{formula}' could not be parsed against adata.obs columns. "
            f"Available columns: {list(obs_data.columns)}\nPatsy error: {e}"
        ) from e

    coef_names_full = list(X_full_df.columns)
    X_full = X_full_df.values

    # Add intercept explicitly (patsy dmatrices with -1 removes it; re-add)
    if "Intercept" not in coef_names_full:
        X_full = np.column_stack([np.ones(X_full.shape[0]), X_full])
        coef_names_full = ["Intercept"] + coef_names_full

    n_params = X_full.shape[1]

    # Null model: drop contrast_column term
    if contrast_column is None:
        # Infer: last term that is not Intercept or cdr
        non_cdr_coefs = [
            c for c in coef_names_full
            if c != "Intercept" and cdr_key not in c
        ]
        contrast_column = non_cdr_coefs[-1] if non_cdr_coefs else coef_names_full[-1]

    null_col_mask = np.array(
        [contrast_column not in c for c in coef_names_full], dtype=bool
    )
    X_null = X_full[:, null_col_mask]
    null_coef_names = [c for c, keep in zip(coef_names_full, null_col_mask) if keep]

    # ---- ZlmFitter ----
    fitter = ZlmFitter(method=method)
    discrete_fitter = fitter.discrete_fitter

    # ---- Gene-wise fitting ----
    # R: zeroinf.R — mapply(.zlm, ...)
    if verbose:
        gene_iter = tqdm(range(n_genes), desc="Fitting ZLM", unit="genes")
    else:
        gene_iter = range(n_genes)

    if n_jobs == 1 or n_genes < 50:
        # Single-threaded (for small n_genes or debug)
        results = []
        for g in gene_iter:
            y = X_expr[:, g]
            res = _fit_one_gene(
                g, y, X_full, X_null, coef_names_full, null_coef_names,
                discrete_fitter, n_params,
            )
            results.append(res)
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one_gene)(
                g, X_expr[:, g], X_full, X_null, coef_names_full,
                null_coef_names, discrete_fitter, n_params,
            )
            for g in gene_iter
        )

    # ---- Assemble ZlmFit arrays ----
    coef_C   = np.vstack([r["coef_C"]   for r in results])   # (n_genes, n_params)
    coef_D   = np.vstack([r["coef_D"]   for r in results])
    vcov_C   = np.stack( [r["vcov_C"]   for r in results])   # (n_genes, n_params, n_params)
    vcov_D   = np.stack( [r["vcov_D"]   for r in results])
    loglik_C      = np.array([r["loglik_C"]      for r in results])
    loglik_D      = np.array([r["loglik_D"]      for r in results])
    loglik_C_null = np.array([r["loglik_C_null"] for r in results])
    loglik_D_null = np.array([r["loglik_D_null"] for r in results])
    df_C          = np.array([r["df_C"]          for r in results])
    df_D          = np.array([r["df_D"]          for r in results])
    conv_C        = np.array([r["conv_C"]         for r in results])
    conv_D        = np.array([r["conv_D"]         for r in results])
    n_expressing  = np.array([r["n_expressing"]  for r in results])

    # ---- Empirical Bayes shrinkage ----
    # R: zeroinf.R — ebayes(zlmfit, ...)
    if use_ebayes:
        residuals_mat = np.full((X_full.shape[0], n_genes), np.nan)
        df_resid_arr = np.array([r["df_resid_C"] for r in results], dtype=float)
        for g, r in enumerate(results):
            pos_count = r["n_expressing"]
            if pos_count > 0 and len(r["residuals"]) == pos_count:
                # Place residuals back into full cell positions
                pos_idx = np.where(X_expr[:, g] > 0)[0]
                residuals_mat[pos_idx, g] = r["residuals"]

        eb_stats = get_ss_g_r_ng(residuals_mat, df_resid_arr)
        eb_result = ebayes(eb_stats)

        # Apply shrinkage: replace per-gene sigma2 in vcov_C
        # R: ebayes-helpers.R — vcov_C scaled by posterior sigma2 / sample sigma2
        for g in range(n_genes):
            raw_sigma2 = results[g]["sigma2"]
            if raw_sigma2 is not None and np.isfinite(raw_sigma2) and raw_sigma2 > 0:
                shrink_factor = eb_result.sigma2_post[g] / raw_sigma2
                vcov_C[g] *= shrink_factor

    return ZlmFit(
        gene_names=gene_names,
        coef_names=coef_names_full,
        coef_C=coef_C,
        coef_D=coef_D,
        vcov_C=vcov_C,
        vcov_D=vcov_D,
        loglik_C=loglik_C,
        loglik_D=loglik_D,
        loglik_C_null=loglik_C_null,
        loglik_D_null=loglik_D_null,
        df_C=df_C,
        df_D=df_D,
        converge_C=conv_C,
        converge_D=conv_D,
        n_expressing=n_expressing,
        formula=formula,
    )
