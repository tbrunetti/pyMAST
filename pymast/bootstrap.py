"""
pymast/bootstrap.py
===================
Bootstrap ZLM fitting for uncertainty quantification.

Reimplements R MAST's ``bootZlm()`` function from ``ZlmFit-bootstrap.R``.
Performs parametric bootstrap resampling of cells, refitting the ZLM model
``n_boot`` times to estimate sampling distributions of ZLM statistics.

Bootstrap results feed into :func:`pymast.gsea.gsea_after_boot` for
bootstrap-based Gene Set Enrichment Analysis.

R reference: RGLab/MAST main branch, R/ZlmFit-bootstrap.R
             https://github.com/RGLab/MAST/blob/main/R/ZlmFit-bootstrap.R

Citation: Efron B & Tibshirani RJ (1994). An Introduction to the Bootstrap.
Chapman & Hall. DOI: 10.1007/978-1-4899-4541-9
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from tqdm import tqdm

from .zlm import zlm


# ---------------------------------------------------------------------------
# BootZlmResult
# ---------------------------------------------------------------------------

@dataclass
class BootZlmResult:
    """Result of bootstrapped ZLM fitting.

    Attributes
    ----------
    coef_C_boot : np.ndarray
        Bootstrapped continuous coefficients (n_boot × n_genes × n_params).
    coef_D_boot : np.ndarray
        Bootstrapped discrete coefficients (n_boot × n_genes × n_params).
    logfc_boot : np.ndarray
        Bootstrapped log fold changes (n_boot × n_genes).
    gene_names : list[str]
        Gene names.
    coef_names : list[str]
        Coefficient names.
    contrast : np.ndarray
        The contrast used.
    n_boot : int
        Number of bootstrap iterations.
    """
    coef_C_boot: np.ndarray
    coef_D_boot: np.ndarray
    logfc_boot: np.ndarray
    gene_names: list[str]
    coef_names: list[str]
    contrast: np.ndarray
    n_boot: int


# ---------------------------------------------------------------------------
# boot_zlm
# R: ZlmFit-bootstrap.R — bootZlm(zlmfit, hypothesis, ...)
# ---------------------------------------------------------------------------

def boot_zlm(
    adata: AnnData,
    formula: str = "~ group + cdr",
    contrast: np.ndarray | None = None,
    *,
    n_boot: int = 500,
    seed: int = 42,
    layer: str | None = None,
    method: str = "bayesglm",
    n_jobs: int = 1,
    verbose: bool = True,
) -> BootZlmResult:
    """Bootstrap ZLM fits for uncertainty quantification and GSEA.

    Resamples cells with replacement ``n_boot`` times, refitting the ZLM
    model each time to obtain a sampling distribution of the ZLM statistics
    (primarily log fold changes for each gene).

    R: ZlmFit-bootstrap.R ``bootZlm(zlmfit, hypothesis, R=n_boot, ...)``
       Uses non-parametric bootstrap (resample cells with replacement).

    Parameters
    ----------
    adata : AnnData
        Input data (same as used for the original fit).
    formula : str
        Model formula (same as used for the original fit).
    contrast : np.ndarray or None
        Contrast vector (n_params,) or matrix (n_params, k) specifying what
        to test. If ``None``, uses the second column of the identity contrast.
    n_boot : int
        Number of bootstrap iterations (R default: 500).
    seed : int
        Random seed for reproducibility.
    layer : str or None
        Expression layer.
    method : str
        Discrete fitter method: ``"bayesglm"`` or ``"glm"``.
    n_jobs : int
        Parallelism within each ZLM fit (passed to :func:`~pymast.zlm.zlm`).
    verbose : bool
        Show progress bar.

    Returns
    -------
    BootZlmResult
    """
    rng = np.random.default_rng(seed)
    n_obs = adata.n_obs

    # Fit original model to get structure
    if verbose:
        print("[boot_zlm] Fitting original ZLM for reference structure...")
    original_fit = zlm(
        adata,
        formula=formula,
        layer=layer,
        method=method,
        n_jobs=n_jobs,
        verbose=False,
    )

    n_genes = original_fit.n_genes
    n_params = original_fit.n_params
    gene_names = original_fit.gene_names
    coef_names = original_fit.coef_names

    if contrast is None:
        # Default: test second parameter (first non-intercept)
        c = np.zeros(n_params)
        c[1] = 1.0
        contrast = c

    contrast_arr = np.asarray(contrast, dtype=np.float64)
    if contrast_arr.ndim == 1:
        contrast_arr = contrast_arr[:, np.newaxis]

    # Storage arrays: (n_boot, n_genes, n_params)
    coef_C_boot = np.zeros((n_boot, n_genes, n_params))
    coef_D_boot = np.zeros((n_boot, n_genes, n_params))
    logfc_boot = np.zeros((n_boot, n_genes))

    boot_iter = tqdm(range(n_boot), desc="Bootstrapping ZLM", unit="boot") if verbose else range(n_boot)

    for b in boot_iter:
        # R: ZlmFit-bootstrap.R — sample(1:nrow, replace=TRUE)
        idx = rng.integers(0, n_obs, size=n_obs)
        adata_boot = adata[idx].copy()

        try:
            boot_fit = zlm(
                adata_boot,
                formula=formula,
                layer=layer,
                method=method,
                use_ebayes=False,   # Skip ebayes in bootstrap for speed
                n_jobs=n_jobs,
                verbose=False,
            )
            coef_C_boot[b] = boot_fit.coef_C
            coef_D_boot[b] = boot_fit.coef_D

            # Compute logFC for this bootstrap replicate
            logfc_df = boot_fit.get_log_fc(contrast_arr)
            logfc_boot[b] = logfc_df["log2FC"].values

        except Exception:
            # On failure, use original coefficients as fallback
            coef_C_boot[b] = original_fit.coef_C
            coef_D_boot[b] = original_fit.coef_D
            logfc_boot[b] = np.nan

    return BootZlmResult(
        coef_C_boot=coef_C_boot,
        coef_D_boot=coef_D_boot,
        logfc_boot=logfc_boot,
        gene_names=gene_names,
        coef_names=coef_names,
        contrast=contrast_arr,
        n_boot=n_boot,
    )
