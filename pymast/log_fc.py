"""
pymast/log_fc.py
================
Log fold change computation from fitted ZLM results.

Reimplements the R MAST ``getLogFC()`` function from ``ZlmFit-logFC.R``,
which extracts log fold changes from the continuous component coefficients
and the discrete component (detection rate).

R reference: RGLab/MAST main branch, R/ZlmFit-logFC.R
             https://github.com/RGLab/MAST/blob/main/R/ZlmFit-logFC.R

Citation: Love MI, Huber W, Anders S (2014). Moderated estimation of fold
change and dispersion for RNA-seq data with DESeq2.
Genome Biology 15:550. DOI: 10.1186/s13059-014-0550-8
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

# ---------------------------------------------------------------------------
# get_log_fc
# R: ZlmFit-logFC.R — getLogFC(zlmfit, contrast)
# ---------------------------------------------------------------------------

def get_log_fc(
    coef_C: np.ndarray,
    coef_D: np.ndarray,
    contrast: np.ndarray,
    coef_names: list[str],
    gene_names: list[str],
) -> pd.DataFrame:
    """Compute log fold changes from ZLM fitted coefficients.

    Computes:
    - **Continuous logFC** (``logFC_C``): the contrast applied to the
      continuous component coefficients. Since the continuous component
      is linear on log-scale, this is directly the log fold change in
      expression among expressing cells.
    - **Discrete logFC** (``logFC_D``): change in log-odds of detection.
      The fold change in detection probability is computed from the logistic
      model coefficients.
    - **Combined logFC** (``logFC``): weighted combination following the
      MAST definition: ``logFC = logFC_C + log2(p1) - log2(p0)``
      where p0, p1 are predicted detection probabilities in reference and
      test groups.

    R: ZlmFit-logFC.R ``getLogFC(object, contrast)`` — applies contrast
       to coefficients of continuous and discrete components.

    Parameters
    ----------
    coef_C : np.ndarray
        Continuous component coefficient matrix (n_genes × n_params).
    coef_D : np.ndarray
        Discrete component coefficient matrix (n_genes × n_params).
    contrast : np.ndarray
        Contrast vector or matrix (n_params,) or (n_params, k).
        For a simple two-group test, this is a (n_params, 1) column vector.
    coef_names : list[str]
        Names of the coefficients (columns of coef_C, coef_D).
    gene_names : list[str]
        Names of genes (rows of coef_C, coef_D).

    Returns
    -------
    pd.DataFrame
        Indexed by gene names with columns:
        ``["logFC_C", "logFC_D", "log2FC", "intercept_C", "intercept_D"]``.
    """
    contrast = np.asarray(contrast, dtype=np.float64)
    coef_C = np.asarray(coef_C, dtype=np.float64)  # (n_genes, n_params)
    coef_D = np.asarray(coef_D, dtype=np.float64)  # (n_genes, n_params)

    if contrast.ndim == 1:
        contrast = contrast[:, np.newaxis]

    # R: ZlmFit-logFC.R L~10: logFC.C <- coef.C %*% contrast
    # Continuous component logFC = linear contrast of continuous coefs
    logfc_C = (coef_C @ contrast).squeeze()   # (n_genes,)

    # R: ZlmFit-logFC.R L~15: logFC.D <- coef.D %*% contrast
    # Discrete component: contrast on log-odds scale
    logfc_D = (coef_D @ contrast).squeeze()   # (n_genes,)

    # Intercept index (first coefficient is assumed to be the intercept)
    intercept_idx = coef_names.index("Intercept") if "Intercept" in coef_names else 0

    intercept_C = coef_C[:, intercept_idx]   # (n_genes,)
    intercept_D = coef_D[:, intercept_idx]   # (n_genes,)

    # Predicted detection probability in reference group (contrast = 0):
    # p0 = expit(intercept_D)
    # Predicted detection in test group (contrast applied):
    # p1 = expit(intercept_D + logFC_D)
    # R: ZlmFit-logFC.R L~22-28
    p0 = expit(intercept_D)
    p1 = expit(intercept_D + logfc_D)

    # Combined logFC on log2 scale
    # R: ZlmFit-logFC.R: logFC = logFC.C / log(2) + log2(p1) - log2(p0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = (
            logfc_C / np.log(2.0)
            + np.where(p1 > 0, np.log2(np.clip(p1, 1e-15, None)), -np.inf)
            - np.where(p0 > 0, np.log2(np.clip(p0, 1e-15, None)), -np.inf)
        )

    return pd.DataFrame(
        {
            "logFC_C": logfc_C,     # continuous component log fold change (natural)
            "logFC_D": logfc_D,     # discrete log-odds change
            "log2FC": log2fc,       # combined log2 fold change (scanpy compatible)
            "intercept_C": intercept_C,
            "intercept_D": intercept_D,
        },
        index=gene_names,
    )
