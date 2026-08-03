"""
pymast/utils.py
===============
Utility functions: Cellular Detection Rate (CDR), frequency tables,
and Fluidigm qPCR Ct → Et conversion.

R reference: RGLab/MAST main branch, R/UtilityFunctions.R
             https://github.com/RGLab/MAST/blob/main/R/UtilityFunctions.R
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .anndata_utils import get_expression_matrix


# ---------------------------------------------------------------------------
# Cellular Detection Rate (CDR)
# R: UtilityFunctions.R — colSums(assay > 0) / nrow(assay)
# ---------------------------------------------------------------------------

def compute_cdr(
    adata: AnnData,
    layer: str | None = None,
    *,
    key_added: str = "cdr",
    inplace: bool = True,
) -> np.ndarray | None:
    """Compute the Cellular Detection Rate (CDR) for each cell.

    CDR is defined as the fraction of genes detected (expression > 0) per cell.
    It is used as a nuisance covariate in the MAST hurdle model to control for
    global differences in library complexity.

    R: UtilityFunctions.R — ``colSums(assay > 0) / nrow(assay)``
    (note: R MAST uses genes as rows; in Python/AnnData genes are columns)

    Parameters
    ----------
    adata : AnnData
        Input single-cell data.
    layer : str or None
        Expression layer to use. ``None`` uses ``adata.X``.
    key_added : str
        Column name to add to ``adata.obs``. Default ``"cdr"``.
    inplace : bool
        If ``True``, add CDR to ``adata.obs[key_added]`` and return ``None``.
        If ``False``, return the CDR array without modifying ``adata``.

    Returns
    -------
    np.ndarray or None
        CDR values (one per cell) if ``inplace=False``, else ``None``.
    """
    X = get_expression_matrix(adata, layer=layer)  # (n_obs, n_vars)
    # CDR = fraction of genes with expression > 0 per cell
    cdr = (X > 0).sum(axis=1) / X.shape[1]         # shape: (n_obs,)
    cdr = cdr.astype(np.float64)

    if inplace:
        adata.obs[key_added] = cdr
        return None
    return cdr


# ---------------------------------------------------------------------------
# Frequency table
# R: UtilityFunctions.R — freq()
# ---------------------------------------------------------------------------

def freq(adata: AnnData, layer: str | None = None) -> pd.DataFrame:
    """Compute the fraction of cells expressing each gene.

    Equivalent to the R MAST ``freq()`` function which returns the frequency
    (proportion of non-zero cells) per gene.

    R: UtilityFunctions.R ``freq()`` — ``rowMeans(assay > 0)``

    Parameters
    ----------
    adata : AnnData
        Input data.
    layer : str or None
        Expression layer to use. ``None`` uses ``adata.X``.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame indexed by gene names with column ``"freq"``.
    """
    X = get_expression_matrix(adata, layer=layer)   # (n_obs, n_vars)
    gene_freq = (X > 0).mean(axis=0)               # fraction per gene
    return pd.DataFrame({"freq": gene_freq}, index=list(adata.var_names))


# ---------------------------------------------------------------------------
# Fluidigm Ct → Et conversion
# R: UtilityFunctions.R — computeEtFromCt()
# ---------------------------------------------------------------------------

def compute_et_from_ct(
    ct: np.ndarray | pd.DataFrame,
    ct_threshold: float = 40.0,
    et_threshold: float = 0.0,
) -> np.ndarray | pd.DataFrame:
    """Convert Fluidigm Ct (cycle threshold) values to Et (expression threshold) values.

    The conversion is: ``Et = (ct_threshold - Ct)``, then values below
    ``et_threshold`` are set to ``et_threshold`` (i.e. clipped at zero by default).

    Used for Fluidigm qPCR data prior to MAST analysis.
    **Lowest priority feature** — all returned values should be loaded into AnnData
    before passing to pyMAST analysis functions.

    R: UtilityFunctions.R ``computeEtFromCt()``
       ``etFromCt <- threshold - ctMat``
       ``etFromCt[etFromCt < et.threshold] <- et.threshold``

    Parameters
    ----------
    ct : np.ndarray or pd.DataFrame
        Matrix of Ct values (cells × genes). Missing values (no amplification)
        should be represented as ``ct_threshold``.
    ct_threshold : float
        The Ct threshold used in qPCR (default: 40 cycles).
        R default: 40.
    et_threshold : float
        Minimum Et value after transformation (default: 0.0).
        R default: 0.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Et values in the same format as input (preserves DataFrame structure).
    """
    is_df = isinstance(ct, pd.DataFrame)
    ct_arr = np.asarray(ct, dtype=np.float64)

    # R: UtilityFunctions.R L~45: etFromCt <- threshold - ctMat
    et = ct_threshold - ct_arr

    # R: UtilityFunctions.R L~46: etFromCt[etFromCt < et.threshold] <- et.threshold
    et = np.clip(et, a_min=et_threshold, a_max=None)

    if is_df:
        return pd.DataFrame(et, index=ct.index, columns=ct.columns)
    return et


# ---------------------------------------------------------------------------
# thresholdSCRNA — soft thresholding for scRNA-seq
# R: UtilityFunctions.R — thresholdSCRNA()
# ---------------------------------------------------------------------------

def threshold_scRNA(
    X: np.ndarray,
    *,
    min_threshold: float = 0.0,
    hard: bool = True,
) -> np.ndarray:
    """Apply detection threshold to a scRNA-seq expression matrix.

    Equivalent to R ``thresholdSCRNA()`` which sets very low expression values
    (below the threshold) to the threshold floor, effectively treating them as
    undetected.

    R: UtilityFunctions.R ``thresholdSCRNA()``

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (cells × genes).
    min_threshold : float
        Minimum expression value. Values below this are set to it.
    hard : bool
        If ``True``, apply hard thresholding (clip). If ``False``, soft threshold
        (subtract threshold, clip at 0).

    Returns
    -------
    np.ndarray
        Thresholded expression matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    if hard:
        return np.clip(X, a_min=min_threshold, a_max=None)
    else:
        return np.clip(X - min_threshold, a_min=0.0, a_max=None)
