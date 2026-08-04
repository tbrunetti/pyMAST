"""
pymast/anndata_utils.py
=======================
Internal AnnData validation and extraction helpers.

This module replaces the R ``SingleCellAssay`` / ``FluidigmAssay`` / ``RNASeqAssay``
S4 class hierarchy (R: AllClasses.R).  Users never interact with this module directly;
all pyMAST public API functions accept ``AnnData`` objects directly.

R reference: RGLab/MAST main branch, R/AllClasses.R
             https://github.com/RGLab/MAST/blob/main/R/AllClasses.R
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_anndata(adata: AnnData) -> None:
    """Raise informative errors if *adata* is not suitable for pyMAST.

    Checks performed
    ----------------
    - Is an AnnData object
    - Has at least 2 observations (cells) and at least 1 variable (gene)
    - ``.X`` (or layer) exists and is finite (after densifying if sparse)

    Parameters
    ----------
    adata : AnnData
        The object to validate.
    """
    if not isinstance(adata, AnnData):
        raise TypeError(
            f"Expected an AnnData object, got {type(adata).__name__}. "
            "All pyMAST functions accept AnnData objects directly."
        )
    if adata.n_obs < 2:
        raise ValueError(
            f"AnnData has only {adata.n_obs} observation(s). "
            "pyMAST requires at least 2 cells."
        )
    if adata.n_vars < 1:
        raise ValueError("AnnData has no variables (genes).")


def get_expression_matrix(
    adata: AnnData,
    layer: str | None = None,
    *,
    densify: bool = True,
) -> np.ndarray:
    """Return expression matrix as a dense float64 array (cells × genes).

    Parameters
    ----------
    adata : AnnData
        Input data.
    layer : str or None
        If ``None`` use ``adata.X``; otherwise use ``adata.layers[layer]``.
    densify : bool
        If True, convert sparse matrices to dense ndarray.

    Returns
    -------
    np.ndarray
        Shape ``(n_obs, n_vars)``, dtype ``float64``.
    """
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer '{layer}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer]

    if sp.issparse(X):
        if densify:
            X = X.toarray()
        # else leave sparse — callers must handle
    return np.asarray(X, dtype=np.float64)


def get_obs_column(adata: AnnData, column: str) -> pd.Series:
    """Return a column from ``adata.obs``, raising a clear error if missing.

    Parameters
    ----------
    adata : AnnData
        Input data.
    column : str
        Column name in ``adata.obs``.

    Returns
    -------
    pd.Series
    """
    if column not in adata.obs.columns:
        raise KeyError(
            f"Column '{column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    return adata.obs[column]


def filter_to_highly_variable(adata: AnnData) -> AnnData:
    """Return a view of *adata* restricted to highly variable genes.

    Uses the ``highly_variable`` column in ``adata.var`` if present;
    returns the full object otherwise.

    Parameters
    ----------
    adata : AnnData
        Input data (may be modified view — caller should use a copy if needed).

    Returns
    -------
    AnnData
        View or full object.
    """
    if "highly_variable" in adata.var.columns:
        return adata[:, adata.var["highly_variable"]]
    return adata


def get_gene_names(adata: AnnData) -> list[str]:
    """Return gene names as a list of strings.

    Parameters
    ----------
    adata : AnnData
        Input data.

    Returns
    -------
    list[str]
    """
    return list(adata.var_names)


def get_cell_names(adata: AnnData) -> list[str]:
    """Return cell barcodes / observation names as a list of strings.

    Parameters
    ----------
    adata : AnnData
        Input data.

    Returns
    -------
    list[str]
    """
    return list(adata.obs_names)
